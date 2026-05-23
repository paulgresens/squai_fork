import os
import re
import time
import json
import random
import plyvel
import requests
import gc
import torch
import io
import fcntl
import numpy as np
from dotenv import load_dotenv
from config import DB_PATH

load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# --- CONFIGURATION ---
SCADS_API_KEY = os.getenv("SCADS_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
INPUT_FILE = "all_paper_ids.txt"
OUTPUT_FILE = "generatedQuestions4.jsonl"
CACHE_FILE="alreadyUsedArxivIds4.txt"
ERROR_CACHE_FILE = "errorAtTheseArxivIds4.txt"
# MODEL = "Qwen/Qwen2.5-72B-Instruct"
# MODEL = "unsloth/Qwen2.5-72B-Instruct"
# MODEL = "meta-llama/Llama-3.3-70B-Instruct" # even worse
#MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct" # best performance it seems
# SCADS_API_MODEL = "openai/gpt-oss-120b"
SCADS_API_MODEL = "moonshotai/Kimi-K2.6"
SEMANTIC_SCHOLAR_API_BLOCK_FILE = "semanticScholarApiLock.txt"
DB_LOCK_FILE = "dbLock.txt"

# MODEL = "unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit"
# JUDGING_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
PAPER_CHARACTER_LIMIT=25000
QUESTIONS_TO_GENERATE = 200



class APILockManager:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.file_obj = None

    def lock(self):
        """Acquires the lock. Will freeze the script here if another script holds it."""
        # Open the file manually and KEEP IT OPEN.
        self.file_obj = open(self.lock_file, "a", encoding="utf-8")
        fcntl.flock(self.file_obj, fcntl.LOCK_EX)
        self.file_obj.write("LOCKED")
        self.file_obj.flush() # Ensure it writes to disk immediately
        print("🔒 Lock acquired.")

    def unlock(self):
        """Explicitly releases the lock so other scripts can proceed."""
        if self.file_obj and not self.file_obj.closed:
            # Release the lock and close the file
            fcntl.flock(self.file_obj, fcntl.LOCK_UN)
            self.file_obj.close()
            self.file_obj = None
            print("🔓 Lock released.")


# Instantiate your lock manager once at the top of your script
api_lock = APILockManager(SEMANTIC_SCHOLAR_API_BLOCK_FILE)
db_lock = APILockManager(DB_LOCK_FILE)

if not os.path.exists(SEMANTIC_SCHOLAR_API_BLOCK_FILE):
    with open(SEMANTIC_SCHOLAR_API_BLOCK_FILE, "a", encoding="utf-8") as f:
        f.write("lock")
if not os.path.exists(DB_LOCK_FILE):
    with open(DB_LOCK_FILE, "a", encoding="utf-8") as f:
        f.write("lock")

PROMPT_TEMPLATE = """
You are generating a scientific 2-hop question-answer-evidence (Q-A) Tuple.
You are given 5 candidate scientific papers in the following format: 
[
  {
    "ArXiv": string,
    "text": string,
  }
]

Your task is to select the best pair of papers and generate ONE question that requires connected reasoning across exactly TWO papers.

PAPER SELECTION
First, select the best pair of papers.
The selected pair must satisfy:
-Paper A provides an intermediate entity, method, dataset, variable, or result
-Paper B uses, evaluates, extends, contrasts with, explains, or depends on that intermediate element
-The final answer requires combining both papers
-Do NOT select papers that are only loosely related or redundant.

REQUIREMENTS
A valid question MUST: 
-require exactly two reasoning steps (2-hop)
-use exactly two supporting papers
-require sequential reasoning:
    -Step 2 must depend on the result of Step 1
    -If Step 1 is removed, Step 2 should not be solvable
-NOT be answerable from either paper alone
-be self-contained and unambiguous
-require combining information across both papers
-only use information that is present in the papers, do not invent hypothetical connections or applications. Only use existing connections.
-question cannot contain explicit references to the papers or its content such as "in this paper", "the proposed methods" or similar

AVOID
Do NOT generate:
-multi-part questions (e.g., "What is X and what is Y?")
-vague or generic questions (e.g., "implications", "advantages")
-literature summary questions
-questions answerable from a single paper
-questions where papers are only topically related but not logically connected
-Do NOT use external scientific knowledge, commonsense assumptions, or unstated domain knowledge, only use the information that is present in the papers

OUTPUT FORMAT
{
    "usedPapers" : [
        {
            "arXiv": <paperId>,
            "role": <bridgeEvidence / bridgeAnswer depending on the papers role> 
        },
        {
            "arXiv": <paperId>,
            "role": <bridgeEvidence/ bridgeAnswer depending on the papers role>
        }
    ],
    "rejectedPapers": [<paperId1>, <paperId2>, <paperId3>],
    "reasoning": {
        "step1": <Explain your reasoning for step1 on paper A>,
        "step2": <use step1 reasoning to derive or support the final answer>,
        "connectionExplanation": <explain why the steps are connected and why step2 requires the result of step1>,
    }, 
    "questionDraft" : "<one clear, self-contained question requiring 2-hop reasoning, this serves as a draft for you final question. You can but don't have to explicitely reference the papers here to help formulate a better question.>",
    "question" : <decontextualize the questionDraft, so that it contains no explicit reference to the papers or external figures. Every reference to a paper should instead directly name the concept, phenomenon, method or similar that you are referring to, briefly describing it if needed. The question cannot contain explicit references to the papers or its content such as "in this paper", "the proposed methods" or similar.>",
    "answerWithPaperReferences" : <long-form paragraph integrating Paper A and Paper B with citations like [Paper A], [Paper B]>,
    "answerWithoutPaperReferences": <long-form paragraph that answers the question, without referencing the papers, but directly incorporatestheir information, so that it is standalone understandable>,
    "isNotSingleHop": <explain why the question you generated is not single hop>
}

Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
Paper Texts:
"""

JUDGING_PROMPT_TEMPLATE ="""
You are evaluating a scientific question-answer (Q-A) example.
Your goal is to assess whether it is a valid 2-hop, evidence-grounded scientific multi-hop question.
A valid example must:
-require combining information from exactly TWO papers
-involve dependent reasoning (Step 2 must require Step 1)
-not be decomposable into independent sub-questions
-be fully answerable from the provided evidence
-be self-contained and unambiguous

IMPORTANT:
-For each criterion, assign one label:
-GOOD = fully satisfies the criterion
-BORDERLINE = partially satisfies the criterion
-BAD = does not satisfy the criterion
-Be strict: only assign GOOD if the criterion is clearly and fully satisfied.

INPUT:
Question: {question} 
Answer: {answer} 
bridgeEvidencePaperText: {paper1Text}
bridgeAnswerPaperText: {paper2Text}
Reasoning Steps: {reasoningSteps}

EVALUATION CRITERIA
A. Reasoning Structure
Multi-hop Validity: Does answering the question require combining both papers?
-GOOD: Both papers are strictly required
-BORDERLINE: Both papers contribute but one may be sufficient
-BAD: Only one paper is sufficient (single-hop)

Dependency Strength: Does Step 2 depend on the result of Step 1?
GOOD: Step 2 strictly requires Step 1
BORDERLINE: Partial dependence
BAD: Steps are independent (disconnected reasoning)

Non-Decomposability: Is the question NOT decomposable into independent sub-questions?
-GOOD: Cannot be split; requires joint reasoning
-BORDERLINE: Partially decomposable
-BAD: Clearly decomposable into independent sub-questions

B. Evidence Grounding: Evidence DistributionAre both papers required and non-redundant?
-GOOD: Each paper contributes distinct, necessary information
-BORDERLINE: Some overlap or redundancy
-BAD: One paper is sufficient; the other is redundant

Answerability: Is the answer fully supported by the provided evidence?
-GOOD: Fully supported by cited evidence
-BORDERLINE: Partially supported
-BAD: Not supported or contradicts the evidence

C. Dataset Quality: 
Decontextualization
Is the question self-contained and unambiguous? The question cannot contain explicit references to the papers or its content such as "in this paper", "the proposed methods", " this approach" or similar.
-GOOD: Fully self-contained; all entities clearly defined
-BORDERLINE: Minor ambiguity
-BAD: Not understandable without external context

OUTPUT FORMAT
{{
    "multiHopValidity": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<explain your judgement for multiHopValidity>"
    }},
    "dependencyStrength": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<explain your judgement for dependencyStrength>"
    }},
    "nonDecomposability": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<explain your judgement for nonDecomposability>"
    }},
    "evidenceDistribution": {{
        "judgement":"<GOOD / BORDERLINE / BAD>",
        "explanation": "<explain your judgement for evidenceDistribution>"

    }},
    "answerability": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<explain your judgement for answerability>"

    }},
    "decontextualization": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<explain your judgement for decontextualization>"

    }},
    "confidence": <0 to 1 rate how confident you are in the judgements>
}}
Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
"""

EXPERIMENTERER_PROMPT = """
You are testing whether a scientific multi-hop question truly requires the provided papers.
You are given:
a question
a subset of papers (one or more papers may be missing)
Your task is to determine whether the question can still be fully answered using ONLY the provided papers.

IMPORTANT:
-Use only the information from the given papers
-Do NOT rely on external knowledge
-Be strict: if any critical information is missing, the answer is NOT recoverable
-Partial or speculative answers count as NOT answerable

INPUT:
Question: {question}
Papers: {paperTexts}

OUTPUT FORMAT:
{{
    "answerable": <true/false - answer this as a boolean value>,
    "explanation": "<if true: explain how the answer was obtained from the available papers, if false:explain what critical information is missing and why the question cannot be answered >",
    "confidence":  <0 to 1 rate how confident you are>
}}
Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
"""

EXPERIMENTERER_CONNECTION_PROMPT ="""
You are given 2 scientific paper texts as well as a logical connection that exists between that papers. Your task is to determine wether that connection is explicitely mentioned in one of the 2 papers, or if is drawn based on their content, but not explicitely mentioned. 

IMPORTANT:
-Use only the information from the given papers
-Do NOT rely on external knowledge
-Be strict: if any critical information is missing, the answer is NOT recoverable
-Partial or speculative answers count as NOT answerable

INPUT:
paper1: {paper1}
paper2: {paper2}
connection: {connection}

OUTPUT FORMAT
{{
    "explanation" : "<describe your reasoning>",
    "isPresent": <true/ false depending on if the connection is explicitely mentioned in at least one of the papers, false if not>
}}
Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
"""

def build_prompt(texts):
    return PROMPT_TEMPLATE + json.dumps(texts)

def build_judging_prompt(question,answer,paper1Text,paper2Text,reasoningSteps):
    return JUDGING_PROMPT_TEMPLATE.format(question=question,answer=answer,paper1Text=paper1Text,paper2Text=paper2Text,reasoningSteps=reasoningSteps)

def buildExperimentererPromps(question,paperTexts):
    return EXPERIMENTERER_PROMPT.format(question=question, paperTexts=paperTexts)
def buildExperimentererConnectionPrompt(paper1, paper2,connection):
    return EXPERIMENTERER_CONNECTION_PROMPT.format(paper1=paper1, paper2=paper2, connection=connection)

def get_all_squai_arxiv_ids():
    allArxivIds = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            clean_id = line.strip().replace('"', '')
            if clean_id:
                allArxivIds.append(clean_id)
    return allArxivIds
    
# TODO ADD DELAY OR STRIKED BY ARXIV

def getCategoryFromArxiv(arxiv_id):
    base_url = "https://export.arxiv.org/api/query"
    params = {
        "id_list": arxiv_id,
        "max_results": 1
    }
    headers = {
        "User-Agent": "ResearchScript/1.0",
        "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        xml_text = response.text

        # Regex to find the primary category (e.g., term="cs.LG")
        match = re.search(r'<arxiv:primary_category\s+term="([^"]+)"', xml_text)
        category = match.group(1)
        #check if this is really correct, maybe take secondary category
        time.sleep(10)
        return category
    
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching metadata for {arxiv_id}: {e}")
        status_code = e.response.status_code
        if (status_code == 429):
            print("429 - sleeping for 5min")
            time.sleep(300)
        else:
            print("errorCode: " + str(status_code))
            time.sleep(10)
        return None
    except Exception as e:
        time.sleep(10)
        return None


def askScadsApiLLM(prompt):
    url = "https://llm.scads.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SCADS_API_KEY}"
    }

    # 3. Create your prompt payload
    payload = {
        "model": SCADS_API_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0
    }

    # 4. Send the request and print the answer
    response = requests.post(url, headers=headers, json=payload)
    print("-----------------")
    print(response)
    print("-----------------")
    # Convert the response to JSON and extract the text
    data = response.json()
    return (data["choices"][0]["message"]["content"])



def get_cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def getPaperReferencesAndEmbeddingFromSemanticScholar(arxiv_id):
    fields = "title,references.title,references.externalIds,references.year,references.url,embedding.specter_v2"
    url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
    headers = {
        "User-Agent": "ResearchScript/1.0",
        "x-api-key": SEMANTIC_SCHOLAR_API_KEY
    }
    api_lock.lock()
    print("getting references and vectors for: " + arxiv_id)
    try:
        response = requests.get(url, params={"fields": fields, "limit": 1000}, headers=headers)
        time.sleep(5)
        return response.json()
    except Exception as e:
        print(f"Semantic Scholar Fetch failed: {e}")
        time.sleep(5)
        return {"error": str(e)}
    finally:
        api_lock.unlock()




def getPaperFullText(arxivId):
    try:
        db_lock.lock()
        print("-------DB_BOOT_TIME")
        print(time.time())
        db = plyvel.DB(DB_PATH, create_if_missing=False)
        print(time.time())
        print("-------DB_BOOT_TIME")


        content = db.get(arxivId.encode('utf-8'))
        if content is None: 
            return None
        return (content.decode('utf-8'), arxivId)[0]
    except:
        print("SOMEHOW GOT AN ERROR USING THE DB")
    finally:
        if 'db' in locals():
            print("CLOSING THE DB AGAIN")
            db.close()
        db_lock.unlock()
    


def clean_and_parse_json(text):
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        json_str = text[start_idx : end_idx + 1]
        
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError as e:
            return None
    else:
        return None

def generateQuestion(arxivId, allSquaiArxivIds): #, judgingAgent
    starting_id = arxivId
    startingPaperFullText = getPaperFullText(starting_id)
    if startingPaperFullText is None:
        return None
    paper_data = getPaperReferencesAndEmbeddingFromSemanticScholar(starting_id)
    if "references" not in paper_data:
        return None

    if not paper_data.get("embedding") or ("vector" not in paper_data["embedding"]): 
         return None
         
    startingPaperSpecterEmbedding = paper_data["embedding"]["vector"]

    # Filter references that exist in your SQuAI ID list
    valid_refs = [
        p for p in paper_data["references"]
        if p.get("externalIds") and p["externalIds"].get("ArXiv") in allSquaiArxivIds
    ]

    print(f"References: {len(paper_data['references'])}")
    print(f"References in squai dataset: {len(valid_refs)}")

    if len(valid_refs) < 4:
        return None

    paperCosineSimilarity = []

    for ref in valid_refs:
        paperId = ref["externalIds"]["ArXiv"]
        paperMeta = getPaperReferencesAndEmbeddingFromSemanticScholar(paperId)
       
        if not paperMeta.get("embedding") or ("vector" not in paperMeta["embedding"]): 
            continue
        embedding = paperMeta["embedding"]["vector"]
        # calculate cosine similarity with the original paper here
        cosineSimilarity = get_cosine_similarity(startingPaperSpecterEmbedding, embedding)
        
        paperCosineSimilarity.append({
            "paperId": paperId,
            "cosineSimilarity": cosineSimilarity
        })

    if len(paperCosineSimilarity) < 4:
        return None
    
    papersInThreshold = [ ref for ref in paperCosineSimilarity if 0.70 <= ref["cosineSimilarity"] <= 0.9]
    print(f"papers in threshold similarity: {len(papersInThreshold)}")
    

    final_papers = [{
        "ArXiv": starting_id,
        "text":  startingPaperFullText,
        "untruncatedTextLength": len(startingPaperFullText),
    }]
    while len(final_papers) < 5 and len(papersInThreshold ) > 0:
        potentialPaper = random.choice(papersInThreshold)
        arxivId = potentialPaper["paperId"]
        paperFullText = getPaperFullText(arxivId)
        if paperFullText is not None:
            final_papers.append({
            "ArXiv": arxivId,
            "text": paperFullText,
            "untruncatedTextLength": len(paperFullText),
            "cosineSimilarity" : potentialPaper["cosineSimilarity"]
            })
        papersInThreshold.remove(potentialPaper)

    if len(final_papers) < 5: 
        return None
    
    papersWithLessThanCharacterLimit = [p for p in final_papers if len(p["text"]) <= PAPER_CHARACTER_LIMIT]
    papersWithMoreThanCharacterLimit = [p for p in final_papers if len(p["text"]) > PAPER_CHARACTER_LIMIT]
    totalCharsInSub50kPapers= 0
    totalCharsInAbove50kPapers = 0
    adaptiveCharacterLimit = 0

    for p in papersWithLessThanCharacterLimit:
        totalCharsInSub50kPapers += len(p["text"])
    
    charactersLeftFromSub50kPapers = len(papersWithLessThanCharacterLimit) * PAPER_CHARACTER_LIMIT - totalCharsInSub50kPapers
    
    for p in papersWithMoreThanCharacterLimit:
        totalCharsInAbove50kPapers += len(p["text"])
    
    # print("paper with less than 50k characters: " + str(len(papersWithLessThan50k)))
    # print("paper with more than 50k characters: " + str(len(papersWithMoreThan50k)))
    # print("characters left from sub 50k papers: " + str(charactersLeftFromSub50kPapers))
    for p in papersWithMoreThanCharacterLimit:
        text = p["text"]
        adaptiveCharacterLimit = int(PAPER_CHARACTER_LIMIT + (len(text) / totalCharsInAbove50kPapers) * charactersLeftFromSub50kPapers)
        snipLength = int(adaptiveCharacterLimit / 2)
        p["text"] = "First " + str(snipLength) + " characters: "+  text[:snipLength] + "   Last " + str(snipLength) + " characters: " + text[-snipLength:]

    finalPapersAdjustedLength = papersWithLessThanCharacterLimit + papersWithMoreThanCharacterLimit


    
    for paper in finalPapersAdjustedLength:    
        print("Arxiv: " + str(paper["ArXiv"]) + "   adjusted Text length: " + str(len(paper["text"])))
    
    clean_papers_for_prompt = [
        {"ArXiv": p["ArXiv"], "text": p["text"]} 
        for p in finalPapersAdjustedLength
    ]


    with open("logPapers.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(clean_papers_for_prompt) + "\n")

    prompt = build_prompt(clean_papers_for_prompt)
    print("asking llm")
    # llmanswer = agent.generate(prompt)
    print(str(time.time()))
    llmanswer = askScadsApiLLM(prompt)
    print("ANSWERER")
    print(llmanswer)
    cleanedAndParsedJson = clean_and_parse_json(llmanswer) 

    if not cleanedAndParsedJson:
        return None
    if len(cleanedAndParsedJson["usedPapers"]) != 2:
        print("NOT ENOUGH PAPERS USED")
        return None
    papersUsedForGenerationWithCategory = []
    for paper in finalPapersAdjustedLength:
        papersUsedForGenerationWithCategory.append({
            "ArXiv" : paper["ArXiv"],
            "category": None, #getCategoryFromArxiv(paper["ArXiv"]),
            "cosineSimilarity": paper["cosineSimilarity"] if "cosineSimilarity" in paper else None
        })
    cleanedAndParsedJson["papersInputtedForGeneration"] = papersUsedForGenerationWithCategory
    cleanedAndParsedJson["anchorPaper"] = starting_id
    cleanedAndParsedJson["adaptiveCharacterLimit"] = adaptiveCharacterLimit

    bridgeEvidencePaperId = next((paper for paper in cleanedAndParsedJson["usedPapers"] if paper.get("role") == "bridgeEvidence"), None)["arXiv"]
    print(str(bridgeEvidencePaperId))
    bridgeAnswerPaperId = next((paper for paper in cleanedAndParsedJson["usedPapers"] if paper.get("role") == "bridgeAnswer"), None)["arXiv"]
    print(str(bridgeAnswerPaperId))

    print("getting paper texts")
    bridgeEvidencePaperText = next((p for p in finalPapersAdjustedLength if p.get("ArXiv") == bridgeEvidencePaperId), None).get("text")
    bridgeAnswerPaperText = next((p for p in finalPapersAdjustedLength if p.get("ArXiv") == bridgeAnswerPaperId), None).get("text")
    print("got paper texts")

    cleanedAndParsedJson["bridgeEvidencePaperText"] = bridgeEvidencePaperText
    cleanedAndParsedJson["bridgeAnswerPaperText"] = bridgeAnswerPaperText

    # judgementPrompt = build_judging_prompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answerWithoutPaperReferences"], bridgeEvidencePaperText, bridgeAnswerPaperText, json.dumps(cleanedAndParsedJson["reasoning"]))
    
    # judgementResult = judgingAgent.generate(judgementPrompt)
    # judgementResultParsed = clean_and_parse_json(judgementResult) 
    # print("JUDGE\n")
    # print(json.dumps(judgementResultParsed))
    # cleanedAndParsedJson["judgementResult"] = judgementResultParsed
    # free_gpu_memory()

    # experimenterPromptEvidence = buildExperimentererPromps("paper1: " + cleanedAndParsedJson["question"], bridgeEvidencePaperText)
    # experimenterPromptEvidenceExperimentorResult = judgingAgent.generate(experimenterPromptEvidence)
    # experimenterPromptEvidenceExperimentorResultParsed = clean_and_parse_json(experimenterPromptEvidenceExperimentorResult) 
    # free_gpu_memory()
    # experimenterPromptAnswer = buildExperimentererPromps("paper1: " + cleanedAndParsedJson["question"], bridgeAnswerPaperText)
    # experimenterPromptAnswerExperimentorResult = judgingAgent.generate(experimenterPromptAnswer)
    # experimenterPromptAnswerExperimentorResultParsed = clean_and_parse_json(experimenterPromptAnswerExperimentorResult) 
    # free_gpu_memory()

    # bothPaperTexts = "EvidencePaperText:\n" + bridgeEvidencePaperText + "\n" + "BridgeAnswerText" +bridgeAnswerPaperText
    # experimenterPromptBoth = buildExperimentererPromps(cleanedAndParsedJson["question"], bothPaperTexts)
    # experimenterPromptBothResult = judgingAgent.generate(experimenterPromptBoth)
    # experimenterPromptBothResultParsed = clean_and_parse_json(experimenterPromptBothResult) 
    # free_gpu_memory()

    # experimentererConnectionPrompt =  buildExperimentererConnectionPrompt(bridgeEvidencePaperText, bridgeAnswerPaperText, cleanedAndParsedJson["reasoning"]["connectionExplanation"])
    # experimentererConnectionResult = judgingAgent.generate(experimentererConnectionPrompt)
    # experimentererConnectionResultParsed = clean_and_parse_json(experimentererConnectionResult)
    # free_gpu_memory()

    # cleanedAndParsedJson["experimenterPromptEvidenceExperimentorResult"] = experimenterPromptEvidenceExperimentorResultParsed
    # cleanedAndParsedJson["experimenterPromptAnswerExperimentorResult"] = experimenterPromptAnswerExperimentorResultParsed
    # cleanedAndParsedJson["experimenterPromptBothResult"] = experimenterPromptBothResultParsed
    # cleanedAndParsedJson["experimentererConnectionResult"] = experimentererConnectionResultParsed


    # usageJudgeResult = []
    # paperLengths = []
    # for paper in finalPapersAdjustedLength:
    #     prompt = build_judging_prompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answer"], paper["text"])
    #     judgementResult = judgingAgent.generate(prompt)
    #     match = re.search(r'<verdict>\s*(true|false)\s*</verdict>', judgementResult, re.IGNORECASE)
    #     judgement = None
    #     if match:
    #          judgement =  match.group(1).lower() == "true"
    #     usageJudgeResult.append({"ArXiv": paper["ArXiv"], "wasUsed": judgement})
    #     paperLengths.append({"ArXiv": paper["ArXiv"], "textLength": len(paper["text"])})
    #     torch.cuda.empty_cache()
    # cleanedAndParsedJson["usageJudgeResult"] = usageJudgeResult
    # cleanedAndParsedJson["paperTextLengths"] = paperLengths
    return cleanedAndParsedJson

def main():
    # agent = LLMAgent(MODEL)
    # judgingAgent = LLMAgent(JUDGING_MODEL)
    all_squai_ids = get_all_squai_arxiv_ids()
    
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            allUsedArxivIds = [line.strip() for line in f if line.strip()]
    else:
        allUsedArxivIds = []
    print("loaded " + str(len(allUsedArxivIds)) + " already used arxiv ids from previous runs")

    allSuccessFullyUsedArxivIds = []
    generatedQuestions = []
    

    while (len(generatedQuestions) < QUESTIONS_TO_GENERATE):
        randomArxiv = random.sample(all_squai_ids, 1)[0]
        if (randomArxiv in allUsedArxivIds):
            continue
        print("trying: " + randomArxiv)

        # category = getCategoryFromArxiv(randomArxiv)
        # physicsCategories = ["astro-ph", "cond-mat","gr-qc", "hep-ex", "hep-lat", "hep-ph",  "hep-th", "math-ph", "nlin.", "nucl-ex", "nucl-th",  "physics",  "quant-ph"]
        
        # pfusch später fixen mit else True
        # isPhysics = any(category.startswith(cat) for cat in physicsCategories) if category is not None else True
        # if isPhysics is not None and isPhysics:
        #     continue

        question = None
        try:
            question = generateQuestion(randomArxiv, all_squai_ids) # removed parameter , judgingAgent here
        except torch.OutOfMemoryError:
            with open(ERROR_CACHE_FILE, "a", encoding="utf-8") as f:
                    f.write("OOM ERROR" + randomArxiv + "\n")    
            gc.collect() 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            gc.collect() 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with open(ERROR_CACHE_FILE, "a", encoding="utf-8") as f:
                f.write("NOT A CUDA OOM ERROR: " + str(e) +  "\n" + randomArxiv + "\n")
            continue
        if question:
            generatedQuestions.append(question)
            allSuccessFullyUsedArxivIds.append(randomArxiv)
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(question, ensure_ascii=False) + "\n") 
            print("Successfully generated " + str(len(generatedQuestions)) + " / " + str(QUESTIONS_TO_GENERATE))
        allUsedArxivIds.append(randomArxiv)
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(randomArxiv + "\n")

        print("-" * 60)
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()



# fix lost arxiv categories
# curl -L "https://export.arxiv.org/api/query?id_list=2412.15670&max_results=1" \
#     -H "User-Agent: ResearchScript/1.0"