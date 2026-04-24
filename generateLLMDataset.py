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
import numpy as np
from dotenv import load_dotenv
import time
from local_agent import LLMAgent
from config import DB_PATH

def free_gpu_memory():
    """Force garbage collection and cle ar GPU cache."""
    # Attempt to clear globals if they exist here (failsafe)
    if 'agent' in globals():
        del globals()['agent']
    if 'model' in globals():
        del globals()['model']
    if 'tokenizer' in globals():
        del globals()['tokenizer']
        
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("✅ GPU Memory Cleared.")



load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# --- CONFIGURATION ---
SCADS_API_KEY = os.getenv("SCADS_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
INPUT_FILE = "all_paper_ids.txt"
OUTPUT_FILE = "generatedQuestions.jsonl"
CACHE_FILE="alreadyUsedArxivIds.txt"
ERROR_CACHE_FILE = "errorAtTheseArxivIds.txt"
MODEL = "Qwen/Qwen2.5-72B-Instruct"
JUDGING_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
PAPER_CHARACTER_LIMIT=25000
QUESTIONS_TO_GENERATE = 200

PROMPT_TEMPLATE = """
You will be be provided 5 scientific paper text, which share same topic that they are talking about. They will be provided in the following format:
[
  {
    "ArXiv": string,
    "text": string,
  }
]

You should do the following:
Step 1: Read the given scientific paper texts and extract a list of 10 topics where the papers overlap, contract or complement each other. Focus on important concepts or entities within the papers. Avoid using generic or broad words.
Step 2: Use the Topics from Step 1 to generate 1 scientific question-answer pair, with the following requirements.
Requirements:
-question should be based on the information provided in multiple of the papers
-try generating a questions, that requires multiple papers to answer
-question must be context independently answerable, so no reference to a specific paper or entities that you can only understand with the specific paper. The question should have the same character, as if you would ask an scientific expert in their field something, without having a specific paper in mind. Do not refer to external sources like figures or tables.
-question cannot contain explicit references to the papers or its content such as "in this paper", "the proposed methods" or similar
-prioritize a question that requires synthesizing, resolving, applying, or evaluating information across papers
-question should be a complex scientific question that needs in depth knowledge in that area, avoid just asking a simple or definitional question
-try answering the question as specific as possible
-add the arxiv ids of the papers, that contain the relevant information for answering the question
-only add the arxiv id of the paper if it really did contribute significantly to the answer of the question

provide your answer, stricly following this json format:
{
    "topic overlaps" : [
    <TOPIC OVERLAP 1>,
    <TOPIC OVERLAP 2>,
    ...
    ],
    "question": "<YOUR_QUESTION>",
    "answer": "<YOUR ANSWER>",
    "usageJudgementGeneratorLLM":[<ARXIV_ID_1>, <ARXIV_ID_2>, ...]  
}
Do not deviate from this schema. Dont add the keywords you generated. Do not add any preciding information like ```json. Only Answer with the valid json
Paper Texts:
"""


JUDGING_PROMPT_TEMPLATE = """
You will be provided a scientific question, an answer, and a scientific paper text. 
Your task is to evaluate whether the specific information in the scientific paper text is necessary to answer the question.

Step 1: Read the question and the synthesized answer. Identify the core scientific claims being made.
Step 2: Scan the scientific paper text. 
Step 3: Determine if the paper provides explicit evidence, data, or mechanisms that directly support the answer. Do not accept mere keyword overlap.

Output your final verdict inside a <verdict> tags. The verdict must be exactly "true" or "false".

Output Example:
    <Your thinking process described in step 1-3 here>
    <verdict>true</verdict>

question: {question}
answer: {answer}
scientific paper text: {scientificPaperText} 
"""

def build_prompt(texts):
    return PROMPT_TEMPLATE + json.dumps(texts)

def build_judging_prompt(question,answer,scientificPaperText):
    return JUDGING_PROMPT_TEMPLATE.format(question=question,answer=answer,scientificPaperText=scientificPaperText)

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
        time.sleep(6)
        return category
    
    except Exception as e:
        print(f"Error fetching metadata for {arxiv_id}: {e}")
        return None



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
    try:
        response = requests.get(url, params={"fields": fields, "limit": 1000}, headers=headers)
        time.sleep(2)
        return response.json()
    except Exception as e:
        print(f"Semantic Scholar Fetch failed: {e}")
        return {"error": str(e)}



def getPaperFullText(db, arxivId):
    content = db.get(arxivId.encode('utf-8'))
    if content is None: 
        return None
    return (content.decode('utf-8'), arxivId)[0]


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

def generateQuestion(arxivId, allSquaiArxivIds, db, agent, judgingAgent):
    starting_id = arxivId
    startingPaperFullText = getPaperFullText(db,starting_id)
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
        "text":  startingPaperFullText 
    }]
    while len(final_papers) < 5 and len(papersInThreshold ) > 0:
        potentialPaper = random.choice(papersInThreshold)
        arxivId = potentialPaper["paperId"]
        paperFullText = getPaperFullText(db,arxivId)
        if paperFullText is not None:
            final_papers.append({
            "ArXiv": arxivId,
            "text": paperFullText,
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

    prompt = build_prompt(clean_papers_for_prompt)
    print("asking llm")
    llmanswer = agent.generate(prompt)

    cleanedAndParsedJson = clean_and_parse_json(llmanswer) 
    torch.cuda.empty_cache()
    if not cleanedAndParsedJson:
        return None
    
    papersUsedForGenerationWithCategory = []
    for paper in finalPapersAdjustedLength:
        papersUsedForGenerationWithCategory.append({
            "ArXiv" : paper["ArXiv"],
            "category": getCategoryFromArxiv(paper["ArXiv"]),
            "cosineSimilarity": paper["cosineSimilarity"] if "cosineSimilarity" in paper else None
        })
    cleanedAndParsedJson["papersInputtedForGeneration"] = papersUsedForGenerationWithCategory
    cleanedAndParsedJson["anchorPaper"] = starting_id
    cleanedAndParsedJson["adaptiveCharacterLimit"] = adaptiveCharacterLimit


    usageJudgeResult = []
    paperLengths = []
    for paper in finalPapersAdjustedLength:
        prompt = build_judging_prompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answer"], paper["text"])
        judgementResult = judgingAgent.generate(prompt)
        match = re.search(r'<verdict>\s*(true|false)\s*</verdict>', judgementResult, re.IGNORECASE)
        judgement = None
        if match:
             judgement =  match.group(1).lower() == "true"
        usageJudgeResult.append({"ArXiv": paper["ArXiv"], "wasUsed": judgement})
        paperLengths.append({"ArXiv": paper["ArXiv"], "textLength": len(paper["text"])})
        torch.cuda.empty_cache()
    cleanedAndParsedJson["usageJudgeResult"] = usageJudgeResult
    cleanedAndParsedJson["paperTextLengths"] = paperLengths
    return cleanedAndParsedJson

def main():
    db = plyvel.DB(DB_PATH, create_if_missing=False)
    agent = LLMAgent(MODEL)
    judgingAgent = LLMAgent(JUDGING_MODEL)
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

        category = getCategoryFromArxiv(randomArxiv)
        physicsCategories = ["astro-ph", "cond-mat","gr-qc", "hep-ex", "hep-lat", "hep-ph",  "hep-th", "math-ph", "nlin.", "nucl-ex", "nucl-th",  "physics",  "quant-ph"]
        
        isPhysics = any(category.startswith(cat) for cat in physicsCategories)
        if isPhysics:
            continue

        question = None
        try:
            question = generateQuestion(randomArxiv, all_squai_ids, db, agent, judgingAgent)
        except torch.OutOfMemoryError:
            with open(ERROR_CACHE_FILE, "a", encoding="utf-8") as f:
                    f.write(randomArxiv + "\n")    
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
    free_gpu_memory()
    main()



# fix lost arxiv categories
# curl -L "https://export.arxiv.org/api/query?id_list=2412.15670&max_results=1" \
#     -H "User-Agent: ResearchScript/1.0"