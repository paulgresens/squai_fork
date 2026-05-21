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

load_dotenv()
# --- CONFIGURATION ---
SCADS_API_KEY = os.getenv("SCADS_API_KEY")

def free_gpu_memory():
    """Force garbage collection and clear GPU cache."""
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



# --- CONFIGURATION ---
SCADS_API_MODEL = "openai/gpt-oss-120b"
INPUT_FILE = "generatedQuestionsWithoutJudgement.jsonl"
OUTPUT_FILE = "generatedQuestionsWithJudgement.jsonl"
JUDGING_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

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
Assesses whether answering the question requires integrating information from both papers. Thus, this criterion is only satisfied, if neither paper provides sufficient information for deriving the complete answer.
-GOOD: Both papers are strictly required, the question cannot be answered using only one paper
-BORDERLINE: Both papers contribute but one may be sufficient
-BAD: Only one paper is sufficient (single-hop)

Dependency Strength: 
evaluates whether the steps in the reasoning chain are sequentially dependent, in order for the conclusion to be drawn. Thus, the second reasoning step cannot be solved without first applying or interpreting the first one.
Step 2 must require the result, entity, method, dataset, variable, or conclusion obtained in Step 1.
GOOD: Step 2 strictly requires Step 1
BORDERLINE: Partial dependence
BAD: Steps are independent (disconnected reasoning)

Non-Decomposability: Is the question NOT decomposable into independent sub-questions?
Verifies that questions cannot be broken down into independent single-hop sub-questions. Instead, reasoning steps must be connected in such a way, that separately solving them would not be sufficient for deriving an answer.
-GOOD: Cannot be split; requires joint reasoning
-BORDERLINE: Partially decomposable
-BAD: Clearly decomposable into independent sub-questions

B. Evidence Grounding
Evidence Distribution: Are both papers required and non-redundant?
Examines whether both papers contribute distinct and necessary evidence, ensuring that required evidence is spread across both papers and questions cannot be answered by using a single dominant source, with redundant information.
-GOOD: Each paper contributes distinct, necessary information
-BORDERLINE: Some overlap or redundancy
-BAD: One paper is sufficient; the other is redundant

Answerability: Is the answer fully supported by the provided evidence?
Assesses whether the answer is fully supported and obtainable by only relying on information present in the  papers, without requiring external knowledge or speculative inference, not explicitly supported by either paper. 
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
You are testing whether a scientific multi-hop question can be fully answered using only the provided papers.
You are given:
-a question
a subset of papers (one or more papers may be missing)

You are NOT checking whether the topic is mentioned.
You are checking whether the COMPLETE reasoning chain needed to answer the question is explicitly supported by the provided papers.

IMPORTANT:
-Use only the information from the given papers
-Do NOT rely on external knowledge
-Be strict: if any critical information is missing, the answer is NOT recoverable
-Every step required to answer must be explicitly supported by the text
-If ANY reasoning step is missing, mark as NOT answerable
-Mentioning related concepts is NOT sufficient
-Partial reasoning is NOT sufficient

INPUT:
Question: {question}
Papers: {paperTexts}

OUTPUT FORMAT:
{{
    "explanation": "<strict reasoning: list and explain which exact steps are supported and which are missing>",
    "answerable": <true/false - answer this as a boolean value based on your explanation on wether the question is fully answerable or not. Answer with true if it is, else with false>,
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
    "explanation": "<describe your reasoning>",
    "isPresent": <true/ false depending on if the connection is explicitely mentioned in at least one of the papers, false if not>
}}
Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
"""

def build_judging_prompt(question,answer,paper1Text,paper2Text,reasoningSteps):
    return JUDGING_PROMPT_TEMPLATE.format(question=question,answer=answer,paper1Text=paper1Text,paper2Text=paper2Text,reasoningSteps=reasoningSteps)

def buildExperimentererPromps(question,paperTexts):
    return EXPERIMENTERER_PROMPT.format(question=question, paperTexts=paperTexts)
def buildExperimentererConnectionPrompt(paper1, paper2,connection):
    return EXPERIMENTERER_CONNECTION_PROMPT.format(paper1=paper1, paper2=paper2, connection=connection)

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
    time.sleep(10)
    return (data["choices"][0]["message"]["content"])


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

def addJudgementToQuestion(question):
    cleanedAndParsedJson = question

    bridgeEvidencePaperText = cleanedAndParsedJson["bridgeEvidencePaperText"]
    bridgeAnswerPaperText = cleanedAndParsedJson["bridgeAnswerPaperText"]

    judgementPrompt = build_judging_prompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answerWithoutPaperReferences"], bridgeEvidencePaperText, bridgeAnswerPaperText, json.dumps(cleanedAndParsedJson["reasoning"]))
    
    # judgementResult = judgingAgent.generate(judgementPrompt)
    judgementResult = askScadsApiLLM(judgementPrompt)
    judgementResultParsed = clean_and_parse_json(judgementResult) 
    print("JUDGE\n")
    print(json.dumps(judgementResultParsed))
    cleanedAndParsedJson["judgementResult"] = judgementResultParsed
    free_gpu_memory()

    experimenterPromptEvidence = buildExperimentererPromps("paper1: " + cleanedAndParsedJson["question"], bridgeEvidencePaperText)
    # experimenterPromptEvidenceExperimentorResult = judgingAgent.generate(experimenterPromptEvidence)
    experimenterPromptEvidenceExperimentorResult = askScadsApiLLM(experimenterPromptEvidence)
    experimenterPromptEvidenceExperimentorResultParsed = clean_and_parse_json(experimenterPromptEvidenceExperimentorResult) 
    free_gpu_memory()
    experimenterPromptAnswer = buildExperimentererPromps("paper1: " + cleanedAndParsedJson["question"], bridgeAnswerPaperText)
    # experimenterPromptAnswerExperimentorResult = judgingAgent.generate(experimenterPromptAnswer)
    experimenterPromptAnswerExperimentorResult = askScadsApiLLM(experimenterPromptAnswer)
    experimenterPromptAnswerExperimentorResultParsed = clean_and_parse_json(experimenterPromptAnswerExperimentorResult) 
    free_gpu_memory()

    bothPaperTexts = "EvidencePaperText:\n" + bridgeEvidencePaperText + "\n" + "BridgeAnswerText" + "\n" + bridgeAnswerPaperText
    experimenterPromptBoth = buildExperimentererPromps(cleanedAndParsedJson["question"], bothPaperTexts)
    # experimenterPromptBothResult = judgingAgent.generate(experimenterPromptBoth)
    experimenterPromptBothResult = askScadsApiLLM(experimenterPromptBoth)
    experimenterPromptBothResultParsed = clean_and_parse_json(experimenterPromptBothResult) 
    free_gpu_memory()

    # experimentererConnectionPrompt =  buildExperimentererConnectionPrompt(bridgeEvidencePaperText, bridgeAnswerPaperText, cleanedAndParsedJson["reasoning"]["connectionExplanation"])
    # experimentererConnectionResult = judgingAgent.generate(experimentererConnectionPrompt)
    # experimentererConnectionResultParsed = clean_and_parse_json(experimentererConnectionResult)
    # free_gpu_memory()

    cleanedAndParsedJson["experimenterPromptEvidenceExperimentorResult"] = experimenterPromptEvidenceExperimentorResultParsed
    cleanedAndParsedJson["experimenterPromptAnswerExperimentorResult"] = experimenterPromptAnswerExperimentorResultParsed
    cleanedAndParsedJson["experimenterPromptBothResult"] = experimenterPromptBothResultParsed
    # cleanedAndParsedJson["experimentererConnectionResult"] = experimentererConnectionResultParsed
    return cleanedAndParsedJson

def main():
    questionsWithoutJudgement = 0
    # judgingAgent = LLMAgent(JUDGING_MODEL)
    alreadyJudgedQuestions = []
    with open(OUTPUT_FILE, "r") as in_file:
        for j, lineAlready in enumerate(in_file):
            lineAlready = lineAlready.strip()
            if not lineAlready:
                continue # Skip empty lines
            question = clean_and_parse_json(lineAlready)
            alreadyJudgedQuestions.append(question["anchorPaper"])
    judgedQuestion = len(alreadyJudgedQuestions)

    with open(INPUT_FILE, "r") as in_file:
        for i, line in enumerate(in_file):
           
            line = line.strip()
            if not line:
                continue # Skip empty lines
            question = clean_and_parse_json(line)

            questionAnchorPaper = question["anchorPaper"]
            
            if questionAnchorPaper in alreadyJudgedQuestions:            
                print("Already judged question with anchor paper: " + questionAnchorPaper)
                continue
            else: 
                questionsWithoutJudgement += 1
                finishedQuestionWithJudgement = addJudgementToQuestion(question)
                print("Questions for anchor paper not judged yet: " + questionAnchorPaper)
                                                                
            with open(OUTPUT_FILE, "a") as file: # Using "a" to append each new question
                # json.dump automatically formats your dictionary and writes it to the file
                json.dump(finishedQuestionWithJudgement, file)
                file.write("\n") # Add a newline so the next JSON object starts on a new line
    print("-----------------------------")
    print("already judged:  " + str(judgedQuestion))
    print("added judgement: " + str(questionsWithoutJudgement))
    print("total:           " + str(judgedQuestion + questionsWithoutJudgement ))
    print("-----------------------------")        

if __name__ == "__main__":
    free_gpu_memory()
    main()
