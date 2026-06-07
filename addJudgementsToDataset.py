import os
import time
import json
import requests
import gc
import torch
import numpy as np
from dotenv import load_dotenv
import time

load_dotenv()
# --- CONFIGURATION ---
SCADS_API_KEY = os.getenv("SCADS_API_KEY")

# --- CONFIGURATION ---
SCADS_API_MODEL = "openai/gpt-oss-120b"
INPUT_FILE = "./dump/mergedVersion3.jsonl"
OUTPUT_FILE = "./dump/mergedVersion3Judged.jsonl"

JUDGING_PROMPT_TEMPLATE ="""
You are evaluating a scientific question-answer (Q-A) example.
Your goal is to assess whether it is a valid 2-hop, evidence-grounded scientific multi-hop question.
A valid example must:
-require combining information from exactly TWO papers
-involve dependent reasoning (Step 2 must require Step 1)
-not be decomposable into independent sub-questions
-be fully answerable from the provided evidence
-be self-contained and unambiguous

The provided Reasoning Steps are not ground truth. They are only the generator's claimed reasoning chain. Verify the claim directly against the two paper texts. If the reasoning steps overstate dependency, invent a connection, or make Paper A seem more necessary than it really is, penalize the relevant criteria.

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
-GOOD: Both papers are strictly required; neither paper alone provides enough information to answer the complete question.
-BORDERLINE: Both papers contribute, but one paper may be sufficient to answer the main question, or the second paper mainly adds background/context.
-BAD: Only one paper is sufficient; the question is effectively single-hop.
Important: If one paper only identifies, introduces, defines, or names an artifact that the other paper already uses, mentions, evaluates, optimizes, cites, or compares against, this is not enough for GOOD. The answer must require combining non-redundant evidence from both papers.

Dependency Strength: Does Step 2 depend on the result of Step 1?
-GOOD: Step 2 strictly requires applying a proposition derived from Step 1. Step 1 must provide more than a named entity; it must provide a mechanism, condition, definition, limitation, assumption, guarantee, interpretation, objective, result, or similar.
-BORDERLINE: Step 2 is related to Step 1, but the dependency is weak, mostly entity-linking, or one step can mostly be solved without the other.
-BAD: Steps are independent, disconnected, or Step 1 only identifies a name/entity used in Step 2.

Important: If Step 1 can be solved mainly by matching a distinctive phrase in the question to nearly identical wording in one paper, and Step 2 only tracks the same artifact in the other paper, do not assign GOOD.

Non-Decomposability: Is the question NOT decomposable into independent sub-questions?
-GOOD: The question cannot be split into independent single-hop questions; solving one part changes, constrains, or enables solving the other.
-BORDERLINE: The question is partially decomposable, but the sub-answers still need some linking or interpretation.
-BAD: The question clearly decomposes into independent sub-questions whose answers can simply be concatenated.

B. Evidence Grounding
Evidence Distribution: Are both papers required and non-redundant?
-GOOD: Each paper contributes distinct, necessary information. One paper provides an intermediate proposition, and the other provides evidence whose meaning depends on applying that proposition.
-BORDERLINE: Both papers contribute, but there is overlap, redundancy, or one paper is dominant while the other only adds support.
-BAD: One paper is sufficient; the other is redundant, only background, or only provides the name/source of an artifact.

Very Important: Artifact reuse alone is insufficient. If Paper A introduces an artifact and Paper B merely uses, applies, evaluates, compares against, cites, optimizes, or extends that artifact, assign BAD or BORDERLINE unless the question requires applying a non-trivial proposition about that artifact.
If the answer can be produced by first identifying an artifact from Paper A and then looking up how Paper B uses that artifact, the Evidence grounding judgement has to be bad.

Answerability: Is the answer fully supported by the provided evidence?
-GOOD: Fully supported by the provided evidence from the papers; no external knowledge, speculation, or unsupported inference is needed.
-BORDERLINE: Partially supported, but some part of the answer requires mild inference, is under-specified, or is not directly grounded.
-BAD: Not supported, contradicted by the evidence, or requires external knowledge/speculation.
C. Dataset Quality: 
Decontextualization: Is the question self-contained and unambiguous?
The question cannot contain explicit references to the papers or their content such as "in this paper", "the proposed method", "this approach", "the authors", or similar.
-GOOD: Fully self-contained; all entities/concepts needed to understand the question are clearly named or described; no explicit paper references.
-BORDERLINE: Mostly self-contained, but contains minor ambiguity, an underspecified phrase, or a concept that is described but not clearly identifiable.
-BAD: Not understandable without external context, relies on paper-specific references, or is ambiguous.

Important: Technical terminology from the papers is allowed when necessary. However, the question should not simply copy distinctive phrasing from a paper in order to make Step 1 a lexical lookup.

OUTPUT FORMAT
{{
    "multiHopValidity": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the multiHopValidity criteria and justify jour judgement>"
    }},
    "dependencyStrength": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the dependencyStrength criteria and justify jour judgement>"
    }},
    "nonDecomposability": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the nonDecomposability criteria and justify jour judgement>"
    }},
    "evidenceDistribution": {{
        "judgement":"<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the evidenceDistribution criteria and justify jour judgement>"

    }},
    "answerability": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the answerability criteria and justify jour judgement>"

    }},
    "decontextualization": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the decontextualization criteria and justify jour judgement>"

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
    time.sleep(2)
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
    start_time = time.perf_counter()
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

    experimenterPromptEvidence = buildExperimentererPromps("paper1: " + cleanedAndParsedJson["question"], bridgeEvidencePaperText)
    experimenterPromptEvidenceExperimentorResult = askScadsApiLLM(experimenterPromptEvidence)
    experimenterPromptEvidenceExperimentorResultParsed = clean_and_parse_json(experimenterPromptEvidenceExperimentorResult) 

    experimenterPromptAnswer = buildExperimentererPromps("paper1: " + cleanedAndParsedJson["question"], bridgeAnswerPaperText)
    experimenterPromptAnswerExperimentorResult = askScadsApiLLM(experimenterPromptAnswer)
    experimenterPromptAnswerExperimentorResultParsed = clean_and_parse_json(experimenterPromptAnswerExperimentorResult) 

    bothPaperTexts = "EvidencePaperText:\n" + bridgeEvidencePaperText + "\n" + "BridgeAnswerText" + "\n" + bridgeAnswerPaperText
    experimenterPromptBoth = buildExperimentererPromps(cleanedAndParsedJson["question"], bothPaperTexts)
    experimenterPromptBothResult = askScadsApiLLM(experimenterPromptBoth)
    experimenterPromptBothResultParsed = clean_and_parse_json(experimenterPromptBothResult) 

    cleanedAndParsedJson["experimenterPromptEvidenceExperimentorResult"] = experimenterPromptEvidenceExperimentorResultParsed
    cleanedAndParsedJson["experimenterPromptAnswerExperimentorResult"] = experimenterPromptAnswerExperimentorResultParsed
    cleanedAndParsedJson["experimenterPromptBothResult"] = experimenterPromptBothResultParsed
    # cleanedAndParsedJson["experimentererConnectionResult"] = experimentererConnectionResultParsed
    end_time = time.perf_counter()
    print("time for all requests: " + str(end_time - start_time))
    return cleanedAndParsedJson

def main():
    questionsWithoutJudgement = 0
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
    main()
