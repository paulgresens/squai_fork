#!/usr/bin/env python3
import json
import os
import logging
import plyvel

import multiprocessing as mp
from hybrid_retriever import Retriever



######retriever config for gettimg the full texts
from get_paths import get_main_data_dir
# Configuration paths
MAIN_DATA_DIR = get_main_data_dir()
DATA_DIR = f"{MAIN_DATA_DIR}_extended_data"
E5_INDEX_DIR = f"{MAIN_DATA_DIR}/faiss_index"
BM25_INDEX_DIR = f"{MAIN_DATA_DIR}/bm25_retriever"
DB_PATH = f"{MAIN_DATA_DIR}/full_text_db"
DEFAULT_RETRIEVER = "hybrid"
db_path_to_use = DB_PATH

alt_db_path = os.path.join(os.path.dirname(__file__), "local_db")
db = plyvel.DB(alt_db_path, create_if_missing=False)


DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.65
SCADS_API_KEY = os.getenv("SCADS_API_KEY")
# OUTPUT_FILE = "contextExtractionResult.jsonl"
INPUT_FILE="contextResultsToJudge.jsonl"
# OUTPUT_FILE_WITHOUT_GOLD="contextWithoutGold.jsonl"
# OUTPUT_FILE_WITH_GOLD="contextWithGold.jsonl"
# OUTPUT_FILE_SUCCESS="outputDone.jsonl"
# OUTPUT_FILE = "contextExtractionResultWithoutGoldGroundTruth.jsonl"

def initialize_retriever(
    retriever_type: str,
    e5_index_dir: str,
    bm25_index_dir: str,
    db_path: str,
    top_k: int,
    alpha: float = 0.65,
    db=None,
):
    """Initialize the retriever with strategy and alpha support"""
    print(f"Initializing {retriever_type} retriever with alpha={alpha}...")
    return Retriever(
        e5_index_dir, bm25_index_dir, top_k=top_k, strategy=retriever_type, alpha=alpha
    )

retriever = initialize_retriever(
    retriever_type=DEFAULT_RETRIEVER,
    e5_index_dir=E5_INDEX_DIR,
    bm25_index_dir=BM25_INDEX_DIR,
    db_path=DB_PATH,
    top_k=DEFAULT_TOP_K,
    alpha=DEFAULT_ALPHA,
)
from scadsApiAgent import ScadsApiAgent
zaiAgent = ScadsApiAgent("zai-org/GLM-5.2-FP8")
gptOssAgent = ScadsApiAgent("openai/gpt-oss-120b")


referencesNativeKey = "referencesNative"
referencesKeys = ["referencesBiencoderTop1","referencesBM25Top1","referencesBiencoderTop10Bm25Top1","referencesBM25Top10BiencoderTop1","referencesBiencoderTop10CrossEncoderTop1","referencesBM25Top10CrossEncoderTop1","referencesBiencoderAndBm25Top1","referencesBiencoderAndBm25Top10CrossEncoderTop1","referencesWithLLM"]


def judgeClaim():
    return {}

def clean_and_parse_json(text):
    if text is None:
        return None
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

questions = []

with open(INPUT_FILE, "r") as in_file:
        for j, lineAlready in enumerate(in_file):
            lineAlready = lineAlready.strip()
            if not lineAlready:
                continue # Skip empty lines
            q = clean_and_parse_json(lineAlready)
            questions.append(q)
counter = 1
for question in questions:
    nonGoldPaperMapping = {
        entry["docId"]: entry["paperId"]
        for entry in question["answerMeta"]["paperInformationWithoutGold"]
    }
    print(json.dumps(list(nonGoldPaperMapping.values())))

    paperTextsNonGold = retriever.get_full_texts(
        list(nonGoldPaperMapping.values()), db=db
    )

    print(json.dumps(paperTextsNonGold))
    modelAnswerWithoutGold = question["answerMeta"]["modelAnswer"]

    print("Paper: " + str(counter) + "    (" + question["generationMeta"]["anchorPaper"] + ")")


    ### without gold papers case
    ###built in squai extraction
    for sentence in question["answerMeta"]["modelAnswer"]:
        documentId = sentence["documentId"]
        extractedSourceSentence = question["withoutGold"]["referencesNative"][str(documentId)]["contextPassage"]
        if "contextJudgementsWithoutGold" not in question:
            question["contextJudgementsWithoutGold"] = {}
        if "referencesNative" not in question["contextJudgementsWithoutGold"]:
            question["contextJudgementsWithoutGold"]["referencesNative"] = {}
        if (documentId not in question["contextJudgementsWithoutGold"]["referencesNative"]):
            question["contextJudgementsWithoutGold"]["referencesNative"][documentId] = []
        judgement = judgeClaim()
        question["contextJudgementsWithoutGold"]["referencesNative"][documentId].append(judgement)                        


    ### go over every of my extractions
    for refKey in referencesKeys:
        # print("judging extraction method " + refKey)
        quoteCounter = {
            item["documentId"]: 0
            for item in question["answerMeta"]["modelAnswer"]
        }
        for sentence in question["answerMeta"]["modelAnswer"]:
            documentId = sentence["documentId"]
            extractedSourceSentence = question["withoutGold"][refKey][str(documentId)][quoteCounter[documentId]]["contextPassage"]
            # print(extractedSourceSentence)
            quoteCounter[documentId] += 1

            if "contextJudgementsWithoutGold" not in question:
                question["contextJudgementsWithoutGold"] = {}
            if refKey not in question["contextJudgementsWithoutGold"]:
                question["contextJudgementsWithoutGold"][refKey] = {}
            if (documentId not in question["contextJudgementsWithoutGold"][refKey]):
                question["contextJudgementsWithoutGold"][refKey][documentId] = []
            # todo replace placeholder function
            judgement = judgeClaim()
            question["contextJudgementsWithoutGold"][refKey][documentId].append(judgement)                        
        # print("-----------------")




    ### with gold papers case
    modelAnswerWithGold = question["answerMeta"]["mddelAnswerWithGold"]

    goldPaperMapping = {
        entry["docId"]: entry["paperId"]
        for entry in question["answerMeta"]["papersInformationGoldAnswer"]
    }


    ###built in squai extraction
    for sentence in question["answerMeta"]["mddelAnswerWithGold"]:
        documentId = sentence["documentId"]
        extractedSourceSentence = question["withGold"]["referencesNative"][str(documentId)]["contextPassage"]
        if "contextJudgementsWithGold" not in question:
            question["contextJudgementsWithGold"] = {}
        if "referencesNative" not in question["contextJudgementsWithGold"]:
            question["contextJudgementsWithGold"]["referencesNative"] = {}
        if (documentId not in question["contextJudgementsWithGold"]["referencesNative"]):
            question["contextJudgementsWithGold"]["referencesNative"][documentId] = []
        judgement = judgeClaim()
        question["contextJudgementsWithGold"]["referencesNative"][documentId].append(judgement)   



    ### go over every of my extractions
    for refKey in referencesKeys:
        # print("judging extraction method with gold " + refKey)
        quoteCounter = {
            item["documentId"]: 0
            for item in question["answerMeta"]["mddelAnswerWithGold"]
        }
        for sentence in question["answerMeta"]["mddelAnswerWithGold"]:
            documentId = sentence["documentId"]
            extractedSourceSentence = question["withGold"][refKey][str(documentId)][quoteCounter[documentId]]["contextPassage"]
            # print(extractedSourceSentence)
            quoteCounter[documentId] += 1

            if "contextJudgementsWithGold" not in question:
                question["contextJudgementsWithGold"] = {}
            if refKey not in question["contextJudgementsWithGold"]:
                question["contextJudgementsWithGold"][refKey] = {}
            if (documentId not in question["contextJudgementsWithGold"][refKey]):
                question["contextJudgementsWithGold"][refKey][documentId] = []
            # todo replace placeholder function
            judgement = judgeClaim()
            question["contextJudgementsWithGold"][refKey][documentId].append(judgement)                        
        # print("-----------------")


    print("#############################################################################################################################")
    print("without: ")
    print(json.dumps(question["contextJudgementsWithoutGold"])) 
    print("with: ")
    print(json.dumps(question["contextJudgementsWithGold"])) 
    print("#############################################################################################################################")


    
    print(json.dumps(nonGoldPaperMapping))
    print(json.dumps(quoteCounter))
    print("-----------------")
    print(json.dumps(goldPaperMapping))
    print('####################')
    counter+=1






    