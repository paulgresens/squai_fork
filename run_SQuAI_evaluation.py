#!/usr/bin/env python3
import json
import os
import re
import gc
import logging
import plyvel
import fcntl
import torch
from dotenv import load_dotenv

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerCorrectness, AnswerRelevancy, Faithfulness, ContextRelevance
from entailment_agent import EntailmentChecker



import multiprocessing as mp
from hybrid_retriever import Retriever

load_dotenv()
#this needs to be changed for every instance
OUTPUT_FILE = "evaluationResult.jsonl"

class APILockManager:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.file_obj = None

    def lock(self):
        """Acquires the lock. Will freeze the script here if another script holds it."""
        # Open the file manually and KEEP IT OPEN.
        self.file_obj = open(self.lock_file, "a", encoding="utf-8")
        fcntl.flock(self.file_obj, fcntl.LOCK_EX)
        # self.file_obj.write("LOCKED")
        # self.file_obj.flush() # Ensure it writes to disk immediately
        print("🔒 Lock acquired.")

    def unlock(self):
        """Explicitly releases the lock so other scripts can proceed."""
        if self.file_obj and not self.file_obj.closed:
            # Release the lock and close the file
            fcntl.flock(self.file_obj, fcntl.LOCK_UN)
            self.file_obj.close()
            self.file_obj = None
            print("🔓 Lock released.")




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
LOCK_FILE = "evaluationLockFile.json"
STATE_TRACKING_FILE = "evaluationStateTracking.json"
DB_LOCK_FILE = "dbLock.txt"
lock = APILockManager(LOCK_FILE)
db_lock = APILockManager(DB_LOCK_FILE)

if not os.path.exists(DB_LOCK_FILE):
    with open(DB_LOCK_FILE, "a", encoding="utf-8") as f:
        f.write("lock")

DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.65
SCADS_API_KEY = os.getenv("PUBLIC_SCADS_KEY")
INPUT_FILE="contextResultsToJudge.jsonl"

lock.lock()
if not os.path.exists(STATE_TRACKING_FILE):
    with open(STATE_TRACKING_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"finished": [], "current": []}))

lock.unlock()

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

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


ragasClient = AsyncOpenAI(
    base_url = "https://llm.scads.ai/v1",
    api_key = SCADS_API_KEY
)
ragasLLM = llm_factory("meta-llama/Llama-3.3-70B-Instruct", client=ragasClient, max_tokens=16000)
ragasEmbeddings = embedding_factory("openai", model="Qwen/Qwen3-Embedding-4B", client=ragasClient)
answerCorrectnessScorer = AnswerCorrectness(llm=ragasLLM, embeddings=ragasEmbeddings)
answerRelevancyScorer = AnswerRelevancy(llm=ragasLLM, embeddings=ragasEmbeddings)
faithfulnessScorer = Faithfulness(llm=ragasLLM)
contextRelevanceScorer = ContextRelevance(llm=ragasLLM)
entailmentChecker = EntailmentChecker()

referencesNativeKey = "referencesNative"
referencesKeys = ["referencesBiencoderTop1","referencesBM25Top1","referencesBiencoderTop10Bm25Top1","referencesBM25Top10BiencoderTop1","referencesBiencoderTop10CrossEncoderTop1","referencesBM25Top10CrossEncoderTop1","referencesBiencoderAndBm25Top1","referencesBiencoderAndBm25Top10CrossEncoderTop1","referencesWithLLM"]


def judgeClaim(sentence,context,query, paperTexts):
# paper text looks like this: {paperId: "paperTExtBla"}

    print("------------JUDGING THIS-----------")
    print("sentence: " + json.dumps(sentence))
    print("context: " + extractedSourceSentences)
    print("original question: " + questionText)
    print("-----------------------------------")

    # floating context window 1-5 sentences, per paper
    paperSpans = {}
    for paperId, documentText in paperTexts.items():
        raw_splits = re.split(r"([.!?]+)", documentText)
        documentSentences = []

        # Loop through splits and re-attach punctuation to the previous sentence
        for i in range(0, len(raw_splits) - 1, 2):
            sent = raw_splits[i].strip()
            # Get the punctuation that follows (if it exists)
            punct = raw_splits[i+1].strip() if i+1 < len(raw_splits) else ""
            if sent:
                documentSentences.append(f"{sent}{punct}")

        window_2 = [" ".join(documentSentences[i : i + 2]) for i in range(len(documentSentences) - 1)]
        window_3 = [" ".join(documentSentences[i : i + 3]) for i in range(len(documentSentences) - 2)]
        window_4 = [" ".join(documentSentences[i : i + 4]) for i in range(len(documentSentences) - 3)]
        window_5 = [" ".join(documentSentences[i : i + 5]) for i in range(len(documentSentences) - 4)]

        paperSpans[paperId] = documentSentences + window_2 + window_3 + window_4 + window_5

    raw_context_splits = re.split(r"([.!?]+)", context)
    contextSentences = []

    # Loop through splits and re-attach punctuation to the previous sentence
    for i in range(0, len(raw_context_splits) - 1, 2):
        sent = raw_context_splits[i].strip()
        punct = raw_context_splits[i+1].strip() if i+1 < len(raw_context_splits) else ""
        if sent:
            contextSentences.append(f"{sent}{punct}")

    # All contiguous sub-windows, every length from 1 up to len(contextSentences)
    # (not capped at 5, unlike paperSpans above - this scales to however many
    # sentences the extracted context actually has).
    contextWindows = []
    for start in range(len(contextSentences)):
        for end in range(start + 1, len(contextSentences) + 1):
            contextWindows.append(" ".join(contextSentences[start:end]))


    result = {}

    noise = entailmentChecker.get_entailments_for_spans(contextWindows, sentence["sentence"])
    result["noise"] = noise

    # faithfulness = faithfulnessScorer.score(user_input=query, response=sentence["sentence"], retrieved_contexts=[context]).to_dict()
    # result["faithfulness"] = faithfulness 
    
    # contextRelevance = contextRelevanceScorer.score(user_input=query,retrieved_contexts=[context]).to_dict()
    # result["contextRelevance"] = contextRelevance
    
    # entailment = entailmentChecker.check_entailment(context, sentence["sentence"])
    # result["entailment"] = entailment

    # entailmentAlternatives = entailmentChecker.get_top_entailments_per_paper(paperSpans, sentence["sentence"])
    # result["entailmentAlternatives"] = entailmentAlternatives

    print(json.dumps(result))
    return result

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
    lock.lock()
    with open(STATE_TRACKING_FILE, "r", encoding="utf-8") as f:
        lockContent = json.load(f)
    currentQuestionAnchor = question["generationMeta"]["anchorPaper"]
    print("LOCK FILE STATE")
    print(json.dumps(lockContent))
    if currentQuestionAnchor in  lockContent["finished"] or currentQuestionAnchor in lockContent["current"]:
        lock.unlock()
        continue
    lockContent["current"].append(currentQuestionAnchor)


    with open(STATE_TRACKING_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(lockContent))

    lock.unlock()
    try:
        nonGoldPaperMapping = {
            entry["docId"]: entry["paperId"]
            for entry in question["answerMeta"]["paperInformationWithoutGold"]
        }

        db_lock.lock()
        try:
            db = plyvel.DB(db_path_to_use, create_if_missing=False)
            paperTextsNonGold = retriever.get_full_texts(
                list(nonGoldPaperMapping.values()), db=db
            )
        finally:
            db.close()
            db_lock.unlock()


        paper_texts_by_id_non_gold = {
            paper_id: text
            for text, paper_id in paperTextsNonGold
        }

        modelAnswerWithoutGold = question["answerMeta"]["modelAnswer"]
        question["quoteJudgement"] = {}
        question["quoteJudgement"]["withoutGold"] = {}
        question["quoteJudgement"]["withGold"] = {}


        ################################################## without gold papers case
        #scope whole answer - ground truth answer

        modelAnswerWithoutQuotation =  " ".join(
            item["sentence"] for item in question["answerMeta"]["modelAnswer"]
        )
        goldGroundTruthAnswer = question["generationMeta"]["answerWithoutPaperReferences"]
        questionText = question["generationMeta"]["question"]

        ####### WITHOUT GOLD - ANSWER CORRECTNESS - ANSWER RELEVANCE
        withoutGoldPaperAnswerCorrectness = answerCorrectnessScorer.score(
                user_input=questionText,
                response=modelAnswerWithoutQuotation,
                reference=goldGroundTruthAnswer
        )
        question["quoteJudgement"]["withoutGold"]["answerCorrectness"] = withoutGoldPaperAnswerCorrectness.to_dict()

        withoutGoldPaperAnswerRelevancy = answerRelevancyScorer.score(
                user_input=questionText,
                response=modelAnswerWithoutQuotation,
        )
        question["quoteJudgement"]["withoutGold"]["answerRelevancy"] = withoutGoldPaperAnswerRelevancy.to_dict()
        print("without gold: ")
        print('answer correctness' + str(withoutGoldPaperAnswerCorrectness))
        print('answer relevancy' + str(withoutGoldPaperAnswerRelevancy))
        print("---------")
        ####### WITHOUT GOLD - ANSWER CORRECTNESS - ANSWER RELEVANCE



        print("STARTING - JUDGING WITHOUT GOLD")

        ###built in squai extraction
        for sentence in question["answerMeta"]["modelAnswer"]:
            documentId = sentence["documentId"]
            extractedSourceSentences = question["withoutGold"]["referencesNative"][str(documentId)]["contextPassage"]
            # if "contextJudgementsWithoutGold" not in question:
            #     question["contextJudgementsWithoutGold"] = {}
            # if "referencesNative" not in question["contextJudgementsWithoutGold"]:
            #     question["contextJudgementsWithoutGold"]["referencesNative"] = {}
            # if (documentId not in question["contextJudgementsWithoutGold"]["referencesNative"]):
            #     question["contextJudgementsWithoutGold"]["referencesNative"][documentId] = []



            judgement = judgeClaim(sentence=sentence, context=extractedSourceSentences, query=questionText, paperTexts=paper_texts_by_id_non_gold)

            question["withoutGold"]["referencesNative"][str(documentId)].setdefault("judgement", []).append(judgement)
            # question["contextJudgementsWithoutGold"]["referencesNative"][documentId].append(judgement)                        


        ### go over every of my extractions
        for refKey in referencesKeys:
            quoteCounter = {
                item["documentId"]: 0
                for item in question["answerMeta"]["modelAnswer"]
            }
            for sentence in question["answerMeta"]["modelAnswer"]:
                documentId = sentence["documentId"]
                extractedSourceSentences = question["withoutGold"][refKey][str(documentId)][quoteCounter[documentId]]["contextPassage"]

                # if "contextJudgementsWithoutGold" not in question:
                #     question["contextJudgementsWithoutGold"] = {}
                # if refKey not in question["contextJudgementsWithoutGold"]:
                #     question["contextJudgementsWithoutGold"][refKey] = {}
                # if (documentId not in question["contextJudgementsWithoutGold"][refKey]):
                #     question["contextJudgementsWithoutGold"][refKey][documentId] = []
                # todo replace placeholder function
                judgement = judgeClaim(sentence=sentence, context=extractedSourceSentences, query=questionText, paperTexts=paper_texts_by_id_non_gold)
                question["withoutGold"][refKey][str(documentId)][quoteCounter[documentId]]["judgement"] = judgement
                
                quoteCounter[documentId] += 1
        print("FINISHED - JUDGING WITHOUT GOLD")

        ##################################################


        ################################################## with gold papers case
        goldPaperMapping = {
            entry["docId"]: entry["paperId"]
            for entry in question["answerMeta"]["papersInformationGoldAnswer"]
        }

        db_lock.lock()
        try:
            db = plyvel.DB(db_path_to_use, create_if_missing=False)
            paperTextsGold = retriever.get_full_texts(
                list(goldPaperMapping.values()), db=db
            )
        finally:
            db.close()
            db_lock.unlock()

        paper_texts_by_id_gold = {
            paper_id: text
            for text, paper_id in paperTextsGold
        }

        modelAnswerWithoutQuotation =  " ".join(
            item["sentence"] for item in question["answerMeta"]["mddelAnswerWithGold"]
        )
        goldGroundTruthAnswer = question["generationMeta"]["answerWithoutPaperReferences"]
        questionText = question["generationMeta"]["question"]

        ####### WITH GOLD - ANSWER CORRECTNESS - ANSWER RELEVANCE
        withGoldPaperAnswerCorrectness = answerCorrectnessScorer.score(
                user_input=questionText,
                response=modelAnswerWithoutQuotation,
                reference=goldGroundTruthAnswer
        )
        question["quoteJudgement"]["withGold"]["answerCorrectness"] = withGoldPaperAnswerCorrectness.to_dict()

        withGoldPaperAnswerRelevancy = answerRelevancyScorer.score(
                user_input=questionText,
                response=modelAnswerWithoutQuotation,
        )
        question["quoteJudgement"]["withGold"]["answerRelevancy"] = withGoldPaperAnswerRelevancy.to_dict()
        ####### WITH GOLD - ANSWER CORRECTNESS - ANSWER RELEVANCE




        ###built in squai extraction
        print("STARTING - JUDGING WITH GOLD")
        for sentence in question["answerMeta"]["mddelAnswerWithGold"]:
            documentId = sentence["documentId"]
            extractedSourceSentences = question["withGold"]["referencesNative"][str(documentId)]["contextPassage"]
            # if "contextJudgementsWithGold" not in question:
            #     question["contextJudgementsWithGold"] = {}
            # if "referencesNative" not in question["contextJudgementsWithGold"]:
            #     question["contextJudgementsWithGold"]["referencesNative"] = {}
            # if (documentId not in question["contextJudgementsWithGold"]["referencesNative"]):
            #     question["contextJudgementsWithGold"]["referencesNative"][documentId] = []
            judgement = judgeClaim(sentence=sentence, context=extractedSourceSentences, query=questionText, paperTexts=paper_texts_by_id_gold)

            question["withGold"]["referencesNative"][str(documentId)].setdefault("judgement", []) .append(judgement)

            # question["contextJudgementsWithGold"]["referencesNative"][documentId].append(judgement)   



        ### go over every of my extractions
        for refKey in referencesKeys:
            quoteCounter = {
                item["documentId"]: 0
                for item in question["answerMeta"]["mddelAnswerWithGold"]
            }
            for sentence in question["answerMeta"]["mddelAnswerWithGold"]:
                documentId = sentence["documentId"]
                extractedSourceSentences = question["withGold"][refKey][str(documentId)][quoteCounter[documentId]]["contextPassage"]

                # if "contextJudgementsWithGold" not in question:
                #     question["contextJudgementsWithGold"] = {}
                # if refKey not in question["contextJudgementsWithGold"]:
                #     question["contextJudgementsWithGold"][refKey] = {}
                # if (documentId not in question["contextJudgementsWithGold"][refKey]):
                #     question["contextJudgementsWithGold"][refKey][documentId] = []
                # todo replace placeholder function
                judgement = judgeClaim(sentence=sentence, context=extractedSourceSentences, query=questionText, paperTexts=paper_texts_by_id_gold)
                question["withGold"][refKey][str(documentId)][quoteCounter[documentId]]["judgement"] = judgement

                # question["contextJudgementsWithGold"][refKey][documentId].append(judgement)                        
                quoteCounter[documentId] += 1
        print("FINISHED - JUDGING WITH GOLD")
        
        ##################################################
        counter+=1


        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(question, ensure_ascii=False) + "\n") 

        lock.lock()
        with open(STATE_TRACKING_FILE, "r", encoding="utf-8") as f:
            lockContent = json.load(f)
        lockContent["current"].remove(currentQuestionAnchor)
        lockContent["finished"].append(currentQuestionAnchor)
        with open(STATE_TRACKING_FILE, "w", encoding="utf-8") as file:
            file.write(json.dumps(lockContent))
        
        lock.unlock()

    except Exception as e : 
        print("EXCEPTION: ")
        print(e)

        lock.lock()
        with open(STATE_TRACKING_FILE, "r", encoding="utf-8") as f:
            lockContent = json.load(f)
        lockContent["current"].remove(currentQuestionAnchor)
        with open(STATE_TRACKING_FILE, "w", encoding="utf-8") as file:
            file.write(json.dumps(lockContent))

        lock.unlock()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
