#!/usr/bin/env python3
"""
Add a ragas faithfulness score ("faithfulness") to each individual claim
judgement on an already-produced evaluation results file (the output of
run_SQuAI_evaluation_attrscore.py), the same way run_SQuAI_evaluation.py
computes it per-claim via judgeClaim() - except here it's a standalone
recompute pass instead of part of the full generation+judging pipeline.

Every other metric already present in each judgement (contextRelevance,
entailment, noise, entailment2, attrScore, answerCorrectness,
answerRelevancy) is left untouched - this script only adds/overwrites the
"faithfulness" key.
"""
import json
import math
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

load_dotenv()

INPUT_FILE = "evaluationResultCombinedWithAttrScore.jsonl"
OUTPUT_FILE = "evaluationResultCombinedWithFaithfulness.jsonl"

SCADS_API_KEY = os.getenv("PUBLIC_SCADS_KEY")

ragasClient = AsyncOpenAI(
    base_url="https://llm.scads.ai/v1",
    api_key=SCADS_API_KEY,
    max_retries=8,
    timeout=300.0,
)
ragasLLM = llm_factory("meta-llama/Llama-3.3-70B-Instruct", client=ragasClient, max_tokens=16000, max_retries=8, timeout=300.0)
faithfulnessScorer = Faithfulness(llm=ragasLLM)

referencesNativeKey = "referencesNative"
referencesKeys = ["referencesBiencoderTop1","referencesBM25Top1","referencesBiencoderTop10Bm25Top1","referencesBM25Top10BiencoderTop1","referencesBiencoderTop10CrossEncoderTop1","referencesBM25Top10CrossEncoderTop1","referencesBiencoderAndBm25Top1","referencesBiencoderAndBm25Top10CrossEncoderTop1","referencesWithLLM"]


def missingValue(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def compute_faithfulness(context, claim, query):
    while True:
        try:
            faithfulness = faithfulnessScorer.score(
                user_input=query, response=claim, retrieved_contexts=[context]
            ).to_dict()
        except Exception as e:
            print(f"compute_faithfulness: ragasClient call failed ({e}), retrying in 5 minutes...")
            time.sleep(300)
            continue

        if not missingValue(faithfulness.get("result")):
            return faithfulness

        print("compute_faithfulness: missing/invalid value for faithfulness, retrying in 5 minutes...")
        time.sleep(300)


def set_faithfulness(judgement, context, claim, query, stats):
    """
    Overwrites `judgement["faithfulness"]` if it already exists, or adds it
    if this judgement never had one. Plain dict assignment handles both
    cases identically - this wrapper just tracks which happened, for a
    visible replaced-vs-added count at the end.
    """
    if "faithfulness" in judgement:
        stats["replaced"] += 1
    else:
        stats["added"] += 1
    judgement["faithfulness"] = compute_faithfulness(context, claim, query)


def recompute_native(question, side_key, model_answer_key, query, stats):
    """
    side_key: "withoutGold" or "withGold"
    model_answer_key: "modelAnswer" or "mddelAnswerWithGold" (matches the
    original scripts' field name, typo included)

    referencesNative stores ONE contextPassage per documentId, with a
    "judgement" list accumulated in the order sentences citing that
    documentId were encountered in model_answer_key - so a per-document
    counter reconstructs the same index mapping used when the list was built.
    """
    native = question[side_key][referencesNativeKey]
    per_doc_index = {}

    for sentence in question["answerMeta"][model_answer_key]:
        documentId = str(sentence["documentId"])
        entry = native[documentId]
        idx = per_doc_index.get(documentId, 0)
        per_doc_index[documentId] = idx + 1

        judgement = entry["judgement"][idx]
        set_faithfulness(judgement, entry["contextPassage"], sentence["sentence"], query, stats)


def recompute_ref_key(question, side_key, model_answer_key, refKey, query, stats):
    """
    Non-native reference keys store a list of {contextPassage, judgement}
    entries per documentId, one per citation occurrence, indexed by a
    per-document counter in the same order as model_answer_key.
    """
    quoteCounter = {}

    for sentence in question["answerMeta"][model_answer_key]:
        documentId = str(sentence["documentId"])
        idx = quoteCounter.get(documentId, 0)
        quoteCounter[documentId] = idx + 1

        entry = question[side_key][refKey][documentId][idx]
        set_faithfulness(entry["judgement"], entry["contextPassage"], sentence["sentence"], query, stats)


def recompute_question(question, stats):
    query = question["generationMeta"]["question"]

    recompute_native(question, "withoutGold", "modelAnswer", query, stats)
    for refKey in referencesKeys:
        recompute_ref_key(question, "withoutGold", "modelAnswer", refKey, query, stats)

    recompute_native(question, "withGold", "mddelAnswerWithGold", query, stats)
    for refKey in referencesKeys:
        recompute_ref_key(question, "withGold", "mddelAnswerWithGold", refKey, query, stats)


def main():
    start = time.time()
    processed = 0
    failed = 0
    stats = {"replaced": 0, "added": 0}

    with open(INPUT_FILE, "r", encoding="utf-8") as in_file, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for lineNumber, line in enumerate(in_file, start=1):
            line = line.strip()
            if not line:
                continue

            question = json.loads(line)
            anchor = question.get("generationMeta", {}).get("anchorPaper", f"line {lineNumber}")

            try:
                recompute_question(question, stats)
            except Exception as e:
                failed += 1
                print(f"FAILED on {anchor}: {e}")
            else:
                out_file.write(json.dumps(question, ensure_ascii=False) + "\n")
                out_file.flush()
                processed += 1

            elapsed = time.time() - start
            print(f"processed={processed} failed={failed} elapsed={elapsed:.1f}s")

    print(
        f"DONE. processed={processed} failed={failed} "
        f"faithfulness_replaced={stats['replaced']} faithfulness_added={stats['added']} "
        f"elapsed={time.time() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
