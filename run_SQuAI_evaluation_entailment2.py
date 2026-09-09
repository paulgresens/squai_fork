#!/usr/bin/env python3
"""
Add a second entailment model's per-sentence check ("entailment2") onto an
already-produced evaluation results file (the output of
run_SQuAI_evaluation*.py).

Every other metric already present in each judgement (faithfulness,
contextRelevance, entailment, noise, answerCorrectness, answerRelevancy) is
left untouched - this script only adds/overwrites the "entailment2" key.
"""
import json
import time

from entailment_agent import EntailmentChecker

INPUT_FILE = "evaluationResultCombinedWithNoise.jsonl"
OUTPUT_FILE = "evaluationResultCombinedWithEntailment2.jsonl"

ENTAILMENT2_MODEL_NAME = "google/t5_xxl_true_nli_mixture"
entailmentChecker2 = EntailmentChecker(model_name=ENTAILMENT2_MODEL_NAME)

referencesNativeKey = "referencesNative"
referencesKeys = ["referencesBiencoderTop1","referencesBM25Top1","referencesBiencoderTop10Bm25Top1","referencesBM25Top10BiencoderTop1","referencesBiencoderTop10CrossEncoderTop1","referencesBM25Top10CrossEncoderTop1","referencesBiencoderAndBm25Top1","referencesBiencoderAndBm25Top10CrossEncoderTop1","referencesWithLLM"]


def compute_entailment2(context, claim):
    return entailmentChecker2.check_entailment(context, claim)


def set_entailment2(judgement, context, claim, stats):
    """
    Overwrites `judgement["entailment2"]` if it already exists, or adds it
    if this judgement never had one. Plain dict assignment handles both
    cases identically - this wrapper just tracks which happened, for a
    visible replaced-vs-added count at the end.
    """
    if "entailment2" in judgement:
        stats["replaced"] += 1
    else:
        stats["added"] += 1
    judgement["entailment2"] = compute_entailment2(context, claim)


def recompute_native(question, side_key, model_answer_key, stats):
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
        set_entailment2(judgement, entry["contextPassage"], sentence["sentence"], stats)


def recompute_ref_key(question, side_key, model_answer_key, refKey, stats):
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
        set_entailment2(entry["judgement"], entry["contextPassage"], sentence["sentence"], stats)


def recompute_question(question, stats):
    recompute_native(question, "withoutGold", "modelAnswer", stats)
    for refKey in referencesKeys:
        recompute_ref_key(question, "withoutGold", "modelAnswer", refKey, stats)

    recompute_native(question, "withGold", "mddelAnswerWithGold", stats)
    for refKey in referencesKeys:
        recompute_ref_key(question, "withGold", "mddelAnswerWithGold", refKey, stats)


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
        f"entailment2_replaced={stats['replaced']} entailment2_added={stats['added']} "
        f"elapsed={time.time() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
