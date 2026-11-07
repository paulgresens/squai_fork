#!/usr/bin/env python3
"""
Recompute only the `noise` metric on an already-produced evaluation results
file (the output of run_SQuAI_evaluation*.py), using the corrected
sentence-splitting logic, and run the entailment checker on GPU for speed.

Every other metric already present in each judgement (faithfulness,
contextRelevance, entailment, noise, answerCorrectness, answerRelevancy) is
left untouched - this script only adds/overwrites the "noise3" key.
"""
import json
import re
import time

from attrscore_agent import AttrScoreChecker

INPUT_FILE = "evaluationResultCombinedWithNoise2.jsonl"
OUTPUT_FILE = "evaluationResultCombinedWithNoise3.jsonl"

# AttrScoreChecker.check_attribution scores one span at a time - there's no
# batched equivalent of EntailmentChecker.get_entailments_for_spans, so
# there's no BATCH_SIZE here to tune. device="cuda" for Capella (not "mps" -
# this isn't running locally anymore).
attrScoreChecker = AttrScoreChecker(device="cuda")

referencesNativeKey = "referencesNative"
referencesKeys = ["referencesBiencoderTop1","referencesBM25Top1","referencesBiencoderTop10Bm25Top1","referencesBM25Top10BiencoderTop1","referencesBiencoderTop10CrossEncoderTop1","referencesBM25Top10CrossEncoderTop1","referencesBiencoderAndBm25Top1","referencesBiencoderAndBm25Top10CrossEncoderTop1","referencesWithLLM"]


def build_context_windows(context):
    raw_context_splits = re.split(r"([.!?]+)", context)
    contextSentences = []

    # Loop through splits and re-attach punctuation to the previous sentence
    for i in range(0, len(raw_context_splits), 2):
        sent = raw_context_splits[i].strip()
        punct = raw_context_splits[i + 1].strip() if i + 1 < len(raw_context_splits) else ""
        if sent:
            contextSentences.append(f"{sent}{punct}")

    # All contiguous sub-windows, every length from 1 up to len(contextSentences)
    contextWindows = []
    for start in range(len(contextSentences)):
        for end in range(start + 1, len(contextSentences) + 1):
            contextWindows.append(" ".join(contextSentences[start:end]))

    return contextWindows


def compute_noise(context, claim):
    contextWindows = build_context_windows(context)
    return [attrScoreChecker.check_attribution(window, claim) for window in contextWindows]


PROGRESS_INTERVAL_SECONDS = 10


def maybe_print_progress(stats):
    """
    Prints a cumulative evaluated count on a wall-clock cadence, independent
    of question/citation size - a per-question print alone can go quiet for
    a long time if one question has many citations.
    """
    now = time.time()
    if now - stats["last_print"] >= PROGRESS_INTERVAL_SECONDS:
        evaluated = stats["added"] + stats["replaced"]
        print(f"progress: evaluated={evaluated}")
        stats["last_print"] = now


def set_noise(judgement, context, claim, stats):
    """
    Overwrites `judgement["noise3"]` if it already exists (e.g. a retry of
    this script), or adds it if this judgement never had one. Plain dict
    assignment handles both cases identically - this wrapper just tracks
    which happened, for a visible replaced-vs-added count at the end.
    """
    if "noise3" in judgement:
        stats["replaced"] += 1
    else:
        stats["added"] += 1
    judgement["noise3"] = compute_noise(context, claim)
    maybe_print_progress(stats)


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
        set_noise(judgement, entry["contextPassage"], sentence["sentence"], stats)


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
        set_noise(entry["judgement"], entry["contextPassage"], sentence["sentence"], stats)


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
    stats = {"replaced": 0, "added": 0, "last_print": start}

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
            print(
                f"question {lineNumber} done ({anchor}): processed={processed} failed={failed} "
                f"evaluated={stats['added'] + stats['replaced']} "
                f"elapsed={elapsed:.1f}s"
            )

    print(
        f"DONE. processed={processed} failed={failed} "
        f"noise3_replaced={stats['replaced']} noise3_added={stats['added']} "
        f"elapsed={time.time() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
