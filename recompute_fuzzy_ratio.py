#!/usr/bin/env python3
"""
Standalone recompute of the hallucination-check fuzzy ratio
(EnhancedCitationHandler._extract_context_passages_using_llm, "variant 10" /
referencesWithLLM) on an already-produced generation output file, without
re-running the LLM.

Why this is possible at all: the only LLM-dependent part of that check is
`contextForSentence` (the quote the model claims it extracted) - that's
already stored per entry in the output. The score itself is just
rapidfuzz.fuzz.partial_ratio(contextForSentence, documentText), which needs
no model call. `documentText` is NOT stored in the output (only its length,
under answerMeta.paperFullTextLength) - this script re-fetches it from the
full-text DB by `paperId` (which IS stored per entry) and rebuilds it with
the exact same cleaning/truncation steps the original run used, so the
recomputed score matches what would have been computed inline.

Structure assumed for each input record (one per question), matching what
run_SQuAI.py writes and what EnhancedCitationHandler.py produces:

    record["withoutGold"]["referencesWithLLM"] = {
        "<documentId>": [
            {
                "contextPassage": "...",
                "paperId": "<arxiv id>",
                "hallucinationCheck": {
                    "contextForSentence": "...",
                    "partialRatioWithSourceText": <float>,
                },
            },
            ...
        ],
        ...
    }
    record["withGold"]["referencesWithLLM"] = { ... same shape ... }

"withoutGold" and "withGold" are recomputed separately because they come
from different retrieval runs (the gold run force-includes the two ground
truth papers) - but since every entry already carries its own `paperId`,
both sides are resolved the same way here: by paper id, not by position.

Input file may be either:
  - a single JSON object (one record, e.g. a pretty-printed testi.json), or
  - a JSON array of records, or
  - a JSONL file (one compact record per line, e.g. raw run_SQuAI.py output)
Output is always written as JSONL, one record per line, to a separate file
- the input file is never modified. Each hallucinationCheck entry gets two
new fields added alongside the original partialRatioWithSourceText:
recomputedPartialRatioWithSourceText and recomputeDelta.

Entries where contextForSentence is exactly MODEL_DECLINED_RESPONSE are
excluded from the recompute entirely (recomputedPartialRatioWithSourceText
is set to None, modelDeclinedToAnswer is set to True, no delta is computed).
That string is not something the LLM ever "decided" to write - it's a
hardcoded fallback in ScadsApiAgent.generate/FalconAgent.generate substituted
whenever the underlying API call returned an empty completion, so scoring it
for fuzzy/hallucination purposes would be scoring an API failure, not a bad
extraction.

Usage:
    python recompute_fuzzy_ratio.py --input testi.json --output testi.recomputed.jsonl --db_path /path/to/full_text_db
"""
import argparse
import json
import logging
import re
import time

import plyvel
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("recompute_fuzzy_ratio")

INPUT_FILE = "evaluationResultCombinedWithNoise3.jsonl"
OUTPUT_FILE = "evaluationTotalWithFuzzyRecomputed.jsonl"

REFERENCES_WITH_LLM_KEY = "referencesWithLLM"
SIDE_KEYS = ("withoutGold", "withGold")

# Hardcoded fallback in ScadsApiAgent.generate/FalconAgent.generate when the
# underlying API call returns an empty completion - not a real extraction.
MODEL_DECLINED_RESPONSE = "I don't have enough information to provide a specific answer."

# must match EnhancedCitationHandler._extract_context_passages_using_llm
MAX_DOCUMENT_TEXT_CHARS = 999999999
TRUNCATION_HEAD_CHARS = 64000
TRUNCATION_TAIL_CHARS = 64000
TRUNCATION_MARKER = "\n\n... [TEXT OMITTED FOR LENGTH] ...\n\n"


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse {path} as JSON or JSONL (line {line_number}): {e}")
        return records

    if isinstance(data, list):
        return data
    return [data]


def clean_document_text(raw_text):
    """Mirrors prepareDocumentsForFinalAnswerGenerator's clean_text step
    plus the 128k-char head/tail truncation applied right before the
    hallucination check in EnhancedCitationHandler.py."""
    clean_text = re.sub(r"Content for [^:]*:\s*\n", "", raw_text).strip()
    if len(clean_text) > MAX_DOCUMENT_TEXT_CHARS:
        clean_text = (
            clean_text[:TRUNCATION_HEAD_CHARS]
            + TRUNCATION_MARKER
            + clean_text[-TRUNCATION_TAIL_CHARS:]
        )
    return clean_text


def get_document_text(paper_id, db, cache, stats):
    if paper_id in cache:
        return cache[paper_id]

    try:
        content = db.get(paper_id.encode("utf-8"))
    except Exception as e:
        logger.warning(f"db.get failed for paperId={paper_id}: {e}")
        content = None

    if content is None:
        logger.warning(f"no full text found for paperId={paper_id}")
        stats["missing_document"] += 1
        stats["missing_paper_ids"].add(paper_id)
        cache[paper_id] = None
        return None

    raw_text = content.decode("utf-8")
    cache[paper_id] = clean_document_text(raw_text)
    stats["papers_loaded"] += 1
    logger.info(f"loaded full text for paperId={paper_id} ({stats['papers_loaded']} papers loaded so far)")
    return cache[paper_id]


def recompute_partial_ratio(contextForSentence, documentText):
    # Variable names mirror EnhancedCitationHandler.py exactly (the names
    # there are swapped relative to their content, but fuzz.partial_ratio
    # is symmetric so this does not change the result).
    cleanedDocumentText = re.sub(r"\s+", "", contextForSentence).lower()
    cleanedContextSentence = re.sub(r"\s+", "", documentText).lower()
    return fuzz.partial_ratio(cleanedContextSentence, cleanedDocumentText)


def recompute_side(record, side_key, db, cache, stats):
    refs = record.get(side_key, {}).get(REFERENCES_WITH_LLM_KEY)
    if not refs:
        return

    for documentId, entries in refs.items():
        for entry in entries:
            hallucinationCheck = entry.get("hallucinationCheck")
            if hallucinationCheck is None:
                continue

            paperId = entry.get("paperId")
            contextForSentence = hallucinationCheck.get("contextForSentence", entry.get("contextPassage"))

            if contextForSentence == MODEL_DECLINED_RESPONSE:
                hallucinationCheck["recomputedPartialRatioWithSourceText"] = None
                hallucinationCheck["modelDeclinedToAnswer"] = True
                stats["declined"] += 1
                continue

            documentText = get_document_text(paperId, db, cache, stats)
            if documentText is None:
                hallucinationCheck["recomputedPartialRatioWithSourceText"] = None
                continue

            recomputed = recompute_partial_ratio(contextForSentence, documentText)
            hallucinationCheck["recomputedPartialRatioWithSourceText"] = recomputed

            original = hallucinationCheck.get("partialRatioWithSourceText")
            if isinstance(original, (int, float)):
                delta = recomputed - original
                hallucinationCheck["recomputeDelta"] = delta
                stats["deltas"].append(delta)

            stats["recomputed"] += 1


def recompute_record(record, db, cache, stats):
    for side_key in SIDE_KEYS:
        recompute_side(record, side_key, db, cache, stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=INPUT_FILE, help="Generation output file (JSON, JSON array, or JSONL)")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Where to write the annotated records (JSONL) - input file is never modified")
    parser.add_argument("--db_path", required=True, help="Path to the full-text LevelDB database")
    args = parser.parse_args()

    logger.info(f"Opening full-text database at {args.db_path}...")
    db = plyvel.DB(args.db_path, create_if_missing=False)

    logger.info(f"Loading records from {args.input}...")
    records = load_records(args.input)
    logger.info(f"Loaded {len(records)} record(s)")

    document_text_cache = {}
    stats = {"recomputed": 0, "missing_document": 0, "papers_loaded": 0, "deltas": [], "missing_paper_ids": set(), "declined": 0}

    start = time.time()
    with open(args.output, "w", encoding="utf-8") as out_file:
        for i, record in enumerate(records, start=1):
            try:
                recompute_record(record, db, document_text_cache, stats)
            except Exception as e:
                logger.error(f"FAILED on record {i}: {e}")
                continue

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
            logger.info(f"processed record {i}/{len(records)} (recomputed={stats['recomputed']}, missing_document={stats['missing_document']}, declined={stats['declined']})")

    db.close()

    deltas = stats["deltas"]
    papers_attempted = stats["papers_loaded"] + stats["missing_document"]
    logger.info(f"DONE in {time.time() - start:.1f}s")
    logger.info(f"  papers successfully loaded: {stats['papers_loaded']}/{papers_attempted}")
    logger.info(f"  papers missing/failed:      {stats['missing_document']}/{papers_attempted}")
    if stats["missing_paper_ids"]:
        logger.info(f"  missing paperIds ({len(stats['missing_paper_ids'])}): {sorted(stats['missing_paper_ids'])}")
    logger.info(f"  entries recomputed:        {stats['recomputed']}")
    logger.info(f"  entries excluded (model declined to answer): {stats['declined']}")
    if deltas:
        changed = [d for d in deltas if abs(d) > 1e-6]
        logger.info(f"  entries with a changed score: {len(changed)}/{len(deltas)}")
        logger.info(f"  mean |delta|: {sum(abs(d) for d in deltas) / len(deltas):.2f}")
        logger.info(f"  max |delta|:  {max(abs(d) for d in deltas):.2f}")
    logger.info(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
