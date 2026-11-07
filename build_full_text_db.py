#!/usr/bin/env python3
"""
Build the full-text LevelDB (config.DB_PATH) that recompute_fuzzy_ratio.py
and UnifiedArxivRetriever.get_full_texts expect, directly from the raw
unarXive 2024 JSONL dump:
https://huggingface.co/datasets/ines-besrour/unarxive_2024

Each input line is one paper record with "paper_id", "abstract.text" and
"sections" (dict of heading -> {"text": ...}); inline {{cite:<hash>}}
markers are stripped. The LevelDB value is the plain concatenated text,
matching what UnifiedArxivRetriever.get_full_texts decodes directly with
no JSON wrapper.

Usage - stream straight from the recombined tar.gz, no intermediate
extraction to disk:

    cat unarXive_2024.tar.gz.part_aa unarXive_2024.tar.gz.part_ab \
        unarXive_2024.tar.gz.part_ac | tar -xzO | \
        python3 build_full_text_db.py --db-path /path/to/full_text_db

Or point it at already-extracted .jsonl files:

    python3 build_full_text_db.py --db-path /path/to/full_text_db --input extracted/*.jsonl
"""
import argparse
import json
import re
import sys

import plyvel

CITE_MARKER_RE = re.compile(r"\{\{cite:[0-9a-f]+\}\}")


def paper_full_text(record):
    parts = []
    abstract = record.get("abstract", {}).get("text", "")
    if abstract:
        parts.append(abstract)
    for section in record.get("sections", {}).values():
        text = section.get("text", "") if isinstance(section, dict) else ""
        if text:
            parts.append(text)
    full_text = "\n\n".join(parts)
    return CITE_MARKER_RE.sub("", full_text)


def ingest_lines(lines, db, stats):
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["bad_lines"] += 1
            continue

        paper_id = record.get("paper_id")
        if not paper_id:
            stats["missing_id"] += 1
            continue

        full_text = paper_full_text(record)
        db.put(paper_id.encode("utf-8"), full_text.encode("utf-8"))
        stats["written"] += 1

        if stats["written"] % 50000 == 0:
            print(
                f"progress: written={stats['written']} "
                f"bad_lines={stats['bad_lines']} missing_id={stats['missing_id']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--input", nargs="*", help="One or more .jsonl files; omit to read from stdin")
    args = parser.parse_args()

    db = plyvel.DB(args.db_path, create_if_missing=True)
    stats = {"written": 0, "bad_lines": 0, "missing_id": 0}

    try:
        if args.input:
            for path in args.input:
                with open(path, "r", encoding="utf-8") as f:
                    ingest_lines(f, db, stats)
        else:
            ingest_lines(sys.stdin, db, stats)
    finally:
        db.close()

    print(f"DONE. written={stats['written']} bad_lines={stats['bad_lines']} missing_id={stats['missing_id']}")


if __name__ == "__main__":
    main()
