#!/usr/bin/env python3
import plyvel
import argparse
import json
import time
import datetime
import requests
import os
from tqdm import tqdm
import logging
import numpy as np
import random
import string
import re
from typing import List, Tuple, Dict, Union
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing as mp
from performance_monitor import monitor, time_block
from agents.QuestionSplitter import QuestionSplitter 
from agents.PaperTitleExtractor import PaperTitleExtractor 
from agents.EnhancedCitationHandler import EnhancedCitationHandler
from agents.util import initialize_retriever, load_datamorgana_questions, format_enhanced_result_to_schema,write_enhanced_result_to_json, write_enhanced_results_to_jsonl
from agents.types import GeneratedAnswerFormat
from entailment_agent import EntailmentChecker
import time


logger = logging.getLogger("Enhanced_4Agent_RAG")
SCADS_API_KEY = os.getenv("SCADS_API_KEY")
# OUTPUT_FILE = "contextExtractionResult.jsonl"
INPUT_FILE="contextResultsToJudge.jsonl"
# OUTPUT_FILE_WITHOUT_GOLD="contextWithoutGold.jsonl"
# OUTPUT_FILE_WITH_GOLD="contextWithGold.jsonl"
# OUTPUT_FILE_SUCCESS="outputDone.jsonl"
# OUTPUT_FILE = "contextExtractionResultWithoutGoldGroundTruth.jsonl"


from scadsApiAgent import ScadsApiAgent
zaiAgent = ScadsApiAgent("zai-org/GLM-5.2-FP8")
gptOssAgent = ScadsApiAgent("openai/gpt-oss-120b")

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
            question = clean_and_parse_json(lineAlready)
            questions.append(question["generationMeta"]["anchorPaper"])

for question in questions:
    print(question["anchorPaper"])
    modelAnswerWithoutGold = question["answerMeta"]["modelAnswer"]

    nonGoldPaperMapping = {
        entry["docId"]: entry["paperId"]
        for entry in question["paperInformationWithoutGold"]
    }


    modelAnswerWithGold = question["answerMeta"]["mddelAnswerWithGold"]

    goldPaperMapping = {
        entry["docId"]: entry["paperId"]
        for entry in question["papersInformationGoldAnswer"]
    }
    print(json.dumps(nonGoldPaperMapping))
    print("-----------------")
    print(json.dumps(goldPaperMapping))
    print('####################')






    