#!/usr/bin/env python3
"""
Enhanced 4-Agent RAG System with Question Splitting and Parallel Processing
- Agent 1: Question Splitter
- Agent 2: Answer Generator from abstracts
- Agent 3: Document Evaluator
- Agent 4: Final Answer Generator with citations
"""
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

# Import configuration
from config import E5_INDEX_DIR, BM25_INDEX_DIR, DB_PATH
logger = logging.getLogger("Enhanced_4Agent_RAG")

# Your existing logging setup (unchanged)
def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/enhanced_4agent_rag_{timestamp}_{random_str}.log"


os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("Enhanced_4Agent_RAG")



def askScadsApiLLM(prompt):
    url = "https://llm.scads.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SCADS_API_KEY}"
    }

    # 3. Create your prompt payload
    payload = {
        "model": "MiniMaxAI/MiniMax-M3-MXFP8",
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
    # Convert the response to JSON and extract the text
    try:
        data = response.json()
    except:
        return None

    time.sleep(2)
    return (data["choices"][0]["message"]["content"])


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



class Enhanced4AgentRAG:
    """
    Enhanced 4-Agent RAG System with Question Splitting, Parallel Processing, and Context Management
    """
    def __init__(
        self,
        retriever,
        n=0.0,
        index_dir="test_index",
        max_workers=4,
        max_context_chars=35000,
    ):
        """Initialize with enhanced 4-agent architecture and context management"""

        self.retriever = retriever
        self.n = n
        self.index_dir = index_dir
        self.max_workers = max_workers
        self.max_context_chars = max_context_chars  # Conservative limit for Falcon-10B

        logger.info(f"Context limit set to {max_context_chars} characters")

        # Initialize agents - defaults disabled for now, because judge and answer generator should be different models
        # if isinstance(agent_model, str):
        #     if "falcon" in agent_model.lower() and falcon_api_key:
        #         from api_agent import FalconAgent
        #         self.agent1 = FalconAgent(falcon_api_key)  # Question Splitter
        #     else:
        #         from local_agent import LLMAgent
        #         self.agent1 = LLMAgent(agent_model)  # Question Splitter
        #         logger.info(f"Using local LLM agents with model {agent_model}")
        # else:
        #     self.agent1 = agent_model  # Question Splitter
        #     logger.info("Using pre-initialized agent for all four agent roles")
        
        # Question Splitter, Answer Generator, Document Evaluator, Final Answer Generator, Content Extractor, Judge
        # from local_agent import LLMAgent
        ##werte
        # agentsSet = {
        #     questionSplitterModel, answerGeneratorModel ,documentEvaluatorModel ,finalAnswerGeneratorModel ,contentExtractorModel, judgeModel  
        # }

        # model als key, generierter Agent als Wert 
        # agents = {agent: LLMAgent(agent) for agent in agentsSet}

        # self.agentMapping = {
        #     'questionSplitterModel': agents[questionSplitterModel],
        #     'answerGeneratorModel': agents[answerGeneratorModel],
        #     'documentEvaluatorModel': agents[documentEvaluatorModel],
        #     'finalAnswerGeneratorModel': agents[finalAnswerGeneratorModel],
        #     'contentExtractorModel': agents[contentExtractorModel],
        #     'judgeModel': agents[judgeModel],
        # }
        from scadsApiAgent import MinimaxAgent
        self.scadsApiAgent = MinimaxAgent(6000)

        self.question_splitter = QuestionSplitter(self.scadsApiAgent, logger)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # self.entailmentChecker = EntailmentChecker()
        
        logger.info("Enhanced 4-agent pre-warming...")
        # try:
        #     # Warm up retriever
        #     dummy_abstracts = self.retriever.retrieve_abstracts("test", top_k=1)
        #     logger.info("Retriever pre-warmed")

        #     # Warm up agents
        #     if hasattr(self.agent1, "generate"):
        #         self.agent1.generate("test")
        #         logger.info("All agents pre-warmed")

        # except Exception as e:
        #     logger.warning(f"Pre-warming had issues: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 chars per token"""
        return len(text) // 4

    def createAnswerGeneratorPrompt(self, query, document):
        """Agent-2 prompt: Answer generation from abstracts"""
        return f"""You are an accurate and reliable AI assistant that can answer questions with the help of external documents. You should only provide the correct answer without repeating the question and instruction.
            Document: {document}
            Question: {query}
            Answer:"""

    def createDocumentEvaluatorPrompt(self, query, document, answer):
        """Agent-3 prompt: Document evaluation"""
        return f"""You are a noisy document evaluator that can judge if the external document is noisy for the query with unrelated or misleading information. Given a retrieved Document, a Question, and an Answer generated by an LLM (LLM Answer), you should judge whether both the following two conditions are reached: (1) the Document provides specific information for answering the Question; (2) the LLM Answer directly answers the question based on the retrieved Document. Please note that external documents may contain noisy or factually incorrect information. If the information in the document does not contain the answer, you should point it out with evidence. You should answer with exactly "Yes" or "No" with evidence of your judgment, where "No" means one of the conditions (1) and (2) are unreached and indicates it is a noisy document.
            Document: {document}        
            Question: {query}       
            LLM Answer: {answer}
            Remember the only answer possibilies are the strings "Yes" and "No"
            Is this document relevant and supportive for answering the question?"""

    def prepareDocumentsForFinalAnswerGenerator(
        self,
        full_texts: List[Tuple[str, str]],
        citation_handler,
        was_split: bool = False,
    ) -> List[str]:
        """
        Prepare documents for Agent 4 with dynamic context length management

        Args:
            full_texts: List of (document_text, doc_id) tuples
            citation_handler: Citation handler instance
            was_split: Whether the original question was split into sub-questions

        Returns:
            List of formatted document strings ready for the prompt
        """
        docs_with_citations = []
        total_chars = 0
        documents_used = 0

        # Dynamic context allocation - top + bottom extraction approach
        if was_split:
            # Conservative: Target ~4K total per paper
            top_chars = 2500  # Top of paper (title, abstract, intro start)
            bottom_chars = 1500  # Bottom of paper (conclusion, results)
            strategy = "CONSERVATIVE (split questions)"
            target_per_paper = "~4K"
        else:
            # Generous: Target ~8K total per paper
            top_chars = 5000  # More from top (title, abstract, intro)
            bottom_chars = 3000  # More from bottom (conclusion, results)
            strategy = "GENEROUS (single question)"
            target_per_paper = "~8K"

        logger.info(
            f"Preparing documents for Agent 4 (context limit: {self.max_context_chars} chars)"
        )
        logger.info(
            f"   Context strategy: {strategy} - targeting {target_per_paper} chars per paper"
        )
        logger.info(
            f"   Extraction: TOP({top_chars} chars) + BOTTOM({bottom_chars} chars) + Title"
        )
        cleanFullDocumentTexts = {}
        for i, (doc_text, doc_id) in enumerate(full_texts):
            # New approach: Extract from top and bottom of paper
            condensed_content = []

            # Extract title first (if available)
            title = PaperTitleExtractor.extract_title_from_text(doc_text, doc_id)
            if title and not title.startswith("Document "):
                condensed_content.append(f"Title: {title}")

            # Remove "Content for [paper_id]:" line and other metadata for cleaner extraction
            clean_text = doc_text
            # Remove the "Content for" line
            clean_text = re.sub(r"Content for [^:]*:\s*\n", "", clean_text)
            # Remove any leading whitespace/newlines
            clean_text = clean_text.strip()

            # TOP EXTRACTION: Get beginning of paper (naturally includes abstract, intro start)
            top_text = clean_text[:top_chars]
            if len(clean_text) > top_chars:
                # Find a good breaking point (end of sentence)
                break_point = top_text.rfind(". ")
                if (
                    break_point > top_chars * 0.8
                ):  # If we find a sentence end in the last 20%
                    top_text = top_text[: break_point + 1]
                else:
                    top_text += "..."

            condensed_content.append(f"[TOP {len(top_text)} chars]: {top_text}")

            # BOTTOM EXTRACTION: Get end of paper (naturally includes conclusion, results)
            if (
                len(clean_text) > top_chars + 100
            ):  # Only add bottom if there's enough remaining content
                bottom_text = clean_text[-bottom_chars:]
                if len(clean_text) > bottom_chars:
                    # Find a good starting point (beginning of sentence)
                    start_point = bottom_text.find(". ")
                    if (
                        start_point > 0 and start_point < bottom_chars * 0.2
                    ):  # If we find sentence start in first 20%
                        bottom_text = bottom_text[start_point + 2 :]  # +2 to skip ". "
                    else:
                        bottom_text = "..." + bottom_text

                condensed_content.append(
                    f"[BOTTOM {len(bottom_text)} chars]: {bottom_text}"
                )

            condensed_text = "\n\n".join(condensed_content)

            # Check if adding this document would exceed context limit
            estimated_doc_size = len(condensed_text) + 200  # +200 for formatting

            if (
                total_chars + estimated_doc_size > self.max_context_chars
                and documents_used > 0
            ):
                logger.info(
                    f"Context limit reached. Using {documents_used} out of {len(full_texts)} documents"
                )
                break

            # Add document with citation
            citation_num = citation_handler.add_document(condensed_text, doc_id)

            cleanFullDocumentTexts[int(citation_num)] = clean_text
            # Get paper info for better document labeling
            paper_info = citation_handler.citation_to_doc[citation_num]["paper_info"]
            doc_title = (
                paper_info["title"][:80] + "..."
                if len(paper_info["title"]) > 80
                else paper_info["title"]
            )

            formatted_doc = (
                f'Document [{citation_num}] - "{doc_title}":\n{condensed_text}'
            )
            docs_with_citations.append(formatted_doc)

            total_chars += estimated_doc_size
            documents_used += 1

            logger.info(
                f"  Added doc [{citation_num}]: {doc_title[:60]}... ({len(condensed_text)} chars)"
            )

        logger.info(
            f"Total context size: {total_chars} chars (~{self._estimate_tokens(str(total_chars))} tokens)"
        )
        logger.info(f"Using {documents_used}/{len(full_texts)} documents for Agent 4")
        logger.info("HERE CLEAN FULL TEXT DOCUMENT TEXT")
        logger.info(json.dumps({key: len(text) for key, text in cleanFullDocumentTexts.items()}))

        return docs_with_citations, cleanFullDocumentTexts

    def createFinalAnswerGeneratorPrompt(
        self, original_query, full_texts, citation_handler, was_split: bool = False
    ):
        """Agent-4 prompt with context-aware document preparation"""

        # Prepare documents with dynamic context management based on question splitting
        docs_with_citations,cleanFullDocumentTexts = self.prepareDocumentsForFinalAnswerGenerator(
            full_texts, citation_handler, was_split
        )

        docs_text = "\n\n" + "=" * 50 + "\n\n".join(docs_with_citations)

        # Count available citation numbers
        available_citations = [str(i) for i in range(1, len(docs_with_citations) + 1)]
        citation_examples = ", ".join(available_citations)

        return f"""
        You are an accurate and reliable AI assistant.
        You will be given will be given chunks from a variety of scientific papers, representing their topmost and bottom most sections, as well as a question. The papers are preselected to provide as much information as possible for you to be able to answer the question on the basis of the information provided there. You can always assume that the information from the paper is trustworthy and logically correct. Use the information that is provided in the paper as ground truth to answer the question.

        For your generated answer and for it to be easily verifiable it is also important to add context. Each claim that you make when answering the question must reference one or multiple documents, on whose information you justify making that claim.

        Input Format:
        Your input will have the following structure, repeated for each individual document, that you should use for answering:

        "Document [DOCUMENT_NUMBER]" - "TITLE OF THE PAPER"
        [TOP K chars]: "THE TEXT THAT MAKES UP THE TOP K CHARACTERS OF THE PAPER"
        [BOTTOM L chars]: "THE TEXT THAT MAKES UP THE BOTTOM L CHARACTERS OF THE PAPER"

        The document number provided here is the one that you must use for referencing the paper and justifying your answer. References are only allowed to be provided in the json. The Answer Sentences itself should not contain any kind of hint towards the documentId which their claim is based upon.

        Referencing instructions
        References must be added for every sentence, linking the document where the underlying information for justifying the claim in the sentence lies.  
        Try referencing multiple papers, so that the best fitting context for each sentence is chosen.

        You are only allowed to use The DOCUMENT_NUMBER for that reference, which is provided to you. 
        You are only allowed to add one document as a reference for every sentence you produce.


        Output Format:
        After generating your answer text, split the text into sentences. You are only allowed to answer in the following json schema, which is an array of all the sentences in your answer with their respective DOCUMENT_NUMBER as second key of the object.
        
        [
          {{
            "sentence": <The first generated answer sentence>,
            "documentId": <The DOCUMENT_NUMBER for justifying the first sentence> 
          }},
          {{
            "sentence": <The second generated answer sentence>,
            "documentId": <The DOCUMENT_NUMBER for justifying the second sentence> 
          }},
          ...
        ]
        
        correct:
        [
          {{
            "sentence": "We propose LexBoost that first builds a network of dense neighbors (a corpus graph) using a dense retrieval approach while indexing.",
            "documentId": 2 
          }},
          {{
            "sentence": "We show theoretically and empirically that the performance for dense representations decreases quicker than sparse representations for increasing index sizes.",
            "documentId": 1 
          }},
          {{
            "sentence": "However, these approaches suffer from the lexical gap problem. To overcome this issue, dense representations have been proposed : Queries and documents are mapped to a dense vector space and relevant documents are retrieved.",
            "documentId": 2 
          }}
        ]
        wrong:
        [
          {{
            "sentence": "We propose LexBoost that first builds a network of dense neighbors (a corpus graph) using a dense retrieval approach while indexing [3].",
            "documentId": 2 
          }},
          {{
            "sentence": "We show theoretically and empirically that the performance for dense representations decreases quicker than sparse representations for increasing index sizes as dicussed in document 1.",
            "documentId": 1 
          }}
        ]
        Do not add anything else to your response, stricly follow the json schema. Do not add a reference section, comments or further explanations. Also do not add "Answer: " or similar before providing your answer. Your only answer should be the json.
        
        Documents: {docs_text}
        Question: {original_query}
        """,cleanFullDocumentTexts
    
    def _log_retrieved_papers(
        self, query: str, retrieved_abstracts: List[Tuple], phase: str = "RETRIEVAL"
    ):
        """Log the titles of retrieved papers with improved title extraction"""
        if not retrieved_abstracts:
            logger.info(f"{phase}: No papers retrieved for query: {query[:50]}...")
            return

        logger.info(
            f"{phase}: Retrieved {len(retrieved_abstracts)} papers for query: {query[:50]}..."
        )
        logger.info("=" * 80)

        for i, (abstract_text, doc_id) in enumerate(retrieved_abstracts, 1):
            # Extract title using improved utility
            title = PaperTitleExtractor.extract_title_from_text(abstract_text, doc_id)
            formatted_title = PaperTitleExtractor.format_title_for_log(
                title, max_length=70
            )

            logger.info(f"  [{i:2d}] {formatted_title}")
            logger.info(f"       Doc ID: {doc_id}")

        logger.info("=" * 80)

    def _log_filtered_papers(
        self, query: str, filtered_abstracts: List[Tuple], scores: List[float]
    ):
        """Log the titles of papers that passed Agent 3 filtering"""
        if not filtered_abstracts:
            logger.info(
                f"FILTERING: No papers passed Agent 3 filter for query: {query[:50]}..."
            )
            return

        logger.info(
            f"FILTERING: {len(filtered_abstracts)} papers passed Agent 3 filter for query: {query[:50]}..."
        )
        logger.info("=" * 80)

        # Sort by score for display
        combined = list(zip(filtered_abstracts, scores))
        combined.sort(key=lambda x: x[1], reverse=True)

        for i, ((abstract_text, doc_id, _), score) in enumerate(combined, 1):
            # Extract title using improved utility
            title = PaperTitleExtractor.extract_title_from_text(abstract_text, doc_id)
            formatted_title = PaperTitleExtractor.format_title_for_log(
                title, max_length=65
            )

            logger.info(f"  ✓ [{i:2d}] {formatted_title} (score: {score:.3f})")
            logger.info(f"        Doc ID: {doc_id}")

        logger.info("=" * 80)

    def _log_context_usage(
        self, full_texts: List[Tuple], docs_used: int, was_split: bool = False
    ):
        """Log context usage statistics with dynamic strategy info"""
        total_chars = sum(len(text) for text, _ in full_texts)
        avg_chars = total_chars // len(full_texts) if full_texts else 0

        strategy = (
            "CONSERVATIVE (split questions)"
            if was_split
            else "GENEROUS (single question)"
        )
        chars_per_paper = (
            "TOP(2.5K)+BOTTOM(1.5K)" if was_split else "TOP(5K)+BOTTOM(3K)"
        )

        logger.info(f"   CONTEXT USAGE [{strategy}]:")
        logger.info(f"   Available papers: {len(full_texts)}")
        logger.info(f"   Papers sent to Agent 4: {docs_used}")
        logger.info(f"   Total characters available: {total_chars:,}")
        logger.info(f"   Average per paper: {avg_chars:,} chars")
        logger.info(f"   Context limit: {self.max_context_chars:,} chars")
        logger.info(f"   Strategy: {chars_per_paper}")

        if total_chars > self.max_context_chars:
            logger.info(
                f"Full papers exceed context limit by {total_chars - self.max_context_chars:,} chars"
            )
            logger.info(f"Using condensed sections to fit within limits")

    def createJudgePrompt(self, claim, context):
        return  f""" 
        You are an accurate and reliable LLM-as-a-judge worker. Your tasks is to evaluate how well a claim from a generated answer is supported by the context in form of a couple of extracted sentences from a scientific paper. You can always assume that the information in the paper and thus in the context is factually correct.
        Input Format:
        You will recieve the following input:
        - Claim: A single sentence generated by an arbitrary LLM.
        - Context: A couple of sentences extracted from a scientific paper that may or may not justify the claim.
        Evaluation Instructions:
        Evaluate how well the claim is grounded in the context. For each of the provided criteria provided below, you should rate on a scale of 1-5 how well it is fullfilled. The scores have the following meaning:
        - 1: Not Fullfilled - the criteria is not fullfilled
        - 2: Partly Fullfilled - the criteria is partly fullfilled, but there are gaps, ambiguity or weaknesses
        - 3: Fully Fullfilled - the criteria is fully satisfied
        Use the following criterias for your rating:
        1. Faithfulness:
           Faithfulness measures how well the claim made is based on the factual content of the context. This includes but not ends with:
           - Is the claim the logical consequence of the facts in the context?
           - Does the claim overgeneralize and thus produce unfounded conclusions, not based on the facts in the context?
           - Does the claim include interpretations of the information on the context, which lack further data to be correctly and justifiedly drawn?
           A good rating here means, that the claim made is based on the factual content, a bad rating means it is not.
        2. Relevance:
           Relevance measures how well the topic in the context overlaps thematically with that in the generated claim and how precisely they address the same topics This includes:
           - Is the topic of the claim also part of the topic of the context?
           - Is each topic that is present in the claim also discussed in the context?
           - Does the context include extra information not needed for justifying the claim?
           A good rating here means, that the context is relevenant for the topic, a bad rating means it is not.
        3. Consistency:
           Consistency measures if the claim is logically consistent in respect to the information in the claim. This includes:
           - Does the claim contradict the information in the context?
           - Does the claim contain conclusions or generalizations that are contradicted by the context?
           - Does the information in the context allow a conclusion that directly contradicts the information in the claim?
           A good rating here means, that the claim is logically consistent, a bad rating means it is not.
        4. Support Coverage:
           Support Coverage measures wether the the context includes sufficient information to justify the claim. This includes:
           - Is each part of the claim justified by information in the context?
           A good rating here means, that the context includes sufficient information, a bad rating means it does not.
        5. Paraphrase Robustness:
           Paraphrase Robustness measures wether the semantic meaning of the claim and the context are the same, even if they are worded differently. This includes:
           - Does the claim convey the same meaning as the context?
           - Does the claim include the same semantic entities as the context?
           - Can the claim be interpreted differently or does it meaning differ from that of the context due to different wording?
           A good rating here means, that the semantic meaning of the claim and the context are the same, a bad rating means they are not.
        6. Ambiguity Level:
           Ambiguity Level measures wether the information in the context could be interpreted differently as done in the claim. This includes:
           - Is the information in the context ambiguous with respect to the claim?
           - Are there parts of the context that could be interpreted differently than done in the claim?
           A good rating here means, that the information in the context is unambiguous, a bad rating means it is not.
        Output Format:
        You will answer ONLY in form of the JSON Schema provided below.
        {{
          "faithfulness" : <1-5>,
          "relevance" : <1-5>,
          "consistency" :<1-5>,
          "supportCoverage" : <1-5>,
          "paraphraseRobustness" : <1-5>,
          "ambiguityLevel": <1-5>,
        }}
        Dont add any extra information, explanation or thoughts. Strictly follow the JSON format. Your only answer should be the json.
        Each "<1-5>" bracket should be replaced with the score of the corresponding metric, also following valid JSON structure. 
        Claim: "{claim}"
        Context: "{context}"
        """

    # def judgeClaim (self, sentence: str, context: str)->str:
    #     prompt = self.createJudgePrompt(sentence, context)
    #     return self.agentMapping["judgeModel"].generate(prompt)
    
    def checkEntailment(self, sentence:str, context:str):
        return self.entailmentChecker.check_entailment(context, sentence)


    ANSWER_KEYS = [
        "faithfulness",
        "relevance",
        "consistency",
        "supportCoverage",
        "paraphraseRobustness",
        "ambiguityLevel",
    ]

    OneContextForOneSource = Dict[str, str]
    
    # MultipleContextsForOneSource = Dict[str, Dict[str, List[str]]]
    # def judgeContextWithReferences (self, answerObject: GeneratedAnswerFormat, context: MultipleContextsForOneSource, multiple: bool = True ) -> str:
    #     result = []
    #     totalResult = {}

    #     for answerEntry in answerObject:
    #         citationNumber = answerEntry["documentId"]
    #         #todo add this entailment checker in again
    #         # entailment = self.checkEntailment(answerEntry["sentence"], context[citationNumber][0])

    #         judgement = json.loads(self.judgeClaim(answerEntry["sentence"], context[citationNumber][0]["context"]))

    #         #if multiple delete, so the next occurance gets judged by its respective context
    #         if (multiple):
    #             del context[citationNumber][0]
    #         result.append(judgement)
    #         for answerKey in self.ANSWER_KEYS:
    #             totalResult.setdefault(answerKey, [judgement[answerKey]]).append(judgement[answerKey])
    #     for answerKey, values in totalResult.items():
    #         totalResult[answerKey] = sum(values) / len(values)    
    #     result.append(totalResult)
    #     return result


    '''
    answerObject
     [
          {{
            "sentence": string
            "documentId": number 
          }}, 
          ...
    ]
    contexts:
    {
  "referencesNative": {
    "2": {
      "title": "Unknown Title",
      "paperId": "arXiv:cond-mat/0211218",
      "contextPassage": "We generate packings by both pouring and sedimentation and examine how the final state depends on the method of construction. The vertical stress becomes depth-independent for deep piles and we compare these stress depth-profiles to the classical Janssen theory."
    }
  },
  "referencesWithCosineSimilarity": {
    "4": [
      {"context": "Title: Particle Shape Effects on the Stress Response of Granular Packings\n\n[TOP 4794 chars]: Particle Shape Effects on the Stress Response of Granular Packings\nabstract: We present measurements of the stress response of packings formed from a wide range of particle shapes. Besides spheres these include convex shapes such as the Platonic solids, truncated tetrahedra, and triangular bipyramids, as well as more complex, non-convex geometries such as hexapods with various arm lengths, dolos, and tetrahedral frames."}
    ],
  },
  "referencesWithCosineSimilarityAndKeywordMatching": {
    "4": [
      {"context": "Title: Particle Shape Effects on the Stress Response of Granular Packings\n\n[TOP 4794 chars]: Particle Shape Effects on the Stress Response of Granular Packings\nabstract: We present measurements of the stress response of packings formed from a wide range of particle shapes. Besides spheres these include convex shapes such as the Platonic solids, truncated tetrahedra, and triangular bipyramids, as well as more complex, non-convex geometries such as hexapods with various arm lengths, dolos, and tetrahedral frames."}
    ]
  },
  "referencesWithCosineSimilarityAndCrossEncoder": {
    "4": [
      { "context": "Title: Particle Shape Effects on the Stress Response of Granular Packings\n\n[TOP 4794 chars]: Particle Shape Effects on the Stress Response of Granular Packings\nabstract: We present measurements of the stress response of packings formed from a wide range of particle shapes. Besides spheres these include convex shapes such as the Platonic solids, truncated tetrahedra, and triangular bipyramids, as well as more complex, non-convex geometries such as hexapods with various arm lengths, dolos, and tetrahedral frames. All particles were 3D-printed in hard resin. Well-defined initial packing states were established through preconditioning by cyclic loading under given confinement pressure. Starting from such initial states, stress-strain relationships for axial compression were obtained at four different confining pressures for each particle type."},
    ],
  },
  "referencesWithBiencoderAndBm25": {
    "4": [
      {"context": "Title: Particle Shape Effects on the Stress Response of Granular Packings\n\n[TOP 4794 chars]: Particle Shape Effects on the Stress Response of Granular Packings\nabstract: We present measurements of the stress response of packings formed from a wide range of particle shapes. Besides spheres these include convex shapes such as the Platonic solids, truncated tetrahedra, and triangular bipyramids, as well as more complex, non-convex geometries such as hexapods with various arm lengths, dolos, and tetrahedral frames."}
    ],  
  },
  "referencesWithBiencoderAndBm25AndCrossEncoder": {
    "4": [
      {"context": "Title: Particle Shape Effects on the Stress Response of Granular Packings\n\n[TOP 4794 chars]: Particle Shape Effects on the Stress Response of Granular Packings\nabstract: We present measurements of the stress response of packings formed from a wide range of particle shapes. Besides spheres these include convex shapes such as the Platonic solids, truncated tetrahedra, and triangular bipyramids, as well as more complex, non-convex geometries such as hexapods with various arm lengths, dolos, and tetrahedral frames. All particles were 3D-printed in hard resin. Well-defined initial packing states were established through preconditioning by cyclic loading under given confinement pressure. Starting from such initial states, stress-strain relationships for axial compression were obtained at four different confining pressures for each particle type."},
    ],
  },
  "referencesWithLLM": {
    "4": [
      {"context": "Besides spheres these include convex shapes such as the Platonic solids, truncated tetrahedra, and triangular bipyramids, as well as more complex, non-convex geometries such as hexapods with various arm lengths, dolos, and tetrahedral frames."},
    ]
  }
}

    
    '''
    method_keys = [
            "referencesBiencoderTop1",
            "referencesBiencoderTop10Bm25Top1",
            "referencesBiencoderTop10CrossEncoderTop1",
            "referencesBiencoderAndBm25Top1",
            "referencesBiencoderAndBm25Top10CrossEncoderTop1",
            "referencesWithLLM"
        ]
    def judgeContextsWithReferences(self, answerObject: GeneratedAnswerFormat, contexts ):
        for answerEntry in answerObject:
            # TODO native entries judgement missing
            for method in self.method_keys: 
                context_list_at_that_document_key = contexts[method][answerEntry["documentId"]]
                unjudged = next((item for item in context_list_at_that_document_key if "judgement" not in item), None)
                unjudged["sentence"] = answerEntry["sentence"]
                unjudged["judgement"] = json.loads(self.judgeClaim(answerEntry["sentence"], unjudged["context"]))
                unjudged["entailment"] = self.entailmentChecker.check_entailment(unjudged["context"], answerEntry["sentence"])
        return contexts
    
    def addMeanJudgements(self, contexts):
        judgementKeys = [
            "faithfulness", 
            "relevance",
            "consistency",
            "supportCoverage",
            "paraphraseRobustness",
            "ambiguityLevel"
        ]

        for method in self.method_keys:
            if method not in contexts: continue

            all_judgements = [
                entry["judgement"] 
                for doc_entries in contexts[method].values() 
                if isinstance(doc_entries, list)
                for entry in doc_entries 
                if "judgement" in entry
            ]
            
            if not all_judgements:
                contexts[method]["meanJudgements"] = {k: 0 for k in judgementKeys}
                continue

            totals = {k: 0.0 for k in judgementKeys}
            
            for judgement_obj in all_judgements:
                for key in judgementKeys:
                    val = judgement_obj.get(key, 0)
                    totals[key] += float(val)

            count = len(all_judgements)
            contexts[method]["meanJudgements"] = {
                key: round(totals[key] / count, 2) 
                for key in judgementKeys
            }


        totals = {k: 0.0 for k in judgementKeys}
        for method in self.method_keys:
            for judgementKey in judgementKeys:
                totals[judgementKey] += contexts[method]["meanJudgements"][judgementKey]
        for key in totals:
            totals[key] /= len(self.method_keys)
        contexts["meanJudgement"] = totals
        return contexts

    def _process_single_question(self, query: str, db=None) -> Tuple[List[Tuple], List]:
        """Process a single question and return (abstracts, filtered_documents)"""

        # PHASE 1: Retrieve ABSTRACTS for Agent2 & Agent3 filtering
        with time_block(f"retrieve_abstracts_{query[:20]}"):
            logger.info(f"Retrieving abstracts for: {query[:50]}...")
            retrieved_abstracts = self.retriever.retrieve_abstracts(query, top_k=10) # increase to 10 for precision @10

            # ✨ NEW: Log retrieved papers with titles
            self._log_retrieved_papers(query, retrieved_abstracts, "RETRIEVAL")

        # Step 2: Agent-2 generates answers from ABSTRACTS
        with time_block(f"agent2_generation_{query[:20]}"):
            logger.info(
                f"Agent-2 generating answers from abstracts for: {query[:50]}..."
            )
            doc_answers = []
            for abstract_text, doc_id in tqdm(retrieved_abstracts):
                prompt = self.createAnswerGeneratorPrompt(query, abstract_text)
                answer = self.scadsApiAgent.generate(prompt)
                # answer = self.agent1.generate(prompt)
                doc_answers.append((abstract_text, doc_id, answer))

            print("agent2 generation result-----------------------------------")
            a = {
                "retrieved abstracts:": retrieved_abstracts,
                "query" : query,
                "doc_answers": doc_answers,
            }
            print(json.dumps(a, indent=2, ensure_ascii=False)) 
            print("-----------------------------------")

        # Step 3: Agent-3 evaluates documents using ABSTRACTS
        with time_block(f"documentEvaluation{query[:20]}"):
            logger.info(f"Agent-3 evaluating abstracts for: {query[:50]}...")
            scores = []
            for abstract_text, doc_id, answer in tqdm(doc_answers):
                prompt = self.createDocumentEvaluatorPrompt(query, abstract_text, answer)
                text_answer, log_probs = self.scadsApiAgent.generate(prompt, True)
                
                # log_probs = self.scadsApiAgent.get_log_probs(prompt, ["Yes", "No"])
                # log_probs = self.agent1.get_log_probs(prompt, ["Yes", "No"])
                score = log_probs["Yes"] - log_probs["No"]
                scores.append(score)

        # Step 4: Calculate adaptive judge bar
        tau_q = np.mean(scores)
        sigma = np.std(scores)
        adjusted_tau_q = tau_q - self.n * sigma
        logger.info(
            f"Adaptive judge bar for '{query[:30]}...': tau_q={tau_q:.4f}, adjusted: {adjusted_tau_q:.4f}"
        )

        # Step 5: Filter documents based on abstract evaluation
        filtered_doc_ids = []
        filtered_abstracts = []
        for i, (abstract_text, doc_id, _) in enumerate(doc_answers):
            if scores[i] >= adjusted_tau_q:
                filtered_doc_ids.append(doc_id)
                filtered_abstracts.append((abstract_text, doc_id, scores[i]))

        filtered_abstracts.sort(key=lambda x: x[2], reverse=True)
        
        print("agent3 result-----------------------------------")
        a = {
            "tau_q": tau_q,
            "sigma": sigma,
            "adjusted_tau_q": adjusted_tau_q,
            "doc_answers": doc_answers,
            "scores": scores,
        }
        print(json.dumps(a, indent=2, ensure_ascii=False)) 
        print("-----------------------------------")
            

        # ✨ NEW: Log filtered papers with titles and scores
        self._log_filtered_papers(
            query, filtered_abstracts, [x[2] for x in filtered_abstracts]
        )

        return retrieved_abstracts, filtered_doc_ids

    def answer_query(self, item, db=None, choices=None, should_split=None, sub_questions=None):
        """
        ENHANCED: Process query with 4-agent approach, question splitting, and parallel processing
        """
        query = item["question"]
        # answer = item["answer"]
        generationMeta = item
        # topicOverlapsInThePapers = item["topic overlaps"]
        # papersLLMActuallyUsedForQuestionGeneration = item["usageJudgementGeneratorLLM"]
        # papersInputtedForGeneration = item["papersInputtedForGeneration"]

        with time_block("total_4agent_processing"):
            logger.info(f"Processing query with enhanced 4-agent approach: {query}")

            # Initialize enhanced citation handler
            citation_handler = EnhancedCitationHandler(self.scadsApiAgent, logger, self.index_dir)
            
            # If should_split and sub_questions are not provided, analyze the query
            if should_split is None or sub_questions is None:
                logger.info("Analyzing query for splitting...")
                should_split, sub_questions = self.question_splitter.analyze_and_split(query)
            
            if should_split and sub_questions:
                logger.info(f"Processing {len(sub_questions)} sub-questions in parallel")
                questions_to_process = sub_questions
            else:
                logger.info("Processing single question")
                questions_to_process = [query]

            # PHASE 2: Parallel Processing of Questions
            all_filtered_doc_ids = []

            if len(questions_to_process) > 1:
                # Parallel processing using thread pool
                with time_block("parallel_question_processing"):
                    logger.info(
                        f"Processing {len(questions_to_process)} questions in parallel"
                    )

                    # Submit all questions for parallel processing
                    future_to_question = {}
                    for sub_query in questions_to_process:
                        future = self.executor.submit(
                            self._process_single_question, sub_query, db
                        )
                        future_to_question[future] = sub_query

                    # Collect results
                    for future in as_completed(future_to_question):
                        sub_query = future_to_question[future]
                        try:
                            retrieved_abstracts, filtered_doc_ids = future.result()
                            all_filtered_doc_ids.extend(filtered_doc_ids)
                            logger.info(
                                f"Completed processing: {sub_query[:50]}... -> {len(filtered_doc_ids)} docs"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error processing sub-question '{sub_query}': {e}"
                            )
            else:
                # Single question processing
                retrieved_abstracts, filtered_doc_ids = self._process_single_question(
                    questions_to_process[0], db
                )
                all_filtered_doc_ids = filtered_doc_ids

            # Remove duplicates while preserving order
            seen = set()
            unique_filtered_doc_ids = []
            for doc_id in all_filtered_doc_ids:
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_filtered_doc_ids.append(doc_id)

            logger.info(
                f"Total unique filtered documents: {len(unique_filtered_doc_ids)}"
            )

            # PHASE 3: Get FULL TEXTS for Agent4
            with time_block("get_full_texts"):
                logger.info("Retrieving FULL texts for final answer generation...")
                # HERE all unique filtered doc ids used for answering
                if unique_filtered_doc_ids:
                    full_texts = self.retriever.get_full_texts(
                        unique_filtered_doc_ids, db=db
                    )

                    # Enhanced logging with context awareness
                    logger.info(
                        f"FINAL ANSWER GENERATION: Retrieved {len(full_texts)} papers:"
                    )
                    logger.info("=" * 80)
                    for i, (doc_text, doc_id) in enumerate(full_texts, 1):
                        title = PaperTitleExtractor.extract_title_from_text(
                            doc_text, doc_id
                        )
                        formatted_title = PaperTitleExtractor.format_title_for_log(
                            title, max_length=70
                        )
                        char_count = len(doc_text)
                        logger.info(
                            f"[{i:2d}] {formatted_title} ({char_count:,} chars)"
                        )
                        logger.info(f"Doc ID: {doc_id}")
                    logger.info("=" * 80)

                    # Log context usage
                    estimated_docs_used = min(
                        len(full_texts),
                        self.max_context_chars // (4000 if should_split else 8000),
                    )
                    self._log_context_usage(
                        full_texts, estimated_docs_used, should_split
                    )

                else:
                    logger.warning("No documents passed the filter, using fallback")
                    # Fallback to some documents from the original query
                    fallback_abstracts, fallback_ids = self._process_single_question(
                        query, db
                    )
                    full_texts = self.retriever.get_full_texts(fallback_ids[:3], db=db)
                    if not full_texts:
                        # Last resort: use abstracts
                        full_texts = [
                            (abstract_text, doc_id)
                            for abstract_text, doc_id in fallback_abstracts[:3]
                        ]

                    # Log fallback papers
                    logger.info(f"FALLBACK: Using {len(full_texts)} papers:")
                    for i, (doc_text, doc_id) in enumerate(full_texts, 1):
                        title = PaperTitleExtractor.extract_title_from_text(
                            doc_text, doc_id
                        )
                        formatted_title = PaperTitleExtractor.format_title_for_log(
                            title, max_length=70
                        )
                        logger.info(f"[{i:2d}] {formatted_title}")

            # PHASE 4: Agent-4 generates final answer with context management
            with time_block("finalAnswerGeneration"):
                strategy_info = (
                    "CONSERVATIVE (split questions)"
                    if should_split
                    else "GENEROUS (single question)"
                )
                logger.info(
                    f"Agent-4 generating final answer with context-aware citations... [{strategy_info}]"
                )
                prompt,cleanFullDocumentTexts = self.createFinalAnswerGeneratorPrompt(
                    query, full_texts, citation_handler, should_split
                )
                answerGenerationStart = time.time()
                unsafe_answer = self.scadsApiAgent.generate(prompt)       
                answerWithoutIllegalBackslashes = re.sub(r'\\(?![nrt"\\u])', r'\\\\', unsafe_answer)
                raw_answer = json.loads(answerWithoutIllegalBackslashes)
                answerGenerationEnd = time.time()
            # metrics

            paperInformationUsedForAnswering = citation_handler._get_papers_used_in_answer(raw_answer)
            # recallAt1 = float(arxivId in unique_filtered_doc_ids[:1])
            # recallAtMaxK = 1 if (arxivId in unique_filtered_doc_ids[:10]) else 0.0
            # reciprocalRank = 1/ ((unique_filtered_doc_ids.index(arxivId) + 1)) if arxivId in unique_filtered_doc_ids else 0
            
            '''
                context extraction variants:
                1: native squai
                2: Biencoder (top 1), floating window up to 5 sentences
                3: Biencoder selecting top 10, floating window up to 5 sentences --> get top 1 with keyword/BM25 matching
                4: Biencoder selecting top 10, floating window up to 5 sentences --> get top 1 using cross encoder
                5: Biencoder + keyword/BM25  on floating windows (combine score using RRF) to get top 1
                6: Biencoder + keyword/BM25  on floating windows (combine score using RRF) to get top 10 --> select top 1 with cross encoder
                7: llama LLM gets whole paper and extract context (top 1)
            '''
            
            #variant 1 - native squai context extraction
            references, referencesDuration = citation_handler.format_references(raw_answer,cleanFullDocumentTexts)

            #variant 2 - Biencoder (selects top 1) out of floating windows of up to 5 sentences
            referencesBiencoderTop1, referencesBiencoderTop1Duration = citation_handler.referencesBiencoderTop1(raw_answer,cleanFullDocumentTexts)
            
            #variant 3 - BM25 selects top 1 out of floating window up to 5 sentences
            referencesBM25Top1, referencesBM25Top1Duration = citation_handler.referencesBM25Top1(raw_answer,cleanFullDocumentTexts)

            #variant 4 - Biencoder selects top 10 (out of floating windows), then BM25 selects top 1   
            referencesBiencoderTop10Bm25Top1, referencesBiencoderTop10Bm25Top1Duration = citation_handler.referencesBiencoderTop10Bm25Top1(raw_answer,cleanFullDocumentTexts)

            #variant 5 - BM25 selects top 10 (out of floating windows), then Biencoder selects top 1
            referencesBM25Top10BiencoderTop1,referencesBM25Top10BiencoderTop1Duration = citation_handler.referencesBM25Top10BiencoderTop1(raw_answer,cleanFullDocumentTexts)

            #variant 6 - Biencoder and bm25 select top 1 (using RRF)
            referencesBiencoderAndBm25Top1, referencesBiencoderAndBm25Top1Duration = citation_handler.referencesBiencoderAndBm25Top1(raw_answer,cleanFullDocumentTexts)
            
            #variant 7 - Biencoder selects top 10 (out of floating windows) top 10, cross encoder selects top 1
            referencesBiencoderTop10CrossEncoderTop1, referencesBiencoderTop10CrossEncoderTop1Duration = citation_handler.referencesBiencoderTop10CrossEncoderTop1(raw_answer,cleanFullDocumentTexts)

            #variant 8 - BM25 selects top 10 (out of floating windows) top 10, cross encoder selects top 1
            referencesBM25Top10CrossEncoderTop1, referencesBM25Top10CrossEncoderTop1Duration = citation_handler.referencesBM25Top10CrossEncoderTop1(raw_answer,cleanFullDocumentTexts)

            #variant 9 - BiEncoder and keyword bm25 select top 10 together, cross encoder selects top 1
            referencesBiencoderAndBm25Top10CrossEncoderTop1, referencesBiencoderAndBm25Top10CrossEncoderTop1Duration = citation_handler.referencesBiencoderAndBm25Top10CrossEncoderTop1(raw_answer,cleanFullDocumentTexts)

            # variant 10 - extract the context using LLM (prompt to extract the best fit)
            referencesWithLLM, referencesWithLLMDuration = citation_handler._extract_context_passages_using_llm(raw_answer,cleanFullDocumentTexts)


            contexts = {
                "referencesNative": references,
                "referencesBiencoderTop1": referencesBiencoderTop1,
                "referencesBM25Top1": referencesBM25Top1,
                "referencesBiencoderTop10Bm25Top1": referencesBiencoderTop10Bm25Top1,
                "referencesBM25Top10BiencoderTop1": referencesBM25Top10BiencoderTop1,
                "referencesBiencoderTop10CrossEncoderTop1": referencesBiencoderTop10CrossEncoderTop1,
                "referencesBM25Top10CrossEncoderTop1": referencesBM25Top10CrossEncoderTop1,
                "referencesBiencoderAndBm25Top1": referencesBiencoderAndBm25Top1,
                "referencesBiencoderAndBm25Top10CrossEncoderTop1": referencesBiencoderAndBm25Top10CrossEncoderTop1,
                "referencesWithLLM": referencesWithLLM,
                "generationMeta": generationMeta,
                "answerMeta": {
                    "papersRetrievedBySQuAI": unique_filtered_doc_ids,
                    "paperInformationUsedForAnswering": paperInformationUsedForAnswering,
                    "modelAnswer" : raw_answer,
                    "paperFullTextLength": {key: len(text) for key, text in cleanFullDocumentTexts.items()},
                    "duration": {
                        "answerGeneration": answerGenerationEnd - answerGenerationStart,
                        "referencesNativeDuration": referencesDuration,
                        "referencesBiencoderTop1Duration": referencesBiencoderTop1Duration,
                        "referencesBM25Top1Duration": referencesBM25Top1Duration,
                        "referencesBiencoderTop10Bm25Top1Duration": referencesBiencoderTop10Bm25Top1Duration,
                        "referencesBM25Top10BiencoderTop1Duration": referencesBM25Top10BiencoderTop1Duration,
                        "referencesBiencoderTop10CrossEncoderTop1Duration": referencesBiencoderTop10CrossEncoderTop1Duration,
                        "referencesBM25Top10CrossEncoderTop1Duration": referencesBM25Top10CrossEncoderTop1Duration,
                        "referencesBiencoderAndBm25Top1Duration": referencesBiencoderAndBm25Top1Duration,
                        "referencesBiencoderAndBm25Top10CrossEncoderTop1Duration": referencesBiencoderAndBm25Top10CrossEncoderTop1Duration,
                        "referencesWithLLMDuration": referencesWithLLMDuration,
                    }
                }
            }

            # contextsWithJudgement = self.judgeContextsWithReferences(raw_answer,contexts)

            # contextsWithMeanJudgements = self.addMeanJudgements(contextsWithJudgement)

            with open("contextExtractionResult.jsonl", "a", encoding="utf-8") as f:
                # json.dump(contextsWithMeanJudgements,f, indent=2, ensure_ascii=False)
                f.write(json.dumps(contexts, ensure_ascii=False) + "\n")

            #remove for now as this could delete parts of the Answer that include this word
            # Remove any references Agent-4 might have added
            '''
            if "Reference" in raw_answer:
                raw_answer = re.split(r"References", raw_answer)[0]
            '''
            citation_map = citation_handler.get_citation_map()

            # Enhanced debug info
            debug_info = {
                "original_query": query,
                "was_split": should_split,
                "sub_questions": sub_questions if should_split else [],
                "questions_processed": len(questions_to_process),
                "total_filtered_docs": len(unique_filtered_doc_ids),
                "full_texts_retrieved": len(full_texts),
                "total_citations": len(citation_map),
                "citation_map": citation_map,
                "passages_used": self._extract_passages_used(
                    raw_answer, citation_handler
                ),
                "document_metadata": self._extract_document_metadata(citation_handler),
                "context_stats": {
                    "max_context_chars": self.max_context_chars,
                    "total_chars_available": sum(len(text) for text, _ in full_texts),
                    "docs_available": len(full_texts),
                    "estimated_docs_used": min(
                        len(full_texts),
                        self.max_context_chars // (4000 if should_split else 8000),
                    ),
                    "strategy": (
                        "CONSERVATIVE (split questions)"
                        if should_split
                        else "GENEROUS (single question)"
                    ),
                    "chars_per_paper_limit": (
                        "TOP(2.5K)+BOTTOM(1.5K)"
                        if should_split
                        else "TOP(5K)+BOTTOM(3K)"
                    ),
                },
                "performance_stats": (
                    monitor.get_stats() if hasattr(monitor, "get_stats") else {}
                ),
            }

        return raw_answer, debug_info

    def _extract_passages_used(
        self, answerObject: GeneratedAnswerFormat, citation_handler: EnhancedCitationHandler
    ):
        """Extract the specific passages used in the answer"""
        # Find all citations in the answer
        #pfusch
        answer_text = " ".join(dataEntry["sentence"][:-1] + " [" + str(dataEntry["documentId"]) + "]" + dataEntry["sentence"][-1]  for dataEntry in answerObject if "sentence" in dataEntry)
        used_citations = {dataEntry["documentId"] for dataEntry in answerObject if "documentId" in dataEntry}
        
        passages_used = []
        for citation_num in used_citations:
            if citation_num in citation_handler.citation_to_doc:
                doc_info = citation_handler.citation_to_doc[citation_num]

                # Extract context passage for this citation
                context_passage = citation_handler._extract_context_passage(
                    answer_text, doc_info["text"], citation_num
                )

                passages_used.append(
                    {
                        "citation_num": citation_num,
                        "doc_id": doc_info["doc_id"],
                        "paper_title": doc_info["paper_info"]["title"],
                        "paper_id": doc_info["paper_info"]["paper_id"],
                        "authors": doc_info["paper_info"]["authors"],
                        "year": doc_info["paper_info"]["year"],
                        "context_passage": context_passage,
                        "passage_preview": (
                            context_passage[:200] + "..."
                            if len(context_passage) > 200
                            else context_passage
                        ),
                    }
                )

        return passages_used

    def _extract_document_metadata(self, citation_handler: EnhancedCitationHandler):
        """Extract document metadata for all citations"""
        metadata = {}
        for citation_num, doc_info in citation_handler.citation_to_doc.items():
            metadata[citation_num] = {
                "doc_id": doc_info["doc_id"],
                "paper_info": doc_info["paper_info"],
            }
        return metadata

    def close(self):
        """Clean up resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
        logger.info("Enhanced 4-Agent RAG system closed")
   

def main():
    """Main function with enhanced 4-agent support"""
    # DEFAULT_GENERATOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    #DEFAULT_GENERATOR_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
    DEFAULT_GENERATOR_MODEL = "MiniMaxAI/MiniMax-M3-MXFP8"
    DEFAULT_JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

    parser = argparse.ArgumentParser(
        description="Enhanced 4-Agent RAG with Question Splitting and Parallel Processing"
    )
    # parser.add_argument(
    #     "--questionSplitterModel",
    #     type=str,
    #     default=DEFAULT_GENERATOR_MODEL,
    #     help="Model used for the question splitter",
    # )
    # parser.add_argument(
    #     "--answerGeneratorModel",
    #     type=str,
    #     default=DEFAULT_GENERATOR_MODEL,
    #     help="Model used for the answer generator",
    # )
    # parser.add_argument(
    #     "--documentEvaluatorModel",
    #     type=str,
    #     default=DEFAULT_GENERATOR_MODEL,
    #     help="Model used for the document evaluator",
    # )
    # parser.add_argument(
    #     "--finalAnswerGeneratorModel",
    #     type=str,
    #     default=DEFAULT_GENERATOR_MODEL,
    #     help="Model used for the final answer generator",
    # )
    # parser.add_argument(
    #     "--contentExtractorModel",
    #     type=str,
    #     default=DEFAULT_GENERATOR_MODEL,
    #     help="Model used for the context extractor",
    # )
    # parser.add_argument(
    #     "--judgeModel",
    #     type=str,
    #     default=DEFAULT_JUDGE_MODEL,
    #     help="Model used for the judge",
    # )

    parser.add_argument(
        "--n", type=float, default=0.5, help="Adjustment factor for adaptive judge bar"
    )
    parser.add_argument(
        "--retriever_type",
        choices=["e5", "bm25", "hybrid"],
        default="hybrid",
        help="Type of retriever",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="test_index",
        help="Directory containing metadata",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="quick_test_questions.jsonl",
        help="File containing questions",
    )
    parser.add_argument(
        "--single_question", type=str, default=None, help="Process a single question"
    )
    parser.add_argument(
        "--output_format",
        choices=["json", "jsonl", "debug"],
        default="jsonl",
        help="Output format",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help="Path to LevelDB database (overrides config)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Weight for E5 in hybrid mode (0.0=BM25 only, 1.0=E5 only)",
    )

    args = parser.parse_args()

    # Use custom DB path if provided, otherwise use config default
    db_path_to_use = args.db_path if args.db_path else DB_PATH

    logger.info(f"Opening database at {db_path_to_use}...")
    try:
        db = plyvel.DB(db_path_to_use, create_if_missing=False)
        logger.info("Database opened successfully")
    except Exception as e:
        logger.error(f"Failed to open database: {e}")
        # Try alternative path if permission denied
        alt_db_path = os.path.join(os.path.dirname(__file__), "local_db")
        logger.info(f"Trying alternative database path: {alt_db_path}")
        db = plyvel.DB(alt_db_path, create_if_missing=True)

    retriever = initialize_retriever(
        args.retriever_type,
        E5_INDEX_DIR,
        BM25_INDEX_DIR,
        DB_PATH,
        args.top_k,
        args.alpha,
    )

    logger.info(
        f"Initializing enhanced 4-agent RAG with n={args.n}, max_workers={args.max_workers}..."
    )
    ragent = Enhanced4AgentRAG(
        retriever,
        # questionSplitterModel=args.questionSplitterModel,
        # answerGeneratorModel=args.answerGeneratorModel,
        # documentEvaluatorModel=args.documentEvaluatorModel,
        # finalAnswerGeneratorModel=args.finalAnswerGeneratorModel,
        # contentExtractorModel=args.contentExtractorModel,
        # judgeModel=args.judgeModel,
        n=args.n,
        index_dir=args.index_dir,
        max_workers=args.max_workers,
    )

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "debug"), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process single question
    if args.single_question:
        logger.info(
            f"\nProcessing single question with enhanced 4-agent system: {args.single_question}"
        )
        start_time = time.time()

        try:
            should_split, sub_questions = ragent.question_splitter.analyze_and_split(
                args.single_question
            )
            cited_answer, debug_info = ragent.answer_query(args.single_question, db, should_split, sub_questions)
            process_time = time.time() - start_time

            result = {
                "id": f"single_question_4agent_{args.retriever_type}",
                "question": args.single_question,
                "model_answer": cited_answer,
                "was_split": debug_info["was_split"],
                "sub_questions": debug_info["sub_questions"],
                "questions_processed": debug_info["questions_processed"],
                "total_citations": debug_info["total_citations"],
                "total_filtered_docs": debug_info["total_filtered_docs"],
                "full_texts_retrieved": debug_info["full_texts_retrieved"],
                "passages_used": debug_info["passages_used"],
                "document_metadata": debug_info["document_metadata"],
                "process_time": process_time,
                "retriever_type": args.retriever_type,
            }

            logger.info(f"Cited Answer: {cited_answer}")
            # logger.info(f"References: {references}")
            logger.info(f"Was Split: {debug_info['was_split']}")
            if debug_info["was_split"]:
                logger.info(f"Sub-questions: {debug_info['sub_questions']}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(f"Citations used: {debug_info['total_citations']}")


            # save comparison of the context extraction
            # passages = {citation_num: [data["contextPassage"]] for citation_num, data in references.items()}

            # judge the passages (only the native squai solution)            
            
            # nativePassagesJudgement = ragent.judgeContextWithReferences(cited_answer,passages, False)
            # llmExtractedPassagesJudgement = ragent.judgeContextWithReferences(cited_answer, referencesWithLLM)
            # cosineSimilarityExtractedPassagesJudgement = ragent.judgeContextWithReferences(cited_answer,referencesWithCosineSimilarity)

            # Save result
            if args.output_format == "debug":
                debug_output_file = os.path.join(
                    args.output_dir,
                    "debug",
                    f"enhanced_4agent_single_{args.retriever_type}_debug_{timestamp}.json",
                )
                with open(debug_output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Debug result saved to {debug_output_file}")
            else:
                output_file = os.path.join(
                    args.output_dir,
                    f"enhanced_4agent_single_{args.retriever_type}_{timestamp}.json",
                )
                write_enhanced_result_to_json(result, output_file)

        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
        finally:
            ragent.close()
            retriever.close()

        return

    # Process question file
    questions = load_datamorgana_questions(args.data_file)
    if not questions:
        logger.error("No questions found. Exiting.")
        return

    results = []

    for i, item in enumerate(questions):
        question_id = item.get("id", i + 1)
        logger.info(
            f"\nProcessing question {i+1}/{len(questions)} with enhanced 4-agent system: {item['question']}"
        )
        start_time = time.time()

        
        try:
            should_split, sub_questions = ragent.question_splitter.analyze_and_split(
                item["question"]
            )
            cited_answer, debug_info = ragent.answer_query(item, db, should_split, sub_questions)
            process_time = time.time() - start_time

            result = {
                "id": question_id,
                "question": item["question"],
                "model_answer": cited_answer,
                "was_split": debug_info["was_split"],
                "sub_questions": debug_info["sub_questions"],
                "questions_processed": debug_info["questions_processed"],
                "total_citations": debug_info["total_citations"],
                "total_filtered_docs": debug_info["total_filtered_docs"],
                "full_texts_retrieved": debug_info["full_texts_retrieved"],
                "passages_used": debug_info["passages_used"],
                "document_metadata": debug_info["document_metadata"],
                "process_time": process_time,
                "retriever_type": args.retriever_type,
            }
            results.append(result)

            logger.info(f"Cited Answer: {cited_answer[:200]}...")
            # logger.info(f"References: {references}")
            logger.info(f"Was Split: {debug_info['was_split']}")
            if debug_info["was_split"]:
                logger.info(f"Sub-questions: {debug_info['sub_questions']}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(f"Citations used: {debug_info['total_citations']}")


            # passages = {citation_num: [data["contextPassage"]] for citation_num, data in references.items()}

            # judge the passages (only the native squai solution)            
            
            #this is the multiple questions thing
            # nativePassagesJudgement = ragent.judgeContextWithReferences(cited_answer,passages, False)
            # llmExtractedPassagesJudgement = ragent.judgeContextWithReferences(cited_answer, referencesWithLLM)
            # cosineSimilarityExtractedPassagesJudgement = ragent.judgeContextWithReferences(cited_answer,referencesWithCosineSimilarity)

            # Save debug info
            debug_output_file = os.path.join(
                args.output_dir,
                "debug",
                f"enhanced_4agent_question_{question_id}_{args.retriever_type}_debug_{timestamp}.json",
            )
            with open(debug_output_file, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}", exc_info=True)

    # Clean up
    ragent.close()
    retriever.close()

    # Save all results
    random_num = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

    if results:
        if args.output_format == "jsonl":
            output_file = os.path.join(
                args.output_dir,
                f"enhanced_4agent_answers_{args.retriever_type}_{timestamp}_{random_num}.jsonl",
            )
            write_enhanced_results_to_jsonl(results, output_file)
        elif args.output_format == "json":
            for result in results:
                question_id = result["id"]
                output_file = os.path.join(
                    args.output_dir,
                    f"enhanced_4agent_answer_{question_id}_{args.retriever_type}_{random_num}.json",
                )
                write_enhanced_result_to_json(result, output_file)
        else:  # debug
            output_file = os.path.join(
                args.output_dir,
                "debug",
                f"enhanced_4agent_all_results_{args.retriever_type}_debug_{timestamp}.json",
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Debug results saved to {output_file}")

    logger.info(f"\nProcessed {len(results)} questions with enhanced 4-agent system.")

    if results:
        avg_time = sum(r["process_time"] for r in results) / len(results)
        avg_filtered = sum(r["total_filtered_docs"] for r in results) / len(results)
        avg_citations = sum(r["total_citations"] for r in results) / len(results)
        avg_full_texts = sum(r["full_texts_retrieved"] for r in results) / len(results)
        split_count = sum(1 for r in results if r["was_split"])

        logger.info(f"Average processing time: {avg_time:.2f} seconds")
        logger.info(
            f"Questions split: {split_count}/{len(results)} ({split_count/len(results)*100:.1f}%)"
        )
        logger.info(f"Average filtered documents: {avg_filtered:.1f}")
        logger.info(f"Average citations: {avg_citations:.1f}")
        logger.info(f"Average full texts used: {avg_full_texts:.1f}")

    db.close()
    logger.info("Database closed")


if __name__ == "__main__":
    main()
