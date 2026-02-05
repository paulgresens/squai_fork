import json
from pathlib import Path
import re
from agents import PaperTitleExtractor
from typing import Dict
from logging import Logger
from agents.types import GeneratedAnswerFormat
from text_cleaner import DocumentTextCleaner
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import time
from rapidfuzz import fuzz
class EnhancedCitationHandler:
    """Enhanced citation handler with proper metadata extraction and context passages"""

    def __init__(self, llmAgent, logger: Logger, index_dir: str = "test_index"):
        self.doc_to_citation = {}
        self.citation_to_doc = {}
        self.next_citation_num = 1
        self.index_dir = Path(index_dir)
        ##todo der pfusch muss weg
        self.llmAgent = llmAgent
        self.logger = logger
        

        # Load arXiv papers for better metadata
        self.arxiv_papers = self._load_arxiv_papers()

        # Connect to metadata database
        self.metadata_db = self._connect_metadata_db()
        self.bitransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.crossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        cleaner = DocumentTextCleaner()
        for entry in self.citation_to_doc.values():
            if "text" in entry and isinstance(entry["text"], str):
                entry["text"] = cleaner.clean_for_citation_matching(entry["text"])       

    def _connect_metadata_db(self):
        """Connect to metadata database"""
        try:
            import sqlite3

            db_path = self.index_dir / "index_store.db"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            return conn
        except:
            return None

    def _load_arxiv_papers(self):
        """Load arXiv papers for metadata extraction"""
        papers = {}
        try:
            jsonl_files = list(self.index_dir.glob("*.jsonl"))

            for jsonl_file in jsonl_files:
                with open(jsonl_file, "r") as f:
                    for line in f:
                        try:
                            paper = json.loads(line.strip())
                            paper_id = paper.get("paper_id", "")

                            metadata = paper.get("metadata", {})
                            title = metadata.get("title", "Unknown Title")
                            authors = metadata.get("authors", "Unknown")

                            # Extract year from versions
                            year = "Unknown"
                            versions = paper.get("versions", [])
                            if versions:
                                created = versions[0].get("created", "")
                                year_match = re.search(r"(\d{4})", created)
                                if year_match:
                                    year = year_match.group(1)

                            # Format authors properly
                            if "authors_parsed" in paper:
                                authors_list = paper["authors_parsed"]
                                if authors_list and len(authors_list) > 0:
                                    first_author = authors_list[0]
                                    if len(first_author) >= 2:
                                        formatted_author = (
                                            f"{first_author[0]}, {first_author[1][0]}."
                                            if first_author[1]
                                            else first_author[0]
                                        )
                                        if len(authors_list) > 1:
                                            authors = f"{formatted_author} et al."
                                        else:
                                            authors = formatted_author

                            papers[paper_id] = {
                                "title": title,
                                "authors": authors,
                                "year": year,
                                "paper_id": paper_id,
                                "abstract": paper.get("abstract", {}).get("text", ""),
                            }
                        except:
                            continue

            return papers
        except:
            return {}

    def _extract_document_title_improved(self, doc_text: str, doc_id: str) -> str:
        """Use the PaperTitleExtractor for consistency"""
        return PaperTitleExtractor.extract_title_from_text(doc_text, doc_id)

    def _extract_paper_info(
        self, doc_text: str, doc_id: str, metadata: Dict = None
    ) -> Dict:
        """Enhanced paper metadata extraction with improved title extraction"""
        paper_info = {
            "title": "Unknown Title",
            "authors": "Unknown",
            "venue": "arXiv",
            "year": "Unknown",
            "paper_id": doc_id,
        }

        try:
            # Use improved title extraction
            paper_info["title"] = self._extract_document_title_improved(
                doc_text, doc_id
            )

            # Extract from JSON in document text
            if "{" in doc_text and '"metadata"' in doc_text:
                try:
                    json_match = re.search(r'\{.*?"metadata".*?\}', doc_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        paper_data = json.loads(json_str)

                        if "metadata" in paper_data:
                            meta = paper_data["metadata"]
                            if "authors" in meta:
                                paper_info["authors"] = meta["authors"]

                        if "paper_id" in paper_data:
                            paper_info["paper_id"] = paper_data["paper_id"]

                        # Extract year from versions
                        if "versions" in paper_data and paper_data["versions"]:
                            created = paper_data["versions"][0].get("created", "")
                            year_match = re.search(r"(\d{4})", created)
                            if year_match:
                                paper_info["year"] = year_match.group(1)

                        self.logger.debug(
                            f"Extracted metadata from JSON in text for {doc_id}"
                        )

                except Exception as e:
                    self.logger.debug(f"JSON parsing failed for {doc_id}: {e}")

            # Match with loaded arXiv papers by paper_id
            if doc_id in self.arxiv_papers:
                arxiv_data = self.arxiv_papers[doc_id]
                # Update info but keep improved title if it's better
                if (
                    paper_info["title"] == "Unknown Title"
                    or paper_info["title"] == f"Document {doc_id}"
                ):
                    paper_info["title"] = arxiv_data["title"]
                if paper_info["authors"] == "Unknown":
                    paper_info["authors"] = arxiv_data["authors"]
                if paper_info["year"] == "Unknown":
                    paper_info["year"] = arxiv_data["year"]
                self.logger.debug(
                    f"Enhanced metadata for {doc_id} from arXiv papers database"
                )

            # Final cleanup
            if len(paper_info["title"]) > 150:
                paper_info["title"] = paper_info["title"][:150] + "..."

            # Ensure we have a paper_id
            if not paper_info["paper_id"]:
                paper_info["paper_id"] = doc_id

        except Exception as e:
            self.logger.debug(f"Error extracting metadata for {doc_id}: {e}")

        return paper_info

    def _basic_text_cleaning(self, text: str) -> str:
        """Basic text cleaning for citation context"""
        # Remove JSON-like section markers
        text = re.sub(r"'section':\s*'[^']*',\s*'text':\s*'", "", text)
        text = re.sub(r"^\s*\{.*?'text':\s*'", "", text)
        text = re.sub(r"\{[^}]*\}", "", text)

        # Remove technical markup
        text = re.sub(r"\{\{[^}]+\}\}", "[REF]", text)
        text = re.sub(r"\$[^$]+\$", "[MATH]", text)
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "[LATEX]", text)

        # Clean whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def _extract_context_passage(
        self, answer_text: str, document_text: str, citation_num: int
    ) -> str:
        """Extract specific sentence(s) used in the answer plus context"""
        try:
            # Clean the document text first
            try:
                from text_cleaner import DocumentTextCleaner

                cleaner = DocumentTextCleaner()
                clean_doc_text = cleaner.clean_for_citation_matching(document_text)
            except ImportError:
                clean_doc_text = self._basic_text_cleaning(document_text)
            
            # NEW: Remove the [TOP xxx chars]: and [BOTTOM xxx chars]: prefixes
            clean_doc_text = re.sub(r'\[TOP \d+ chars\]:\s*', '', clean_doc_text)
            clean_doc_text = re.sub(r'\[BOTTOM \d+ chars\]:\s*', '', clean_doc_text)
            
            # Find all sentences in the clean document
            sentences = re.split(r"[.!?]+", clean_doc_text)

            sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

            # Look for content that appears in the answer near this citation
            citation_pattern = f"\\[{citation_num}\\]"
            citation_matches = list(re.finditer(citation_pattern, answer_text))

            if not citation_matches:
                # Fallback: return first few clean sentences
                return (
                    ". ".join(sentences[:2]) + "."
                    if sentences
                    else clean_doc_text[:200] + "..."
                )

            # For each citation, find the preceding text that likely came from this document
            relevant_sentences = set()

            for match in citation_matches:
                # Get text before this citation (up to 150 chars back)
                start_pos = max(0, match.start() - 150)
                context_text = answer_text[start_pos : match.start()].strip()

                # Find the sentence in context_text that likely came from the document
                context_sentences = re.split(r"[.!?]+", context_text)

                for context_sent in context_sentences[
                    -2:
                ]:  # Last 1-2 sentences before citation
                    if len(context_sent.strip()) < 15:
                        continue

                    # Find similar sentences in the document
                    context_words = set(context_sent.lower().split())

                    for i, doc_sent in enumerate(sentences):
                        doc_words = set(doc_sent.lower().split())

                        # Check word overlap
                        overlap = len(context_words.intersection(doc_words))
                        overlap_ratio = overlap / max(len(context_words), 1)

                        if overlap_ratio > 0.25 or overlap > 4:  # Good match
                            # Add this sentence plus context (±1 sentence)
                            start_idx = max(0, i - 1)
                            end_idx = min(len(sentences), i + 2)

                            for j in range(start_idx, end_idx):
                                relevant_sentences.add(j)

            if relevant_sentences:
                # Sort and build context passage
                sorted_indices = sorted(relevant_sentences)
                context_parts = [sentences[i] for i in sorted_indices]
                result = ". ".join(context_parts) + "."

                # Limit length
                if len(result) > 500:
                    result = result[:500] + "..."
                print("------------------------ctitation result")
                print(result)
                print("----------------------------------")
                return result

            # Fallback: return beginning of clean document
            fallback = ". ".join(sentences[:2]) + "."
            return fallback if len(fallback) < 300 else fallback[:300] + "..."

        except Exception as e:
            self.logger.debug(f"Error extracting context passage: {e}")
            # Simple fallback with basic cleaning
            clean_text = self._basic_text_cleaning(document_text)
            return clean_text[:200] + "..." if len(clean_text) > 200 else clean_text

    def create_context_extraction_using_llm_prompt(
        self, answer:str, documentText: str,
    )->str:

     return f"""
        You will be given the text of a scientific paper and a claim which was generated based on the content of the paper. 
        You can assume that all information in the paper is factually correct. Your job is, that for the given claim you should extract the section of the paper which provides the 
        best factual basis for justifying the statement. For that you should at least extract one sentence and up to 5 sentences which must be consecutively present in the paper. 
        The extraction can only contain sentences that are directly quoted as they are written in the document. You are not allowed to make any adjustments whatsoever to them.
        Here is a list of criteria that defines what a good
        extraction looks like:

        -The claim is the logical consequence of the facts in the context

        -The claim and the context cover the same topic
        -each topic that is present in the claim is also covered in the context

        -the context does not contradict the claim
        -the claim includes only information provided in the context or correct generalizations or interpretations of that information

        -the context includes every piece of information needed to justify each individual piece of the claim

        -the context and the claim convey the same meaning, even though the claim might be worded differently
        -the context includes every semantic entity that is present in the claim

        -the information in the context is unambiguous and its interpretation can only lead to the logical conclusions that are drawn in the claim

        Search for the context, that fullfills all those criteria the best.

        Input: 
        -text of a scientific paper
        -claim generated by an LLM on the basis of this paper

        Output:
        -The extracted context

        Answer with only the extracted sentences, exactly as they are consecutively! written in the document
        You are not allowed to hallucinate or make any adjustments to the sentences. You are also now allowed to add additional comments or headllines.
        Strictly reply with only those sentences. 
        Try to select as few sentences as possible. The response must include every information needed to fullfill the criteria, but should not contain any unnecessary information,
        that is not needed to justify the claim.


        Scientific Paper Text : "{documentText}"
        Claim: "{answer}"
        """

    #extract context for a single sentence from a document text
    def _extract_context_for_sentence_using_llm(self, sentence:str, document_text: str)->str:
        prompt = self.create_context_extraction_using_llm_prompt(
            sentence, document_text
        )
        return self.llmAgent.generate(prompt)
        
    # extract context passages for an answer text from the according paper
    def _extract_context_passages_using_llm(self, answerObject: GeneratedAnswerFormat) -> str:
        # answerTextSentences =  re.split(r"[.!?]+", answerObject)
        # sentences_with_citations = [s.strip() for s in answerTextSentences if re.search(r"\[\d+\]", s)]            
        start = time.time()
        result = {}
        for answerEntry in answerObject:
            doc_id = answerEntry["documentId"]
            documentData = self.citation_to_doc[doc_id]
            paper_info = documentData["paper_info"]
            paperId = paper_info.get("paper_id")
            
            documentText = documentData.get("text")
            contextForSentence = self._extract_context_for_sentence_using_llm(answerEntry["sentence"], documentText)

            # hallucination check
            cleanedDocumentText =  self.normalize(documentText)
            cleanedContextSentence = self.normalize(contextForSentence)

            partialRatio = fuzz.partial_ratio(cleanedContextSentence, cleanedDocumentText)

            result.setdefault(doc_id, []).append({"contextPassage":contextForSentence, "paperId": paperId ,"hallucinationCheck": {
                # "cleanedDocumentText": cleanedDocumentText,
                "cleanedContextSentence": cleanedContextSentence,
                "partialRatio": partialRatio
            }})
        end = time.time()
        return result, end-start    
    
    def normalize(self,text:str):
         lowercaseText = text.lower()
         textOnlyAlphanumeric = re.sub(r'[^\w\s]', '', lowercaseText)
         textCleaned = re.sub(r'\s+', ' ', textOnlyAlphanumeric).strip()
         return textCleaned


    def extract_top_k_contexts(self, documentId: int, answerSentence: str, top_k:int = 1, useBM25HybridRetrieval: bool = False):
        bitransformer = self.bitransformer
        documentText = self.citation_to_doc[documentId].get("text")
        print("--------------------")
        print(json.dumps(self.citation_to_doc[documentId]))
        print("----------------------")

        #floating context window 1-5 sentences
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

        contextWindows = documentSentences + window_2 + window_3 + window_4 + window_5
        
        answerSentenceEncoding = bitransformer.encode(answerSentence)
        
        similarities = []
        for context in contextWindows:
             similarities.append({
                "bitransformer_score": bitransformer.similarity(answerSentenceEncoding , bitransformer.encode(context)).item(),
                "context": context 
            })
             
        if (useBM25HybridRetrieval):
            tokenized_answer = answerSentence.split(" ")
            tokenized_context_windows = [context_window.split(" ") for context_window in contextWindows]

            bm25 = BM25Okapi(tokenized_context_windows)    
            bm25_scores = bm25.get_scores(tokenized_answer)
            for i, score in enumerate(bm25_scores):
                similarities[i]["bm25_score"] = score
            
            # Sort by Dense Score to find Dense Rank
            similarities.sort(key=lambda x: x["bitransformer_score"], reverse=True)
            for rank, item in enumerate(similarities):
                item["dense_rank"] = rank

            # Sort by BM25 Score to find Sparse Rank
            similarities.sort(key=lambda x: x["bm25_score"], reverse=True)
            for rank, item in enumerate(similarities):
                item["sparse_rank"] = rank

            # 3. Iterate and Calculate Final RRF Score
            k = 60
            for item in similarities:
                # The Formula: 1 / (k + rank1) + 1 / (k + rank2)
                item["rrf_score"] = (1 / (k + item["dense_rank"] + 1)) + (1 / (k + item["sparse_rank"] + 1))
            
            # 4. Final Sort by the combined score
            similarities.sort(key=lambda x: x["rrf_score"], reverse=True)

            #todo debug this

            top_k_results = similarities[:top_k]

            return [item["context"] for item in top_k_results]

        sorted_similarities = sorted(similarities, key=lambda x: x["bitransformer_score"], reverse=True)
        top_k_results = sorted_similarities[:top_k]

        return [item["context"] for item in top_k_results]



    def extract_context_using_cosine_similarity(self, answerObject:GeneratedAnswerFormat):        
        start = time.time()
        result = {}
        for answerObjectEntry in answerObject:
            doc_id = answerObjectEntry["documentId"]
            documentData = self.citation_to_doc[doc_id]
            paper_info = documentData["paper_info"]
            paperId = paper_info.get("paper_id")

            bestContext = self.extract_top_k_contexts(answerObjectEntry["documentId"], answerObjectEntry["sentence"], 1, False)
            result.setdefault(answerObjectEntry["documentId"], []).append({"contextPassage": bestContext[0], "paperId": paperId})
        end = time.time()
        return result, end-start 
    

    def extract_context_using_cosine_similarity_top_10_and_keyword_matching(self,answerObject:GeneratedAnswerFormat):
        start = time.time()
        result = {}
        for answerObjectEntry in answerObject:
            best10Contexts = self.extract_top_k_contexts(answerObjectEntry["documentId"], answerObjectEntry["sentence"], 10, False)

            doc_id = answerObjectEntry["documentId"]
            documentData = self.citation_to_doc[doc_id]
            paper_info = documentData["paper_info"]
            paperId = paper_info.get("paper_id")

            tokenized_corpus = [context.split(" ") for context in best10Contexts]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_answer_sentence = answerObjectEntry["sentence"].split(" ")
            bm25_scores = bm25.get_scores(tokenized_answer_sentence)

            best_idx = list(bm25_scores).index(max(bm25_scores))
            best_context = best10Contexts[best_idx]
            result.setdefault(answerObjectEntry["documentId"], []).append({"contextPassage":best_context, "paperId": paperId})
        end = time.time()
        return result, end-start
    
    def extract_context_using_cosine_similarity_top_10_and_cross_encoder(self, answerObject:GeneratedAnswerFormat):
        start = time.time()
        result = {}
        for answerObjectEntry in answerObject:
            best10Contexts = self.extract_top_k_contexts(answerObjectEntry["documentId"], answerObjectEntry["sentence"], 10, False)

            doc_id = answerObjectEntry["documentId"]
            documentData = self.citation_to_doc[doc_id]
            paper_info = documentData["paper_info"]
            paperId = paper_info.get("paper_id")

            model_inputs = [[answerObjectEntry["sentence"], context] for context in best10Contexts]
            scores = self.crossEncoder.predict(model_inputs).tolist()

            max_score = max(scores)
            best_idx = scores.index(max_score)
            
            best_context = best10Contexts[best_idx]
            result.setdefault(answerObjectEntry["documentId"], []).append({"contextPassage":best_context, "paperId": paperId})
        end = time.time()
        return result, end-start
    
    def extract_context_using_cosine_similarity_and_bm25(self, answerObject: GeneratedAnswerFormat):
        start = time.time()
        result = {}
        for answerObjectEntry in answerObject:

            doc_id = answerObjectEntry["documentId"]
            documentData = self.citation_to_doc[doc_id]
            paper_info = documentData["paper_info"]
            paperId = paper_info.get("paper_id")

            bestContext = self.extract_top_k_contexts(answerObjectEntry["documentId"], answerObjectEntry["sentence"], 1, True)
            result.setdefault(answerObjectEntry["documentId"], []).append({"contextPassage":bestContext[0], "paperId": paperId})
        end = time.time()
        return result, end-start
    
    def extract_context_using_biencoder_and_bm25_and_cross_encoder(self, answerObject: GeneratedAnswerFormat):
        start = time.time()
        result = {}
        for answerObjectEntry in answerObject:
            best10Contexts = self.extract_top_k_contexts(answerObjectEntry["documentId"], answerObjectEntry["sentence"], 10, True)
            
            doc_id = answerObjectEntry["documentId"]
            documentData = self.citation_to_doc[doc_id]
            paper_info = documentData["paper_info"]
            paperId = paper_info.get("paper_id")
            
            model_inputs = [[answerObjectEntry["sentence"], context] for context in best10Contexts]
            scores = self.crossEncoder.predict(model_inputs).tolist()

            max_score = max(scores)
            best_idx = scores.index(max_score)
            
            best_context = best10Contexts[best_idx]
            result.setdefault(answerObjectEntry["documentId"], []).append({"contextPassage":best_context, "paperId": paperId})
        end = time.time()
        return result, end-start
    
    def format_references(self, answerObject: GeneratedAnswerFormat = None) -> str:
        start = time.time()
        """Format references with proper metadata and context passages"""
        if not self.citation_to_doc:
            return ""

        # Get all available citations
        citations_to_show = set(self.citation_to_doc.keys())
        
        if answerObject and len(answerObject):
            uniqueDocIds = {dataEntry["documentId"] for dataEntry in answerObject if "documentId" in dataEntry}
            if len(uniqueDocIds):
                citations_to_show = uniqueDocIds.intersection(
                    set(self.citation_to_doc.keys())
                )
        if not citations_to_show:
            return ""

        references = {}

        for citation_num in sorted(citations_to_show):
            doc_info = self.citation_to_doc[citation_num]
            paper_info = doc_info["paper_info"]

            # Format academic reference
            # reference.append(f"[{citation_num}]")

            # Add title in quotes
            title = paper_info["title"].replace('Title:', "").replace('"', "").replace("'", "")

            # Add venue and year with paper ID
            if paper_info.get("paper_id") and paper_info["paper_id"] != "Unknown":
                if str(paper_info["paper_id"]).startswith("arXiv:"):
                    paper_id =  paper_info['paper_id']
                else:
                    paper_id = f"arXiv:{paper_info['paper_id']}"
            else:
                paper_id = f"{paper_info['venue']}"
            
            answer_text = " ".join(dataEntry["sentence"] for dataEntry in answerObject if "sentence" in dataEntry)
            # Add context passage with actual sentences used
            if answer_text:
                context_passage = self._extract_context_passage(
                    answer_text, doc_info["text"], citation_num
                ) 
            else:
                context_passage = (
                    doc_info["text"][:300] + "..."
                    if len(doc_info["text"]) > 300
                    else doc_info["text"]
                )

            references[citation_num] = {"title":title, "paperId": paper_id, "contextPassage":context_passage}
        end = time.time()    
        return references, end-start

    def add_document(self, doc_text: str, doc_id: str, metadata: Dict = None) -> int:
        """Add a document and return its citation number"""

        if doc_id not in self.doc_to_citation:
            citation_num = self.next_citation_num
            self.doc_to_citation[doc_id] = citation_num

            paper_info = self._extract_paper_info(doc_text, doc_id, metadata)

            self.citation_to_doc[citation_num] = {
                "doc_id": doc_id,
                "paper_info": paper_info,
                "text": doc_text,
            }

            self.next_citation_num += 1
            self.logger.debug(
                f"Added document {doc_id} as citation [{citation_num}]: {paper_info['title'][:50]}..."
            )
            return citation_num
        else:
            return self.doc_to_citation[doc_id]

    def get_citation_map(self) -> Dict[str, int]:
        """Get mapping from doc_id to citation number"""
        return self.doc_to_citation.copy()
