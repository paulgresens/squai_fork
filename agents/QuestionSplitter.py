import logging
from typing import List, Tuple
from performance_monitor import monitor, time_block
from logging import Logger

class QuestionSplitter:
    """
    Agent 1: Intelligent Question Splitting Agent
    Detects complex queries with multiple sub-questions and splits them appropriately
    """

    def __init__(self, agent_model, logger: Logger):
        self.agent = agent_model
        self.logger = logger
        self.logger.info("Agent 1 (Question Splitter) initialized")

    def _create_splitting_prompt(self, query: str) -> str:
        """Create prompt for question splitting analysis"""
        return f"""You are an intelligent question analyzer. Your task is to determine if a query contains multiple distinct sub-questions that would benefit from separate retrieval and research.

SPLITTING CRITERIA:
- Split if query contains multiple distinct topics connected by "and", "also", "what about"
- Split if query asks for COMPARISONS or DIFFERENCES between concepts (e.g., "difference between X and Y")
- Split if query asks for comparisons PLUS evaluation/preference (e.g., "which is better")
- Split if query has multiple question words (what, how, why, when, where, which)
- Split if query asks about a concept AND its implications/effects/applications
- DO NOT split simple clarifications or related aspects of the same topic

Examples:

Query: "What is quantum computing and how is it used in cryptography?"
Split: YES
Sub-questions: ["What is quantum computing?", "How is quantum computing used in cryptography?"]

Query: "What is the difference between dense and sparse retrieval and which one is better suited for RAG?"
Split: YES
Sub-questions: ["What is dense retrieval?", "What is sparse retrieval?", "Which retrieval method is better suited for RAG systems?"]

Query: "Compare transformers and RNNs and explain which is better for sequence modeling"
Split: YES
Sub-questions: ["What are transformers?", "What are RNNs?", "Which architecture is better for sequence modeling?"]

Query: "What is page rank algorithm and who invented it?"
Split: YES
Sub-questions: ["What is page rank algorithm?", "Who invented page rank algorithm?"]

Query: "How does BERT work and what is GPT-3?"
Split: YES  
Sub-questions: ["How does BERT work?", "What is GPT-3?"]

Query: "What are the advantages and disadvantages of federated learning?"
Split: YES
Sub-questions: ["What are the advantages of federated learning?", "What are the disadvantages of federated learning?"]

Query: "What is reinforcement learning?"
Split: NO
Sub-questions: []

Query: "Explain how attention mechanism works in transformers"
Split: NO
Sub-questions: []

Query: "What are neural networks and how do they learn and what are CNNs?"
Split: YES
Sub-questions: ["What are neural networks?", "How do neural networks learn?", "What are CNNs?"]

Now analyze this query:
Query: "{query}"

IMPORTANT: For comparison questions with evaluation (like "difference between X and Y and which is better"), always split into:
1. Explanation of concept X
2. Explanation of concept Y  
3. Comparison/evaluation question

Respond with ONLY this format:
Split: YES/NO
Sub-questions: [list of questions] (empty list if Split: NO)"""

    def analyze_and_split(self, query: str) -> Tuple[bool, List[str]]:
        """
        Analyze query and split into sub-questions if beneficial
        Always uses LLM for accurate analysis

        Returns:
            Tuple of (should_split: bool, sub_questions: List[str])
        """
        # Handle time_block if it exists (for compatibility)
        try:
            # Check if time_block is available
            if 'time_block' in globals():
                with time_block("agent1_question_splitting"):
                    return self._perform_split_analysis(query)
            else:
                return self._perform_split_analysis(query)
        except:
            # Fallback if time_block causes any issues
            return self._perform_split_analysis(query)

    def _perform_split_analysis(self, query: str) -> Tuple[bool, List[str]]:
        """
        Internal method to perform the actual split analysis using LLM
        """
        self.logger.info(f"Agent 1: Analyzing query for splitting: {query}")
        
        # Basic validation - only skip extremely short queries
        if len(query.strip()) < 10:  # Less than 10 characters is too short
            self.logger.info("Query too short for meaningful splitting")
            return False, []
        
        try:
            # Always use LLM for analysis (no heuristics)
            self.logger.info("Using LLM to analyze if query should be split...")
            
            prompt = self._create_splitting_prompt(query)
            response = self.agent.generate(prompt)
            
            # Parse response
            should_split, sub_questions = self._parse_splitting_response(response, query)
            
            if should_split:
                self.logger.info(f"Agent 1: LLM decided to split into {len(sub_questions)} sub-questions: {sub_questions}")
            else:
                self.logger.info("Agent 1: LLM decided no splitting needed")
            
            return should_split, sub_questions
            
        except Exception as e:
            self.logger.error(f"Error in LLM splitting analysis: {e}")
            # On error, don't split
            return False, []

    def _parse_splitting_response(self, response: str, original_query: str) -> Tuple[bool, List[str]]:
        """Parse the LLM response for splitting decision"""
        try:
            lines = response.strip().split('\n')
            should_split = False
            sub_questions = []

            for line in lines:
                line = line.strip()
                if line.startswith("Split:"):
                    should_split = "YES" in line.upper()
                elif line.startswith("Sub-questions:"):
                    # Extract list from the line
                    list_part = line.split(":", 1)[1].strip()
                    if list_part and list_part != "[]":
                        # Parse the list - handle both ["q1", "q2"] and simple comma-separated
                        try:
                            if list_part.startswith("[") and list_part.endswith("]"):
                                # JSON-like format
                                sub_questions = json.loads(list_part)
                            else:
                                # Comma-separated format
                                sub_questions = [
                                    q.strip().strip('"').strip("'")
                                    for q in list_part.split(",")
                                ]
                        except:
                            self.logger.warning(f"Failed to parse sub-questions: {list_part}")
                            sub_questions = []

            # Validation: ensure sub-questions are meaningful
            if should_split and sub_questions:
                valid_questions = []
                for q in sub_questions:
                    q = q.strip()
                    # Clean up and validate
                    if len(q) > 10:
                        # Add question mark if missing
                        if not q.endswith("?"):
                            q = q + "?"
                        valid_questions.append(q)

                if len(valid_questions) < 2:
                    self.logger.info("Not enough valid sub-questions, keeping original")
                    return False, []

                return True, valid_questions

            return False, []

        except Exception as e:
            self.logger.warning(f"Error parsing splitting response: {e}")
            return False, []

    def _quick_split_check(self, query: str) -> bool:
        """
        DEPRECATED: Keep for backward compatibility but not used
        This method is kept in case other parts of the code reference it
        """
        # Always return False since we're using LLM directly now
        return False
