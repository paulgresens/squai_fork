import logging
from hybrid_retriever import Retriever
import json 

logger = logging.getLogger("Enhanced_4Agent_RAG")

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
    logger.info(f"Initializing {retriever_type} retriever with alpha={alpha}...")
    return Retriever(
        e5_index_dir, bm25_index_dir, top_k=top_k, strategy=retriever_type, alpha=alpha
    )

def load_datamorgana_questions(file_path):
    """Load questions from file"""
    is_jsonl = file_path.lower().endswith(".jsonl")

    try:
        questions = []

        if is_jsonl:
            logger.info(f"Loading questions from JSONL file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        question = json.loads(line)
                        if "id" not in question:
                            question["id"] = line_num
                        questions.append(question)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON at line {line_num}: {e}")
        else:
            logger.info(f"Loading questions from JSON file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list):
                    questions = data
                    for i, question in enumerate(questions):
                        if "id" not in question:
                            question["id"] = i + 1
                elif isinstance(data, dict):
                    if "questions" in data:
                        questions = data["questions"]
                    elif "question" in data:
                        questions = [data]
                    else:
                        questions = [data]

        logger.info(f"Loaded {len(questions)} questions")
        return questions

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading questions: {e}")
        return []


def format_enhanced_result_to_schema(result):
    """Format result with enhanced 4-agent information"""
    formatted_result = {
        "id": result.get("id", 0),
        "question": result.get("question", ""),
        "answer": result.get("model_answer", ""),
        "was_split": result.get("was_split", False),
        "sub_questions": result.get("sub_questions", []),
        "questions_processed": result.get("questions_processed", 1),
        "citation_count": result.get("total_citations", 0),
        "total_filtered_docs": result.get("total_filtered_docs", 0),
        "full_texts_used": result.get("full_texts_retrieved", 0),
        "processing_time": result.get("process_time", 0),
        "retriever_type": result.get("retriever_type", "hybrid"),
        "passages_used": result.get("passages_used", []),
        "document_metadata": result.get("document_metadata", {}),
    }

    return formatted_result

def write_enhanced_result_to_json(result, output_file):
    """Write single enhanced result to JSON file"""
    formatted_result = format_enhanced_result_to_schema(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Enhanced result written to {output_file}")

def write_enhanced_results_to_jsonl(results, output_file):
    """Write enhanced results to JSONL file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            formatted_result = format_enhanced_result_to_schema(result)
            f.write(json.dumps(formatted_result, ensure_ascii=False) + "\n")
    logger.info(f"Enhanced results written to {output_file}")
