import logging
import re
import json
from typing import  Dict

logger = logging.getLogger("Enhanced_4Agent_RAG")

class PaperTitleExtractor:
    """
    Utility class for extracting paper titles from document text
    IMPROVED: Handles LevelDB storage format where title is on second line
    """

    @staticmethod
    def extract_title_from_text(doc_text: str, doc_id: str, ) -> str:
        """
        Extract paper title from document text using multiple patterns
        IMPROVED: Handles "Content for [paper_id]:\n[Title]" format from LevelDB
        """
        try:
            # Method 1: NEW - Handle LevelDB format: "Content for [paper_id]:\n[Title]"
            leveldb_pattern = r"Content for [^:]*:\s*\n([^\n]+)"
            match = re.search(leveldb_pattern, doc_text)
            if match:
                title_candidate = match.group(1).strip()
                # Validate it looks like a title (not abstract or other content)
                if (
                    len(title_candidate) > 10
                    and len(title_candidate) < 300
                    and not title_candidate.lower().startswith(
                        ("abstract:", "introduction:", "the abstract", "in this", "we ")
                    )
                ):

                    logger.debug(
                        f"Extracted title from LevelDB format: {title_candidate[:50]}..."
                    )
                    return title_candidate

            # Method 2: Look for title in first few lines (for direct title format)
            lines = doc_text.split("\n")
            for i, line in enumerate(lines[:5]):
                line = line.strip()

                # Skip empty lines and common headers
                if not line or line.lower().startswith(
                    ("content for", "time taken", "opening")
                ):
                    continue

                # Check if this line looks like a title
                if (
                    len(line) > 10
                    and len(line) < 300
                    and not line.lower().startswith(
                        (
                            "abstract:",
                            "introduction:",
                            "the abstract",
                            "in this",
                            "we ",
                            "this paper",
                            "{",
                        )
                    )
                    and not re.match(r"^\d+", line)  # Not starting with numbers
                    and not line.endswith(":")  # Not a section header
                    and line.count(" ") >= 2
                ):  # At least 3 words

                    logger.debug(f"Extracted title from line {i+1}: {line[:50]}...")
                    return line

            # Method 3: Look for "Content for [paper_id]:" pattern (legacy)
            content_pattern = r"Content for [^:]*:\s*\n([^\n]+)"
            match = re.search(content_pattern, doc_text)
            if match:
                title_candidate = match.group(1).strip()
                if len(title_candidate) > 10 and len(title_candidate) < 300:
                    title_candidate = re.sub(r'^["\']|["\']$', "", title_candidate)
                    title_candidate = re.sub(r"^\W+|\W+$", "", title_candidate)
                    if len(title_candidate) > 10:
                        return title_candidate

            # Method 4: Look for "Title. {" pattern
            title_brace_pattern = r"^([^.]+)\.\s*\{"
            match = re.search(title_brace_pattern, doc_text.strip(), re.MULTILINE)
            if match:
                title_candidate = match.group(1).strip()
                if (
                    len(title_candidate) > 10
                    and len(title_candidate) < 300
                    and not title_candidate.lower().startswith(
                        ("the ", "this ", "in ", "we ", "abstract", "introduction")
                    )
                ):
                    title_candidate = re.sub(r'^["\']|["\']$', "", title_candidate)
                    if len(title_candidate) > 10:
                        return title_candidate

            # Method 5: Extract from cleaned first sentence
            clean_text = re.sub(r"\{[^}]*\}", "", doc_text)
            clean_text = re.sub(r"Content for [^:]+:\s*", "", clean_text)
            clean_text = clean_text.strip()

            first_sentence = clean_text.split("\n")[0].strip()
            if ". {" in first_sentence:
                first_sentence = first_sentence.split(". {")[0].strip()
            elif ". " in first_sentence and len(first_sentence.split(". ")[0]) < 200:
                first_sentence = first_sentence.split(". ")[0].strip()

            if (
                len(first_sentence) > 15
                and len(first_sentence) < 300
                and not first_sentence.lower().startswith(
                    (
                        "content for",
                        "time taken",
                        "opening",
                        "the ",
                        "this ",
                        "in ",
                        "we ",
                        "abstract",
                        "introduction",
                    )
                )
                and not re.match(r"^\d+", first_sentence)
            ):
                return first_sentence

            # Method 6: Try JSON metadata
            if "{" in doc_text and '"title"' in doc_text:
                try:
                    json_match = re.search(r'\{.*?"title".*?\}', doc_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        metadata = json.loads(json_str)
                        if "title" in metadata and len(metadata["title"]) > 10:
                            return metadata["title"]
                except:
                    pass

            # Fallback: use first substantial line
            for line in lines[:5]:
                line = line.strip()
                if len(line) > 15 and len(line) < 200:
                    return line[:150] + "..." if len(line) > 150 else line

            return f"Document {doc_id}"

        except Exception as e:
            logger.debug(f"Error extracting title for {doc_id}: {e}")
            return f"Document {doc_id}"

    @staticmethod
    def format_title_for_log(title: str, max_length: int = 80) -> str:
        """Format title for logging with length limit"""
        if len(title) <= max_length:
            return title
        return title[: max_length - 3] + "..."

    @staticmethod
    def extract_paper_sections(
        full_text: str, max_chars_per_section: int = 10000
    ) -> Dict[str, str]:
        """
        Extract key sections from full paper text for better context utilization

        Args:
            full_text: The full paper text
            max_chars_per_section: Limit for introduction and conclusion extraction (abstract is kept full)

        Returns:
            Dict with 'title', 'abstract', 'introduction', 'conclusion' keys
            Note: Abstract is returned in full (no artificial limits)
        """
        sections = {}

        # Extract title (first line after "Content for")
        title_match = re.search(r"Content for [^:]*:\s*\n([^\n]+)", full_text)
        if title_match:
            sections["title"] = title_match.group(1).strip()

        # Extract abstract (keep full abstract - they're naturally short and important)
        abstract_match = re.search(
            r"abstract:\s*(.+?)(?:\n\n|\nintroduction|\nrelated work|\nmethodology)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            # Keep full abstract - no artificial limits since they're naturally concise
            sections["abstract"] = abstract_text

        # Extract introduction (can be long and informative)
        intro_match = re.search(
            r"(?:^|\n)introduction[:\n]\s*(.+?)(?:\n\n[A-Z]|\nrelated work|\nmethodology|\nconclusion)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if intro_match:
            intro_text = intro_match.group(1).strip()
            sections["introduction"] = intro_text[:max_chars_per_section]

        # Extract conclusion (moderate length, important summary)
        conclusion_match = re.search(
            r"(?:^|\n)conclusion[s]?[:\n]\s*(.+?)(?:\n\n[A-Z]|\nreferences|\nacknowledgments|$)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if conclusion_match:
            conclusion_text = conclusion_match.group(1).strip()
            sections["conclusion"] = conclusion_text[:max_chars_per_section]

        return sections
