import re
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
import docx2txt

class RecursiveCharacterTextSplitter:
    """
    A smart splitter that recursively splits text on a list of separators
    until chunks are smaller than the target chunk_size, while keeping an overlap.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Splits the text recursively based on separators."""
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        # If text is small enough, return as-is
        if len(text) <= self.chunk_size:
            return [text]

        # Use the first separator that actually splits the text
        if not separators:
            # No separators left, force-split by chunk_size
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i:i + self.chunk_size])
            return chunks

        separator = separators[0]
        next_separators = separators[1:]

        # Split text by separator
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split)
            # If a single split is larger than chunk_size, recurse on it
            if split_len > self.chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Recurse on the oversized split
                sub_chunks = self._split(split, next_separators)
                chunks.extend(sub_chunks)
                continue

            # Check if adding this split exceeds size limit
            addition_len = split_len + (len(separator) if current_chunk else 0)
            if current_length + addition_len > self.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                
                # Handle overlap: backtrack to include trailing elements that fit in overlap
                overlap_chunk = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    prev_len = len(prev) + (len(separator) if overlap_chunk else 0)
                    if overlap_len + prev_len <= self.chunk_overlap:
                        overlap_chunk.insert(0, prev)
                        overlap_len += prev_len
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(split)
            current_length += split_len + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        # Filter out empty chunks
        return [c.strip() for c in chunks if c.strip()]


def parse_pdf(file_path: Path) -> str:
    """Extracts text content from a PDF file."""
    reader = PdfReader(file_path)
    text_parts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(f"--- Page {i+1} ---\n{page_text}")
    return "\n\n".join(text_parts)


def parse_docx(file_path: Path) -> str:
    """Extracts text content from a Word Document file."""
    return docx2txt.process(str(file_path))


def parse_text_or_code(file_path: Path) -> str:
    """Reads a plain text file, markdown file, or source code file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_file(file_path: Path) -> str:
    """Dispatches parsing to the appropriate extractor based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix in (".docx", ".doc"):
        return parse_docx(file_path)
    elif suffix in (".txt", ".md", ".markdown", ".py", ".js", ".ts", ".html", ".css", ".json", ".sh", ".yml", ".yaml", ".ini", ".conf"):
        return parse_text_or_code(file_path)
    else:
        # Fallback to general text reading
        try:
            return parse_text_or_code(file_path)
        except Exception:
            return ""


def chunk_document(file_path: Path, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Parses a document and splits it into metadata-enriched chunks.
    """
    content = parse_file(file_path)
    if not content:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_text(content)

    chunks_data = []
    for idx, text in enumerate(raw_chunks):
        chunks_data.append({
            "text": text,
            "metadata": {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "chunk_index": idx,
                "category": file_path.parent.name  # e.g., 'docs', 'papers', 'notes'
            }
        })
    return chunks_data


if __name__ == "__main__":
    # Quick test of text splitter
    test_text = "Hello world. " * 100
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text(test_text)
    print(f"Split test text into {len(chunks)} chunks.")
    if chunks:
        print(f"First chunk length: {len(chunks[0])}, content: {chunks[0][:50]}...")
