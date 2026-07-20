# ============================================================
#
# Splits clean text into optimized chunks for our RAG system.
#
# Why chunking matters:
# LLMs have a context window limit (max text they can read).
# We cannot send a 194-page document to an LLM at once.
# We split into chunks, find the relevant ones, and send only those.
#
# Good chunking = better answers.
# Bad chunking = missed context = wrong answers.
# ============================================================

import re
import hashlib
from typing import List, Dict


# Optimized chunk sizes for petroleum documents
# These values were chosen based on:
# - Typical length of a DDR section (operations log entry)
# - LLM context window efficiency
# - Semantic coherence (keeping related info together)
CHUNK_SIZE = 600          # Characters per chunk
CHUNK_OVERLAP = 120       # Overlap between chunks (20% of chunk size)
MIN_CHUNK_SIZE = 50       # Skip chunks smaller than this


class SmartChunker:
    """
    Intelligent text chunker optimized for petroleum documents.

    Features:
    - Splits at natural boundaries (paragraphs, sections)
    - Maintains overlap to prevent missing cross-boundary answers
    - Adds metadata to each chunk for better retrieval
    - Handles petroleum-specific section markers
    """

    def __init__(self,
                 chunk_size=CHUNK_SIZE,
                 chunk_overlap=CHUNK_OVERLAP):
        """
        Initialize chunker with size parameters.

        chunk_size: Target number of characters per chunk
        chunk_overlap: How many chars to repeat between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Petroleum document section markers
        # We prefer to split AT these markers
        # This keeps sections coherent instead of cutting through them
        self.section_markers = [
            # DDR section headers
            r'DEPTH SUMMARY',
            r'DRILLING PARAMETERS',
            r'MUD.*REPORT',
            r'FORMATION DESCRIPTION',
            r'OPERATIONS LOG',
            r'INCIDENTS.*NON-PRODUCTIVE',
            r'SAFETY REPORT',
            r'NEXT 24-HOUR',
            # General section markers
            r'SECTION \d+',
            r'CHAPTER \d+',
            r'\d+\.\d+\s+[A-Z][A-Z\s]+',  # Numbered sections like "2.1 DRILLING"
        ]

        # Compile markers for speed
        self.section_pattern = re.compile(
            '|'.join(self.section_markers),
            re.IGNORECASE | re.MULTILINE
        )

    def generate_chunk_id(self, text: str, doc_name: str,
                          chunk_index: int) -> str:
        """
        Generate a unique, stable ID for each chunk.

        Uses MD5 hash of content + document + index.
        Same chunk always gets same ID → safe to re-index.
        """
        content = f"{doc_name}::{chunk_index}::{text[:100]}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def split_by_sections(self, text: str) -> List[str]:
        """
        Split text into sections at natural petroleum document boundaries.

        First we try to split at section headers.
        If sections are too long, we split them further.
        If too short, we merge them.
        """
        # Find all section marker positions
        section_starts = [0]  # Always start with position 0
        for match in self.section_pattern.finditer(text):
            section_starts.append(match.start())

        # Extract sections
        sections = []
        for i, start in enumerate(section_starts):
            end = section_starts[i + 1] if i + 1 < len(section_starts) else len(text)
            section = text[start:end].strip()
            if len(section) >= MIN_CHUNK_SIZE:
                sections.append(section)

        return sections if sections else [text]

    def split_section_into_chunks(self, section: str,
                                  doc_name: str,
                                  chunk_index_offset: int) -> List[Dict]:
        """
        Split a single section into fixed-size overlapping chunks.

        Why overlapping?
        Answer: "The stuck pipe occurred at 1,654 feet with 3.5 hours NPT"
        If "1,654 feet" is at the END of chunk 3 and
        "3.5 hours NPT" is at the START of chunk 4,
        without overlap no single chunk has the full answer.
        With overlap, chunk 3 ends with some of chunk 4's content.
        """
        chunks = []
        start = 0
        chunk_index = chunk_index_offset

        while start < len(section):
            # Calculate end position
            end = start + self.chunk_size

            # If not at the end, try to break at a sentence boundary
            if end < len(section):
                # Look for the last sentence ending (. ! ?) within the chunk
                # This prevents cutting in the middle of a sentence
                last_period = max(
                    section.rfind('. ', start, end),
                    section.rfind('.\n', start, end),
                    section.rfind('\n\n', start, end)
                )

                if last_period > start + (self.chunk_size // 2):
                    # Found a good break point past halfway through the chunk
                    end = last_period + 1
                # If no good break found, just cut at chunk_size (acceptable)

            chunk_text = section[start:end].strip()

            if len(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text),
                    "start_char": start,
                    "end_char": end
                })
                chunk_index += 1

            # Move start forward, accounting for overlap
            # We go back by chunk_overlap characters to create overlap
            start = end - self.chunk_overlap

            # Safety check: ensure we always move forward
            if start >= end:
                start = end

        return chunks

    def chunk_document(self, text: str, doc_name: str) -> List[Dict]:
        """
        MAIN METHOD: Split a full document into chunks.

        Process:
        1. Split into sections at petroleum section headers
        2. Split each section into overlapping fixed-size chunks
        3. Add metadata to each chunk
        4. Generate unique ID for each chunk

        Returns: List of chunk dictionaries, each containing:
            - id: Unique chunk identifier
            - text: The chunk content
            - document: Source document name
            - chunk_index: Position in document
            - char_count: Length of chunk
        """
        if not text or len(text.strip()) < MIN_CHUNK_SIZE:
            print(f"    ⚠️  Text too short to chunk: {len(text)} chars")
            return []

        # Step 1: Split into sections
        sections = self.split_by_sections(text)

        # Step 2: Split each section into chunks
        all_chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self.split_section_into_chunks(
                section, doc_name, chunk_index
            )
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Step 3: Add metadata and IDs to all chunks
        final_chunks = []
        for chunk_data in all_chunks:
            chunk_id = self.generate_chunk_id(
                chunk_data["text"], doc_name, chunk_data["chunk_index"]
            )
            final_chunks.append({
                "id": chunk_id,
                "text": chunk_data["text"],
                "document": doc_name,
                "chunk_index": chunk_data["chunk_index"],
                "char_count": chunk_data["char_count"]
            })

        print(f"    Chunking complete: {len(final_chunks)} chunks created")
        return final_chunks