"""
Text chunking module for audio generation.

This module provides functionality to chunk text into segments suitable for
TTS processing, with special handling for academic and mathematical content.
"""

import re
import logging
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Constants
MAX_CHUNK_DURATION_SEC = 35  # Safe margin below Chatterbox's 40-second limit
AVG_WORDS_PER_MINUTE = 150
AVG_CHARS_PER_WORD = 5
CHARS_PER_SECOND = (AVG_WORDS_PER_MINUTE * AVG_CHARS_PER_WORD) / 60

# Regular expressions for identifying special content
MATH_EXPR_PATTERN = r'\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{equation\}.*?\\end\{equation\}'
CITATION_PATTERN = r'\[[\d,\s-]+\]|\([\w\s]+,\s*\d{4}\)'
FIGURE_REF_PATTERN = r'Figure \d+|Fig\. \d+|Table \d+|Tab\. \d+'
SECTION_HEADING_PATTERN = r'^#+\s+.*$|^.*\n[=-]+\n'


class TextChunker:
    """
    Chunks text into segments suitable for TTS processing.
    """

    def __init__(
        self,
        max_duration_sec: float = MAX_CHUNK_DURATION_SEC,
        chars_per_second: float = CHARS_PER_SECOND,
        strategy: str = "smart"
    ):
        """
        Initialize the text chunker.

        Args:
            max_duration_sec: Maximum duration in seconds for each chunk.
            chars_per_second: Estimated characters per second in speech.
            strategy: Chunking strategy to use ("duration", "sentences", or "smart").
        """
        self.max_duration_sec = max_duration_sec
        self.chars_per_second = chars_per_second
        self.strategy = strategy
        self.max_chars_per_chunk = int(max_duration_sec * chars_per_second)
        
        logger.info(
            f"Initialized TextChunker with strategy={strategy}, "
            f"max_duration={max_duration_sec}s, "
            f"max_chars_per_chunk={self.max_chars_per_chunk}"
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into segments suitable for TTS processing.

        Args:
            text: The text to chunk.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []
            
        if self.strategy == "duration":
            return self._chunk_by_duration(text)
        elif self.strategy == "sentences":
            return self._chunk_by_sentences(text)
        elif self.strategy == "smart":
            return self._chunk_smart(text)
        else:
            logger.warning(f"Unknown chunking strategy: {self.strategy}. Using 'smart' strategy.")
            return self._chunk_smart(text)

    def _chunk_by_duration(self, text: str) -> List[str]:
        """
        Chunk text based on estimated duration.

        Args:
            text: The text to chunk.

        Returns:
            A list of text chunks.
        """
        chunks = []
        current_chunk = ""
        
        for paragraph in text.split("\n\n"):
            # If adding this paragraph would exceed the max chars, start a new chunk
            if len(current_chunk) + len(paragraph) > self.max_chars_per_chunk and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                
            # If the paragraph itself exceeds max chars, split it further
            if len(paragraph) > self.max_chars_per_chunk:
                # Add any existing content to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    
                # Split the paragraph into smaller chunks
                words = paragraph.split()
                temp_chunk = ""
                
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > self.max_chars_per_chunk:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = word
                    else:
                        temp_chunk += " " + word if temp_chunk else word
                
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                # Add paragraph with a newline
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        logger.info(f"Chunked text into {len(chunks)} chunks using duration strategy")
        return chunks

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences, keeping related sentences together.

        Args:
            text: The text to chunk.

        Returns:
            A list of text chunks.
        """
        # Split text into sentences
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # If adding this sentence would exceed the max chars, start a new chunk
            if len(current_chunk) + len(sentence) > self.max_chars_per_chunk and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                
            # If the sentence itself exceeds max chars, use duration-based chunking for it
            if len(sentence) > self.max_chars_per_chunk:
                # Add any existing content to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    
                # Use duration-based chunking for this long sentence
                sentence_chunks = self._chunk_by_duration(sentence)
                chunks.extend(sentence_chunks)
            else:
                # Add sentence with appropriate spacing
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        logger.info(f"Chunked text into {len(chunks)} chunks using sentence strategy")
        return chunks

    def _chunk_smart(self, text: str) -> List[str]:
        """
        Smart chunking that preserves semantic units and handles special content.

        Args:
            text: The text to chunk.

        Returns:
            A list of text chunks.
        """
        # First, identify and protect special content
        protected_text, protected_regions = self._protect_special_content(text)
        
        # Split text into paragraphs
        paragraphs = protected_text.split("\n\n")
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if this is a section heading
            is_heading = bool(re.match(SECTION_HEADING_PATTERN, paragraph, re.MULTILINE))
            
            # If adding this paragraph would exceed the max chars or it's a heading, start a new chunk
            if ((current_length + len(paragraph) > self.max_chars_per_chunk and current_chunk) or 
                (is_heading and current_chunk)):
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_length = 0
                
            # If the paragraph itself exceeds max chars, split it into sentences
            if len(paragraph) > self.max_chars_per_chunk:
                # Add any existing content to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_length = 0
                    
                # Split the paragraph into sentences
                sentence_endings = r'(?<=[.!?])\s+'
                sentences = re.split(sentence_endings, paragraph)
                
                temp_chunk = ""
                temp_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # If adding this sentence would exceed the max chars, start a new temp chunk
                    if temp_length + len(sentence) > self.max_chars_per_chunk and temp_chunk:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = ""
                        temp_length = 0
                        
                    # If the sentence itself exceeds max chars, use duration-based chunking
                    if len(sentence) > self.max_chars_per_chunk:
                        # Add any existing content to chunks
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = ""
                            temp_length = 0
                            
                        # Use duration-based chunking for this long sentence
                        sentence_chunks = self._chunk_by_duration(sentence)
                        chunks.extend(sentence_chunks)
                    else:
                        # Add sentence with appropriate spacing
                        if temp_chunk:
                            temp_chunk += " " + sentence
                        else:
                            temp_chunk = sentence
                        temp_length = len(temp_chunk)
                
                # Add the last temp chunk if it's not empty
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_length = temp_length
            else:
                # Add paragraph with a newline
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_length = len(current_chunk)
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Restore protected content
        chunks = [self._restore_special_content(chunk, protected_regions) for chunk in chunks]
        
        logger.info(f"Chunked text into {len(chunks)} chunks using smart strategy")
        return chunks

    def _protect_special_content(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Identify and protect special content like math expressions and citations.

        Args:
            text: The text to process.

        Returns:
            A tuple containing the text with placeholders and a dictionary mapping
            placeholders to original content.
        """
        protected_regions = {}
        
        # Function to replace matches with placeholders
        def replace_with_placeholder(pattern, prefix):
            nonlocal text, protected_regions
            
            matches = re.finditer(pattern, text, re.DOTALL)
            for i, match in enumerate(matches):
                placeholder = f"__{prefix}_{i}__"
                protected_regions[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
                
        # Protect math expressions
        replace_with_placeholder(MATH_EXPR_PATTERN, "MATH")
        
        # Protect citations
        replace_with_placeholder(CITATION_PATTERN, "CITE")
        
        # Protect figure references
        replace_with_placeholder(FIGURE_REF_PATTERN, "FIG")
        
        return text, protected_regions

    def _restore_special_content(self, text: str, protected_regions: Dict[str, str]) -> str:
        """
        Restore protected content from placeholders.

        Args:
            text: The text with placeholders.
            protected_regions: Dictionary mapping placeholders to original content.

        Returns:
            The text with original content restored.
        """
        for placeholder, original in protected_regions.items():
            text = text.replace(placeholder, original)
            
        return text