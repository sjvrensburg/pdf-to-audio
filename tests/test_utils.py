"""Tests for utility functions."""

import pytest
from pdf_to_audio.utils import estimate_tokens, split_chunk, post_process_output, clean_base64_image


def test_estimate_tokens():
    """Test token estimation function."""
    text = "This is a test sentence with eight words."
    tokens = estimate_tokens(text)
    # Should be approximately 8 / 0.75 = ~10.67, rounded to 10
    assert tokens == 10


def test_split_chunk_small():
    """Test chunk splitting with small content."""
    content = "This is a small paragraph.\n\nThis is another paragraph."
    chunks = split_chunk(content, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0] == content


def test_split_chunk_large():
    """Test chunk splitting with large content."""
    # Create content that exceeds token limit
    large_paragraph = " ".join(["word"] * 1000)  # ~1333 tokens
    content = f"{large_paragraph}\n\n{large_paragraph}"
    
    chunks = split_chunk(content, max_tokens=500)
    assert len(chunks) > 1


def test_post_process_output():
    """Test output post-processing."""
    messy_text = "This  has   multiple    spaces.\n\n\n\nAnd too many line breaks."
    cleaned = post_process_output(messy_text)
    
    assert "  " not in cleaned  # No double spaces
    assert "\n\n\n" not in cleaned  # No triple line breaks


def test_clean_base64_image():
    """Test base64 image cleaning."""
    # Test with data prefix
    base64_with_prefix = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    cleaned = clean_base64_image(base64_with_prefix)
    assert not cleaned.startswith("data:image/")
    
    # Test with clean base64
    clean_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    result = clean_base64_image(clean_base64)
    assert result == clean_base64
    
    # Test with empty string
    assert clean_base64_image("") == ""
    assert clean_base64_image(None) == ""