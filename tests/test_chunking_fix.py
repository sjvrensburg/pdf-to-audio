"""Tests for the chunking fix in stages 2-4 to prevent content loss in long documents."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.pdf_to_audio.core import process_stage_in_chunks
from src.pdf_to_audio.constants import MATH_PROMPT


class TestChunkingFix:
    """Test the chunking functionality for long documents."""

    def test_process_small_document_single_call(self):
        """Test that small documents are processed in a single API call."""
        # Create a small text (< 3200 tokens estimated)
        small_text = "This is a small document. " * 100  # ~500 chars, ~125 estimated tokens

        # Mock provider
        mock_provider = Mock()
        mock_response = "Processed: " + small_text

        with patch('src.pdf_to_audio.core.make_api_call', return_value=mock_response) as mock_api_call:
            result = process_stage_in_chunks(
                text=small_text,
                stage_name="Test stage",
                system_prompt="Test prompt",
                provider=mock_provider,
                max_tokens=4000,
                verbose=False
            )

            # Should make exactly 1 API call for small documents
            assert mock_api_call.call_count == 1
            assert "Processed:" in result

    def test_process_large_document_multiple_chunks(self):
        """Test that large documents are split into chunks and processed."""
        # Create a large text (> 3200 tokens estimated)
        # Each paragraph is ~200 chars, total ~20,000 chars = ~5,000 estimated tokens
        paragraph = "This is a longer paragraph with more content to simulate a real document. " * 10
        large_text = "\n\n".join([paragraph] * 100)  # ~20,000 chars

        # Mock provider
        mock_provider = Mock()

        # Mock API calls to return the input with a prefix
        def mock_api_call_side_effect(provider, messages):
            return "Processed: " + messages[1]["content"]

        with patch('src.pdf_to_audio.core.make_api_call', side_effect=mock_api_call_side_effect) as mock_api_call:
            result = process_stage_in_chunks(
                text=large_text,
                stage_name="Test stage",
                system_prompt="Test prompt",
                provider=mock_provider,
                max_tokens=4000,
                verbose=False
            )

            # Should make multiple API calls for large documents
            assert mock_api_call.call_count > 1
            # Result should contain the prefix for each chunk
            assert "Processed:" in result
            # Result should be non-empty
            assert len(result) > 0

    def test_process_handles_api_errors_gracefully(self):
        """Test that processing continues even if some chunks fail."""
        text = "Test content. " * 1000  # Large enough to require chunking

        # Mock provider
        mock_provider = Mock()

        # Mock API calls where some fail
        call_count = [0]
        def mock_api_call_side_effect(provider, messages):
            call_count[0] += 1
            if call_count[0] == 2:
                # Second call fails
                raise Exception("API error")
            return messages[1]["content"]

        with patch('src.pdf_to_audio.core.make_api_call', side_effect=mock_api_call_side_effect):
            # Should not raise an exception
            result = process_stage_in_chunks(
                text=text,
                stage_name="Test stage",
                system_prompt="Test prompt",
                provider=mock_provider,
                max_tokens=4000,
                verbose=False
            )

            # Result should still contain content
            assert len(result) > 0
            assert "Test content" in result

    def test_content_preservation_across_chunks(self):
        """Test that all content is preserved when processing in chunks."""
        # Create structured content with distinct sections
        sections = [
            "Section 1: Introduction\n\nThis is the introduction. " * 50,
            "Section 2: Methods\n\nThis describes the methods. " * 50,
            "Section 3: Results\n\nHere are the results. " * 50,
            "Section 4: Discussion\n\nThis is the discussion. " * 50,
            "Section 5: Conclusion\n\nThis is the conclusion. " * 50,
        ]
        large_text = "\n\n".join(sections)

        # Mock provider to return input unchanged
        mock_provider = Mock()

        with patch('src.pdf_to_audio.core.make_api_call', side_effect=lambda p, m: m[1]["content"]):
            result = process_stage_in_chunks(
                text=large_text,
                stage_name="Test stage",
                system_prompt="Test prompt",
                provider=mock_provider,
                max_tokens=4000,
                verbose=False
            )

            # All sections should be present in the result
            assert "Section 1: Introduction" in result
            assert "Section 2: Methods" in result
            assert "Section 3: Results" in result
            assert "Section 4: Discussion" in result
            assert "Section 5: Conclusion" in result

    def test_mathematical_content_preservation(self):
        """Test that mathematical content is preserved across chunks."""
        # Create content with math tags
        content = []
        for i in range(50):
            content.append(f"This is paragraph {i} with math: <MATH>x squared plus y squared equals z squared</MATH>. ")

        large_text = "\n\n".join(content)

        # Mock provider to return input unchanged
        mock_provider = Mock()

        with patch('src.pdf_to_audio.core.make_api_call', side_effect=lambda p, m: m[1]["content"]):
            result = process_stage_in_chunks(
                text=large_text,
                stage_name="Math processing",
                system_prompt=MATH_PROMPT,
                provider=mock_provider,
                max_tokens=4000,
                verbose=False
            )

            # Count MATH tags in input and output
            input_math_tags = large_text.count("<MATH>")
            output_math_tags = result.count("<MATH>")

            # Should preserve all math tags
            assert output_math_tags == input_math_tags

            # Should preserve all paragraph numbers
            for i in range(50):
                assert f"paragraph {i}" in result

    def test_empty_text_handling(self):
        """Test handling of empty or very small text."""
        mock_provider = Mock()

        with patch('src.pdf_to_audio.core.make_api_call', return_value="") as mock_api_call:
            result = process_stage_in_chunks(
                text="",
                stage_name="Test stage",
                system_prompt="Test prompt",
                provider=mock_provider,
                max_tokens=4000,
                verbose=False
            )

            # Should handle empty text gracefully (may result in chunking)
            assert mock_api_call.call_count >= 1
            assert result == ""
