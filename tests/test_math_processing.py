"""Tests for mathematical content processing."""

import re
import unittest
from unittest.mock import patch, MagicMock

import torch

from pdf_to_audio.audio.chunking import MATH_TAG_PATTERN
from pdf_to_audio.core import generate_audio


class TestMathProcessing(unittest.TestCase):
    """Test cases for mathematical content processing."""

    def test_math_tag_pattern(self):
        """Test the regular expression for math tags."""
        # Test basic math tag
        text = "This is a test with <MATH>a squared plus b squared equals c squared</MATH>"
        matches = re.findall(MATH_TAG_PATTERN, text)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], "<MATH>a squared plus b squared equals c squared</MATH>")
        
        # Test multiple math tags
        text = "First equation: <MATH>x equals 5</MATH>. Second equation: <MATH>y equals 10</MATH>"
        matches = re.findall(MATH_TAG_PATTERN, text)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0], "<MATH>x equals 5</MATH>")
        self.assertEqual(matches[1], "<MATH>y equals 10</MATH>")
        
        # Test nested math tags (should only match the outer ones)
        text = "<MATH>outer <MATH>inner</MATH> content</MATH>"
        matches = re.findall(MATH_TAG_PATTERN, text)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], "<MATH>outer <MATH>inner</MATH> content</MATH>")
        
    @patch('pdf_to_audio.core.ChatterboxTTSEngine')
    @patch('pdf_to_audio.core.load_config')
    @patch('pdf_to_audio.core.merge_with_args')
    @patch('pdf_to_audio.core.get_math_tts_settings')
    @patch('pdf_to_audio.core.temp_directory')
    @patch('pdf_to_audio.core.TextChunker')
    @patch('pdf_to_audio.core.AudioConcatenator')
    @patch('pdf_to_audio.core.AudioProcessor')
    @patch('pdf_to_audio.core.AudioFormatHandler')
    @patch('pdf_to_audio.core.shutil')
    def test_generate_audio_with_math_content(
        self, mock_shutil, mock_format_handler, mock_processor, mock_concatenator, 
        mock_chunker, mock_temp_dir, mock_get_math_settings, mock_merge_args, 
        mock_load_config, mock_tts_engine
    ):
        """Test generating audio with math content."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.output_audio = "output.mp3"
        
        mock_config = {
            "mistral": {"text_model": "mistral-small-latest"},
            "tts": {
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "math_tts_scale": 0.75,
                "audio_format": "mp3",
                "voice_path": None,
                "chunk_strategy": "smart"
            },
            "general": {"temp_dir": None, "force_cpu": False}
        }
        mock_merge_args.return_value = mock_config
        
        mock_math_settings = {"exaggeration": 0.375, "cfg_weight": 0.375}
        mock_get_math_settings.return_value = mock_math_settings
        
        # Mock the temp directory context manager
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"
        
        # Mock the TTS engine
        mock_engine = MagicMock()
        mock_engine.sample_rate = 24000
        mock_engine.generate_audio.return_value = (torch.zeros((1, 1000)), 24000)
        mock_tts_engine.return_value = mock_engine
        
        # Mock the chunker
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk_text.return_value = [
            "This is regular text.",
            "This contains <MATH>a squared plus b squared equals c squared</MATH>.",
            "<MATH>x equals 5</MATH> and <MATH>y equals 10</MATH>."
        ]
        mock_chunker.return_value = mock_chunker_instance
        
        # Mock the concatenator
        mock_concatenator_instance = MagicMock()
        mock_concatenator.return_value = mock_concatenator_instance
        
        # Mock the processor
        mock_processor_instance = MagicMock()
        mock_processor_instance.normalize_volume.return_value = torch.zeros((1, 1000))
        mock_processor_instance.add_pause.return_value = torch.zeros((1, 1000))
        mock_processor.return_value = mock_processor_instance
        
        # Mock the format handler
        mock_format_handler_instance = MagicMock()
        mock_format_handler_instance.convert_format.return_value = "output.mp3"
        mock_format_handler.return_value = mock_format_handler_instance
        
        # Call the function
        text = """
        This is a test document with mathematical content.
        
        Here's an equation: <MATH>a squared plus b squared equals c squared</MATH>.
        
        And another one: <MATH>x equals 5</MATH>.
        """
        result = generate_audio(text, mock_args)
        
        # Verify the result
        self.assertEqual(result, "output.mp3")
        
        # Verify that the TTS engine was called with different settings for math content
        calls = mock_engine.generate_audio.call_args_list
        
        # At least one call should use regular settings and one should use math settings
        regular_settings_used = False
        math_settings_used = False
        
        for call in calls:
            args, kwargs = call
            if "settings" in kwargs:
                if kwargs["settings"] == mock_math_settings:
                    math_settings_used = True
                elif kwargs["settings"] != mock_math_settings:
                    regular_settings_used = True
        
        self.assertTrue(regular_settings_used, "Regular TTS settings were not used")
        self.assertTrue(math_settings_used, "Math-specific TTS settings were not used")


if __name__ == "__main__":
    unittest.main()