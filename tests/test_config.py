"""Tests for the configuration module."""

import os
import tempfile
import unittest
from unittest.mock import patch

import yaml

from pdf_to_audio.config import load_config, merge_with_args, get_math_tts_settings


class TestConfig(unittest.TestCase):
    """Test cases for the configuration module."""

    def test_load_config_default(self):
        """Test loading default configuration."""
        config = load_config(None)
        self.assertEqual(config["mistral"]["text_model"], "mistral-small-latest")
        self.assertEqual(config["tts"]["exaggeration"], 0.5)
        self.assertEqual(config["tts"]["math_tts_scale"], 0.75)
        self.assertEqual(config["general"]["pages_per_chunk"], 1)

    def test_load_config_from_file(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
            yaml.dump({
                "mistral": {"text_model": "mistral-large-latest"},
                "tts": {"exaggeration": 0.7, "math_exaggeration": 0.4},
                "general": {"pages_per_chunk": 3}
            }, temp_file)
            temp_file_path = temp_file.name

        try:
            # Load the config
            config = load_config(temp_file_path)
            
            # Check that values from the file were loaded
            self.assertEqual(config["mistral"]["text_model"], "mistral-large-latest")
            self.assertEqual(config["tts"]["exaggeration"], 0.7)
            self.assertEqual(config["tts"]["math_exaggeration"], 0.4)
            self.assertEqual(config["general"]["pages_per_chunk"], 3)
            
            # Check that default values for unspecified options are still present
            self.assertEqual(config["mistral"]["image_model"], "pixtral-12b-latest")
            self.assertEqual(config["tts"]["cfg_weight"], 0.5)
            self.assertEqual(config["general"]["include_images"], False)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    def test_merge_with_args(self):
        """Test merging configuration with command-line arguments."""
        # Create a mock args object
        class MockArgs:
            text_model = "mistral-medium-latest"
            exaggeration = 0.8
            math_tts_scale = 0.6
            verbose = True
            
        args = MockArgs()
        
        # Create a base config
        config = {
            "mistral": {"text_model": "mistral-small-latest", "image_model": "pixtral-12b-latest"},
            "tts": {"exaggeration": 0.5, "cfg_weight": 0.5, "math_tts_scale": 0.75},
            "general": {"pages_per_chunk": 1, "verbose": False}
        }
        
        # Merge with args
        merged = merge_with_args(config, args)
        
        # Check that args values override config values
        self.assertEqual(merged["mistral"]["text_model"], "mistral-medium-latest")
        self.assertEqual(merged["tts"]["exaggeration"], 0.8)
        self.assertEqual(merged["tts"]["math_tts_scale"], 0.6)
        self.assertEqual(merged["general"]["verbose"], True)
        
        # Check that unspecified values remain unchanged
        self.assertEqual(merged["mistral"]["image_model"], "pixtral-12b-latest")
        self.assertEqual(merged["tts"]["cfg_weight"], 0.5)
        self.assertEqual(merged["general"]["pages_per_chunk"], 1)

    def test_get_math_tts_settings(self):
        """Test getting math-specific TTS settings."""
        # Test with explicit math settings
        config = {
            "tts": {
                "exaggeration": 0.6,
                "cfg_weight": 0.4,
                "math_exaggeration": 0.3,
                "math_cfg_weight": 0.7,
                "math_tts_scale": 0.75
            }
        }
        
        math_settings = get_math_tts_settings(config)
        self.assertEqual(math_settings["exaggeration"], 0.3)
        self.assertEqual(math_settings["cfg_weight"], 0.7)
        
        # Test with calculated math settings
        config = {
            "tts": {
                "exaggeration": 0.6,
                "cfg_weight": 0.4,
                "math_exaggeration": None,
                "math_cfg_weight": None,
                "math_tts_scale": 0.75
            }
        }
        
        math_settings = get_math_tts_settings(config)
        self.assertEqual(math_settings["exaggeration"], 0.6 * 0.75)
        self.assertEqual(math_settings["cfg_weight"], 0.4 * 0.75)


if __name__ == "__main__":
    unittest.main()