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
        self.assertEqual(config["llm"]["transform_model"], "mistral-small-latest")
        self.assertEqual(config["tts"]["exaggeration"], 0.5)
        self.assertEqual(config["tts"]["math_tts_scale"], 0.75)
        self.assertEqual(config["general"]["pages_per_chunk"], 1)

    def test_load_config_from_file(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
            yaml.dump({
                "llm": {"transform_model": "mistral-large-latest"},
                "tts": {"exaggeration": 0.7, "math_exaggeration": 0.4},
                "general": {"pages_per_chunk": 3}
            }, temp_file)
            temp_file_path = temp_file.name

        try:
            # Load the config
            config = load_config(temp_file_path)

            # Check that values from the file were loaded
            self.assertEqual(config["llm"]["transform_model"], "mistral-large-latest")
            self.assertEqual(config["tts"]["exaggeration"], 0.7)
            self.assertEqual(config["tts"]["math_exaggeration"], 0.4)
            self.assertEqual(config["general"]["pages_per_chunk"], 3)

            # Check that default values for unspecified options are still present
            self.assertEqual(config["image"]["image_model"], "pixtral-12b-latest")
            self.assertEqual(config["tts"]["cfg_weight"], 0.5)
            self.assertEqual(config["general"]["include_images"], False)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    def test_merge_with_args(self):
        """Test merging configuration with command-line arguments."""
        # Create a mock args object
        class MockArgs:
            transform_model = "mistral-medium-latest"
            transform_provider = None
            math_model = None
            math_provider = None
            citations_model = None
            citations_provider = None
            language_model = None
            language_provider = None
            image_model = None
            exaggeration = 0.8
            cfg_weight = None
            math_exaggeration = None
            math_cfg_weight = None
            math_tts_scale = 0.6
            audio_format = None
            chunk_strategy = None
            global_normalization = False
            voice_clone = None
            pages_per_chunk = None
            include_images = None
            overwrite = None
            verbose = True
            force_cpu = None
            temp_dir = None
            enable_math_refinement = True
            enable_citations_refinement = True
            enable_language_refinement = True

        args = MockArgs()

        # Create a base config
        config = {
            "llm": {
                "transform_model": "mistral-small-latest",
                "transform_provider": "mistral",
                "temperature": 0.2,
                "max_tokens": 4000
            },
            "image": {"image_model": "pixtral-12b-latest"},
            "tts": {
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "math_tts_scale": 0.75,
                "math_exaggeration": None,
                "math_cfg_weight": None,
                "audio_format": "wav",
                "chunk_strategy": "smart",
                "global_normalization": False,
                "audio_prompt_path": None
            },
            "general": {
                "pages_per_chunk": 1,
                "verbose": False,
                "include_images": False,
                "overwrite": False,
                "force_cpu": False,
                "temp_dir": None,
                "enable_math_refinement": True,
                "enable_citations_refinement": True,
                "enable_language_refinement": True
            }
        }

        # Merge with args
        merged = merge_with_args(config, args)

        # Check that args values override config values
        self.assertEqual(merged["llm"]["transform_model"], "mistral-medium-latest")
        self.assertEqual(merged["tts"]["exaggeration"], 0.8)
        self.assertEqual(merged["tts"]["math_tts_scale"], 0.6)
        self.assertEqual(merged["general"]["verbose"], True)

        # Check that unspecified values remain unchanged
        self.assertEqual(merged["image"]["image_model"], "pixtral-12b-latest")
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