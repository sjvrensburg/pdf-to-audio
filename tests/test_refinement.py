"""
Tests for the multi-pass refinement system.
"""

import unittest
from unittest.mock import MagicMock, patch

from pdf_to_audio.refinement.base import RefinementConfig, RefinementPass
from pdf_to_audio.refinement.pipeline import RefinementPipeline
from pdf_to_audio.refinement.passes import (
    MathContentRefinementPass,
    StructureCitationOptimizationPass,
    LanguageStyleRefinementPass,
    AudioSpecificOptimizationPass
)


class TestRefinementConfig(unittest.TestCase):
    """Tests for the RefinementConfig class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = RefinementConfig()
        self.assertTrue(config.enable_math_refinement)
        self.assertTrue(config.enable_structure_citation_optimization)
        self.assertTrue(config.enable_language_style_refinement)
        self.assertTrue(config.enable_audio_specific_optimization)
        self.assertEqual(config.math_refinement_intensity, 0.5)
        self.assertEqual(config.structure_citation_intensity, 0.5)
        self.assertEqual(config.language_style_intensity, 0.5)
        self.assertEqual(config.audio_specific_intensity, 0.5)
        self.assertEqual(config.target_audience, "academic")
        self.assertTrue(config.fallback_on_error)
    
    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        config = RefinementConfig(
            enable_math_refinement=False,
            enable_structure_citation_optimization=False,
            enable_language_style_refinement=False,
            enable_audio_specific_optimization=False,
            math_refinement_intensity=0.7,
            structure_citation_intensity=0.8,
            language_style_intensity=0.9,
            audio_specific_intensity=1.0,
            target_audience="general",
            fallback_on_error=False
        )
        self.assertFalse(config.enable_math_refinement)
        self.assertFalse(config.enable_structure_citation_optimization)
        self.assertFalse(config.enable_language_style_refinement)
        self.assertFalse(config.enable_audio_specific_optimization)
        self.assertEqual(config.math_refinement_intensity, 0.7)
        self.assertEqual(config.structure_citation_intensity, 0.8)
        self.assertEqual(config.language_style_intensity, 0.9)
        self.assertEqual(config.audio_specific_intensity, 1.0)
        self.assertEqual(config.target_audience, "general")
        self.assertFalse(config.fallback_on_error)
    
    def test_from_dict(self):
        """Test creation from a dictionary."""
        config_dict = {
            "refinement": {
                "enable_math_refinement": False,
                "enable_structure_citation_optimization": False,
                "enable_language_style_refinement": False,
                "enable_audio_specific_optimization": False,
                "math_refinement_intensity": 0.7,
                "structure_citation_intensity": 0.8,
                "language_style_intensity": 0.9,
                "audio_specific_intensity": 1.0,
                "target_audience": "general",
                "fallback_on_error": False
            }
        }
        config = RefinementConfig.from_dict(config_dict)
        self.assertFalse(config.enable_math_refinement)
        self.assertFalse(config.enable_structure_citation_optimization)
        self.assertFalse(config.enable_language_style_refinement)
        self.assertFalse(config.enable_audio_specific_optimization)
        self.assertEqual(config.math_refinement_intensity, 0.7)
        self.assertEqual(config.structure_citation_intensity, 0.8)
        self.assertEqual(config.language_style_intensity, 0.9)
        self.assertEqual(config.audio_specific_intensity, 1.0)
        self.assertEqual(config.target_audience, "general")
        self.assertFalse(config.fallback_on_error)


class TestRefinementPipeline(unittest.TestCase):
    """Tests for the RefinementPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_config = {"mistral": {"text_model": "test-model"}}
        self.test_text = "This is a test text with <MATH>x^2</MATH> math content."
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        pipeline = RefinementPipeline()
        self.assertEqual(len(pipeline.passes), 4)
        self.assertIsInstance(pipeline.passes[0], MathContentRefinementPass)
        self.assertIsInstance(pipeline.passes[1], StructureCitationOptimizationPass)
        self.assertIsInstance(pipeline.passes[2], LanguageStyleRefinementPass)
        self.assertIsInstance(pipeline.passes[3], AudioSpecificOptimizationPass)
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = RefinementConfig(
            enable_math_refinement=False,
            enable_structure_citation_optimization=False,
            enable_language_style_refinement=True,
            enable_audio_specific_optimization=True
        )
        pipeline = RefinementPipeline(config=config)
        self.assertEqual(len(pipeline.passes), 4)
        self.assertFalse(pipeline.passes[0].enabled)
        self.assertFalse(pipeline.passes[1].enabled)
        self.assertTrue(pipeline.passes[2].enabled)
        self.assertTrue(pipeline.passes[3].enabled)
    
    @patch('pdf_to_audio.refinement.passes.MathContentRefinementPass.refine')
    @patch('pdf_to_audio.refinement.passes.StructureCitationOptimizationPass.refine')
    @patch('pdf_to_audio.refinement.passes.LanguageStyleRefinementPass.refine')
    @patch('pdf_to_audio.refinement.passes.AudioSpecificOptimizationPass.refine')
    def test_refine_all_passes_enabled(self, mock_audio, mock_language, mock_structure, mock_math):
        """Test refinement with all passes enabled."""
        # Set up mock return values
        mock_math.return_value = "Math refined"
        mock_structure.return_value = "Structure refined"
        mock_language.return_value = "Language refined"
        mock_audio.return_value = "Audio refined"
        
        # Create pipeline and run refinement
        pipeline = RefinementPipeline()
        result = pipeline.refine(self.test_text, self.mock_client, self.mock_config)
        
        # Check that all passes were called
        mock_math.assert_called_once_with(self.test_text, self.mock_client, self.mock_config)
        mock_structure.assert_called_once_with("Math refined", self.mock_client, self.mock_config)
        mock_language.assert_called_once_with("Structure refined", self.mock_client, self.mock_config)
        mock_audio.assert_called_once_with("Language refined", self.mock_client, self.mock_config)
        
        # Check final result
        self.assertEqual(result, "Audio refined")
    
    @patch('pdf_to_audio.refinement.passes.MathContentRefinementPass.refine')
    @patch('pdf_to_audio.refinement.passes.StructureCitationOptimizationPass.refine')
    @patch('pdf_to_audio.refinement.passes.LanguageStyleRefinementPass.refine')
    @patch('pdf_to_audio.refinement.passes.AudioSpecificOptimizationPass.refine')
    def test_refine_some_passes_disabled(self, mock_audio, mock_language, mock_structure, mock_math):
        """Test refinement with some passes disabled."""
        # Set up mock return values
        mock_math.return_value = "Math refined"
        mock_structure.return_value = "Structure refined"
        mock_language.return_value = "Language refined"
        mock_audio.return_value = "Audio refined"
        
        # Create pipeline with some passes disabled
        config = RefinementConfig(
            enable_math_refinement=True,
            enable_structure_citation_optimization=False,
            enable_language_style_refinement=True,
            enable_audio_specific_optimization=False
        )
        pipeline = RefinementPipeline(config=config)
        result = pipeline.refine(self.test_text, self.mock_client, self.mock_config)
        
        # Check that only enabled passes were called
        mock_math.assert_called_once_with(self.test_text, self.mock_client, self.mock_config)
        mock_structure.assert_not_called()
        mock_language.assert_called_once_with("Math refined", self.mock_client, self.mock_config)
        mock_audio.assert_not_called()
        
        # Check final result
        self.assertEqual(result, "Language refined")
    
    @patch('pdf_to_audio.refinement.passes.MathContentRefinementPass.refine')
    def test_refine_with_error_and_fallback(self, mock_math):
        """Test refinement with error and fallback enabled."""
        # Set up mock to raise an exception
        mock_math.side_effect = Exception("Test error")
        
        # Create pipeline with fallback enabled
        config = RefinementConfig(fallback_on_error=True)
        pipeline = RefinementPipeline(config=config)
        result = pipeline.refine(self.test_text, self.mock_client, self.mock_config)
        
        # Check that the original text is returned
        self.assertEqual(result, self.test_text)
    
    @patch('pdf_to_audio.refinement.passes.MathContentRefinementPass.refine')
    def test_refine_with_error_and_no_fallback(self, mock_math):
        """Test refinement with error and fallback disabled."""
        # Set up mock to raise an exception
        mock_math.side_effect = Exception("Test error")
        
        # Create pipeline with fallback disabled
        config = RefinementConfig(fallback_on_error=False)
        pipeline = RefinementPipeline(config=config)
        
        # Check that the exception is propagated
        with self.assertRaises(Exception):
            pipeline.refine(self.test_text, self.mock_client, self.mock_config)


class TestRefinementPasses(unittest.TestCase):
    """Tests for the individual refinement passes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_config = {"mistral": {"text_model": "test-model"}}
        self.test_text = "This is a test text with <MATH>x^2</MATH> math content."
        
        # Mock API response
        self.mock_response = MagicMock()
        self.mock_response.choices = [MagicMock()]
        self.mock_response.choices[0].message.content = "Refined content"
    
    @patch('pdf_to_audio.refinement.passes.make_api_call')
    def test_math_content_refinement_pass(self, mock_api_call):
        """Test the math content refinement pass."""
        # Set up mock API call
        mock_api_call.return_value = self.mock_response
        
        # Create and run the pass
        refinement_pass = MathContentRefinementPass()
        result = refinement_pass.refine(self.test_text, self.mock_client, self.mock_config)
        
        # Check that the API was called with the correct parameters
        mock_api_call.assert_called_once()
        args = mock_api_call.call_args[0]
        self.assertEqual(args[0], self.mock_client)
        self.assertEqual(args[1], "test-model")
        self.assertEqual(args[2][0]["role"], "system")
        self.assertEqual(args[2][1]["role"], "user")
        
        # Check that the result is correct
        self.assertEqual(result, "Refined content")
    
    @patch('pdf_to_audio.refinement.passes.make_api_call')
    def test_structure_citation_optimization_pass(self, mock_api_call):
        """Test the structure and citation optimization pass."""
        # Set up mock API call
        mock_api_call.return_value = self.mock_response
        
        # Create and run the pass
        refinement_pass = StructureCitationOptimizationPass()
        result = refinement_pass.refine(self.test_text, self.mock_client, self.mock_config)
        
        # Check that the API was called with the correct parameters
        mock_api_call.assert_called_once()
        args = mock_api_call.call_args[0]
        self.assertEqual(args[0], self.mock_client)
        self.assertEqual(args[1], "test-model")
        self.assertEqual(args[2][0]["role"], "system")
        self.assertEqual(args[2][1]["role"], "user")
        
        # Check that the result is correct
        self.assertEqual(result, "Refined content")
    
    @patch('pdf_to_audio.refinement.passes.make_api_call')
    def test_language_style_refinement_pass(self, mock_api_call):
        """Test the language and style refinement pass."""
        # Set up mock API call
        mock_api_call.return_value = self.mock_response
        
        # Create and run the pass
        refinement_pass = LanguageStyleRefinementPass()
        result = refinement_pass.refine(self.test_text, self.mock_client, self.mock_config)
        
        # Check that the API was called with the correct parameters
        mock_api_call.assert_called_once()
        args = mock_api_call.call_args[0]
        self.assertEqual(args[0], self.mock_client)
        self.assertEqual(args[1], "test-model")
        self.assertEqual(args[2][0]["role"], "system")
        self.assertEqual(args[2][1]["role"], "user")
        
        # Check that the result is correct
        self.assertEqual(result, "Refined content")
    
    @patch('pdf_to_audio.refinement.passes.make_api_call')
    def test_audio_specific_optimization_pass(self, mock_api_call):
        """Test the audio-specific optimization pass."""
        # Set up mock API call
        mock_api_call.return_value = self.mock_response
        
        # Create and run the pass
        refinement_pass = AudioSpecificOptimizationPass()
        result = refinement_pass.refine(self.test_text, self.mock_client, self.mock_config)
        
        # Check that the API was called with the correct parameters
        mock_api_call.assert_called_once()
        args = mock_api_call.call_args[0]
        self.assertEqual(args[0], self.mock_client)
        self.assertEqual(args[1], "test-model")
        self.assertEqual(args[2][0]["role"], "system")
        self.assertEqual(args[2][1]["role"], "user")
        
        # Check that the result is correct
        self.assertEqual(result, "Refined content")


if __name__ == '__main__':
    unittest.main()