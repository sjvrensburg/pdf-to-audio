"""
Base classes for the multi-pass refinement system.

This module provides the base classes for implementing refinement passes
and configuring the refinement pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class RefinementPass(ABC):
    """
    Base class for all refinement passes.
    
    A refinement pass takes text content, processes it according to specific rules,
    and returns the refined content.
    """
    
    def __init__(self, intensity: float = 0.5, enabled: bool = True):
        """
        Initialize the refinement pass.
        
        Args:
            intensity: The intensity of the refinement (0.0-1.0).
                Higher values result in more aggressive refinement.
            enabled: Whether this pass is enabled.
        """
        self.intensity = max(0.0, min(1.0, intensity))  # Clamp to [0.0, 1.0]
        self.enabled = enabled
        logger.info(f"Initialized {self.__class__.__name__} with intensity={intensity}, enabled={enabled}")
    
    @abstractmethod
    def refine(self, text: str, client: Any, config: Dict[str, Any]) -> str:
        """
        Refine the text content.
        
        Args:
            text: The text content to refine.
            client: The LLM client to use for refinement.
            config: Configuration options for the refinement.
            
        Returns:
            The refined text content.
        """
        pass
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for this refinement pass.
        
        Returns:
            The system prompt as a string.
        """
        return ""
    
    def _get_user_prompt(self, text: str) -> str:
        """
        Get the user prompt for this refinement pass.
        
        Args:
            text: The text content to refine.
            
        Returns:
            The user prompt as a string.
        """
        return text


class RefinementConfig:
    """
    Configuration for the refinement pipeline.
    """
    
    def __init__(
        self,
        enable_math_refinement: bool = True,
        enable_structure_citation_optimization: bool = True,
        enable_language_style_refinement: bool = True,
        enable_audio_specific_optimization: bool = True,
        math_refinement_intensity: float = 0.5,
        structure_citation_intensity: float = 0.5,
        language_style_intensity: float = 0.5,
        audio_specific_intensity: float = 0.5,
        target_audience: str = "academic",
        fallback_on_error: bool = True
    ):
        """
        Initialize the refinement configuration.
        
        Args:
            enable_math_refinement: Whether to enable the math content refinement pass.
            enable_structure_citation_optimization: Whether to enable the structure and citation optimization pass.
            enable_language_style_refinement: Whether to enable the language and style refinement pass.
            enable_audio_specific_optimization: Whether to enable the audio-specific optimization pass.
            math_refinement_intensity: The intensity of the math content refinement (0.0-1.0).
            structure_citation_intensity: The intensity of the structure and citation optimization (0.0-1.0).
            language_style_intensity: The intensity of the language and style refinement (0.0-1.0).
            audio_specific_intensity: The intensity of the audio-specific optimization (0.0-1.0).
            target_audience: The target audience for the refinement ("academic" or "general").
            fallback_on_error: Whether to fall back to the original content if refinement fails.
        """
        self.enable_math_refinement = enable_math_refinement
        self.enable_structure_citation_optimization = enable_structure_citation_optimization
        self.enable_language_style_refinement = enable_language_style_refinement
        self.enable_audio_specific_optimization = enable_audio_specific_optimization
        
        self.math_refinement_intensity = math_refinement_intensity
        self.structure_citation_intensity = structure_citation_intensity
        self.language_style_intensity = language_style_intensity
        self.audio_specific_intensity = audio_specific_intensity
        
        self.target_audience = target_audience
        self.fallback_on_error = fallback_on_error
        
        logger.info(f"Initialized RefinementConfig with target_audience={target_audience}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RefinementConfig':
        """
        Create a RefinementConfig from a dictionary.
        
        Args:
            config_dict: A dictionary containing configuration values.
            
        Returns:
            A RefinementConfig instance.
        """
        refinement_config = config_dict.get("refinement", {})
        
        return cls(
            enable_math_refinement=refinement_config.get("enable_math_refinement", True),
            enable_structure_citation_optimization=refinement_config.get("enable_structure_citation_optimization", True),
            enable_language_style_refinement=refinement_config.get("enable_language_style_refinement", True),
            enable_audio_specific_optimization=refinement_config.get("enable_audio_specific_optimization", True),
            math_refinement_intensity=refinement_config.get("math_refinement_intensity", 0.5),
            structure_citation_intensity=refinement_config.get("structure_citation_intensity", 0.5),
            language_style_intensity=refinement_config.get("language_style_intensity", 0.5),
            audio_specific_intensity=refinement_config.get("audio_specific_intensity", 0.5),
            target_audience=refinement_config.get("target_audience", "academic"),
            fallback_on_error=refinement_config.get("fallback_on_error", True)
        )