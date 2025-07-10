"""
Refinement pipeline for the multi-pass refinement system.

This module provides the main pipeline for applying multiple refinement passes
to text content in sequence.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from .base import RefinementPass, RefinementConfig
from .passes import (
    MathContentRefinementPass,
    StructureCitationOptimizationPass,
    LanguageStyleRefinementPass,
    AudioSpecificOptimizationPass
)

logger = logging.getLogger(__name__)


class RefinementPipeline:
    """
    Pipeline for applying multiple refinement passes to text content.
    
    The pipeline applies each enabled refinement pass in sequence, with
    configurable options for each pass.
    """
    
    def __init__(self, config: Optional[RefinementConfig] = None):
        """
        Initialize the refinement pipeline.
        
        Args:
            config: Configuration for the refinement pipeline.
                If None, default configuration is used.
        """
        self.config = config or RefinementConfig()
        
        # Initialize refinement passes
        self.passes = [
            MathContentRefinementPass(
                intensity=self.config.math_refinement_intensity,
                enabled=self.config.enable_math_refinement
            ),
            StructureCitationOptimizationPass(
                intensity=self.config.structure_citation_intensity,
                enabled=self.config.enable_structure_citation_optimization
            ),
            LanguageStyleRefinementPass(
                intensity=self.config.language_style_intensity,
                enabled=self.config.enable_language_style_refinement
            ),
            AudioSpecificOptimizationPass(
                intensity=self.config.audio_specific_intensity,
                enabled=self.config.enable_audio_specific_optimization
            )
        ]
        
        logger.info(f"Initialized RefinementPipeline with {len(self.passes)} passes")
    
    def refine(self, text: str, client: Any, config: Dict[str, Any]) -> str:
        """
        Apply all enabled refinement passes to the text content.
        
        Args:
            text: The text content to refine.
            client: The LLM client to use for refinement.
            config: Configuration options for the refinement.
            
        Returns:
            The refined text content.
        """
        if not text:
            logger.warning("Empty text provided to refinement pipeline")
            return text
        
        logger.info("Starting refinement pipeline")
        refined_text = text
        
        for i, refinement_pass in enumerate(self.passes):
            if not refinement_pass.enabled:
                logger.info(f"Skipping disabled pass: {refinement_pass.__class__.__name__}")
                continue
            
            logger.info(f"Applying refinement pass {i+1}/{len(self.passes)}: {refinement_pass.__class__.__name__}")
            start_time = time.time()
            
            try:
                refined_text = refinement_pass.refine(refined_text, client, config)
                elapsed_time = time.time() - start_time
                logger.info(f"Completed {refinement_pass.__class__.__name__} in {elapsed_time:.2f} seconds")
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in {refinement_pass.__class__.__name__}: {e}")
                if self.config.fallback_on_error:
                    logger.info(f"Skipping {refinement_pass.__class__.__name__} due to error")
                    continue
                else:
                    raise
        
        logger.info("Refinement pipeline completed")
        return refined_text