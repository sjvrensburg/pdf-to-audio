"""
Multi-pass LLM refinement system for PDF to audio conversion.

This module provides a pipeline for refining LLM-processed content to optimize it
for audio consumption, specifically addressing challenges with mathematical content,
technical abbreviations, citation clusters, complex tables, and dense academic writing style.
"""

from .pipeline import RefinementPipeline
from .passes import (
    MathContentRefinementPass,
    StructureCitationOptimizationPass,
    LanguageStyleRefinementPass,
    AudioSpecificOptimizationPass
)

__all__ = [
    'RefinementPipeline',
    'MathContentRefinementPass',
    'StructureCitationOptimizationPass',
    'LanguageStyleRefinementPass',
    'AudioSpecificOptimizationPass'
]