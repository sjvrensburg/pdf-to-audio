"""
Refinement passes for the multi-pass refinement system.

This module provides specialized refinement passes for different aspects of content optimization:
- Mathematical content refinement
- Structure and citation optimization
- Language and style refinement
- Audio-specific optimization
"""

import logging
import re
from typing import Dict, Any, List, Optional

from ..api import make_api_call
from .base import RefinementPass

logger = logging.getLogger(__name__)


class MathContentRefinementPass(RefinementPass):
    """
    Refinement pass for mathematical content.
    
    This pass converts mathematical notation to speech-friendly descriptions
    while preserving <MATH></MATH> tags.
    """
    
    def refine(self, text: str, client: Any, config: Dict[str, Any]) -> str:
        """
        Refine mathematical content in the text.
        
        Args:
            text: The text content to refine.
            client: The LLM client to use for refinement.
            config: Configuration options for the refinement.
            
        Returns:
            The refined text content.
        """
        if not self.enabled:
            logger.info("Math content refinement pass is disabled, skipping")
            return text
        
        logger.info("Running math content refinement pass")
        
        try:
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._get_user_prompt(text)}
            ]
            
            # Make the API call
            model = config["mistral"]["text_model"]
            response = make_api_call(client, model, messages)
            
            if response and response.choices:
                refined_text = response.choices[0].message.content
                logger.info("Math content refinement completed successfully")
                return refined_text
            else:
                logger.warning("Empty response from LLM during math content refinement")
                return text
                
        except Exception as e:
            logger.error(f"Error during math content refinement: {e}")
            if config.get("refinement", {}).get("fallback_on_error", True):
                logger.info("Falling back to original content")
                return text
            else:
                raise
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for math content refinement.
        
        Returns:
            The system prompt as a string.
        """
        return f"""
You are an expert in converting mathematical content into speech-friendly format for audio consumption. Your task is to refine mathematical expressions and equations to make them clearer and more understandable when read aloud, while maintaining scientific accuracy.

**CRITICAL REQUIREMENT: You MUST preserve all <MATH></MATH> tags exactly as they appear in the input text.**

Guidelines for mathematical content refinement (intensity: {self.intensity}):

1. Convert complex mathematical notation into clear, spoken language descriptions.
2. Ensure all mathematical expressions are unambiguous when heard rather than read.
3. Maintain the exact meaning and precision of the original mathematical content.
4. For equations with multiple components, break them down into logical parts.
5. Use consistent terminology for mathematical operations and symbols.
6. Ensure proper verbalization of subscripts, superscripts, and special symbols.
7. DO NOT remove or modify any <MATH></MATH> tags - these are critical for audio processing.
8. DO NOT add explanations or interpretations beyond what's in the original content.

The intensity parameter ({self.intensity}) controls how aggressively you should refine the content:
- Lower intensity (0.0-0.3): Make minimal changes, focusing only on the most complex expressions.
- Medium intensity (0.3-0.7): Make moderate changes to improve clarity while preserving style.
- Higher intensity (0.7-1.0): Make comprehensive changes to optimize all mathematical content.

Your output should be the complete text with refined mathematical content, keeping all <MATH></MATH> tags intact.
"""
    
    def _get_user_prompt(self, text: str) -> str:
        """
        Get the user prompt for math content refinement.
        
        Args:
            text: The text content to refine.
            
        Returns:
            The user prompt as a string.
        """
        return f"""
Please refine the mathematical content in the following text to make it more suitable for audio consumption. Remember to preserve all <MATH></MATH> tags exactly as they appear.

TEXT TO REFINE:
{text}

IMPORTANT REMINDERS:
1. DO NOT remove or modify any <MATH></MATH> tags.
2. Maintain scientific accuracy while improving clarity for audio.
3. Return the complete text with your refinements.
"""


class StructureCitationOptimizationPass(RefinementPass):
    """
    Refinement pass for structure and citation optimization.
    
    This pass transforms citation clusters into natural language references
    and converts complex tables into concise narrative descriptions.
    """
    
    def refine(self, text: str, client: Any, config: Dict[str, Any]) -> str:
        """
        Refine structure and citations in the text.
        
        Args:
            text: The text content to refine.
            client: The LLM client to use for refinement.
            config: Configuration options for the refinement.
            
        Returns:
            The refined text content.
        """
        if not self.enabled:
            logger.info("Structure and citation optimization pass is disabled, skipping")
            return text
        
        logger.info("Running structure and citation optimization pass")
        
        try:
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._get_user_prompt(text)}
            ]
            
            # Make the API call
            model = config["mistral"]["text_model"]
            response = make_api_call(client, model, messages)
            
            if response and response.choices:
                refined_text = response.choices[0].message.content
                logger.info("Structure and citation optimization completed successfully")
                return refined_text
            else:
                logger.warning("Empty response from LLM during structure and citation optimization")
                return text
                
        except Exception as e:
            logger.error(f"Error during structure and citation optimization: {e}")
            if config.get("refinement", {}).get("fallback_on_error", True):
                logger.info("Falling back to original content")
                return text
            else:
                raise
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for structure and citation optimization.
        
        Returns:
            The system prompt as a string.
        """
        return f"""
You are an expert in optimizing academic content for audio consumption. Your task is to refine the structure, citations, and tables in the text to make them more suitable for listening rather than reading.

**CRITICAL REQUIREMENT: You MUST preserve all <MATH></MATH> tags exactly as they appear in the input text.**

Guidelines for structure and citation optimization (intensity: {self.intensity}):

1. Citation Transformation:
   - Convert citation clusters (e.g., [1,2,3,4]) into natural language references.
   - Example: "[1,2,3]" → "as shown in references one, two, and three"
   - For author-year citations: "(Smith et al., 2020)" → "as Smith and colleagues showed in 2020"

2. Table Optimization:
   - Transform complex tables into concise narrative descriptions.
   - Focus on key insights and trends rather than individual data points.
   - Maintain the table's title and context.
   - Structure the description to flow naturally in spoken form.

3. Section Structure:
   - Ensure clear transitions between sections.
   - Add brief navigational cues where helpful.
   - Maintain the hierarchical structure of the document.

4. Figure References:
   - Ensure figure references are clear and descriptive.
   - Connect figure descriptions naturally to the surrounding text.

5. DO NOT remove or modify any <MATH></MATH> tags - these are critical for audio processing.
6. DO NOT add explanations or interpretations beyond what's in the original content.

The intensity parameter ({self.intensity}) controls how aggressively you should refine the content:
- Lower intensity (0.0-0.3): Make minimal changes, focusing only on the most complex citations and tables.
- Medium intensity (0.3-0.7): Make moderate changes to improve clarity while preserving style.
- Higher intensity (0.7-1.0): Make comprehensive changes to optimize all citations and tables.

Your output should be the complete text with refined structure and citations, keeping all <MATH></MATH> tags intact.
"""
    
    def _get_user_prompt(self, text: str) -> str:
        """
        Get the user prompt for structure and citation optimization.
        
        Args:
            text: The text content to refine.
            
        Returns:
            The user prompt as a string.
        """
        return f"""
Please optimize the structure, citations, and tables in the following text to make it more suitable for audio consumption. Remember to preserve all <MATH></MATH> tags exactly as they appear.

TEXT TO REFINE:
{text}

IMPORTANT REMINDERS:
1. DO NOT remove or modify any <MATH></MATH> tags.
2. Transform citation clusters into natural language references.
3. Convert complex tables into concise narrative descriptions.
4. Return the complete text with your refinements.
"""


class LanguageStyleRefinementPass(RefinementPass):
    """
    Refinement pass for language and style.
    
    This pass expands technical abbreviations, simplifies complex academic sentences,
    and adds conversational elements for improved audio flow.
    """
    
    def refine(self, text: str, client: Any, config: Dict[str, Any]) -> str:
        """
        Refine language and style in the text.
        
        Args:
            text: The text content to refine.
            client: The LLM client to use for refinement.
            config: Configuration options for the refinement.
            
        Returns:
            The refined text content.
        """
        if not self.enabled:
            logger.info("Language and style refinement pass is disabled, skipping")
            return text
        
        logger.info("Running language and style refinement pass")
        
        try:
            # Get target audience from config
            target_audience = config.get("refinement", {}).get("target_audience", "academic")
            
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": self._get_system_prompt(target_audience)},
                {"role": "user", "content": self._get_user_prompt(text)}
            ]
            
            # Make the API call
            model = config["mistral"]["text_model"]
            response = make_api_call(client, model, messages)
            
            if response and response.choices:
                refined_text = response.choices[0].message.content
                logger.info("Language and style refinement completed successfully")
                return refined_text
            else:
                logger.warning("Empty response from LLM during language and style refinement")
                return text
                
        except Exception as e:
            logger.error(f"Error during language and style refinement: {e}")
            if config.get("refinement", {}).get("fallback_on_error", True):
                logger.info("Falling back to original content")
                return text
            else:
                raise
    
    def _get_system_prompt(self, target_audience: str = "academic") -> str:
        """
        Get the system prompt for language and style refinement.
        
        Args:
            target_audience: The target audience for the refinement ("academic" or "general").
            
        Returns:
            The system prompt as a string.
        """
        audience_specific_guidance = ""
        if target_audience.lower() == "academic":
            audience_specific_guidance = """
For academic audience:
- Maintain technical precision and academic terminology.
- Preserve field-specific jargon but ensure it's introduced properly.
- Keep a formal tone while improving flow and clarity.
- Focus on making complex sentences more digestible without oversimplification.
"""
        else:  # general audience
            audience_specific_guidance = """
For general audience:
- Replace specialized jargon with more accessible terminology.
- Add brief explanations for technical concepts where necessary.
- Use more conversational language and simpler sentence structures.
- Focus on making the content accessible to non-experts while maintaining accuracy.
"""
        
        return f"""
You are an expert in refining academic language for audio consumption. Your task is to improve the language and style of the text to make it more suitable for listening rather than reading, targeting a {target_audience} audience.

**CRITICAL REQUIREMENT: You MUST preserve all <MATH></MATH> tags exactly as they appear in the input text.**

Guidelines for language and style refinement (intensity: {self.intensity}):

1. Abbreviation Expansion:
   - Expand technical abbreviations on first use.
   - Example: "SVM" → "Support Vector Machine (SVM)"
   - After first expansion, you may continue using the abbreviation.

2. Sentence Simplification:
   - Break down complex, nested sentences into simpler ones.
   - Convert passive voice to active voice where appropriate.
   - Maintain the original meaning and technical precision.

3. Flow Enhancement:
   - Add natural transitions between ideas.
   - Ensure logical progression of concepts.
   - Add conversational elements for improved audio flow.

4. Clarity Improvements:
   - Replace ambiguous pronouns with specific references.
   - Ensure antecedents are clear for all pronouns.
   - Clarify complex or dense descriptions.

5. DO NOT remove or modify any <MATH></MATH> tags - these are critical for audio processing.
6. DO NOT add explanations or interpretations beyond what's in the original content.

{audience_specific_guidance}

The intensity parameter ({self.intensity}) controls how aggressively you should refine the content:
- Lower intensity (0.0-0.3): Make minimal changes, focusing only on the most complex sentences.
- Medium intensity (0.3-0.7): Make moderate changes to improve clarity while preserving style.
- Higher intensity (0.7-1.0): Make comprehensive changes to optimize all language and style.

Your output should be the complete text with refined language and style, keeping all <MATH></MATH> tags intact.
"""
    
    def _get_user_prompt(self, text: str) -> str:
        """
        Get the user prompt for language and style refinement.
        
        Args:
            text: The text content to refine.
            
        Returns:
            The user prompt as a string.
        """
        return f"""
Please refine the language and style in the following text to make it more suitable for audio consumption. Remember to preserve all <MATH></MATH> tags exactly as they appear.

TEXT TO REFINE:
{text}

IMPORTANT REMINDERS:
1. DO NOT remove or modify any <MATH></MATH> tags.
2. Expand technical abbreviations on first use.
3. Simplify complex sentences while maintaining meaning.
4. Add conversational elements for improved audio flow.
5. Return the complete text with your refinements.
"""


class AudioSpecificOptimizationPass(RefinementPass):
    """
    Refinement pass for audio-specific optimization.
    
    This pass ensures compatibility with Chatterbox TTS, adds optional
    section markers/navigation aids, and performs final cleanup for
    audio-friendly formatting.
    """
    
    def refine(self, text: str, client: Any, config: Dict[str, Any]) -> str:
        """
        Refine the text for audio-specific optimization.
        
        Args:
            text: The text content to refine.
            client: The LLM client to use for refinement.
            config: Configuration options for the refinement.
            
        Returns:
            The refined text content.
        """
        if not self.enabled:
            logger.info("Audio-specific optimization pass is disabled, skipping")
            return text
        
        logger.info("Running audio-specific optimization pass")
        
        try:
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._get_user_prompt(text)}
            ]
            
            # Make the API call
            model = config["mistral"]["text_model"]
            response = make_api_call(client, model, messages)
            
            if response and response.choices:
                refined_text = response.choices[0].message.content
                logger.info("Audio-specific optimization completed successfully")
                return refined_text
            else:
                logger.warning("Empty response from LLM during audio-specific optimization")
                return text
                
        except Exception as e:
            logger.error(f"Error during audio-specific optimization: {e}")
            if config.get("refinement", {}).get("fallback_on_error", True):
                logger.info("Falling back to original content")
                return text
            else:
                raise
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for audio-specific optimization.
        
        Returns:
            The system prompt as a string.
        """
        return f"""
You are an expert in optimizing content for text-to-speech systems. Your task is to perform final refinements to make the text optimal for audio consumption using Chatterbox TTS.

**CRITICAL REQUIREMENT: You MUST preserve all <MATH></MATH> tags exactly as they appear in the input text.**

Guidelines for audio-specific optimization (intensity: {self.intensity}):

1. TTS Compatibility:
   - Ensure content is formatted optimally for Chatterbox TTS.
   - Avoid characters or constructs that might cause TTS issues.
   - DO NOT add manual pause markers - Chatterbox handles these automatically.

2. Navigation Aids:
   - Add clear section markers for easier audio navigation.
   - Ensure transitions between sections are explicit and clear.
   - For longer documents, add brief summaries at section beginnings if helpful.

3. Pronunciation Optimization:
   - Identify and adjust words that might be mispronounced by TTS.
   - Use phonetic spelling or alternative wording for problematic terms.
   - Ensure proper handling of acronyms, numbers, and special terms.

4. Final Cleanup:
   - Remove any remaining formatting artifacts.
   - Ensure consistent spacing and paragraph structure.
   - Check for and fix any remaining issues that might affect audio quality.

5. DO NOT remove or modify any <MATH></MATH> tags - these are critical for audio processing.
6. DO NOT add explanations or interpretations beyond what's in the original content.

The intensity parameter ({self.intensity}) controls how aggressively you should refine the content:
- Lower intensity (0.0-0.3): Make minimal changes, focusing only on critical TTS issues.
- Medium intensity (0.3-0.7): Make moderate changes to improve audio quality.
- Higher intensity (0.7-1.0): Make comprehensive changes to optimize all aspects for audio.

Your output should be the complete text optimized for audio, keeping all <MATH></MATH> tags intact.
"""
    
    def _get_user_prompt(self, text: str) -> str:
        """
        Get the user prompt for audio-specific optimization.
        
        Args:
            text: The text content to refine.
            
        Returns:
            The user prompt as a string.
        """
        return f"""
Please optimize the following text for audio consumption with Chatterbox TTS. Remember to preserve all <MATH></MATH> tags exactly as they appear.

TEXT TO REFINE:
{text}

IMPORTANT REMINDERS:
1. DO NOT remove or modify any <MATH></MATH> tags.
2. Ensure compatibility with Chatterbox TTS.
3. Add section markers/navigation aids as appropriate.
4. Perform final cleanup for audio-friendly formatting.
5. Return the complete text with your refinements.
"""