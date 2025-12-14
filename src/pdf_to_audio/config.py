"""
Configuration module for PDF to audio conversion.

This module provides functionality to load and parse configuration files,
and manage settings with proper precedence (CLI > Config File > Defaults).
"""

import os
import logging
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    # LLM settings (now provider-agnostic)
    "llm": {
        "api_key": None,  # Will use environment variable if None
        "temperature": 0.2,
        # Core text transformation
        "transform_provider": "mistral",
        "transform_model": "mistral-small-latest",
        # Math expression handling
        "math_provider": "mistral",
        "math_model": "mistral-small-latest",
        # Citations and references
        "citations_provider": "mistral",
        "citations_model": "mistral-small-latest",
        # Language and style refinement
        "language_provider": "mistral",
        "language_model": "mistral-small-latest",
        # Max tokens for LLM responses
        "max_tokens": 4000,
    },

    # Image model settings (still Mistral-specific for now)
    "image": {
        "image_model": "pixtral-12b-latest",
    },

    # TTS settings
    "tts": {
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "math_exaggeration": None,  # Will be calculated as 0.75 * exaggeration if None
        "math_cfg_weight": None,    # Will be calculated as 0.75 * cfg_weight if None
        "math_tts_scale": 0.75,     # Default scaling factor for math TTS settings
        "audio_format": "wav",
        "chunk_strategy": "smart",
        "global_normalization": False,
        "audio_prompt_path": None,  # Path to reference audio file for voice cloning
    },

    # General settings
    "general": {
        "temp_dir": None,  # Will use system temp dir if None
        "pages_per_chunk": 1,
        "include_images": False,
        "overwrite": False,
        "verbose": False,
        "force_cpu": False,
        # Pipeline stage enable/disable
        "enable_math_refinement": True,
        "enable_citations_refinement": True,
        "enable_language_refinement": True,
    }
}


def load_config(config_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_file_path: Path to the configuration file.
        
    Returns:
        A dictionary containing the configuration.
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_file_path and os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                
            if user_config:
                # Update the configuration with user-provided values
                _update_nested_dict(config, user_config)
                logger.info(f"Loaded configuration from {config_file_path}")
            else:
                logger.warning(f"Configuration file {config_file_path} is empty or invalid")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file_path}: {e}")
    elif config_file_path:
        logger.warning(f"Configuration file {config_file_path} not found")
    
    return config


def _update_nested_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a nested dictionary with values from another dictionary.
    
    Args:
        d: The dictionary to update.
        u: The dictionary with new values.
        
    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_nested_dict(d[k], v)
        else:
            d[k] = v
    return d


def merge_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Merge configuration with command-line arguments.
    Command-line arguments take precedence over configuration file values.

    Args:
        config: The configuration dictionary.
        args: The command-line arguments.

    Returns:
        The merged configuration dictionary.
    """
    # Create a copy of the config to avoid modifying the original
    merged_config = {
        "llm": config.get("llm", DEFAULT_CONFIG["llm"]).copy(),
        "image": config.get("image", DEFAULT_CONFIG["image"]).copy(),
        "tts": config.get("tts", DEFAULT_CONFIG["tts"]).copy(),
        "general": config.get("general", DEFAULT_CONFIG["general"]).copy(),
    }

    # Update LLM settings (provider and model for each stage)
    if hasattr(args, 'transform_provider') and args.transform_provider is not None:
        merged_config["llm"]["transform_provider"] = args.transform_provider
    if hasattr(args, 'transform_model') and args.transform_model is not None:
        merged_config["llm"]["transform_model"] = args.transform_model
    if hasattr(args, 'math_provider') and args.math_provider is not None:
        merged_config["llm"]["math_provider"] = args.math_provider
    if hasattr(args, 'math_model') and args.math_model is not None:
        merged_config["llm"]["math_model"] = args.math_model
    if hasattr(args, 'citations_provider') and args.citations_provider is not None:
        merged_config["llm"]["citations_provider"] = args.citations_provider
    if hasattr(args, 'citations_model') and args.citations_model is not None:
        merged_config["llm"]["citations_model"] = args.citations_model
    if hasattr(args, 'language_provider') and args.language_provider is not None:
        merged_config["llm"]["language_provider"] = args.language_provider
    if hasattr(args, 'language_model') and args.language_model is not None:
        merged_config["llm"]["language_model"] = args.language_model

    # Update image model settings
    if hasattr(args, 'image_model') and args.image_model is not None:
        merged_config["image"]["image_model"] = args.image_model

    # Update TTS settings
    if hasattr(args, 'exaggeration') and args.exaggeration is not None:
        merged_config["tts"]["exaggeration"] = args.exaggeration
    if hasattr(args, 'cfg_weight') and args.cfg_weight is not None:
        merged_config["tts"]["cfg_weight"] = args.cfg_weight
    if hasattr(args, 'math_exaggeration') and args.math_exaggeration is not None:
        merged_config["tts"]["math_exaggeration"] = args.math_exaggeration
    if hasattr(args, 'math_cfg_weight') and args.math_cfg_weight is not None:
        merged_config["tts"]["math_cfg_weight"] = args.math_cfg_weight
    if hasattr(args, 'math_tts_scale') and args.math_tts_scale is not None:
        merged_config["tts"]["math_tts_scale"] = args.math_tts_scale
    if hasattr(args, 'audio_format') and args.audio_format is not None:
        merged_config["tts"]["audio_format"] = args.audio_format
    if hasattr(args, 'chunk_strategy') and args.chunk_strategy is not None:
        merged_config["tts"]["chunk_strategy"] = args.chunk_strategy
    if hasattr(args, 'global_normalization'):
        merged_config["tts"]["global_normalization"] = args.global_normalization
    if hasattr(args, 'voice_clone') and args.voice_clone is not None:
        merged_config["tts"]["audio_prompt_path"] = args.voice_clone

    # Update general settings
    if hasattr(args, 'pages_per_chunk') and args.pages_per_chunk is not None:
        merged_config["general"]["pages_per_chunk"] = args.pages_per_chunk
    if hasattr(args, 'include_images') and args.include_images is not None:
        merged_config["general"]["include_images"] = args.include_images
    if hasattr(args, 'overwrite') and args.overwrite is not None:
        merged_config["general"]["overwrite"] = args.overwrite
    if hasattr(args, 'verbose') and args.verbose is not None:
        merged_config["general"]["verbose"] = args.verbose
    if hasattr(args, 'force_cpu') and args.force_cpu is not None:
        merged_config["general"]["force_cpu"] = args.force_cpu
    if hasattr(args, 'temp_dir') and args.temp_dir is not None:
        merged_config["general"]["temp_dir"] = args.temp_dir

    # Update LLM stage enable/disable flags
    if hasattr(args, 'enable_math_refinement'):
        merged_config["general"]["enable_math_refinement"] = args.enable_math_refinement
    if hasattr(args, 'enable_citations_refinement'):
        merged_config["general"]["enable_citations_refinement"] = args.enable_citations_refinement
    if hasattr(args, 'enable_language_refinement'):
        merged_config["general"]["enable_language_refinement"] = args.enable_language_refinement

    return merged_config


def get_math_tts_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate math-specific TTS settings based on the configuration.
    
    Args:
        config: The configuration dictionary.
        
    Returns:
        A dictionary containing math-specific TTS settings.
    """
    tts_config = config["tts"]
    
    # Get base TTS settings
    exaggeration = tts_config["exaggeration"]
    cfg_weight = tts_config["cfg_weight"]
    math_tts_scale = tts_config["math_tts_scale"]
    
    # Calculate math-specific settings if not explicitly provided
    math_exaggeration = tts_config["math_exaggeration"]
    if math_exaggeration is None:
        math_exaggeration = exaggeration * math_tts_scale
        
    math_cfg_weight = tts_config["math_cfg_weight"]
    if math_cfg_weight is None:
        math_cfg_weight = cfg_weight * math_tts_scale
    
    return {
        "exaggeration": math_exaggeration,
        "cfg_weight": math_cfg_weight
    }