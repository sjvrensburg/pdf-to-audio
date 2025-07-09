"""
Chatterbox TTS integration module.

This module provides functionality to generate audio from text using the Chatterbox TTS engine.
"""

import os
import logging
from typing import Optional, Union, Dict, Any, Tuple

import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

logger = logging.getLogger(__name__)

# Default TTS settings
DEFAULT_TTS_SETTINGS = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
}

# Settings optimized for academic content
ACADEMIC_TTS_SETTINGS = {
    "exaggeration": 0.3,  # Lower exaggeration for more controlled tone
    "cfg_weight": 0.6,    # Higher cfg_weight for more deliberate pacing
}

# Settings optimized for math-heavy content
MATH_HEAVY_SETTINGS = {
    "exaggeration": 0.2,  # Even lower exaggeration for precise math expressions
    "cfg_weight": 0.7,    # Higher cfg_weight for slower, more deliberate pacing
}


class ChatterboxTTSEngine:
    """
    A wrapper around the Chatterbox TTS engine that provides additional functionality
    for generating audio from text.
    """

    def __init__(self, device: Optional[str] = None, force_cpu: bool = False):
        """
        Initialize the Chatterbox TTS engine.

        Args:
            device: The device to use for inference. If None, will automatically
                   select the best available device (CUDA, MPS, or CPU).
            force_cpu: If True, will force using CPU regardless of GPU availability.
        """
        self.device = "cpu" if force_cpu else self._get_device(device)
        logger.info(f"Initializing Chatterbox TTS engine on device: {self.device}")
        
        try:
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.sample_rate = self.model.sr
            logger.info(f"Chatterbox TTS engine initialized successfully. Sample rate: {self.sample_rate}Hz")
        except Exception as e:
            if self.device != "cpu" and not force_cpu:
                # If initialization fails with GPU, try falling back to CPU
                logger.warning(f"Failed to initialize on {self.device}, falling back to CPU: {e}")
                try:
                    self.device = "cpu"
                    self.model = ChatterboxTTS.from_pretrained(device=self.device)
                    self.sample_rate = self.model.sr
                    logger.info(f"Chatterbox TTS engine initialized successfully on CPU. Sample rate: {self.sample_rate}Hz")
                except Exception as cpu_e:
                    logger.error(f"Failed to initialize Chatterbox TTS engine on CPU: {cpu_e}")
                    raise
            else:
                logger.error(f"Failed to initialize Chatterbox TTS engine: {e}")
                raise

    def _get_device(self, device: Optional[str] = None) -> str:
        """
        Determine the best available device for inference.

        Args:
            device: The requested device. If None, will automatically select.

        Returns:
            The device to use for inference.
        """
        if device is not None:
            return device
            
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def generate_audio(
        self, 
        text: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate audio from text using the Chatterbox TTS engine.

        Args:
            text: The text to convert to speech.
            settings: Optional dictionary of TTS settings to override defaults.

        Returns:
            A tuple containing the audio waveform tensor and sample rate.
        """
        if not text.strip():
            logger.warning("Empty text provided to TTS engine. Returning empty audio.")
            return torch.zeros((1, 1)), self.sample_rate
            
        # Merge default settings with any provided settings
        tts_settings = DEFAULT_TTS_SETTINGS.copy()
        if settings:
            tts_settings.update(settings)
            
        try:
            logger.info(f"Generating audio for text of length {len(text)} with settings: {tts_settings}")
            
            # Generate audio using Chatterbox TTS
            wav = self.model.generate(
                text,
                exaggeration=tts_settings.get("exaggeration", 0.5),
                cfg_weight=tts_settings.get("cfg_weight", 0.5)
            )
            
            logger.info(f"Audio generated successfully. Shape: {wav.shape}")
            return wav, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise

    def save_audio(
        self, 
        audio: torch.Tensor, 
        output_path: str, 
        sample_rate: Optional[int] = None
    ) -> str:
        """
        Save audio to a file.

        Args:
            audio: The audio waveform tensor.
            output_path: The path to save the audio to.
            sample_rate: The sample rate of the audio. If None, uses the model's sample rate.

        Returns:
            The path to the saved audio file.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save the audio
            torchaudio.save(output_path, audio, sample_rate)
            logger.info(f"Audio saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving audio to {output_path}: {e}")
            raise

    def clear_cache(self):
        """
        Clear GPU cache if using CUDA to free up memory.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")