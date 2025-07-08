"""
Voice management module for Chatterbox TTS.

This module provides functionality to register, validate, and manage voice samples
for use with the Chatterbox TTS engine.
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)

# Constants
DEFAULT_VOICES_DIR = os.path.expanduser("~/.pdf_to_audio/voices")
VOICE_INFO_FILE = "voice_info.json"
MIN_VOICE_DURATION_SEC = 3.0  # Minimum recommended duration for voice samples
MAX_VOICE_DURATION_SEC = 30.0  # Maximum recommended duration for voice samples


class VoiceManager:
    """
    Manages voice samples for use with the Chatterbox TTS engine.
    """

    def __init__(self, voices_dir: Optional[str] = None):
        """
        Initialize the voice manager.

        Args:
            voices_dir: Directory to store voice samples. If None, uses the default.
        """
        self.voices_dir = Path(voices_dir or DEFAULT_VOICES_DIR)
        self._ensure_voices_dir()
        self.voice_info = self._load_voice_info()

    def _ensure_voices_dir(self) -> None:
        """Ensure the voices directory exists."""
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        voice_info_path = self.voices_dir / VOICE_INFO_FILE
        if not voice_info_path.exists():
            with open(voice_info_path, 'w') as f:
                json.dump({}, f)

    def _load_voice_info(self) -> Dict:
        """Load voice information from the voice info file."""
        voice_info_path = self.voices_dir / VOICE_INFO_FILE
        try:
            with open(voice_info_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not load voice info from {voice_info_path}. Creating new file.")
            return {}

    def _save_voice_info(self) -> None:
        """Save voice information to the voice info file."""
        voice_info_path = self.voices_dir / VOICE_INFO_FILE
        with open(voice_info_path, 'w') as f:
            json.dump(self.voice_info, f, indent=2)

    def register_voice(
        self, 
        voice_path: str, 
        name: str, 
        description: Optional[str] = None,
        overwrite: bool = False
    ) -> str:
        """
        Register a voice sample for use with the TTS engine.

        Args:
            voice_path: Path to the voice sample file.
            name: Name to identify the voice.
            description: Optional description of the voice.
            overwrite: Whether to overwrite an existing voice with the same name.

        Returns:
            Path to the registered voice sample.
        """
        # Check if voice with this name already exists
        if name in self.voice_info and not overwrite:
            raise ValueError(f"Voice with name '{name}' already exists. Use overwrite=True to replace it.")

        # Validate the voice sample
        validation_result, message = self.validate_voice_sample(voice_path)
        if not validation_result:
            raise ValueError(f"Invalid voice sample: {message}")

        # Copy the voice sample to the voices directory
        voice_filename = f"{name.lower().replace(' ', '_')}.wav"
        target_path = self.voices_dir / voice_filename
        
        try:
            shutil.copy2(voice_path, target_path)
        except Exception as e:
            logger.error(f"Error copying voice sample: {e}")
            raise

        # Get voice metadata
        waveform, sample_rate = torchaudio.load(voice_path)
        duration = waveform.shape[1] / sample_rate

        # Update voice info
        self.voice_info[name] = {
            "path": str(target_path),
            "description": description or "",
            "sample_rate": sample_rate,
            "duration": duration,
            "channels": waveform.shape[0],
        }
        
        self._save_voice_info()
        logger.info(f"Voice '{name}' registered successfully at {target_path}")
        
        return str(target_path)

    def validate_voice_sample(self, voice_path: str) -> Tuple[bool, str]:
        """
        Validate a voice sample for use with the TTS engine.

        Args:
            voice_path: Path to the voice sample file.

        Returns:
            A tuple containing a boolean indicating whether the sample is valid,
            and a message explaining the validation result.
        """
        # Check if file exists
        if not os.path.exists(voice_path):
            return False, f"File does not exist: {voice_path}"

        # Check file format
        if not voice_path.lower().endswith('.wav'):
            return False, "Voice sample must be a WAV file"

        try:
            # Load the audio file
            waveform, sample_rate = torchaudio.load(voice_path)
            
            # Check duration
            duration = waveform.shape[1] / sample_rate
            if duration < MIN_VOICE_DURATION_SEC:
                return False, f"Voice sample is too short ({duration:.2f}s). Minimum recommended duration is {MIN_VOICE_DURATION_SEC}s."
            if duration > MAX_VOICE_DURATION_SEC:
                return False, f"Voice sample is too long ({duration:.2f}s). Maximum recommended duration is {MAX_VOICE_DURATION_SEC}s."
            
            # Check channels (mono is preferred)
            if waveform.shape[0] > 1:
                return True, "Warning: Multi-channel audio detected. Mono audio is recommended for best results."
                
            return True, "Voice sample is valid"
            
        except Exception as e:
            return False, f"Error validating voice sample: {e}"

    def get_voice_path(self, name: str) -> Optional[str]:
        """
        Get the path to a registered voice sample.

        Args:
            name: Name of the voice.

        Returns:
            Path to the voice sample, or None if not found.
        """
        if name in self.voice_info:
            return self.voice_info[name]["path"]
        return None

    def list_voices(self) -> List[Dict]:
        """
        List all registered voices.

        Returns:
            A list of dictionaries containing information about each voice.
        """
        return [
            {
                "name": name,
                **info
            }
            for name, info in self.voice_info.items()
        ]

    def remove_voice(self, name: str) -> bool:
        """
        Remove a registered voice.

        Args:
            name: Name of the voice to remove.

        Returns:
            True if the voice was removed, False otherwise.
        """
        if name not in self.voice_info:
            logger.warning(f"Voice '{name}' not found")
            return False

        # Get the path to the voice sample
        voice_path = self.voice_info[name]["path"]
        
        # Remove the voice sample file
        try:
            os.remove(voice_path)
        except Exception as e:
            logger.error(f"Error removing voice sample file: {e}")
            
        # Remove the voice from the voice info
        del self.voice_info[name]
        self._save_voice_info()
        
        logger.info(f"Voice '{name}' removed successfully")
        return True