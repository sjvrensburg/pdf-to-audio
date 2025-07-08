"""
Audio concatenation module.

This module provides functionality to concatenate multiple audio segments into a single file.
"""

import os
import logging
from typing import List, Optional, Union, Tuple

import torch
import torchaudio
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class AudioConcatenator:
    """
    Concatenates multiple audio segments into a single file.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the audio concatenator.

        Args:
            sample_rate: The sample rate of the audio to concatenate.
        """
        self.sample_rate = sample_rate

    def concatenate_tensors(
        self, 
        audio_tensors: List[torch.Tensor],
        crossfade_ms: int = 100
    ) -> torch.Tensor:
        """
        Concatenate multiple audio tensors into a single tensor.

        Args:
            audio_tensors: List of audio tensors to concatenate.
            crossfade_ms: Duration of crossfade between segments in milliseconds.

        Returns:
            A single concatenated audio tensor.
        """
        if not audio_tensors:
            logger.warning("No audio tensors provided for concatenation")
            return torch.zeros((1, 1))
            
        if len(audio_tensors) == 1:
            return audio_tensors[0]
            
        # Ensure all tensors have the same number of channels
        num_channels = audio_tensors[0].shape[0]
        for i, tensor in enumerate(audio_tensors):
            if tensor.shape[0] != num_channels:
                logger.warning(f"Audio tensor {i} has {tensor.shape[0]} channels, expected {num_channels}")
                # Convert to the expected number of channels (simple average for downmixing)
                if tensor.shape[0] > num_channels:
                    tensor = tensor[:num_channels]
                else:
                    # Duplicate channels for upmixing
                    tensor = tensor.repeat(num_channels // tensor.shape[0] + 1, 1)[:num_channels]
                audio_tensors[i] = tensor
        
        # If crossfade is enabled and we have enough samples
        if crossfade_ms > 0:
            crossfade_samples = int(self.sample_rate * crossfade_ms / 1000)
            result = audio_tensors[0]
            
            for tensor in audio_tensors[1:]:
                if result.shape[1] > crossfade_samples and tensor.shape[1] > crossfade_samples:
                    # Create crossfade weights
                    fade_out = torch.linspace(1, 0, crossfade_samples)
                    fade_in = torch.linspace(0, 1, crossfade_samples)
                    
                    # Apply crossfade
                    result_end = result[:, -crossfade_samples:]
                    tensor_start = tensor[:, :crossfade_samples]
                    
                    # Blend the overlapping region
                    overlap = result_end * fade_out + tensor_start * fade_in
                    
                    # Concatenate with crossfade
                    result = torch.cat([result[:, :-crossfade_samples], overlap, tensor[:, crossfade_samples:]], dim=1)
                else:
                    # If segments are too short for crossfade, just concatenate
                    result = torch.cat([result, tensor], dim=1)
            
            return result
        else:
            # Simple concatenation without crossfade
            return torch.cat(audio_tensors, dim=1)

    def concatenate_files(
        self, 
        audio_files: List[str],
        output_file: str,
        crossfade_ms: int = 100
    ) -> str:
        """
        Concatenate multiple audio files into a single file.

        Args:
            audio_files: List of paths to audio files to concatenate.
            output_file: Path to save the concatenated audio.
            crossfade_ms: Duration of crossfade between segments in milliseconds.

        Returns:
            Path to the concatenated audio file.
        """
        if not audio_files:
            logger.warning("No audio files provided for concatenation")
            return ""
            
        if len(audio_files) == 1:
            # Just copy the single file
            import shutil
            shutil.copy2(audio_files[0], output_file)
            return output_file
            
        # Use pydub for file-based concatenation with crossfade
        try:
            # Load the first segment
            combined = AudioSegment.from_file(audio_files[0])
            
            # Add the remaining segments with crossfade
            for file_path in audio_files[1:]:
                segment = AudioSegment.from_file(file_path)
                combined = combined.append(segment, crossfade=crossfade_ms)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Export the combined audio
            file_format = output_file.split('.')[-1]
            combined.export(output_file, format=file_format)
            
            logger.info(f"Concatenated {len(audio_files)} audio files to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error concatenating audio files: {e}")
            raise

    def load_and_concatenate(
        self, 
        audio_files: List[str],
        crossfade_ms: int = 100
    ) -> Tuple[torch.Tensor, int]:
        """
        Load multiple audio files and concatenate them into a single tensor.

        Args:
            audio_files: List of paths to audio files to concatenate.
            crossfade_ms: Duration of crossfade between segments in milliseconds.

        Returns:
            A tuple containing the concatenated audio tensor and the sample rate.
        """
        if not audio_files:
            logger.warning("No audio files provided for concatenation")
            return torch.zeros((1, 1)), self.sample_rate
            
        # Load all audio files
        audio_tensors = []
        for file_path in audio_files:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
                
            audio_tensors.append(waveform)
            
        # Concatenate the tensors
        concatenated = self.concatenate_tensors(audio_tensors, crossfade_ms)
        
        logger.info(f"Loaded and concatenated {len(audio_files)} audio files")
        return concatenated, self.sample_rate