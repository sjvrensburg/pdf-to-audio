"""
Tests for audio processing functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

import torch
import torchaudio

from pdf_to_audio.audio.concatenation import AudioConcatenator
from pdf_to_audio.audio.formats import AudioFormatHandler
from pdf_to_audio.tts.audio_processing import AudioProcessor


class TestAudioConcatenation:
    """Tests for audio concatenation functionality."""
    
    def test_concatenate_tensors(self):
        """Test concatenating audio tensors."""
        concatenator = AudioConcatenator()
        
        # Create test tensors
        tensor1 = torch.ones((1, 24000))
        tensor2 = torch.ones((1, 24000)) * 0.5
        
        # Concatenate without crossfade
        result = concatenator.concatenate_tensors([tensor1, tensor2], crossfade_ms=0)
        
        assert result.shape == (1, 48000)
        assert torch.all(result[:, :24000] == 1.0)
        assert torch.all(result[:, 24000:] == 0.5)
        
    def test_concatenate_tensors_with_crossfade(self):
        """Test concatenating audio tensors with crossfade."""
        concatenator = AudioConcatenator()
        
        # Create test tensors
        tensor1 = torch.ones((1, 24000))
        tensor2 = torch.ones((1, 24000)) * 0.5
        
        # Concatenate with crossfade
        result = concatenator.concatenate_tensors([tensor1, tensor2], crossfade_ms=100)
        
        # Check shape (should be less than sum due to crossfade)
        assert result.shape[1] < 48000
        
        # Check that the middle has a crossfade (values between 0.5 and 1.0)
        crossfade_samples = int(24000 * 100 / 1000)
        middle_values = result[0, 24000-crossfade_samples:24000]
        assert torch.all(middle_values <= 1.0)
        assert torch.all(middle_values >= 0.5)
        
    def test_empty_input(self):
        """Test handling of empty input."""
        concatenator = AudioConcatenator()
        result = concatenator.concatenate_tensors([])
        
        assert result.shape == (1, 1)
        
    def test_single_tensor(self):
        """Test handling of a single tensor."""
        concatenator = AudioConcatenator()
        tensor = torch.ones((1, 24000))
        result = concatenator.concatenate_tensors([tensor])
        
        assert torch.all(result == tensor)


class TestAudioFormatHandler:
    """Tests for audio format handling."""
    
    def test_get_format_info(self):
        """Test getting format information."""
        handler = AudioFormatHandler()
        
        # Test valid formats
        wav_info = handler.get_format_info("wav")
        assert wav_info["extension"] == ".wav"
        
        mp3_info = handler.get_format_info("mp3")
        assert mp3_info["extension"] == ".mp3"
        
        # Test case insensitivity
        flac_info = handler.get_format_info("FLAC")
        assert flac_info["extension"] == ".flac"
        
        # Test invalid format
        with pytest.raises(ValueError):
            handler.get_format_info("invalid")
            
    def test_list_supported_formats(self):
        """Test listing supported formats."""
        handler = AudioFormatHandler()
        formats = handler.list_supported_formats()
        
        assert len(formats) >= 3  # Should have at least wav, mp3, flac
        assert any(fmt["name"] == "wav" for fmt in formats)
        assert any(fmt["name"] == "mp3" for fmt in formats)
        
    def test_get_extension(self):
        """Test getting file extension for a format."""
        handler = AudioFormatHandler()
        
        assert handler.get_extension_for_format("wav") == ".wav"
        assert handler.get_extension_for_format("mp3") == ".mp3"
        
        with pytest.raises(ValueError):
            handler.get_extension_for_format("invalid")


class TestAudioProcessor:
    """Tests for audio processing functionality."""
    
    def test_normalize_volume(self):
        """Test volume normalization."""
        processor = AudioProcessor()
        
        # Create a quiet audio tensor
        audio = torch.ones((1, 24000)) * 0.1
        
        # Normalize to -16dB
        normalized = processor.normalize_volume(audio, target_db=-16.0)
        
        # Should be louder now
        assert torch.mean(normalized**2) > torch.mean(audio**2)
        
    def test_add_pause(self):
        """Test adding a pause to audio."""
        processor = AudioProcessor()
        
        # Create an audio tensor
        audio = torch.ones((1, 24000))
        
        # Add a 500ms pause
        with_pause = processor.add_pause(audio, duration_ms=500)
        
        # Should be longer by 500ms (12000 samples at 24kHz)
        assert with_pause.shape[1] == audio.shape[1] + int(24000 * 500 / 1000)
        
        # The added part should be silence
        assert torch.all(with_pause[:, -int(24000 * 500 / 1000):] == 0)
        
    def test_reduce_noise(self):
        """Test noise reduction."""
        processor = AudioProcessor()

        # Create an audio tensor with some "noise"
        audio = torch.ones((1, 24000)) * 0.1
        # Add some "signal"
        audio[0, 5000:10000] = 0.5

        # Apply noise reduction
        # threshold = max_amp * reduction_amount = 0.5 * 0.25 = 0.125
        # This will zero out values with abs < 0.125 (i.e., 0.1)
        denoised = processor.reduce_noise(audio, reduction_amount=0.25)

        # Low values should be zeroed out
        assert torch.all(denoised[:, 0:5000] == 0)
        assert torch.all(denoised[:, 10000:] == 0)

        # Signal should remain
        assert torch.all(denoised[:, 5000:10000] == 0.5)