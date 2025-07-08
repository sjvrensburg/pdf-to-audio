"""
Tests for the TTS integration.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

import torch

from pdf_to_audio.tts.chatterbox_tts import ChatterboxTTSEngine
from pdf_to_audio.tts.voice_manager import VoiceManager
from pdf_to_audio.tts.audio_processing import AudioProcessor
from pdf_to_audio.audio.chunking import TextChunker
from pdf_to_audio.audio.concatenation import AudioConcatenator
from pdf_to_audio.audio.formats import AudioFormatHandler


class TestChatterboxTTS:
    """Tests for the ChatterboxTTS integration."""
    
    @pytest.fixture
    def mock_chatterbox(self):
        """Mock the ChatterboxTTS class."""
        with patch('pdf_to_audio.tts.chatterbox_tts.ChatterboxTTS') as mock:
            # Configure the mock
            mock_instance = MagicMock()
            mock_instance.sr = 24000
            mock_instance.generate.return_value = torch.zeros((1, 24000))
            mock.from_pretrained.return_value = mock_instance
            yield mock
    
    def test_tts_engine_initialization(self, mock_chatterbox):
        """Test that the TTS engine initializes correctly."""
        engine = ChatterboxTTSEngine()
        assert engine.device in ["cuda", "mps", "cpu"]
        assert engine.sample_rate == 24000
        mock_chatterbox.from_pretrained.assert_called_once()
    
    def test_generate_audio(self, mock_chatterbox):
        """Test audio generation."""
        engine = ChatterboxTTSEngine()
        text = "This is a test sentence."
        audio, sample_rate = engine.generate_audio(text)
        
        assert isinstance(audio, torch.Tensor)
        assert sample_rate == 24000
        engine.model.generate.assert_called_once()
    
    def test_generate_audio_with_voice(self, mock_chatterbox, tmp_path):
        """Test audio generation with voice cloning."""
        # Create a temporary voice file
        voice_path = tmp_path / "test_voice.wav"
        with open(voice_path, 'w') as f:
            f.write("dummy")
            
        engine = ChatterboxTTSEngine()
        text = "This is a test sentence with voice cloning."
        audio, sample_rate = engine.generate_audio(text, voice_path=str(voice_path))
        
        assert isinstance(audio, torch.Tensor)
        assert sample_rate == 24000
        engine.model.generate.assert_called_once()
        
    def test_empty_text_handling(self, mock_chatterbox):
        """Test handling of empty text."""
        engine = ChatterboxTTSEngine()
        audio, sample_rate = engine.generate_audio("")
        
        assert isinstance(audio, torch.Tensor)
        assert audio.shape == (1, 1)  # Should return an empty tensor
        assert sample_rate == 24000
        engine.model.generate.assert_not_called()


class TestTextChunking:
    """Tests for the text chunking functionality."""
    
    def test_duration_chunking(self):
        """Test chunking by estimated duration."""
        chunker = TextChunker(strategy="duration", max_duration_sec=10, chars_per_second=10)
        text = "A" * 200  # 200 characters should be split into at least 2 chunks
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 2
        assert all(len(chunk) <= 100 for chunk in chunks)  # 10 sec * 10 chars/sec = 100 chars
    
    def test_sentence_chunking(self):
        """Test chunking by sentences."""
        chunker = TextChunker(strategy="sentences", max_duration_sec=10, chars_per_second=10)
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = chunker.chunk_text(text)
        
        # Should be chunked by sentences
        assert len(chunks) >= 1
        
    def test_smart_chunking(self):
        """Test smart chunking with special content."""
        chunker = TextChunker(strategy="smart", max_duration_sec=10, chars_per_second=10)
        text = "Introduction. $x^2 + y^2 = z^2$ is the Pythagorean theorem. Figure 1 shows an example."
        chunks = chunker.chunk_text(text)
        
        # Should keep math expressions together
        assert any("$x^2 + y^2 = z^2$" in chunk for chunk in chunks)
        
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")
        
        assert chunks == []


class TestAudioProcessing:
    """Tests for audio processing functionality."""
    
    def test_normalize_volume(self):
        """Test volume normalization."""
        processor = AudioProcessor()
        # Create a test audio tensor
        audio = torch.ones((1, 24000)) * 0.1  # Low volume
        normalized = processor.normalize_volume(audio, target_db=-16.0)
        
        assert normalized.shape == audio.shape
        assert torch.max(torch.abs(normalized)) > torch.max(torch.abs(audio))
    
    def test_add_pause(self):
        """Test adding a pause to audio."""
        processor = AudioProcessor()
        audio = torch.ones((1, 24000))
        with_pause = processor.add_pause(audio, duration_ms=500)
        
        # Should be longer by 500ms (12000 samples at 24kHz)
        assert with_pause.shape[1] == audio.shape[1] + 12000
        # The added part should be silence
        assert torch.all(with_pause[:, -12000:] == 0)


class TestVoiceManager:
    """Tests for the voice manager functionality."""
    
    @pytest.fixture
    def temp_voice_manager(self, tmp_path):
        """Create a temporary voice manager for testing."""
        voices_dir = tmp_path / "voices"
        return VoiceManager(voices_dir=str(voices_dir))
    
    @pytest.fixture
    def mock_torchaudio(self):
        """Mock torchaudio for testing."""
        with patch('pdf_to_audio.tts.voice_manager.torchaudio') as mock:
            mock.load.return_value = (torch.ones((1, 24000)), 24000)
            yield mock
    
    def test_voice_validation(self, temp_voice_manager, mock_torchaudio, tmp_path):
        """Test voice sample validation."""
        # Create a test voice file
        voice_path = tmp_path / "test_voice.wav"
        with open(voice_path, 'w') as f:
            f.write("dummy")
            
        valid, message = temp_voice_manager.validate_voice_sample(str(voice_path))
        assert valid
    
    def test_register_and_list_voices(self, temp_voice_manager, mock_torchaudio, tmp_path):
        """Test registering and listing voices."""
        # Create a test voice file
        voice_path = tmp_path / "test_voice.wav"
        with open(voice_path, 'w') as f:
            f.write("dummy")
            
        # Register the voice
        temp_voice_manager.register_voice(str(voice_path), "test_voice")
        
        # List voices
        voices = temp_voice_manager.list_voices()
        assert len(voices) == 1
        assert voices[0]["name"] == "test_voice"
        
        # Get voice path
        retrieved_path = temp_voice_manager.get_voice_path("test_voice")
        assert retrieved_path is not None
        
        # Remove voice
        success = temp_voice_manager.remove_voice("test_voice")
        assert success
        
        # List voices again
        voices = temp_voice_manager.list_voices()
        assert len(voices) == 0