"""
Audio format handling module.

This module provides functionality to handle different audio formats and conversions.
"""

import os
import logging
from typing import Dict, Optional, Tuple, List

import ffmpeg

logger = logging.getLogger(__name__)

# Audio format specifications
AUDIO_FORMATS = {
    "wav": {
        "extension": ".wav",
        "mime_type": "audio/wav",
        "description": "Waveform Audio File Format (uncompressed)",
        "ffmpeg_options": {
            "acodec": "pcm_s16le",
            "ar": "44100"
        }
    },
    "mp3": {
        "extension": ".mp3",
        "mime_type": "audio/mpeg",
        "description": "MPEG Audio Layer III (compressed)",
        "ffmpeg_options": {
            "acodec": "libmp3lame",
            "ar": "44100",
            "ab": "192k"
        }
    },
    "flac": {
        "extension": ".flac",
        "mime_type": "audio/flac",
        "description": "Free Lossless Audio Codec (lossless compression)",
        "ffmpeg_options": {
            "acodec": "flac",
            "ar": "44100"
        }
    },
    "ogg": {
        "extension": ".ogg",
        "mime_type": "audio/ogg",
        "description": "Ogg Vorbis (compressed)",
        "ffmpeg_options": {
            "acodec": "libvorbis",
            "ar": "44100",
            "ab": "192k"
        }
    },
    "m4a": {
        "extension": ".m4a",
        "mime_type": "audio/mp4",
        "description": "MPEG-4 Audio (compressed)",
        "ffmpeg_options": {
            "acodec": "aac",
            "ar": "44100",
            "ab": "192k"
        }
    }
}


class AudioFormatHandler:
    """
    Handles audio format conversions and metadata.
    """

    def __init__(self):
        """Initialize the audio format handler."""
        pass

    def get_format_info(self, format_name: str) -> Dict:
        """
        Get information about a specific audio format.

        Args:
            format_name: The name of the audio format.

        Returns:
            A dictionary containing format information.
        """
        format_name = format_name.lower()
        if format_name not in AUDIO_FORMATS:
            raise ValueError(f"Unsupported audio format: {format_name}")
            
        return AUDIO_FORMATS[format_name]

    def list_supported_formats(self) -> List[Dict]:
        """
        List all supported audio formats.

        Returns:
            A list of dictionaries containing format information.
        """
        return [
            {
                "name": name,
                **info
            }
            for name, info in AUDIO_FORMATS.items()
        ]

    def convert_format(
        self, 
        input_path: str, 
        output_format: str, 
        output_path: Optional[str] = None,
        quality: str = "medium"
    ) -> str:
        """
        Convert an audio file to a different format.

        Args:
            input_path: Path to the input audio file.
            output_format: The desired output format.
            output_path: Path to save the converted audio file. If None, will use the
                         input path with the new extension.
            quality: Quality setting for compressed formats ("low", "medium", "high").

        Returns:
            The path to the converted audio file.
        """
        # Validate format
        output_format = output_format.lower()
        if output_format not in AUDIO_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        # Determine output path if not provided
        if output_path is None:
            input_dir = os.path.dirname(input_path)
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            output_ext = AUDIO_FORMATS[output_format]["extension"]
            output_path = os.path.join(input_dir, f"{input_name}{output_ext}")
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Get format options
        format_options = AUDIO_FORMATS[output_format]["ffmpeg_options"].copy()
        
        # Adjust quality settings
        if output_format in ["mp3", "ogg", "m4a"]:
            if quality == "low":
                format_options["ab"] = "96k"
            elif quality == "high":
                format_options["ab"] = "320k"
                
        try:
            # Use ffmpeg for conversion
            (
                ffmpeg
                .input(input_path)
                .output(output_path, **format_options)
                .run(quiet=True, overwrite_output=True)
            )
            
            logger.info(f"Converted audio from {input_path} to {output_path} ({output_format} format)")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            raise

    def get_audio_info(self, audio_path: str) -> Dict:
        """
        Get information about an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            A dictionary containing audio information.
        """
        try:
            # Use ffprobe to get audio information
            probe = ffmpeg.probe(audio_path)
            
            # Extract audio stream information
            audio_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'audio'), None)
            
            if audio_stream is None:
                raise ValueError(f"No audio stream found in {audio_path}")
                
            # Extract relevant information
            info = {
                "format": probe['format']['format_name'],
                "duration": float(probe['format']['duration']),
                "size_bytes": int(probe['format']['size']),
                "bit_rate": int(probe['format']['bit_rate']) if 'bit_rate' in probe['format'] else None,
                "sample_rate": int(audio_stream['sample_rate']) if 'sample_rate' in audio_stream else None,
                "channels": int(audio_stream['channels']) if 'channels' in audio_stream else None,
                "codec": audio_stream['codec_name'] if 'codec_name' in audio_stream else None,
            }
            
            # Add metadata if available
            if 'tags' in probe['format']:
                info['metadata'] = probe['format']['tags']
                
            return info
            
        except Exception as e:
            logger.error(f"Error getting audio information: {e}")
            raise

    def get_extension_for_format(self, format_name: str) -> str:
        """
        Get the file extension for a specific audio format.

        Args:
            format_name: The name of the audio format.

        Returns:
            The file extension for the format.
        """
        format_name = format_name.lower()
        if format_name not in AUDIO_FORMATS:
            raise ValueError(f"Unsupported audio format: {format_name}")
            
        return AUDIO_FORMATS[format_name]["extension"]