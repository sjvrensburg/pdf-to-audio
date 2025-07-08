"""Command-line interface for PDF to audio conversion."""

import argparse
import os
import sys
import logging

from mistralai import Mistral

from .api import check_available_models, process_pdf_to_json
from .core import process_document, generate_audio
from .tts.voice_manager import VoiceManager
from .audio.formats import AudioFormatHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert a PDF to TTS-friendly text and optionally generate audio using Mistral API and Chatterbox TTS.",
        prog="pdf-to-audio"
    )
    parser.add_argument("input_pdf", nargs='?', help="Path to the input PDF file.")
    parser.add_argument("output_file", nargs='?', help="Path to the output text file.")
    
    # PDF processing options
    pdf_group = parser.add_argument_group('PDF Processing Options')
    pdf_group.add_argument(
        "--pages_per_chunk",
        type=int,
        default=1,
        help="Number of pages to process at a time (default: 1).",
    )
    pdf_group.add_argument(
        "--include_images",
        action="store_true",
        help="Include image descriptions in the output (default: False).",
    )
    pdf_group.add_argument(
        "--text_model",
        default="mistral-small-latest",
        help="Model for text processing (default: mistral-small-latest).",
    )
    pdf_group.add_argument(
        "--image_model",
        default="pixtral-12b-latest",
        help="Model for image processing (default: pixtral-12b-latest).",
    )
    
    # Audio generation options
    audio_group = parser.add_argument_group('Audio Generation Options')
    audio_group.add_argument(
        "--output_audio",
        help="Path to the output audio file. If provided, audio will be generated.",
    )
    audio_group.add_argument(
        "--voice",
        help="Path to a .wav file for voice cloning.",
    )
    audio_group.add_argument(
        "--list_voices",
        action="store_true",
        help="List currently registered voices and exit.",
    )
    audio_group.add_argument(
        "--register_voice",
        nargs=2,
        metavar=('VOICE_PATH', 'VOICE_NAME'),
        help="Register a voice sample for future use. Requires a path to a .wav file and a name.",
    )
    audio_group.add_argument(
        "--remove_voice",
        metavar='VOICE_NAME',
        help="Remove a registered voice by name.",
    )
    audio_group.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="TTS exaggeration level (0.0-1.0, default: 0.5).",
    )
    audio_group.add_argument(
        "--cfg_weight",
        type=float,
        default=0.5,
        help="CFG weight for TTS (0.0-1.0, default: 0.5).",
    )
    audio_group.add_argument(
        "--audio_format",
        choices=["wav", "mp3", "flac", "ogg", "m4a"],
        default="wav",
        help="Output audio format (default: wav).",
    )
    audio_group.add_argument(
        "--chunk_strategy",
        choices=["duration", "sentences", "smart"],
        default="smart",
        help="Text chunking strategy for audio generation (default: smart).",
    )
    
    # General options
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they exist (default: False).",
    )
    general_group.add_argument(
        "--list_models",
        action="store_true",
        help="List available Mistral models and exit.",
    )
    general_group.add_argument(
        "--list_audio_formats",
        action="store_true",
        help="List supported audio formats and exit.",
    )
    general_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging.",
    )
    
    return parser


def validate_api_key():
    """Validate that the Mistral API key is available."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable is not set.")
        print("Please set your Mistral API key:")
        print("  export MISTRAL_API_KEY='your-api-key-here'")
        sys.exit(1)
    return api_key


def validate_arguments(args, parser):
    """Validate command-line arguments."""
    # Skip validation for utility commands
    if (args.list_models or args.list_voices or args.list_audio_formats or 
        args.register_voice or args.remove_voice):
        return
        
    # Validate required arguments
    if not args.input_pdf:
        print("Error: input_pdf is required")
        parser.print_help()
        sys.exit(1)
        
    # Require at least one output (text or audio)
    if not args.output_file and not args.output_audio:
        print("Error: At least one of output_file or output_audio must be specified")
        parser.print_help()
        sys.exit(1)

    # Check if output files exist and overwrite is not set
    if args.output_file and os.path.exists(args.output_file) and not args.overwrite:
        print(f"Error: {args.output_file} exists. Use --overwrite to replace it.")
        sys.exit(1)
        
    if args.output_audio and os.path.exists(args.output_audio) and not args.overwrite:
        print(f"Error: {args.output_audio} exists. Use --overwrite to replace it.")
        sys.exit(1)
        
    # Validate voice file if provided
    if args.voice and not os.path.exists(args.voice):
        print(f"Error: Voice file not found: {args.voice}")
        sys.exit(1)


def handle_voice_management(args):
    """Handle voice management commands."""
    voice_manager = VoiceManager()
    
    # List registered voices
    if args.list_voices:
        voices = voice_manager.list_voices()
        if not voices:
            print("No voices registered.")
        else:
            print(f"Found {len(voices)} registered voices:")
            for voice in voices:
                print(f"  - {voice['name']}: {voice['description'] or 'No description'} "
                      f"({voice['duration']:.1f}s, {voice['sample_rate']}Hz)")
        return True
        
    # Register a new voice
    if args.register_voice:
        voice_path, voice_name = args.register_voice
        if not os.path.exists(voice_path):
            print(f"Error: Voice file not found: {voice_path}")
            return True
            
        try:
            # Validate the voice sample
            valid, message = voice_manager.validate_voice_sample(voice_path)
            if not valid:
                print(f"Error: Invalid voice sample: {message}")
                return True
                
            # Register the voice
            voice_path = voice_manager.register_voice(
                voice_path, 
                voice_name, 
                overwrite=args.overwrite
            )
            print(f"Voice '{voice_name}' registered successfully at {voice_path}")
        except Exception as e:
            print(f"Error registering voice: {e}")
        return True
        
    # Remove a voice
    if args.remove_voice:
        try:
            success = voice_manager.remove_voice(args.remove_voice)
            if success:
                print(f"Voice '{args.remove_voice}' removed successfully")
            else:
                print(f"Voice '{args.remove_voice}' not found")
        except Exception as e:
            print(f"Error removing voice: {e}")
        return True
        
    return False


def list_audio_formats():
    """List supported audio formats."""
    format_handler = AudioFormatHandler()
    formats = format_handler.list_supported_formats()
    
    print("Supported audio formats:")
    for fmt in formats:
        print(f"  - {fmt['name']}: {fmt['description']} ({fmt['extension']})")
    
    return True


def main():
    """Main entry point for the CLI application."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle voice management commands
    if handle_voice_management(args):
        sys.exit(0)
        
    # List audio formats
    if args.list_audio_formats:
        list_audio_formats()
        sys.exit(0)

    # Initialize Mistral client with API key
    api_key = validate_api_key()
    client = Mistral(api_key=api_key)

    # Test API key validity and optionally list models
    try:
        if args.list_models:
            check_available_models(client)
            sys.exit(0)
        else:
            client.models.list()
    except Exception as e:
        print(f"Invalid API key or connection error: {e}")
        sys.exit(1)

    # Validate arguments
    validate_arguments(args, parser)

    if args.verbose:
        print(f"Using text model: {args.text_model}")
        if args.include_images:
            print(f"Using image model: {args.image_model}")

    # Convert PDF to JSON
    print("Processing PDF with OCR...")
    doc = process_pdf_to_json(client, args.input_pdf)

    # Process the document
    final_transformed_text = process_document(client, doc, args)

    # Write the transformed text to the output file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(final_transformed_text)
        print(f"Transformation complete. TTS-friendly document saved to '{args.output_file}'.")
        print(f"Output file size: {len(final_transformed_text)} characters")

    # Generate audio if requested
    if args.output_audio:
        try:
            audio_path = generate_audio(final_transformed_text, args)
            if audio_path:
                print(f"Audio generation complete. Audio saved to '{audio_path}'.")
            else:
                print("Audio generation failed.")
        except Exception as e:
            print(f"Error generating audio: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()