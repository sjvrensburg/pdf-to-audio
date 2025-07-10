"""Command-line interface for PDF to audio conversion."""
import argparse
import os
import sys
import logging
from mistralai import Mistral
from .api import check_available_models, process_pdf_to_json
from .core import process_document, generate_audio
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
        "--global_normalization",
        action="store_true",
        help="Apply global volume normalization to the entire audio file after concatenation (default: False).",
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
        "--math_exaggeration",
        type=float,
        help="TTS exaggeration level for mathematical content (0.0-1.0). If not provided, defaults to 0.75 * exaggeration.",
    )
    audio_group.add_argument(
        "--math_cfg_weight",
        type=float,
        help="CFG weight for TTS for mathematical content (0.0-1.0). If not provided, defaults to 0.75 * cfg_weight.",
    )
    audio_group.add_argument(
        "--math_tts_scale",
        type=float,
        default=0.75,
        help="Scaling factor for math TTS settings relative to plain text settings (default: 0.75).",
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
    audio_group.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force using CPU for TTS even if GPU is available. Use this if you encounter CUDA errors.",
    )
    
    # Refinement options
    refinement_group = parser.add_argument_group('Content Refinement Options')
    refinement_group.add_argument(
        "--enable_refinement",
        action="store_true",
        help="Enable the multi-pass refinement pipeline (default: True).",
    )
    refinement_group.add_argument(
        "--disable_refinement",
        action="store_true",
        help="Disable the multi-pass refinement pipeline.",
    )
    refinement_group.add_argument(
        "--enable_math_refinement",
        action="store_true",
        help="Enable the mathematical content refinement pass (default: True).",
    )
    refinement_group.add_argument(
        "--disable_math_refinement",
        action="store_true",
        help="Disable the mathematical content refinement pass.",
    )
    refinement_group.add_argument(
        "--enable_structure_citation_optimization",
        action="store_true",
        help="Enable the structure and citation optimization pass (default: True).",
    )
    refinement_group.add_argument(
        "--disable_structure_citation_optimization",
        action="store_true",
        help="Disable the structure and citation optimization pass.",
    )
    refinement_group.add_argument(
        "--enable_language_style_refinement",
        action="store_true",
        help="Enable the language and style refinement pass (default: True).",
    )
    refinement_group.add_argument(
        "--disable_language_style_refinement",
        action="store_true",
        help="Disable the language and style refinement pass.",
    )
    refinement_group.add_argument(
        "--enable_audio_specific_optimization",
        action="store_true",
        help="Enable the audio-specific optimization pass (default: True).",
    )
    refinement_group.add_argument(
        "--disable_audio_specific_optimization",
        action="store_true",
        help="Disable the audio-specific optimization pass.",
    )
    refinement_group.add_argument(
        "--math_refinement_intensity",
        type=float,
        default=0.5,
        help="Intensity of the mathematical content refinement (0.0-1.0, default: 0.5).",
    )
    refinement_group.add_argument(
        "--structure_citation_intensity",
        type=float,
        default=0.5,
        help="Intensity of the structure and citation optimization (0.0-1.0, default: 0.5).",
    )
    refinement_group.add_argument(
        "--language_style_intensity",
        type=float,
        default=0.5,
        help="Intensity of the language and style refinement (0.0-1.0, default: 0.5).",
    )
    refinement_group.add_argument(
        "--audio_specific_intensity",
        type=float,
        default=0.5,
        help="Intensity of the audio-specific optimization (0.0-1.0, default: 0.5).",
    )
    refinement_group.add_argument(
        "--target_audience",
        choices=["academic", "general"],
        default="academic",
        help="Target audience for the refinement (default: academic).",
    )
    # General options
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument(
        "--config_file",
        help="Path to a YAML configuration file with custom settings.",
    )
    general_group.add_argument(
        "--system_prompt",
        help="Custom system prompt to override the default one.",
    )
    general_group.add_argument(
        "--temp_dir",
        help="Directory to use for temporary files. If not provided, system temp directory will be used.",
    )
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
    if args.list_models or args.list_audio_formats:
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
    
    # Process enable/disable flags for refinement
    if hasattr(args, 'disable_refinement') and args.disable_refinement:
        args.enable_refinement = False
    if hasattr(args, 'disable_math_refinement') and args.disable_math_refinement:
        args.enable_math_refinement = False
    if hasattr(args, 'disable_structure_citation_optimization') and args.disable_structure_citation_optimization:
        args.enable_structure_citation_optimization = False
    if hasattr(args, 'disable_language_style_refinement') and args.disable_language_style_refinement:
        args.enable_language_style_refinement = False
    if hasattr(args, 'disable_audio_specific_optimization') and args.disable_audio_specific_optimization:
        args.enable_audio_specific_optimization = False
    
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