"""Command-line interface for PDF to audio conversion."""

import argparse
import os
import sys

from mistralai import Mistral

from .api import check_available_models, process_pdf_to_json
from .core import process_document


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert a PDF to a TTS-friendly text file using Mistral API.",
        prog="pdf-to-audio"
    )
    parser.add_argument("input_pdf", nargs='?', help="Path to the input PDF file.")
    parser.add_argument("output_file", nargs='?', help="Path to the output text file.")
    parser.add_argument(
        "--pages_per_chunk",
        type=int,
        default=1,
        help="Number of pages to process at a time (default: 1).",
    )
    parser.add_argument(
        "--include_images",
        action="store_true",
        help="Include image descriptions in the output (default: False).",
    )
    parser.add_argument(
        "--text_model",
        default="mistral-small-latest",
        help="Model for text processing (default: mistral-small-latest).",
    )
    parser.add_argument(
        "--image_model",
        default="pixtral-12b-latest",
        help="Model for image processing (default: pixtral-12b-latest).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists (default: False).",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
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
    # Validate required arguments if not listing models
    if not args.list_models and (not args.input_pdf or not args.output_file):
        print("Error: input_pdf and output_file are required unless using --list_models")
        parser.print_help()
        sys.exit(1)

    # Check if output file exists and overwrite is not set
    if args.output_file and os.path.exists(args.output_file) and not args.overwrite:
        print(f"Error: {args.output_file} exists. Use --overwrite to replace it.")
        sys.exit(1)


def main():
    """Main entry point for the CLI application."""
    parser = create_parser()
    args = parser.parse_args()

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

    # Write the transformed text to the output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(final_transformed_text)

    print(f"Transformation complete. TTS-friendly document saved to '{args.output_file}'.")
    print(f"Output file size: {len(final_transformed_text)} characters")


if __name__ == "__main__":
    main()