#!/usr/bin/env python3
"""
Debug script to test each stage of the pipeline and save intermediate outputs.
This helps identify where content is being lost in long documents.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_to_audio.api import process_pdf_to_json
from pdf_to_audio.core import process_document

class Args:
    """Mock args object for testing"""
    def __init__(self):
        self.config_file = None
        self.pages_per_chunk = 1
        self.include_images = False
        self.verbose = True
        self.enable_math_refinement = True
        self.enable_citations_refinement = True
        self.enable_language_refinement = True


def main():
    pdf_path = "examples/solar-02-00026-v4.pdf"
    api_key = os.environ.get("MISTRAL_API_KEY")

    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set")
        sys.exit(1)

    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print("=" * 80)
    print("STAGE 0: OCR Processing")
    print("=" * 80)
    doc = process_pdf_to_json(api_key, pdf_path)

    # Save raw OCR output
    with open("debug_stage0_ocr_raw.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(doc['pages']):
            f.write(f"\n{'='*60}\nPAGE {i+1}\n{'='*60}\n")
            f.write(page['markdown'])

    print(f"Saved raw OCR output to debug_stage0_ocr_raw.txt")
    print(f"Total pages: {len(doc['pages'])}")

    # Calculate total character count
    total_chars = sum(len(page['markdown']) for page in doc['pages'])
    print(f"Total characters from OCR: {total_chars:,}")
    print(f"Estimated tokens (chars/4): {total_chars//4:,}")

    print("\n" + "=" * 80)
    print("Processing through full pipeline...")
    print("=" * 80)

    args = Args()
    final_text = process_document(doc, args, api_key)

    # Save final output
    with open("debug_final_output.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"\nSaved final output to debug_final_output.txt")
    print(f"Final character count: {len(final_text):,}")
    print(f"Content retention: {len(final_text)/total_chars*100:.1f}%")

    # Compare with existing output
    existing_output_path = "examples/solar-02-0026-v4.txt"
    if Path(existing_output_path).exists():
        with open(existing_output_path, "r", encoding="utf-8") as f:
            existing_output = f.read()
        print(f"Existing output character count: {len(existing_output):,}")
        print(f"Existing retention: {len(existing_output)/total_chars*100:.1f}%")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print("\nFiles created:")
    print("  - debug_stage0_ocr_raw.txt (raw OCR output)")
    print("  - debug_final_output.txt (final processed text)")


if __name__ == "__main__":
    main()
