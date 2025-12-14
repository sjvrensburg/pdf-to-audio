#!/usr/bin/env python3
"""
Debug script that saves output after EACH pipeline stage to identify where content is lost.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_to_audio.api import process_pdf_to_json, make_api_call
from pdf_to_audio.constants import CORE_TRANSFORM_PROMPT, MATH_PROMPT, CITATIONS_PROMPT, LANGUAGE_STYLE_PROMPT
from pdf_to_audio.utils import split_chunk, post_process_output
from pdf_to_audio.llm_provider import create_llm_provider
from pdf_to_audio.config import load_config, merge_with_args
from tqdm import tqdm


class Args:
    """Mock args object for testing"""
    def __init__(self):
        self.config_file = None
        self.pages_per_chunk = 1
        self.include_images = False
        self.verbose = False


def save_stage_output(stage_name: str, content: str, char_count_original: int):
    """Save stage output and print statistics"""
    filename = f"debug_stage_{stage_name}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    char_count = len(content)
    token_estimate = char_count // 4
    retention = (char_count / char_count_original * 100) if char_count_original > 0 else 0

    print(f"  Saved: {filename}")
    print(f"  Characters: {char_count:,} ({retention:.1f}% of original)")
    print(f"  Estimated tokens: {token_estimate:,}")
    print()


def main():
    pdf_path = "examples/solar-02-00026-v4.pdf"
    api_key = os.environ.get("MISTRAL_API_KEY")

    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set")
        sys.exit(1)

    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    # ========== STAGE 0: OCR ==========
    print("=" * 80)
    print("STAGE 0: OCR Processing")
    print("=" * 80)
    doc = process_pdf_to_json(api_key, pdf_path)

    ocr_text = "\n\n".join(page['markdown'] for page in doc['pages'])
    original_char_count = len(ocr_text)

    save_stage_output("0_ocr", ocr_text, original_char_count)
    print(f"Pages processed: {len(doc['pages'])}\n")

    # Setup config
    args = Args()
    config = load_config(None)
    merged_config = merge_with_args(config, args)

    # Initialize providers
    transform_provider = create_llm_provider(
        provider="mistral",
        model="mistral-small-latest",
        api_key=None,
        temperature=0.2,
        max_tokens=4000,
    )

    math_provider = create_llm_provider(
        provider="mistral",
        model="mistral-small-latest",
        api_key=None,
        temperature=0.2,
        max_tokens=4000,
    )

    citations_provider = create_llm_provider(
        provider="mistral",
        model="mistral-small-latest",
        api_key=None,
        temperature=0.2,
        max_tokens=4000,
    )

    language_provider = create_llm_provider(
        provider="mistral",
        model="mistral-small-latest",
        api_key=None,
        temperature=0.2,
        max_tokens=4000,
    )

    # ========== STAGE 1: Core Text Transformation ==========
    print("=" * 80)
    print("STAGE 1: Core Text Transformation (with chunking)")
    print("=" * 80)

    transformed_chunks = []
    pages = doc['pages']
    pages_per_chunk = 1
    total_chunks = (len(pages) + pages_per_chunk - 1) // pages_per_chunk

    for i in tqdm(range(0, len(pages), pages_per_chunk), desc="Processing chunks", total=total_chunks):
        chunk_pages = pages[i:i + pages_per_chunk]
        modified_markdowns = [page['markdown'] for page in chunk_pages]
        chunk_content = "\n\n".join(modified_markdowns)
        sub_chunks = split_chunk(chunk_content)

        transformed_text = ""
        for sub_chunk in sub_chunks:
            messages = [
                {"role": "system", "content": CORE_TRANSFORM_PROMPT},
                {"role": "user", "content": sub_chunk},
            ]

            try:
                response = make_api_call(transform_provider, messages)
                if response:
                    transformed_text += response + "\n\n"
            except Exception as e:
                print(f"Error processing sub-chunk: {e}")
                continue

        transformed_chunks.append(transformed_text)
        time.sleep(0.5)

    core_transformed_text = "\n\n".join(transformed_chunks)
    core_transformed_text = post_process_output(core_transformed_text)

    save_stage_output("1_core_transform", core_transformed_text, original_char_count)

    # ========== STAGE 2: Math Expression Handling ==========
    print("=" * 80)
    print("STAGE 2: Math Expression Handling (SINGLE API CALL - This is the problem!)")
    print("=" * 80)
    print(f"Input size: {len(core_transformed_text):,} characters ({len(core_transformed_text)//4:,} tokens estimated)")
    print(f"Max output tokens: 4000")
    print()

    messages = [
        {"role": "system", "content": MATH_PROMPT},
        {"role": "user", "content": core_transformed_text},
    ]

    try:
        math_processed_text = make_api_call(math_provider, messages)
        if math_processed_text:
            core_transformed_text = math_processed_text
    except Exception as e:
        print(f"Warning: Math processing failed: {e}")

    save_stage_output("2_math_processed", core_transformed_text, original_char_count)

    # ========== STAGE 3: Citations Optimization ==========
    print("=" * 80)
    print("STAGE 3: Citations Optimization (SINGLE API CALL)")
    print("=" * 80)
    print(f"Input size: {len(core_transformed_text):,} characters ({len(core_transformed_text)//4:,} tokens estimated)")
    print(f"Max output tokens: 4000")
    print()

    messages = [
        {"role": "system", "content": CITATIONS_PROMPT},
        {"role": "user", "content": core_transformed_text},
    ]

    try:
        citations_processed_text = make_api_call(citations_provider, messages)
        if citations_processed_text:
            core_transformed_text = citations_processed_text
    except Exception as e:
        print(f"Warning: Citations processing failed: {e}")

    save_stage_output("3_citations_processed", core_transformed_text, original_char_count)

    # ========== STAGE 4: Language/Style Refinement ==========
    print("=" * 80)
    print("STAGE 4: Language/Style Refinement (SINGLE API CALL)")
    print("=" * 80)
    print(f"Input size: {len(core_transformed_text):,} characters ({len(core_transformed_text)//4:,} tokens estimated)")
    print(f"Max output tokens: 4000")
    print()

    messages = [
        {"role": "system", "content": LANGUAGE_STYLE_PROMPT},
        {"role": "user", "content": core_transformed_text},
    ]

    try:
        language_processed_text = make_api_call(language_provider, messages)
        if language_processed_text:
            core_transformed_text = language_processed_text
    except Exception as e:
        print(f"Warning: Language processing failed: {e}")

    final_text = post_process_output(core_transformed_text)
    save_stage_output("4_final", final_text, original_char_count)

    # ========== Summary ==========
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original OCR: {original_char_count:,} chars")
    print(f"Final output: {len(final_text):,} chars")
    print(f"Content loss: {original_char_count - len(final_text):,} chars ({(1 - len(final_text)/original_char_count)*100:.1f}%)")
    print("\nThe problem: Stages 2-4 use single API calls with 4000 token output limits.")
    print("For long documents, the LLM cannot fit the entire output within 4000 tokens.")
    print("Solution: Implement chunking for stages 2-4 similar to stage 1.")


if __name__ == "__main__":
    main()
