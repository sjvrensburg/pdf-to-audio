#!/usr/bin/env python3
"""
Debug script to trace the transformation pipeline and identify where quality is lost.
"""

import os
import argparse
from pdf_to_audio.api import process_pdf_to_json
from pdf_to_audio.core import process_document
from pdf_to_audio.llm_provider import create_llm_provider
from pdf_to_audio.constants import CORE_TRANSFORM_PROMPT, MATH_PROMPT, CITATIONS_PROMPT, LANGUAGE_STYLE_PROMPT
from pdf_to_audio.config import load_config, merge_with_args

def debug_pipeline():
    """Debug the transformation pipeline step by step."""
    
    api_key = os.environ.get('MISTRAL_API_KEY')
    if not api_key:
        print('MISTRAL_API_KEY not set')
        return
    
    # Load configuration
    config = load_config(None)
    
    # Create args
    args = argparse.Namespace()
    args.pages_per_chunk = 1
    args.include_images = False
    args.config_file = None
    args.verbose = True
    
    merged_config = merge_with_args(config, args)
    
    # Process OCR
    print("=== STAGE 0: OCR OUTPUT ===")
    doc = process_pdf_to_json(api_key, 'examples/solar-02-00026-v4-page4.pdf')
    raw_text = doc['pages'][0]['markdown']
    print(f"Raw OCR length: {len(raw_text)} characters")
    print(raw_text[:500])
    print("...")
    
    # Initialize LLM providers
    transform_provider = create_llm_provider(
        provider=merged_config["llm"].get("transform_provider", "mistral"),
        model=merged_config["llm"].get("transform_model", "mistral-small-latest"),
        api_key=api_key,
        temperature=merged_config["llm"].get("temperature", 0.2),
        max_tokens=merged_config["llm"].get("max_tokens", 4000),
    )
    
    # Stage 1: Core Transform
    print("\n=== STAGE 1: CORE TRANSFORM ===")
    messages = [
        {"role": "system", "content": CORE_TRANSFORM_PROMPT},
        {"role": "user", "content": raw_text},
    ]
    
    try:
        stage1_result = transform_provider.chat_complete(messages)
        print(f"Stage 1 output length: {len(stage1_result)} characters")
        print(stage1_result[:500])
        print("...")
        
        # Save intermediate results
        with open('debug_stage1.txt', 'w') as f:
            f.write(stage1_result)
            
    except Exception as e:
        print(f"Stage 1 error: {e}")
        return
    
    # Stage 2: Math Processing
    print("\n=== STAGE 2: MATH PROCESSING ===")
    math_provider = create_llm_provider(
        provider=merged_config["llm"].get("math_provider", "mistral"),
        model=merged_config["llm"].get("math_model", "mistral-small-latest"),
        api_key=api_key,
        temperature=merged_config["llm"].get("temperature", 0.2),
        max_tokens=merged_config["llm"].get("max_tokens", 4000),
    )
    
    messages = [
        {"role": "system", "content": MATH_PROMPT},
        {"role": "user", "content": stage1_result},
    ]
    
    try:
        stage2_result = math_provider.chat_complete(messages)
        print(f"Stage 2 output length: {len(stage2_result)} characters")
        print(stage2_result[:500])
        print("...")
        
        with open('debug_stage2.txt', 'w') as f:
            f.write(stage2_result)
            
    except Exception as e:
        print(f"Stage 2 error: {e}")
        return
    
    # Stage 3: Citations Processing
    print("\n=== STAGE 3: CITATIONS PROCESSING ===")
    citations_provider = create_llm_provider(
        provider=merged_config["llm"].get("citations_provider", "mistral"),
        model=merged_config["llm"].get("citations_model", "mistral-small-latest"),
        api_key=api_key,
        temperature=merged_config["llm"].get("temperature", 0.2),
        max_tokens=merged_config["llm"].get("max_tokens", 4000),
    )
    
    messages = [
        {"role": "system", "content": CITATIONS_PROMPT},
        {"role": "user", "content": stage2_result},
    ]
    
    try:
        stage3_result = citations_provider.chat_complete(messages)
        print(f"Stage 3 output length: {len(stage3_result)} characters")
        print(stage3_result[:500])
        print("...")
        
        with open('debug_stage3.txt', 'w') as f:
            f.write(stage3_result)
            
    except Exception as e:
        print(f"Stage 3 error: {e}")
        return
    
    # Stage 4: Language Processing
    print("\n=== STAGE 4: LANGUAGE PROCESSING ===")
    language_provider = create_llm_provider(
        provider=merged_config["llm"].get("language_provider", "mistral"),
        model=merged_config["llm"].get("language_model", "mistral-small-latest"),
        api_key=api_key,
        temperature=merged_config["llm"].get("temperature", 0.2),
        max_tokens=merged_config["llm"].get("max_tokens", 4000),
    )
    
    messages = [
        {"role": "system", "content": LANGUAGE_STYLE_PROMPT},
        {"role": "user", "content": stage3_result},
    ]
    
    try:
        stage4_result = language_provider.chat_complete(messages)
        print(f"Stage 4 output length: {len(stage4_result)} characters")
        print(stage4_result[:500])
        print("...")
        
        with open('debug_stage4.txt', 'w') as f:
            f.write(stage4_result)
            
    except Exception as e:
        print(f"Stage 4 error: {e}")
        return
    
    print("\n=== COMPARISON ===")
    print(f"Original OCR: {len(raw_text)} chars")
    print(f"After Stage 1: {len(stage1_result)} chars")
    print(f"After Stage 2: {len(stage2_result)} chars")
    print(f"After Stage 3: {len(stage3_result)} chars")
    print(f"After Stage 4: {len(stage4_result)} chars")

if __name__ == "__main__":
    debug_pipeline()