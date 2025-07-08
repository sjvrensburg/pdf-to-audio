"""Core processing functions for PDF to audio conversion."""

import time

from tqdm import tqdm

from .api import make_api_call
from .constants import SYSTEM_PROMPT
from .image import process_page
from .utils import split_chunk, post_process_output


def process_document(client, doc, args):
    """Process the entire document and return the transformed text."""
    transformed_chunks = []
    pages = doc['pages']
    total_chunks = (len(pages) + args.pages_per_chunk - 1) // args.pages_per_chunk
    
    print(f"Processing {len(pages)} pages in {total_chunks} chunks...")
    
    for i in tqdm(range(0, len(pages), args.pages_per_chunk), desc="Processing chunks", total=total_chunks):
        if args.verbose:
            print(f"\nProcessing chunk {i//args.pages_per_chunk + 1} of {total_chunks}")
        
        chunk_pages = pages[i:i + args.pages_per_chunk]
        modified_markdowns = []
        
        for page_idx, page in enumerate(chunk_pages):
            if args.verbose:
                print(f"Processing page {i + page_idx + 1}")
            
            if args.include_images:
                modified_markdown = process_page(page, client, args.image_model)
            else:
                modified_markdown = page['markdown']
            modified_markdowns.append(modified_markdown)
        
        chunk_content = "\n\n".join(modified_markdowns)
        sub_chunks = split_chunk(chunk_content)
        
        if args.verbose:
            print(f"Split into {len(sub_chunks)} sub-chunks")
        
        transformed_text = ""
        for sub_chunk_idx, sub_chunk in enumerate(sub_chunks):
            if args.verbose:
                print(f"Processing sub-chunk {sub_chunk_idx + 1}/{len(sub_chunks)}")
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sub_chunk},
            ]
            
            try:
                response = make_api_call(client, args.text_model, messages)
                if response and response.choices:
                    transformed_text += response.choices[0].message.content + "\n\n"
            except Exception as e:
                print(f"Error processing sub-chunk: {e}")
                # Continue with remaining chunks
                continue
        
        transformed_chunks.append(transformed_text)
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.5)

    # Combine all transformed chunks into a single text
    final_transformed_text = "\n\n".join(transformed_chunks)
    
    # Post-process the output
    final_transformed_text = post_process_output(final_transformed_text)
    
    return final_transformed_text