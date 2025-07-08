"""Core processing functions for PDF to audio conversion."""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio
from tqdm import tqdm

from .api import make_api_call
from .constants import SYSTEM_PROMPT
from .image import process_page
from .utils import split_chunk, post_process_output
from .audio.chunking import TextChunker
from .audio.concatenation import AudioConcatenator
from .audio.formats import AudioFormatHandler
from .tts.chatterbox_tts import ChatterboxTTSEngine, DEFAULT_TTS_SETTINGS, ACADEMIC_TTS_SETTINGS, MATH_HEAVY_SETTINGS
from .tts.audio_processing import AudioProcessor

logger = logging.getLogger(__name__)


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


def generate_audio(text, args):
    """
    Generate audio from text using the Chatterbox TTS engine.
    
    Args:
        text: The text to convert to speech.
        args: Command-line arguments containing TTS settings.
        
    Returns:
        Path to the generated audio file.
    """
    if not text:
        logger.warning("No text provided for audio generation")
        return None
        
    # Initialize TTS engine
    try:
        tts_engine = ChatterboxTTSEngine()
    except Exception as e:
        logger.error(f"Failed to initialize TTS engine: {e}")
        raise
        
    # Initialize text chunker
    chunker = TextChunker(
        strategy=args.chunk_strategy,
        max_duration_sec=35  # Safe margin below Chatterbox's 40-second limit
    )
    
    # Initialize audio concatenator
    concatenator = AudioConcatenator(sample_rate=tts_engine.sample_rate)
    
    # Initialize audio processor
    audio_processor = AudioProcessor(sample_rate=tts_engine.sample_rate)
    
    # Initialize format handler
    format_handler = AudioFormatHandler()
    
    # Determine TTS settings based on content
    if "math" in text.lower() or "equation" in text.lower():
        tts_settings = MATH_HEAVY_SETTINGS.copy()
    elif any(academic_term in text.lower() for academic_term in 
             ["abstract", "introduction", "methodology", "conclusion", "references"]):
        tts_settings = ACADEMIC_TTS_SETTINGS.copy()
    else:
        tts_settings = DEFAULT_TTS_SETTINGS.copy()
    
    # Override with user-provided settings
    if hasattr(args, 'exaggeration') and args.exaggeration is not None:
        tts_settings["exaggeration"] = args.exaggeration
    if hasattr(args, 'cfg_weight') and args.cfg_weight is not None:
        tts_settings["cfg_weight"] = args.cfg_weight
        
    # Chunk the text
    print("Chunking text for audio generation...")
    text_chunks = chunker.chunk_text(text)
    print(f"Text chunked into {len(text_chunks)} segments")
    
    # Generate audio for each chunk
    print("Generating audio...")
    temp_audio_files = []
    
    for i, chunk in enumerate(tqdm(text_chunks, desc="Generating audio chunks")):
        try:
            # Generate audio
            audio, sample_rate = tts_engine.generate_audio(
                chunk, 
                voice_path=args.voice if hasattr(args, 'voice') else None,
                settings=tts_settings
            )
            
            # Apply audio processing
            audio = audio_processor.normalize_volume(audio)
            
            # Add a pause at the end of each chunk
            audio = audio_processor.add_pause(audio)
            
            # Save to temporary file
            temp_file = f"temp_audio_chunk_{i}.wav"
            tts_engine.save_audio(audio, temp_file, sample_rate)
            temp_audio_files.append(temp_file)
            
            # Clear GPU cache if using CUDA
            tts_engine.clear_cache()
            
        except Exception as e:
            logger.error(f"Error generating audio for chunk {i}: {e}")
            print(f"Error generating audio for chunk {i}: {e}")
            continue
    
    if not temp_audio_files:
        logger.error("No audio chunks were successfully generated")
        return None
        
    # Concatenate audio chunks
    print("Concatenating audio chunks...")
    temp_concatenated = "temp_concatenated.wav"
    concatenator.concatenate_files(temp_audio_files, temp_concatenated)
    
    # Convert to desired format if not WAV
    if hasattr(args, 'audio_format') and args.audio_format.lower() != 'wav':
        print(f"Converting to {args.audio_format} format...")
        output_path = format_handler.convert_format(
            temp_concatenated, 
            args.audio_format, 
            args.output_audio
        )
    else:
        # If WAV or no format specified, just use the concatenated file
        if hasattr(args, 'output_audio') and args.output_audio:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(args.output_audio)), exist_ok=True)
            # Copy the file
            import shutil
            shutil.copy2(temp_concatenated, args.output_audio)
            output_path = args.output_audio
        else:
            output_path = temp_concatenated
    
    # Clean up temporary files
    for temp_file in temp_audio_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    if os.path.exists(temp_concatenated) and temp_concatenated != output_path:
        os.remove(temp_concatenated)
        
    print(f"Audio generation complete: {output_path}")
    return output_path