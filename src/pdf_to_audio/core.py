"""Core processing functions for PDF to audio conversion."""

import os
import re
import time
import logging
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import torch
import torchaudio
from tqdm import tqdm

from .api import make_api_call
from .constants import SYSTEM_PROMPT, DEFAULT_MATH_TTS_SCALE
from .image import process_page
from .utils import split_chunk, post_process_output
from .audio.chunking import TextChunker, MATH_TAG_PATTERN
from .audio.concatenation import AudioConcatenator
from .audio.formats import AudioFormatHandler
from .tts.chatterbox_tts import ChatterboxTTSEngine, DEFAULT_TTS_SETTINGS, ACADEMIC_TTS_SETTINGS, MATH_HEAVY_SETTINGS
from .tts.audio_processing import AudioProcessor
from .config import load_config, merge_with_args, get_math_tts_settings
from .refinement.pipeline import RefinementPipeline
from .refinement.base import RefinementConfig

logger = logging.getLogger(__name__)


@contextmanager
def temp_directory(base_dir=None):
    """
    Context manager for creating and cleaning up a temporary directory.
    
    Args:
        base_dir: Optional base directory to create the temporary directory in.
                 If None, the system's default temporary directory is used.
                 
    Yields:
        The path to the temporary directory.
    """
    temp_dir = tempfile.mkdtemp(prefix="pdf_to_audio_", dir=base_dir)
    try:
        logger.info(f"Created temporary directory: {temp_dir}")
        yield temp_dir
    finally:
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")


def process_document(client, doc, args):
    """Process the entire document and return the transformed text."""
    # Load configuration
    config = load_config(args.config_file if hasattr(args, 'config_file') else None)
    merged_config = merge_with_args(config, args)
    
    # Get system prompt (custom or default)
    system_prompt = merged_config["mistral"]["system_prompt"] or SYSTEM_PROMPT
    
    transformed_chunks = []
    pages = doc['pages']
    pages_per_chunk = merged_config["general"]["pages_per_chunk"]
    total_chunks = (len(pages) + pages_per_chunk - 1) // pages_per_chunk
    
    print(f"Processing {len(pages)} pages in {total_chunks} chunks...")
    
    for i in tqdm(range(0, len(pages), pages_per_chunk), desc="Processing chunks", total=total_chunks):
        if merged_config["general"]["verbose"]:
            print(f"\nProcessing chunk {i//pages_per_chunk + 1} of {total_chunks}")
        
        chunk_pages = pages[i:i + pages_per_chunk]
        modified_markdowns = []
        
        for page_idx, page in enumerate(chunk_pages):
            if merged_config["general"]["verbose"]:
                print(f"Processing page {i + page_idx + 1}")
            
            if merged_config["general"]["include_images"]:
                modified_markdown = process_page(page, client, merged_config["mistral"]["image_model"])
            else:
                modified_markdown = page['markdown']
            modified_markdowns.append(modified_markdown)
        
        chunk_content = "\n\n".join(modified_markdowns)
        sub_chunks = split_chunk(chunk_content)
        
        if merged_config["general"]["verbose"]:
            print(f"Split into {len(sub_chunks)} sub-chunks")
        
        transformed_text = ""
        for sub_chunk_idx, sub_chunk in enumerate(sub_chunks):
            if merged_config["general"]["verbose"]:
                print(f"Processing sub-chunk {sub_chunk_idx + 1}/{len(sub_chunks)}")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sub_chunk},
            ]
            
            try:
                response = make_api_call(client, merged_config["mistral"]["text_model"], messages)
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
    
    # Apply multi-pass refinement if enabled
    if merged_config["refinement"]["enable_refinement"]:
        print("Applying multi-pass refinement...")
        
        # Create refinement config from merged config
        refinement_config = RefinementConfig(
            enable_math_refinement=merged_config["refinement"]["enable_math_refinement"],
            enable_structure_citation_optimization=merged_config["refinement"]["enable_structure_citation_optimization"],
            enable_language_style_refinement=merged_config["refinement"]["enable_language_style_refinement"],
            enable_audio_specific_optimization=merged_config["refinement"]["enable_audio_specific_optimization"],
            math_refinement_intensity=merged_config["refinement"]["math_refinement_intensity"],
            structure_citation_intensity=merged_config["refinement"]["structure_citation_intensity"],
            language_style_intensity=merged_config["refinement"]["language_style_intensity"],
            audio_specific_intensity=merged_config["refinement"]["audio_specific_intensity"],
            target_audience=merged_config["refinement"]["target_audience"],
            fallback_on_error=merged_config["refinement"]["fallback_on_error"]
        )
        
        # Initialize and run the refinement pipeline
        pipeline = RefinementPipeline(config=refinement_config)
        refined_text = pipeline.refine(final_transformed_text, client, merged_config)
        
        # Post-process the refined output again to ensure consistency
        final_transformed_text = post_process_output(refined_text)
        
        print("Multi-pass refinement complete")
    else:
        print("Multi-pass refinement is disabled, skipping")
    
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
    
    # Load configuration
    config = load_config(args.config_file if hasattr(args, 'config_file') else None)
    merged_config = merge_with_args(config, args)
    
    # Get temp directory from config or use system default
    temp_dir_base = merged_config["general"]["temp_dir"]
    
    # Use a context manager for the temporary directory
    with temp_directory(temp_dir_base) as temp_dir:
        # Initialize TTS engine
        try:
            # Check if force_cpu option is set
            force_cpu = merged_config["general"]["force_cpu"]
            tts_engine = ChatterboxTTSEngine(force_cpu=force_cpu)
            if tts_engine.device == "cpu" and not force_cpu:
                print("Note: Using CPU for audio generation. This may be slower than GPU acceleration.")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            print(f"Error initializing TTS engine. Try using --force_cpu option if you're having GPU issues.")
            raise
            
        # Initialize text chunker
        chunker = TextChunker(
            strategy=merged_config["tts"]["chunk_strategy"],
            max_duration_sec=35  # Safe margin below Chatterbox's 40-second limit
        )
        
        # Initialize audio concatenator
        concatenator = AudioConcatenator(sample_rate=tts_engine.sample_rate)
        
        # Initialize audio processor
        audio_processor = AudioProcessor(sample_rate=tts_engine.sample_rate)
        
        # Initialize format handler
        format_handler = AudioFormatHandler()
        
        # Determine base TTS settings based on content
        if "math" in text.lower() or "equation" in text.lower():
            base_tts_settings = MATH_HEAVY_SETTINGS.copy()
        elif any(academic_term in text.lower() for academic_term in 
                ["abstract", "introduction", "methodology", "conclusion", "references"]):
            base_tts_settings = ACADEMIC_TTS_SETTINGS.copy()
        else:
            base_tts_settings = DEFAULT_TTS_SETTINGS.copy()
        
        # Override with user-provided settings
        tts_settings = base_tts_settings.copy()
        if merged_config["tts"]["exaggeration"] is not None:
            tts_settings["exaggeration"] = merged_config["tts"]["exaggeration"]
        if merged_config["tts"]["cfg_weight"] is not None:
            tts_settings["cfg_weight"] = merged_config["tts"]["cfg_weight"]
        
        # Add voice cloning audio prompt path if provided
        if merged_config["tts"]["audio_prompt_path"] is not None:
            tts_settings["audio_prompt_path"] = merged_config["tts"]["audio_prompt_path"]
            print(f"Voice cloning enabled with reference audio: {merged_config['tts']['audio_prompt_path']}")
        
        # Get math-specific TTS settings
        math_tts_settings = get_math_tts_settings(merged_config)
        
        # Add voice cloning to math settings as well
        if merged_config["tts"]["audio_prompt_path"] is not None:
            math_tts_settings["audio_prompt_path"] = merged_config["tts"]["audio_prompt_path"]
        
        # Chunk the text
        print("Chunking text for audio generation...")
        text_chunks = chunker.chunk_text(text)
        print(f"Text chunked into {len(text_chunks)} segments")
        
        # Generate audio for each chunk
        print("Generating audio...")
        temp_audio_files = []
        
        for i, chunk in enumerate(tqdm(text_chunks, desc="Generating audio chunks")):
            try:
                # Check if the chunk contains math content
                contains_math = bool(re.search(MATH_TAG_PATTERN, chunk))
                
                # Process math tags in the chunk
                if contains_math:
                    # Extract math content and non-math content
                    math_segments = re.findall(MATH_TAG_PATTERN, chunk)
                    non_math_segments = re.split(MATH_TAG_PATTERN, chunk)
                    
                    # Generate audio for each segment with appropriate settings
                    segment_audios = []
                    
                    # Process non-math segments first (they might be empty)
                    for j, segment in enumerate(non_math_segments):
                        if segment.strip():
                            # Generate audio with regular settings
                            segment_audio, segment_sr = tts_engine.generate_audio(
                                segment.strip(),
                                settings=tts_settings
                            )
                            segment_audios.append(segment_audio)
                        
                        # Add math segment if available
                        if j < len(math_segments):
                            # Extract content from math tags
                            math_content = math_segments[j].replace("<MATH>", "").replace("</MATH>", "").strip()
                            if math_content:
                                # Generate audio with math-specific settings
                                math_audio, math_sr = tts_engine.generate_audio(
                                    math_content,
                                    settings=math_tts_settings
                                )
                                segment_audios.append(math_audio)
                    
                    # Concatenate all segment audios
                    if segment_audios:
                        # Concatenate along the time dimension (dim=1)
                        audio = torch.cat(segment_audios, dim=1)
                        sample_rate = tts_engine.sample_rate
                    else:
                        # Fallback if no segments were processed
                        audio, sample_rate = tts_engine.generate_audio(
                            chunk.replace("<MATH>", "").replace("</MATH>", ""),
                            settings=tts_settings
                        )
                else:
                    # No math content, use regular settings
                    audio, sample_rate = tts_engine.generate_audio(
                        chunk,
                        settings=tts_settings
                    )
                
                # Apply audio processing
                audio = audio_processor.normalize_volume(audio)
                
                # Add a pause at the end of each chunk
                audio = audio_processor.add_pause(audio)
                
                # Save to temporary file in the temp directory
                temp_file = os.path.join(temp_dir, f"audio_chunk_{i}.wav")
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
        temp_concatenated = os.path.join(temp_dir, "concatenated.wav")
        concatenator.concatenate_files(temp_audio_files, temp_concatenated)
        
        # Apply global volume normalization if enabled
        if merged_config["tts"].get("global_normalization", False):
            print("Applying global volume normalization...")
            # Load the concatenated audio
            waveform, sample_rate = torchaudio.load(temp_concatenated)
            # Apply global normalization
            normalized_waveform = audio_processor.normalize_volume(waveform)
            # Save the normalized audio
            torchaudio.save(temp_concatenated, normalized_waveform, sample_rate)
            
        # Save the text content alongside the audio
        if hasattr(args, 'output_audio') and args.output_audio:
            text_output_path = os.path.splitext(args.output_audio)[0] + ".txt"
            try:
                with open(text_output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Text content saved to: {text_output_path}")
            except Exception as e:
                logger.error(f"Error saving text content: {e}")
                print(f"Error saving text content: {e}")
        
        # Convert to desired format if not WAV
        if merged_config["tts"]["audio_format"].lower() != 'wav':
            print(f"Converting to {merged_config['tts']['audio_format']} format...")
            output_path = format_handler.convert_format(
                temp_concatenated, 
                merged_config["tts"]["audio_format"], 
                args.output_audio
            )
        else:
            # If WAV or no format specified, just use the concatenated file
            if hasattr(args, 'output_audio') and args.output_audio:
                # Ensure output directory exists
                os.makedirs(os.path.dirname(os.path.abspath(args.output_audio)), exist_ok=True)
                # Copy the file
                shutil.copy2(temp_concatenated, args.output_audio)
                output_path = args.output_audio
            else:
                # Create a copy in the current directory before the temp dir is deleted
                output_path = "output.wav"
                shutil.copy2(temp_concatenated, output_path)
                # Also save the text content
                with open("output.txt", 'w', encoding='utf-8') as f:
                    f.write(text)
        
        print(f"Audio generation complete: {output_path}")
        return output_path