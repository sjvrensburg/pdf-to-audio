"""Utility functions for PDF to audio conversion."""

import re


def estimate_tokens(text):
    """Estimate the number of tokens in a text (1 token ≈ 0.75 words)."""
    words = text.split()
    return int(len(words) / 0.75)


def split_chunk(content, max_tokens=4000):
    """Split content into sub-chunks at paragraph or sentence boundaries if it exceeds the token limit."""
    sub_chunks = []
    current_chunk = ""
    paragraphs = content.split("\n\n")
    
    for para in paragraphs:
        if estimate_tokens(current_chunk + para) < max_tokens:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                sub_chunks.append(current_chunk.strip())
            
            if estimate_tokens(para) > max_tokens:
                # Split large paragraph by sentences
                sentences = para.split(". ")
                temp_chunk = ""
                for sent in sentences:
                    if estimate_tokens(temp_chunk + sent) < max_tokens:
                        temp_chunk += sent + ". "
                    else:
                        if temp_chunk:
                            sub_chunks.append(temp_chunk.strip())
                        temp_chunk = sent + ". "
                if temp_chunk:
                    sub_chunks.append(temp_chunk.strip())
            else:
                sub_chunks.append(para)
            current_chunk = ""
    
    if current_chunk:
        sub_chunks.append(current_chunk.strip())
    
    return sub_chunks


def post_process_output(text):
    """Clean up minor inconsistencies in the output."""
    # Fix excessive spacing issues
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading spaces on lines
    text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing spaces on lines
    
    # Fix line breaks - ensure proper paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Clean up weird spacing
    
    # Standardize parentheses descriptions
    text = re.sub(r'\bopen parenthesis\b', 'opening parenthesis', text)
    text = re.sub(r'\bclose parenthesis\b', 'closing parenthesis', text)
    
    # Ensure sections have proper spacing
    text = re.sub(r'(\n##\s*\d+)', r'\n\1', text)  # Add line before sections
    text = re.sub(r'(\n###\s*\d+\.\d+)', r'\n\1', text)  # Add line before subsections
    
    return text.strip()


def clean_base64_image(base64_string):
    """Clean base64 image string to remove any duplicate prefixes."""
    if not base64_string:
        return ""
    
    # Handle case where the string might already have the data: prefix
    if base64_string.startswith('data:image/'):
        # Extract just the base64 part after the comma
        comma_index = base64_string.find(',')
        if comma_index != -1:
            base64_string = base64_string[comma_index + 1:]
    
    # Remove any remaining duplicate prefixes that might have been added
    prefixes_to_remove = [
        'data:image/jpeg;base64,',
        'data:image/png;base64,',
        'data:image/gif;base64,',
        'data:image/webp;base64,'
    ]
    
    for prefix in prefixes_to_remove:
        if base64_string.startswith(prefix):
            base64_string = base64_string[len(prefix):]
    
    return base64_string.strip()