import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# Constants
MAX_TOKENS = 4000  # Adjust based on model's token limit
MAX_RETRIES = 3
RETRY_WAIT_MIN = 4  # seconds
RETRY_WAIT_MAX = 10  # seconds

# Enhanced system prompt for TTS transformation
SYSTEM_PROMPT = r"""
You are an AI assistant specialized in converting academic papers into a text-to-speech (TTS) friendly format. Your task is to transform the provided content, with a particular focus on converting mathematical notation into clear, spoken language. The transformation should preserve the academic tone, technical precision, and overall structure of the document. The output should be easily comprehensible when read aloud.

**Guidelines for Mathematical Notation:**

- **Basic Operations:**
  - \( a + b \) → "a plus b"
  - \( a - b \) → "a minus b"
  - \( a \times b \) or \( a \cdot b \) → "a times b"
  - \( a / b \) → "a divided by b"

- **Equations:** Convert equations into natural language:
  - \( E = mc^2 \) → "E equals m c squared"
  - \( a^2 + b^2 = c^2 \) → "a squared plus b squared equals c squared"

- **Fractions:** Use "over" for fractions:
  - \( \frac{a}{b} \) → "a over b"
  - \( \frac{1}{2} \) → "one half"
  - \( \frac{d^2y}{dx^2} \) → "d squared y over d x squared"

- **Powers and Exponents:**
  - \( x^2 \) → "x squared"
  - \( x^3 \) → "x cubed"
  - \( x^n \) → "x to the power of n"
  - \( e^{-x} \) → "e to the power of negative x"

- **Roots:**
  - \( \sqrt{x} \) → "the square root of x"
  - \( \sqrt[n]{x} \) → "the nth root of x"

- **Integrals:** Describe integrals fully:
  - \( \int f(x) dx \) → "the integral of f of x with respect to x"
  - \( \int_{a}^{b} f(x) dx \) → "the integral from a to b of f of x with respect to x"
  - \( \iint_D f(x,y) dx dy \) → "the double integral over region D of f of x comma y with respect to x and y"

- **Summations and Products:**
  - \( \sum_{i=1}^{n} a_i \) → "the sum from i equals 1 to n of a sub i"
  - \( \prod_{i=1}^{n} a_i \) → "the product from i equals 1 to n of a sub i"

- **Limits:**
  - \( \lim_{x \to a} f(x) \) → "the limit as x approaches a of f of x"
  - \( \lim_{n \to \infty} \) → "the limit as n approaches infinity"

- **Derivatives:**
  - \( f'(x) \) → "f prime of x"
  - \( \frac{df}{dx} \) → "d f over d x"
  - \( \frac{\partial f}{\partial x} \) → "the partial derivative of f with respect to x"

- **Set Notation:**
  - \( \{x \in \mathbb{R} : x > 0\} \) → "the set of x in the real numbers such that x is greater than 0"
  - \( x \in A \) → "x is an element of A"
  - \( A \cup B \) → "A union B"
  - \( A \cap B \) → "A intersection B"

- **Greek Letters:** Spell out Greek letters:
  - \( \alpha \) → "alpha"
  - \( \beta \) → "beta"
  - \( \gamma \) → "gamma"
  - \( \delta \) → "delta"
  - \( \epsilon \) → "epsilon"
  - \( \theta \) → "theta"
  - \( \lambda \) → "lambda"
  - \( \mu \) → "mu"
  - \( \nu \) → "nu"
  - \( \pi \) → "pi"
  - \( \rho \) → "rho"
  - \( \sigma \) → "sigma"
  - \( \tau \) → "tau"
  - \( \phi \) → "phi"
  - \( \chi \) → "chi"
  - \( \psi \) → "psi"
  - \( \omega \) → "omega"

- **Comparison Operators:**
  - \( x < y \) → "x is less than y"
  - \( x \leq y \) → "x is less than or equal to y"
  - \( x > y \) → "x is greater than y"
  - \( x \geq y \) → "x is greater than or equal to y"
  - \( x \neq y \) → "x is not equal to y"
  - \( x \approx y \) → "x is approximately equal to y"

- **Subscripts and Superscripts:**
  - \( x_i \) → "x sub i"
  - \( a_{ij} \) → "a sub i j"
  - \( p_s(x, y) \) → "p sub s of x and y"
  - \( p_t(y | x) \) → "p sub t of y given x"

- **Matrices and Vectors:**
  - Describe matrix dimensions: "a 3 by 3 matrix"
  - \( \mathbf{v} \) → "vector v"
  - \( \det(A) \) → "the determinant of A"
  - \( A^T \) → "A transpose"

- **Special Functions:**
  - \( \sin(x) \) → "sine of x"
  - \( \cos(x) \) → "cosine of x"
  - \( \tan(x) \) → "tangent of x"
  - \( \ln(x) \) → "natural log of x"
  - \( \log(x) \) → "log of x"
  - \( \exp(x) \) → "exponential of x"

**Document Structure:**
- Preserve section headers as "Section [number]: [title]"
- Maintain paragraph breaks and structure
- For numbered equations, say "Equation [number]:"
- For figures, say "Figure [number]: [caption]"
- For tables, say "Table [number]: [caption]"
- Ensure proper spacing and formatting for readability

**Table Handling:**
- Announce tables as "Table [number]: [caption]"
- For each row, clearly describe the content
- Make table content clear and structured for audio consumption

**General Instructions:**
- Use clear, unambiguous language
- Maintain technical precision
- Keep the academic tone
- Preserve proper formatting with consistent line breaks
- Avoid overly complex nested descriptions
- When in doubt, err on the side of clarity over brevity
- For complex expressions, break them into smaller parts if needed
- Maintain consistent spacing between sections and paragraphs

Apply these transformations to convert all mathematical notation in the provided content into speech-friendly text.
"""

def estimate_tokens(text):
    """Estimate the number of tokens in a text (1 token ≈ 0.75 words)."""
    words = text.split()
    return int(len(words) / 0.75)

def split_chunk(content, max_tokens=MAX_TOKENS):
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

@retry(
    stop=stop_after_attempt(MAX_RETRIES), 
    wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
)
def make_api_call(client, model, messages):
    """Make an API call with retry logic for transient errors."""
    try:
        return client.chat.complete(model=model, messages=messages)
    except requests.exceptions.Timeout:
        print("API call timed out. Retrying...")
        raise
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"Rate limit exceeded for model {model}. This might be a capacity issue.")
            print("Consider trying a different model or waiting before retrying.")
        else:
            print(f"HTTP error {e.response.status_code}: {e.response.text}")
        raise
    except Exception as e:
        # Handle SDKError and other exceptions
        if "429" in str(e) or "capacity exceeded" in str(e).lower():
            print(f"Capacity exceeded for model {model}.")
            print("This is likely due to high demand. Try:")
            print("1. Using a different model (e.g., pixtral-12b-latest)")
            print("2. Waiting a few minutes and trying again")
            print("3. Upgrading your Mistral service tier")
            raise
        elif "HTTPValidationError" in str(e):
            print(f"Validation error with model {model}: {e}")
            raise
        print(f"API error: {e}")
        raise

def process_pdf_to_json(client, pdf_path):
    """Convert a PDF file to JSON using Mistral's OCR API with error handling."""
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Upload the file to Mistral API
        uploaded_file = client.files.upload(
            file={
                "file_name": pdf_file.stem,
                "content": pdf_file.read_bytes(),
            },
            purpose="ocr",
        )
    except Exception as e:
        print(f"Error uploading file: {e}")
        sys.exit(1)

    try:
        # Process the PDF using OCR
        pdf_response = client.ocr.process(
            document={
                "type": "file",
                "file_id": uploaded_file.id,
            },
            model="mistral-ocr-latest",
            include_image_base64=True,
        )
    except Exception as e:
        print(f"Error processing OCR: {e}")
        sys.exit(1)

    # Convert the OCR response to a dictionary
    response_dict = json.loads(pdf_response.model_dump_json())
    if 'pages' not in response_dict:
        print("Error: OCR response does not contain 'pages'.")
        sys.exit(1)
    return response_dict

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

def process_image(client, base64_image, model):
    """Describe an image using Mistral's vision capabilities."""
    # Clean the base64 string to avoid double prefixes
    clean_base64 = clean_base64_image(base64_image)
    
    # Validate that we have actual base64 content
    if not clean_base64:
        return "Image could not be processed - no valid base64 data"
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Describe this mathematical diagram or figure in detail for text-to-speech conversion. Focus on what the image shows, any mathematical content, axes, curves, labels, or data presented. Be specific about mathematical elements like equations, graphs, charts, or diagrams. Keep the description clear and suitable for audio conversion."
                },
                {
                    "type": "image_url", 
                    "image_url": f"data:image/jpeg;base64,{clean_base64}"
                }
            ]
        }
    ]
    
    try:
        response = make_api_call(client, model, messages)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Image could not be processed due to an error"

def replace_image_references(markdown, image_descriptions):
    """Replace image references using string replacement instead of regex."""
    # Create a mapping of image IDs to descriptions
    for image_id, description in image_descriptions.items():
        # Use simple string replacement to avoid regex issues
        # Look for the markdown image pattern
        patterns_to_try = [
            f"![{image_id}]({image_id})",
            f"![]({image_id})",
            f"![img]({image_id})",
            f"![image]({image_id})",
            f"![figure]({image_id})",
            f"![chart]({image_id})",
            f"![diagram]({image_id})"
        ]
        
        replacement = f"\n\nImage description: {description}\n\n"
        
        # Try each pattern for replacement
        for pattern in patterns_to_try:
            if pattern in markdown:
                markdown = markdown.replace(pattern, replacement)
                break
        else:
            # If no standard pattern found, try a more general approach
            # Look for any markdown image syntax containing the image_id
            lines = markdown.split('\n')
            for i, line in enumerate(lines):
                if f']({image_id})' in line and line.strip().startswith('!['):
                    lines[i] = replacement.strip()
                    break
            markdown = '\n'.join(lines)
    
    return markdown

def process_page(page, client, image_model):
    """Replace image references in markdown with their descriptions."""
    markdown = page['markdown']
    if 'images' not in page or not page['images']:
        return markdown
    
    # Process all images and create descriptions
    image_descriptions = {}
    
    for image in page['images']:
        image_id = image['id']
        try:
            description = process_image(client, image['image_base64'], image_model)
            image_descriptions[image_id] = description
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            image_descriptions[image_id] = f"[Image {image_id} could not be processed]"
    
    # Replace image references using string replacement
    try:
        markdown = replace_image_references(markdown, image_descriptions)
    except Exception as e:
        print(f"Error replacing image references: {e}")
        # Fallback: append all image descriptions at the end
        for image_id, description in image_descriptions.items():
            markdown += f"\n\nImage description for {image_id}: {description}\n\n"
    
    return markdown

def check_available_models(client):
    """Check which models are available."""
    try:
        models = client.models.list()
        print("Available models:")
        for model in models.data:
            print(f"- {model.id}")
        return [model.id for model in models.data]
    except Exception as e:
        print(f"Error checking models: {e}")
        return []

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert a PDF to a TTS-friendly text file using Mistral API."
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
    args = parser.parse_args()

    # Initialize Mistral client with API key
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable is not set.")
        sys.exit(1)
    
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

    # Validate required arguments if not listing models
    if not args.input_pdf or not args.output_file:
        print("Error: input_pdf and output_file are required unless using --list_models")
        parser.print_help()
        sys.exit(1)

    # Check if output file exists and overwrite is not set
    if os.path.exists(args.output_file) and not args.overwrite:
        print(f"Error: {args.output_file} exists. Use --overwrite to replace it.")
        sys.exit(1)

    if args.verbose:
        print(f"Using text model: {args.text_model}")
        if args.include_images:
            print(f"Using image model: {args.image_model}")

    # Convert PDF to JSON
    print("Processing PDF with OCR...")
    doc = process_pdf_to_json(client, args.input_pdf)

    # Process the document in chunks
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

    # Write the transformed text to the output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(final_transformed_text)

    print(f"Transformation complete. TTS-friendly document saved to '{args.output_file}'.")
    print(f"Output file size: {len(final_transformed_text)} characters")

if __name__ == "__main__":
    main()