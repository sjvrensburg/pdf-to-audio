"""API interaction functions for LLM services using any-llm."""

import json
import sys
from pathlib import Path
from typing import List, Dict

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .constants import MAX_RETRIES, RETRY_WAIT_MIN, RETRY_WAIT_MAX, TEMPERATURE
from .llm_provider import LLMProvider, AnyLLMProvider, create_llm_provider


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
)
def make_api_call(
    provider: LLMProvider,
    messages: List[Dict[str, str]],
    temperature: float = TEMPERATURE
) -> str:
    """
    Make an API call to an LLM provider with retry logic for transient errors.

    Args:
        provider: LLMProvider instance to use for the call
        messages: List of message dicts with 'role' and 'content'
        temperature: Temperature setting for the response

    Returns:
        The assistant's response text

    Raises:
        Various exceptions for API errors
    """
    try:
        return provider.chat_complete(messages=messages, temperature=temperature)
    except requests.exceptions.Timeout:
        print("API call timed out. Retrying...")
        raise
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"Rate limit exceeded. This might be a capacity issue.")
            print("Consider trying a different model or waiting before retrying.")
        else:
            print(f"HTTP error {e.response.status_code}: {e.response.text}")
        raise
    except Exception as e:
        # Handle various LLM exceptions
        if "429" in str(e) or "capacity exceeded" in str(e).lower():
            print(f"Capacity exceeded.")
            print("This is likely due to high demand. Try:")
            print("1. Using a different model")
            print("2. Waiting a few minutes and trying again")
            print("3. Checking your API account/quota")
            raise
        elif "validation" in str(e).lower() or "invalid" in str(e).lower():
            print(f"Validation error: {e}")
            raise
        print(f"API error: {e}")
        raise


def process_pdf_to_json(api_key: str, pdf_path: str, model: str = "mistral-ocr-latest"):
    """
    Convert a PDF file to JSON using Mistral's OCR API with error handling.

    Args:
        api_key: Mistral API key
        pdf_path: Path to the PDF file
        model: OCR model to use (default: mistral-ocr-latest)

    Returns:
        Dictionary with OCR results
    """
    from mistralai import Mistral

    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is required for OCR operations")

    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    client = Mistral(api_key=api_key)

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
            model=model,
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