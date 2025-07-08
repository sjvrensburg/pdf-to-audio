"""API interaction functions for Mistral services."""

import json
import sys
from pathlib import Path

import requests
from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .constants import MAX_RETRIES, RETRY_WAIT_MIN, RETRY_WAIT_MAX, TEMPERATURE


@retry(
    stop=stop_after_attempt(MAX_RETRIES), 
    wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
)
def make_api_call(client, model, messages):
    """Make an API call with retry logic for transient errors."""
    try:
        return client.chat.complete(model=model, messages=messages, temperature=TEMPERATURE)
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