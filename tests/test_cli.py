"""Tests for CLI functionality."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pdf_to_audio.cli import validate_api_key, create_parser


def test_create_parser():
    """Test argument parser creation."""
    parser = create_parser()
    
    # Test with valid arguments
    args = parser.parse_args(['input.pdf', 'output.txt'])
    assert args.input_pdf == 'input.pdf'
    assert args.output_file == 'output.txt'
    assert args.pages_per_chunk == 1
    assert not args.include_images
    assert not args.verbose
    
    # Test with optional arguments
    args = parser.parse_args(['input.pdf', 'output.txt', '--include_images', '--verbose', '--pages_per_chunk', '3'])
    assert args.include_images
    assert args.verbose
    assert args.pages_per_chunk == 3


def test_validate_api_key_missing():
    """Test API key validation when key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(SystemExit):
            validate_api_key()


def test_validate_api_key_present():
    """Test API key validation when key is present."""
    with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test-key'}):
        api_key = validate_api_key()
        assert api_key == 'test-key'