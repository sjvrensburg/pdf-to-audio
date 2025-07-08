# PDF to Audio

A powerful command-line tool that converts PDF documents to TTS-friendly text using Mistral AI's OCR and language models. This tool is specifically designed to handle academic papers with mathematical notation, converting complex formulas and equations into clear, spoken language suitable for text-to-speech conversion.

## Features

- **OCR Processing**: Extracts text from PDF files using Mistral's OCR API
- **Mathematical Notation**: Converts LaTeX math expressions to spoken language
- **Image Descriptions**: Optional processing of figures, charts, and diagrams
- **Chunked Processing**: Handles large documents efficiently
- **Customizable Models**: Choose different Mistral models for text and image processing
- **CLI Interface**: Easy-to-use command-line interface

## Installation

### Using pipx (Recommended)

```bash
pipx install pdf-to-audio
```

### Using pip

```bash
pip install pdf-to-audio
```

### From Source

```bash
git clone https://github.com/sjvrensburg/pdf-to-audio.git
cd pdf-to-audio
pip install .
```

## Prerequisites

1. **Mistral API Key**: You need a Mistral API key to use this tool. Get one from [Mistral AI](https://mistral.ai/).

2. **Set Environment Variable**:
   ```bash
   export MISTRAL_API_KEY='your-api-key-here'
   ```

## Usage

### Basic Usage

```bash
pdf-to-audio input.pdf output.txt
```

### Advanced Usage

```bash
# Include image descriptions
pdf-to-audio input.pdf output.txt --include_images

# Process multiple pages at once
pdf-to-audio input.pdf output.txt --pages_per_chunk 3

# Use different models
pdf-to-audio input.pdf output.txt --text_model mistral-large-latest --image_model pixtral-12b-latest

# Verbose output for debugging
pdf-to-audio input.pdf output.txt --verbose

# Overwrite existing output file
pdf-to-audio input.pdf output.txt --overwrite
```

### Command-Line Options

- `input_pdf`: Path to the input PDF file
- `output_file`: Path to the output text file
- `--pages_per_chunk`: Number of pages to process at a time (default: 1)
- `--include_images`: Include image descriptions in the output
- `--text_model`: Model for text processing (default: mistral-small-latest)
- `--image_model`: Model for image processing (default: pixtral-12b-latest)
- `--overwrite`: Overwrite output file if it exists
- `--list_models`: List available models and exit
- `--verbose`: Enable verbose output for debugging

### List Available Models

```bash
pdf-to-audio --list_models
```

## Mathematical Notation Conversion

The tool automatically converts mathematical notation to spoken language:

- `x^2` → "x squared"
- `∫f(x)dx` → "the integral of f of x with respect to x"
- `α + β` → "alpha plus beta"
- `∑_{i=1}^{n} a_i` → "the sum from i equals 1 to n of a sub i"

## Examples

### Convert a Research Paper

```bash
pdf-to-audio research_paper.pdf paper_audio.txt --include_images --verbose
```

### Process a Large Document in Chunks

```bash
pdf-to-audio large_document.pdf output.txt --pages_per_chunk 5
```

### Use High-Performance Models

```bash
pdf-to-audio paper.pdf output.txt --text_model mistral-large-latest --include_images
```

## Development

### Setting up Development Environment

```bash
git clone https://github.com/sjvrensburg/pdf-to-audio.git
cd pdf-to-audio
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
ruff format .
ruff check .
isort .
```

## License

This project is licensed under the GNU General Public License v3.0 or later. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/sjvrensburg/pdf-to-audio/issues) on GitHub.
