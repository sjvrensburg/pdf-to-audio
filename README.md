# PDF to Audio

A powerful command-line tool that converts PDF documents to TTS-friendly text and high-quality audio using Mistral AI's OCR and language models with Chatterbox TTS. This tool is specifically designed to handle academic papers with mathematical notation, converting complex formulas and equations into clear, spoken language suitable for text-to-speech conversion.

## Features

- **OCR Processing**: Extracts text from PDF files using Mistral's OCR API
- **Mathematical Notation**: Converts LaTeX math expressions to spoken language
- **Image Descriptions**: Optional processing of figures, charts, and diagrams
- **Chunked Processing**: Handles large documents efficiently
- **Customizable Models**: Choose different Mistral models for text and image processing
- **Audio Generation**: Generate high-quality audio files from PDF content using Chatterbox TTS
- **Voice Cloning**: Clone voices from audio samples for personalized TTS
- **Audio Post-Processing**: Volume normalization, natural pauses, and noise reduction
- **Multiple Audio Formats**: Support for WAV, MP3, FLAC, OGG, and M4A formats
- **Intelligent Text Chunking**: Smart chunking strategies for optimal audio generation
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

3. **CUDA Support (Optional)**: For faster audio generation, CUDA-compatible GPU is recommended. The tool will automatically use CUDA if available, or fall back to CPU.

## Usage

### Basic Usage

```bash
# Generate text only
pdf-to-audio input.pdf output.txt

# Generate audio only
pdf-to-audio input.pdf --output_audio output.mp3

# Generate both text and audio
pdf-to-audio input.pdf output.txt --output_audio output.mp3
```

### Advanced Usage

```bash
# Include image descriptions
pdf-to-audio input.pdf output.txt --include_images

# Process multiple pages at once
pdf-to-audio input.pdf output.txt --pages_per_chunk 3

# Use different models
pdf-to-audio input.pdf output.txt --text_model mistral-large-latest --image_model pixtral-12b-latest

# Generate audio with voice cloning
pdf-to-audio input.pdf --output_audio output.mp3 --voice voice_sample.wav

# Generate audio with custom TTS settings
pdf-to-audio input.pdf --output_audio output.mp3 --exaggeration 0.7 --cfg_weight 0.3

# Use a specific audio format
pdf-to-audio input.pdf --output_audio output.flac --audio_format flac

# Use a specific chunking strategy for audio
pdf-to-audio input.pdf --output_audio output.mp3 --chunk_strategy sentences

# Verbose output for debugging
pdf-to-audio input.pdf output.txt --verbose

# Overwrite existing output files
pdf-to-audio input.pdf output.txt --output_audio output.mp3 --overwrite
```

### Voice Management

```bash
# Register a voice for future use
pdf-to-audio --register_voice voice_sample.wav "John Doe"

# List registered voices
pdf-to-audio --list_voices

# Remove a registered voice
pdf-to-audio --remove_voice "John Doe"

# Use a registered voice by name
pdf-to-audio input.pdf --output_audio output.mp3 --voice "John Doe"
```

### Utility Commands

```bash
# List available Mistral models
pdf-to-audio --list_models

# List supported audio formats
pdf-to-audio --list_audio_formats
```

### Command-Line Options

#### PDF Processing Options
- `input_pdf`: Path to the input PDF file
- `output_file`: Path to the output text file
- `--pages_per_chunk`: Number of pages to process at a time (default: 1)
- `--include_images`: Include image descriptions in the output
- `--text_model`: Model for text processing (default: mistral-small-latest)
- `--image_model`: Model for image processing (default: pixtral-12b-latest)

#### Audio Generation Options
- `--output_audio`: Path to the output audio file
- `--voice`: Path to a .wav file for voice cloning
- `--list_voices`: List currently registered voices and exit
- `--register_voice`: Register a voice sample for future use
- `--remove_voice`: Remove a registered voice by name
- `--exaggeration`: TTS exaggeration level (0.0-1.0, default: 0.5)
- `--cfg_weight`: CFG weight for TTS (0.0-1.0, default: 0.5)
- `--audio_format`: Output audio format (wav, mp3, flac, ogg, m4a)
- `--chunk_strategy`: Text chunking strategy (duration, sentences, smart)
- `--force_cpu`: Force using CPU for TTS even if GPU is available

#### General Options
- `--overwrite`: Overwrite output files if they exist
- `--list_models`: List available Mistral models and exit
- `--list_audio_formats`: List supported audio formats and exit
- `--verbose`: Enable verbose output for debugging

## Mathematical Notation Conversion

The tool automatically converts mathematical notation to spoken language:

- `x^2` → "x squared"
- `∫f(x)dx` → "the integral of f of x with respect to x"
- `α + β` → "alpha plus beta"
- `∑_{i=1}^{n} a_i` → "the sum from i equals 1 to n of a sub i"

## Voice Cloning Setup

For best results with voice cloning:

1. **Prepare a high-quality voice sample**:
   - Use a WAV file with clear speech, minimal background noise
   - Optimal duration: 5-15 seconds
   - Sample rate: 44.1kHz or 48kHz
   - Mono audio is preferred

2. **Register the voice**:
   ```bash
   pdf-to-audio --register_voice your_sample.wav "Your Voice Name"
   ```

3. **Use the registered voice**:
   ```bash
   pdf-to-audio input.pdf --output_audio output.mp3 --voice "Your Voice Name"
   ```

## Audio Quality Optimization

- **For academic content**: Use `--exaggeration 0.3 --cfg_weight 0.6` for a more controlled, deliberate tone
- **For math-heavy content**: Use `--exaggeration 0.2 --cfg_weight 0.7` for precise pronunciation of mathematical terms
- **For expressive speech**: Use `--exaggeration 0.7 --cfg_weight 0.3` for more dynamic, engaging narration

## Examples

### Convert a Research Paper to Text and Audio

```bash
pdf-to-audio research_paper.pdf paper_text.txt --output_audio paper_audio.mp3 --include_images --verbose
```

### Process a Large Document in Chunks with Voice Cloning

```bash
pdf-to-audio large_document.pdf --output_audio document_audio.mp3 --pages_per_chunk 5 --voice your_voice.wav
```

### Generate High-Quality Audio for Academic Content

```bash
pdf-to-audio academic_paper.pdf --output_audio academic_audio.flac --audio_format flac --exaggeration 0.3 --cfg_weight 0.6
```

### Generate Audio Using CPU (for CUDA Error Workaround)

```bash
pdf-to-audio input.pdf --output_audio output.mp3 --force_cpu
```

## Troubleshooting

### Common TTS Issues

- **CUDA Out of Memory**: Try using a smaller `--pages_per_chunk` value or process the document in smaller sections
- **CUDA Errors**: If you encounter CUDA errors like "CUDA-capable device(s) is/are busy or unavailable", use the `--force_cpu` option to force CPU processing:
  ```bash
  pdf-to-audio input.pdf --output_audio output.mp3 --force_cpu
  ```
- **Voice Cloning Quality**: Ensure your voice sample is clear, has minimal background noise, and is 5-15 seconds long
- **Slow Audio Generation**: Audio generation is CPU-intensive without CUDA. Consider using a GPU-enabled system

### Audio Generation Failures

- **Check Voice Sample**: Ensure the voice sample is a valid WAV file with appropriate duration
- **Check Disk Space**: Ensure you have sufficient disk space for temporary files during audio generation
- **Check CUDA Installation**: If using CUDA, ensure your drivers and CUDA toolkit are properly installed
- **GPU Issues**: If you're having persistent GPU issues, try the `--force_cpu` option to bypass GPU acceleration

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
