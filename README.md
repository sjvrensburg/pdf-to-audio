# PDF to Audio

A powerful command-line tool that converts PDF documents to TTS-friendly text and high-quality audio using Mistral AI's OCR and language models with Chatterbox TTS. This tool is specifically designed to handle academic papers with mathematical notation, converting complex formulas and equations into clear, spoken language suitable for text-to-speech conversion.

## Features

- **OCR Processing**: Extracts text from PDF files using Mistral's OCR API
- **Mathematical Notation**: Converts LaTeX math expressions to spoken language with specialized TTS settings
- **Image Descriptions**: Optional processing of figures, charts, and diagrams
- **Chunked Processing**: Handles large documents efficiently
- **Customizable Models**: Choose different Mistral models for text and image processing
- **Audio Generation**: Generate high-quality audio files from PDF content using Chatterbox TTS
- **Global Volume Normalization**: Optional global volume normalization for consistent audio levels
- **Audio Post-Processing**: Volume normalization, natural pauses, and noise reduction
- **Multiple Audio Formats**: Support for WAV, MP3, FLAC, OGG, and M4A formats
- **Intelligent Text Chunking**: Smart chunking strategies for optimal audio generation
- **Math-Specific TTS Settings**: Different TTS parameters for mathematical content
- **Customizable System Prompt**: Override the default system prompt with your own
- **Configuration File Support**: Use YAML configuration files for persistent settings
- **Robust Temporary File Handling**: Secure and automatic cleanup of temporary files
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

# Generate audio with global volume normalization
pdf-to-audio input.pdf --output_audio output.mp3 --global_normalization

# Generate audio with custom TTS settings
pdf-to-audio input.pdf --output_audio output.mp3 --exaggeration 0.7 --cfg_weight 0.3

# Use math-specific TTS settings
pdf-to-audio input.pdf --output_audio output.mp3 --math_exaggeration 0.4 --math_cfg_weight 0.6

# Adjust the math TTS scaling factor
pdf-to-audio input.pdf --output_audio output.mp3 --math_tts_scale 0.8

# Use a specific audio format
pdf-to-audio input.pdf --output_audio output.flac --audio_format flac

# Use a specific chunking strategy for audio
pdf-to-audio input.pdf --output_audio output.mp3 --chunk_strategy sentences

# Use a custom system prompt
pdf-to-audio input.pdf output.txt --system_prompt "Your custom system prompt here"

# Use a configuration file
pdf-to-audio input.pdf output.txt --config_file my_config.yaml

# Specify a custom temporary directory
pdf-to-audio input.pdf --output_audio output.mp3 --temp_dir /path/to/temp/dir

# Verbose output for debugging
pdf-to-audio input.pdf output.txt --verbose

# Overwrite existing output files
pdf-to-audio input.pdf output.txt --output_audio output.mp3 --overwrite
```

### Text Output

When generating audio, the tool will automatically save the processed text content alongside the audio file with the same base filename but with a `.txt` extension. For example, if you generate `output.mp3`, the tool will also create `output.txt` containing the text that was used to generate the audio.

```bash
# Generate audio (text file will be created automatically)
pdf-to-audio input.pdf --output_audio output.mp3
```

### Utility Commands

```bash
# List available Mistral models
pdf-to-audio --list_models

# List supported audio formats
pdf-to-audio --list_audio_formats
```

## Configuration File

You can use a YAML configuration file to store your settings. This is especially useful if you have a set of preferred settings that you use frequently. The configuration file can be specified using the `--config_file` option.

An example configuration file is provided in the repository as `config.example.yaml`. You can copy this file and modify it to suit your needs.

```yaml
# Example configuration file for pdf-to-audio

# Mistral API settings
mistral:
  # Model for text processing
  text_model: "mistral-small-latest"
  
  # Model for image processing (if include_images is true)
  image_model: "pixtral-12b-latest"
  
  # Temperature parameter for randomness in responses (0.0-1.0)
  temperature: 0.2
  
  # Custom system prompt (if not provided, the default one will be used)
  # system_prompt: |
  #   Your custom system prompt here.
  #   You can use multiple lines.

# Text-to-Speech settings
tts:
  # TTS exaggeration level (0.0-1.0)
  exaggeration: 0.5
  
  # CFG weight for TTS (0.0-1.0)
  cfg_weight: 0.5
  
  # TTS exaggeration level for mathematical content (0.0-1.0)
  # If not provided, defaults to math_tts_scale * exaggeration
  # math_exaggeration: 0.375
  
  # CFG weight for TTS for mathematical content (0.0-1.0)
  # If not provided, defaults to math_tts_scale * cfg_weight
  # math_cfg_weight: 0.375
  
  # Scaling factor for math TTS settings relative to plain text settings
  math_tts_scale: 0.75
  
  # Output audio format (wav, mp3, flac, ogg, m4a)
  audio_format: "wav"
  
  # Text chunking strategy for audio generation (duration, sentences, smart)
  chunk_strategy: "smart"
  
  # Apply global volume normalization to the entire audio file after concatenation
  global_normalization: false

# General settings
general:
  # Directory to use for temporary files
  # If not provided, system temp directory will be used
  # temp_dir: "/path/to/temp/dir"
  
  # Number of pages to process at a time
  pages_per_chunk: 1
  
  # Include image descriptions in the output
  include_images: false
  
  # Overwrite output files if they exist
  overwrite: false
  
  # Enable verbose output for debugging
  verbose: false
  
  # Force using CPU for TTS even if GPU is available
  force_cpu: false
```

## Mathematical Content Processing

The tool now automatically identifies mathematical content in the text and applies specialized TTS settings to it. This is done by instructing the Mistral AI model to tag mathematical content with `<MATH>` and `</MATH>` markers.

By default, mathematical content is processed with TTS settings that are 75% of the values used for regular text. This can be adjusted using the `--math_tts_scale` option, or by setting specific values with `--math_exaggeration` and `--math_cfg_weight`.

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
- `--global_normalization`: Apply global volume normalization to the entire audio file
- `--exaggeration`: TTS exaggeration level (0.0-1.0, default: 0.5)
- `--cfg_weight`: CFG weight for TTS (0.0-1.0, default: 0.5)
- `--math_exaggeration`: TTS exaggeration level for mathematical content (0.0-1.0)
- `--math_cfg_weight`: CFG weight for TTS for mathematical content (0.0-1.0)
- `--math_tts_scale`: Scaling factor for math TTS settings (default: 0.75)
- `--audio_format`: Output audio format (wav, mp3, flac, ogg, m4a)
- `--chunk_strategy`: Text chunking strategy (duration, sentences, smart)
- `--force_cpu`: Force using CPU for TTS even if GPU is available

#### General Options
- `--config_file`: Path to a YAML configuration file with custom settings
- `--system_prompt`: Custom system prompt to override the default one
- `--temp_dir`: Directory to use for temporary files
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

## Global Volume Normalization

The tool now supports global volume normalization, which ensures consistent audio levels throughout the entire output file:

1. **Enable global normalization**:
   ```bash
   pdf-to-audio input.pdf --output_audio output.mp3 --global_normalization
   ```

2. **How it works**:
   - Without global normalization, each chunk of audio is normalized individually
   - With global normalization, the entire audio file is normalized after all chunks are concatenated
   - This prevents volume "jumps" between different sections of the audio

3. **When to use it**:
   - For longer documents with varying content types
   - When processing documents with mathematical content mixed with regular text
   - When you notice inconsistent volume levels in the output audio

## Audio Quality Optimization

- **For academic content**: Use `--exaggeration 0.3 --cfg_weight 0.6` for a more controlled, deliberate tone
- **For math-heavy content**: Use `--exaggeration 0.2 --cfg_weight 0.7` for precise pronunciation of mathematical terms
- **For expressive speech**: Use `--exaggeration 0.7 --cfg_weight 0.3` for more dynamic, engaging narration

## Examples

### Convert a Research Paper to Text and Audio

```bash
pdf-to-audio research_paper.pdf paper_text.txt --output_audio paper_audio.mp3 --include_images --verbose
```

### Process a Large Document in Chunks with Global Normalization

```bash
pdf-to-audio large_document.pdf --output_audio document_audio.mp3 --pages_per_chunk 5 --global_normalization
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
- **Volume Inconsistencies**: If you notice volume jumps between sections, try using the `--global_normalization` option
- **Slow Audio Generation**: Audio generation is CPU-intensive without CUDA. Consider using a GPU-enabled system

### Audio Generation Failures

- **Check Text Output**: If audio generation fails, check the text output file for any issues with the processed text
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
