# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pdf-to-audio** is a production-grade command-line tool that converts PDF documents to high-quality audio using OCR, multi-stage LLM refinement, and text-to-speech synthesis. It's specifically engineered for academic papers with mathematical content, converting LaTeX and mathematical notation to natural spoken language.

**Key Innovation**: Multi-provider LLM architecture supporting Mistral, OpenAI, Anthropic, and Z.AI for different processing stages, enabling flexible and cost-optimized workflows.

## Architecture Overview

### Data Flow Pipeline
```
PDF Input → Mistral OCR → 4-Stage LLM Refinement → Chatterbox TTS → Audio Output
```

### Core Components

| Module | Purpose |
|--------|---------|
| `src/pdf_to_audio/core.py` | Orchestrates the 4-stage refinement pipeline, coordinates TTS, manages temp directories |
| `src/pdf_to_audio/api.py` | Wraps LLM API calls with retry logic; handles Mistral OCR file uploads |
| `src/pdf_to_audio/llm_provider.py` | Abstract LLM provider interface (any-llm SDK) supporting multiple backends |
| `src/pdf_to_audio/config.py` | Configuration management with precedence: CLI args > YAML config > hardcoded defaults |
| `src/pdf_to_audio/cli.py` | Command-line interface with argument parsing and workflow orchestration |
| `src/pdf_to_audio/constants.py` | Four system prompts: Core Transform, Math, Citations, Language/Style |
| `src/pdf_to_audio/audio/chunking.py` | Three strategies: duration-based, sentence-based, smart (hybrid) |
| `src/pdf_to_audio/audio/concatenation.py` | Audio segment merging with optional crossfading |
| `src/pdf_to_audio/audio/formats.py` | FFmpeg-based format conversion (WAV, MP3, FLAC, OGG, M4A) |
| `src/pdf_to_audio/tts/chatterbox_tts.py` | Chatterbox TTS wrapper with GPU/CPU device fallback |

### Four-Stage Refinement Pipeline

Instead of complex nested prompts, the system uses four focused, sequential LLM passes on the OCR output:

1. **Core Transform** (`CORE_TRANSFORM_PROMPT`): Marks mathematical content with `<MATH></MATH>` tags, preserves document structure
2. **Math Processing** (`MATH_PROMPT`): Converts marked math to spoken language ("x squared", "alpha plus beta"), removes tags
3. **Citations Optimization** (`CITATIONS_PROMPT`): Reformats citation clusters for audio clarity
4. **Language/Style** (`LANGUAGE_STYLE_PROMPT`): Simplifies academic sentences while retaining meaning

**Key Insight**: Each stage is isolated and idempotent. Any stage can be disabled independently via config flags.

### Multi-Provider LLM Architecture

Uses `any-llm-sdk` for provider-agnostic integration. Each processing stage can independently specify a provider and model:

```python
# Different stages can use different providers
transform_stage_provider = create_llm_provider(provider="mistral", model="mistral-small-latest")
math_stage_provider = create_llm_provider(provider="openai", model="gpt-4-turbo")
```

API keys are environment-based (`MISTRAL_API_KEY`, `OPENAI_API_KEY`, etc.), not passed between stages.

### Content Preservation Strategy

Critical bug fix from v0.1.x: Previous versions dropped ~50% of content during math processing.

**Solution**: Tag-based preservation
1. Mark ALL mathematical content with `<MATH>` tags (Stage 1)
2. Process ONLY within tags (Stage 2)
3. Remove tags after verbalization (Stages 3-4)

**Result**: 100% content preservation with proper math conversion.

### Text Chunking for Audio Generation

Three strategies for splitting text before TTS:
- **Duration-based**: Fixed 35-second chunks (safe margin below 40s Chatterbox limit)
- **Sentence-based**: Chunk at sentence boundaries for natural break points
- **Smart** (default): Hybrid approach combining duration limits with section/sentence boundaries and math tag preservation

## Development Commands

### Setup and Installation

```bash
# Clone and install in development mode
git clone https://github.com/sjvrensburg/pdf-to-audio.git
cd pdf-to-audio
pip install -e ".[dev]"

# Or for direct Poetry use
poetry install
```

### Code Quality

```bash
# Format code
ruff format .
isort .

# Lint and type-check
ruff check .
mypy src/

# Run all checks
ruff format . && isort . && ruff check . && mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/pdf_to_audio

# Run a single test file
pytest tests/test_api.py

# Run tests matching a pattern
pytest tests/ -k "test_math"

# Run with verbose output
pytest -v
```

### Common Development Tasks

**Run the CLI on a test PDF**:
```bash
pdf-to-audio tests/fixtures/sample.pdf output.txt --verbose
```

**Generate audio from a PDF**:
```bash
pdf-to-audio input.pdf output.txt --output_audio output.mp3 --verbose
```

**Process with custom refinement settings**:
```bash
pdf-to-audio input.pdf output.txt \
  --math_refinement_intensity 0.8 \
  --structure_citation_intensity 0.6 \
  --language_style_intensity 0.4
```

**List available Mistral models**:
```bash
pdf-to-audio --list_models
```

## Configuration

### YAML Configuration Precedence

```
CLI Arguments > config.yaml file > Hardcoded Defaults
```

### Key Configuration Sections

**Mistral Settings**:
- `mistral.text_model`: Model for text processing (default: `mistral-small-latest`)
- `mistral.image_model`: Model for image processing (default: `pixtral-12b-latest`)
- `mistral.temperature`: LLM randomness (0.0-1.0)

**TTS Settings**:
- `tts.exaggeration`: Speech emphasis (0.0-1.0, default: 0.5)
- `tts.cfg_weight`: Classifier-free guidance weight (0.0-1.0, default: 0.5)
- `tts.math_tts_scale`: Math content scaling factor (default: 0.75)
- `tts.chunk_strategy`: `duration`, `sentences`, or `smart` (default: `smart`)
- `tts.global_normalization`: Normalize entire audio file after concatenation
- `tts.audio_prompt_path`: Reference audio for voice cloning

**Refinement Pipeline**:
- `refinement.enable_refinement`: Master switch (default: true)
- `refinement.enable_math_refinement`: Stage 2 (default: true)
- `refinement.enable_structure_citation_optimization`: Stage 3 (default: true)
- `refinement.enable_language_style_refinement`: Stage 4 (default: true)
- `refinement.math_refinement_intensity`: 0.0-1.0 (default: 0.5)
- `refinement.target_audience`: `academic` or `general` (default: `academic`)

See `config.example.yaml` for full template.

## Important Design Patterns

### Retry Logic with Exponential Backoff

Uses `tenacity` library for resilient API calls:
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=7, min=7, max=15),
    retry=retry_if_exception_type((RequestException, Timeout))
)
```

Handles rate limiting, timeouts, and connection errors automatically.

### GPU/CPU Device Fallback

Graceful degradation: Try CUDA → Try MPS → Fall back to CPU. Force CPU option available for compatibility issues.

### Watermarker Compatibility Patch

In `tts/chatterbox_tts.py`, handles library version mismatches:
```python
if perth.PerthImplicitWatermarker is None:
    perth.PerthImplicitWatermarker = perth.DummyWatermarker
```

### Temporary Directory Management

Uses context managers for automatic cleanup:
```python
with temp_directory(base_dir) as temp_dir:
    # Process files
    # Automatic cleanup on exit
```

### Content-Type Aware TTS

Different TTS parameters for math vs. academic vs. general content:
```python
MATH_HEAVY_SETTINGS = {"exaggeration": 0.2, "cfg_weight": 0.7}  # Slow, deliberate
ACADEMIC_TTS_SETTINGS = {"exaggeration": 0.3, "cfg_weight": 0.6}  # Controlled tone
DEFAULT_TTS_SETTINGS = {"exaggeration": 0.5, "cfg_weight": 0.5}  # Balanced
```

## Technology Stack

- **Python**: 3.11-3.12 (sweet spot for ML/LLM stability)
- **LLM Integration**: `any-llm-sdk` (supports Mistral, OpenAI, Anthropic, Z.AI)
- **OCR**: Mistral API
- **TTS**: Chatterbox TTS with PyTorch backend
- **Audio Processing**: PyDub, FFmpeg, TorchAudio
- **Dependency Management**: Poetry
- **Code Quality**: Ruff, isort, mypy
- **Testing**: Pytest with coverage

## Dependency Notes

- **PyTorch**: CUDA 12.4 pinned via Poetry source priority; automatic CPU fallback
- **GPU Optional**: CUDA acceleration is optional; CPU-only environments fully supported
- **FFmpeg**: External dependency; required for audio format conversion

## Common Issues and Solutions

### CUDA Out of Memory
- Reduce `pages_per_chunk` value
- Process document in smaller sections
- Use `--force_cpu` to bypass GPU

### CUDA Compatibility Errors
- Use `--force_cpu` to force CPU processing
- Check CUDA driver/toolkit installation

### Volume Inconsistencies
- Use `--global_normalization` flag for consistent levels across sections

### Perth Watermarker Warnings
- Normal for newer perth versions; system auto-falls back to DummyWatermarker
- No audio quality impact

## When Making Changes

1. **Update system prompts** (constants.py): Impacts output quality significantly
2. **Modify pipeline stages** (core.py): Affects content preservation and audio quality
3. **Add new chunking strategy** (audio/chunking.py): Test with math-heavy documents
4. **Change LLM provider logic** (llm_provider.py): Ensure backward compatibility
5. **Audio format changes** (audio/formats.py): Test with all supported formats

Always test with math-heavy PDFs to ensure content preservation remains at 100%.

## Repository Structure

```
pdf-to-audio/
├── src/pdf_to_audio/           # Main package
│   ├── api.py                  # LLM/OCR API integration
│   ├── core.py                 # Pipeline orchestration
│   ├── cli.py                  # CLI interface
│   ├── config.py               # Configuration
│   ├── constants.py            # System prompts
│   ├── llm_provider.py         # LLM abstraction
│   ├── audio/                  # Audio processing
│   ├── tts/                    # Text-to-speech
│   └── utils.py
├── tests/                      # Test suite
├── examples/                   # Usage examples
├── notebooks/                  # Development notebooks
├── pyproject.toml              # Poetry config
├── config.example.yaml         # Config template
└── README.md                   # User documentation
```

## Multi-Provider Support

The codebase supports four LLM providers via `any-llm-sdk`:

| Provider | API Key Env Var | Default Models |
|----------|-------------------|---|
| Mistral | `MISTRAL_API_KEY` | `mistral-small-latest` (text), `pixtral-12b-latest` (image) |
| OpenAI | `OPENAI_API_KEY` | `gpt-4-turbo` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus` |
| Z.AI | `Z_AI_API_KEY` | See any-llm docs |

Each processing stage can independently select a provider and model via `llm_provider.py` factory.
