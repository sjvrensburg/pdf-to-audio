# Changelog

All notable changes to the PDF-to-Audio project will be documented in this file.

## [Unreleased]

## [0.2.0] - 2025-12-14

### 🐛 Bug Fixes

#### Critical Fixes
- **Fixed LLM Provider API Key Issue**: All LLM providers were incorrectly receiving the Mistral API key instead of fetching their own provider-specific keys. This prevented the use of alternative providers like OpenAI, Anthropic, etc.
- **Fixed Perth Library Compatibility**: `PerthImplicitWatermarker` was `None` in newer versions of the `perth` library, causing `TypeError: 'NoneType' object is not callable` during TTS initialization. Added automatic fallback to `DummyWatermarker`.

#### Quality Improvements
- **Fixed Math Processing Content Loss**: MATH_PROMPT was causing ~50% content loss by dropping non-mathematical content. Enhanced prompt now preserves all document structure while only converting mathematical notation.
- **Removed Unnecessary Introductions**: CITATIONS_PROMPT and LANGUAGE_STYLE_PROMPT were adding introductory text like "Here's the audio-friendly version..." which is now removed for cleaner output.
- **Improved Math Tag Handling**: LANGUAGE_STYLE_PROMPT now properly removes `<MATH>` tags and handles remaining mathematical notation in the final output.

#### Enhanced Mathematical Notation Support
- **Comprehensive Math Tagging**: CORE_TRANSFORM_PROMPT now identifies and tags ALL mathematical notation including inline math (`$k_{c}$`, `$G_{h}$`) and LaTeX equations.
- **Faithful Content Preservation**: Math processing now preserves 100% of original document content while only converting mathematical notation to spoken language.

### 🚀 Features

#### Multi-Provider LLM Support
- **Provider Flexibility**: Each of the 4 processing stages can now use different LLM providers (Mistral, OpenAI, Anthropic, Z.AI, etc.)
- **Per-Stage Configuration**: Independent provider/model selection for transform, math, citations, and language stages
- **Automatic API Key Management**: Each provider automatically fetches its API key from the appropriate environment variable

#### Enhanced Audio Generation
- **Robust TTS Initialization**: Automatic fallback mechanisms for library compatibility issues
- **High-Quality Output**: Professional-grade audio generation with proper pacing and intonation
- **Mathematical Verbalization**: Proper spoken language conversion of complex mathematical notation

### 🔧 Technical Improvements

#### Architecture
- **Model-Agnostic Design**: Complete refactoring to support any LLM provider via `any-llm` library
- **Simplified Pipeline**: Reduced from 9-20+ LLM calls to 4 focused calls per document
- **Focused Prompts**: Split monolithic prompt into 4 specialized prompts for better quality and reduced hallucinations

#### Code Quality
- **Comprehensive Error Handling**: Robust error handling for API calls, TTS initialization, and audio generation
- **Automatic Fallbacks**: Graceful degradation for library compatibility issues
- **Informative Logging**: Clear logging of fallback mechanisms and processing stages

#### Performance
- **Efficient Processing**: Optimized text chunking and audio generation
- **Memory Management**: Better handling of GPU/CPU resources
- **Resource Cleanup**: Proper cleanup of temporary files and GPU cache

### 📝 Documentation

- **Updated README**: Added comprehensive troubleshooting section for multi-provider support
- **Enhanced Examples**: Added examples for mixed provider usage
- **Technical Details**: Documented mathematical notation preservation improvements
- **Troubleshooting Guide**: Added guidance for common TTS and API issues

### 🔒 Backward Compatibility

- **Mistral Remains Default**: All stages default to Mistral for backward compatibility
- **Existing Commands Work**: No changes required for existing users
- **Configuration Compatibility**: Existing configuration files continue to work
- **CLI Interface Preserved**: All existing command-line options maintained

## [0.1.3] - 2025-11-14

### Initial Refactoring
- **Model-Agnostic Architecture**: Integrated `any-llm` library for provider-agnostic LLM support
- **Simplified Prompts**: Split monolithic 152-line system prompt into 4 focused prompts
- **Removed Refinement Passes**: Deleted entire refinement module (4 refinement passes)
- **Per-Stage Model Selection**: Added provider/model selection for each pipeline stage
- **Configuration Restructuring**: Updated config.py to support new LLM configuration

## [0.1.2] - 2025-07-10

### Initial Release
- **Core Functionality**: PDF to text conversion with Mistral OCR
- **Basic TTS**: Chatterbox TTS integration
- **Mathematical Notation**: Basic math to speech conversion
- **CLI Interface**: Command-line tool with basic options
