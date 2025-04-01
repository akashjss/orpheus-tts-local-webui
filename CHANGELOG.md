# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0] - 2025-03-31

### Added
- Initial release with Gradio web interface
- Support for 8 different voices (tara, leah, jess, leo, dan, mia, zac, zoe)
- Emotive speech generation with natural expressions
- Audio generation with SNAC neural codec
- Local model inference using llama-cpp-python
- Automatic model downloading and caching
- GPU acceleration support (CUDA/MPS)

### Features
- Modern Gradio web interface
- Real-time audio generation
- Adjustable generation parameters:
  - Temperature control
  - Top-p sampling
  - Repetition penalty
- Emotion tags support:
  - giggle
  - laugh
  - chuckle
  - sigh
  - cough
  - sniffle
  - groan
  - yawn
  - gasp
- WAV file export
- Progress tracking during generation

### Technical Details
- Standalone implementation without external dependencies
- Efficient token processing (28 tokens per chunk)
- Automatic model management
- Error handling and user feedback
- Support for Python 3.8+

### Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- CUDA-capable GPU (optional)
- See requirements.txt for package dependencies