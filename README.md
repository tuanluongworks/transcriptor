# Transcriptor - Real-time Audio Transcription

A Windows 11 console application that captures system audio output and transcribes it to text using GPU-accelerated Whisper AI with high accuracy.

## Features

- 🎯 **High Accuracy**: Uses OpenAI's Whisper `large-v2` model for best-in-class transcription
- ⚡ **GPU Accelerated**: Leverages your RTX 3060 for 3-4x faster processing
- 🔊 **System Audio Capture**: Captures speaker output using WASAPI loopback (not microphone)
- 📝 **Timestamped Output**: Generates text files with precise timestamps
- 🔇 **Silence Detection**: Marks silent periods automatically
- 🤖 **Ollama Integration**: Optional AI post-processing for improved formatting
- 🎮 **Real-time Status**: Live audio levels, queue status, and processing metrics
- 💾 **Continuous Recording**: Handles long sessions with automatic file management

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 3060 (6GB VRAM) or better
- **RAM**: 16GB+ recommended
- **Storage**: 5GB for models + space for transcriptions

### Software
- **OS**: Windows 11
- **Python**: 3.10 or 3.11 ([Download](https://www.python.org/downloads/))
- **NVIDIA Driver**: Version 522.06 or later (for GPU mode)
- **CUDA Toolkit**: 12.4 (for GPU mode) ([Download](https://developer.nvidia.com/cuda-downloads))
  - **cuDNN**: Included with PyTorch (no separate installation required)
  - **Note**: CTranslate2 4.5+ requires CUDA 12.x. CUDA 11.8 is not compatible.

> **Note**: The application now defaults to **CPU mode** for better compatibility. GPU acceleration is optional and requires proper CUDA/cuDNN setup.

## Installation

### Quick Start (CPU Mode)

```powershell
cd C:\GithubProjects
cd transcriptor

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4 (for GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt

# Verify audio devices
python transcriptor.py --list-devices

# Start transcription
python transcriptor.py
```

Look for a device with "** LOOPBACK DEVICE **" - this captures your speaker output.

### GPU Setup (Optional - See "GPU Setup" section below for details)

For 3-4x faster transcription, follow the GPU setup instructions after completing the quick start.

## Usage

### Basic Usage

Start transcription with default settings (large-v2 model, 60-second chunks):

```powershell
python transcriptor.py
```

Press `Ctrl+C` to stop recording. Output will be saved to `output/transcription_YYYYMMDD_HHMMSS.txt`.

### Command Line Options

```powershell
# Use faster medium model (trades some accuracy for speed)
python transcriptor.py --model medium.en

# Use shorter chunks for lower latency
python transcriptor.py --chunk-duration 30

# Show silence markers in console
python transcriptor.py --verbose

# Specify output directory
python transcriptor.py --output-dir "D:\Transcriptions"

# Enable GPU mode (requires proper CUDA/cuDNN setup)
python transcriptor.py --device cuda

# List all available options
python transcriptor.py --help
```

### GPU Setup (Optional)

To enable GPU acceleration for 3-4x faster transcription:

1. **Install CUDA Toolkit 12.4**: [Download here](https://developer.nvidia.com/cuda-downloads)
2. **Verify CUDA installation**: `nvcc --version` (should show 12.4)
3. **Install PyTorch with CUDA 12.4**:
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
4. **Verify CUDA is working**:
   ```powershell
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
   ```
   Should output: `CUDA available: True` and `CUDA version: 12.4`
5. **Update config.py**:
   ```python
   WHISPER_DEVICE = "cuda"
   WHISPER_COMPUTE_TYPE = "float16"
   ```
6. **Or use command line**: `python transcriptor.py --device cuda`

> **Important**: CTranslate2 4.5+ requires CUDA 12.x. If you have CUDA 11.8, you must upgrade to CUDA 12.4.

If you see cuDNN errors, the app will automatically fall back to CPU mode.

### Ollama Integration (Optional)

If you have [Ollama](https://ollama.ai) installed:

```powershell
# Enable Ollama for post-processing
python transcriptor.py --ollama

# Enable real-time transcription improvement
python transcriptor.py --ollama --ollama-improve

# Use specific Ollama model
python transcriptor.py --ollama --ollama-model llama2
```

## Output Format

Transcriptions are saved as timestamped text files:

```
================================================================================
Transcriptor - Audio Transcription
Session started: 2025-12-29 14:30:00
Model: large-v2
================================================================================

[2025-12-29 14:30:05] The quick brown fox jumps over the lazy dog (confidence: 95%)
[2025-12-29 14:30:12] This is an example of transcribed audio
[2025-12-29 14:30:20] <silence: 5.2s>
[2025-12-29 14:30:25] And the transcription continues here

================================================================================
Session ended: 2025-12-29 14:35:00
Total lines written: 45
================================================================================
```

## Configuration

Edit `config.py` to customize:

- Model selection
- Chunk duration and overlap
- Silence detection thresholds
- Output formatting
- GPU settings
- Ollama integration

## Performance

With RTX 3060 + faster-whisper:

| Model | Speed | Accuracy | VRAM Usage |
|-------|-------|----------|------------|
| `large-v2` | ~3-4x real-time | 95-98% WER | ~5GB |
| `medium.en` | ~10x real-time | 93-95% WER | ~2GB |
| `small.en` | ~20x real-time | 90-93% WER | ~1GB |

**Example**: For 60-second audio chunks with `large-v2`, expect ~15-20 seconds processing time.

## Troubleshooting

### CUDA Library Errors

#### Error: `Library cublas64_12.dll is not found`

**Root Cause**: CTranslate2 4.5+ requires CUDA 12.x, but you have CUDA 11.8 or PyTorch built with CUDA 11.8.

**Solution**: Install PyTorch with CUDA 12.4:
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Error: `Could not load library cudnn_ops_infer64_8.dll`

**Root Cause**: CTranslate2 versions prior to 4.5.0 required cuDNN 8.x, but modern PyTorch bundles cuDNN 9.x.

**Solution**: Upgrade to CTranslate2 4.5.0 or later (included in requirements.txt):
```powershell
pip install --upgrade ctranslate2 faster-whisper
```

**Key Compatibility Requirements**:
- CTranslate2 4.5+ requires CUDA 12.x and cuDNN 9.x
- PyTorch with CUDA 12.4 includes compatible cuDNN 9.x
- No separate cuDNN installation needed

### CUDA Not Available

```powershell
# Check CUDA installation
nvcc --version

# Verify PyTorch sees CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### No Audio Devices Found

1. Make sure audio output is enabled in Windows
2. Check Windows Sound Settings → Output
3. Run with `--list-devices` to see available devices
4. Verify pyaudiowpatch is installed: `pip install pyaudiowpatch`

### Model Download Issues

First run downloads ~1.5GB model. If it fails:

```powershell
# Manual download
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v2', device='cpu')"
```

### Out of Memory (CUDA)

If you get CUDA out of memory errors:

1. Use smaller model: `--model medium.en`
2. Close other GPU applications
3. Use `--compute-type int8` for lower VRAM usage

### Low Transcription Quality

1. Use `large-v2` model for best accuracy
2. Increase chunk duration: `--chunk-duration 60`
3. Ensure good audio quality (avoid background noise)
4. Enable Ollama improvement: `--ollama --ollama-improve`

## Project Structure

```
transcriptor/
├── transcriptor.py          # Main application
├── audio_capture.py         # WASAPI audio capture
├── transcription_engine.py  # Whisper integration
├── output_handler.py        # File writing with timestamps
├── config.py                # Configuration settings
├── requirements.txt         # Python dependencies
├── utils/
│   ├── vad.py              # Voice Activity Detection
│   └── ollama_client.py    # Ollama integration
└── output/                  # Transcription output directory
```

## Development

### Running Tests

```powershell
# Test audio capture
python audio_capture.py

# Test transcription engine
python transcription_engine.py

# Test output handler
python output_handler.py

# Test VAD
python utils/vad.py

# Test Ollama client
python utils/ollama_client.py
```

### Debug Mode

Enable debug output in `config.py`:

```python
DEBUG_MODE = True
LOG_AUDIO_STATS = True
```

## Advanced Usage

### Custom Models

```powershell
# Use different Whisper models
python transcriptor.py --model tiny.en      # Fastest, lowest accuracy
python transcriptor.py --model base.en      # Fast, decent accuracy
python transcriptor.py --model small.en     # Balanced
python transcriptor.py --model medium.en    # Good accuracy, fast
python transcriptor.py --model large-v2     # Best accuracy (default)
python transcriptor.py --model large-v3     # Latest model
```

### Batch Processing

For processing existing audio files, modify `transcription_engine.py` to load from file instead of stream.

### Integration with Other Tools

The output text files can be processed by:
- Text editors
- Note-taking apps (Obsidian, Notion, etc.)
- Speech analysis tools
- Your Ollama server for summaries/action items

## Known Limitations

- Windows 11 only (uses WASAPI loopback)
- Requires NVIDIA GPU for optimal performance
- English language only (configurable in `config.py`)
- System audio only (not microphone input)
- First run downloads ~1.5GB model

## Future Enhancements

- [ ] Multi-language support
- [ ] Speaker diarization (identify multiple speakers)
- [ ] Real-time subtitle display
- [ ] Export to SRT/VTT format
- [ ] Web interface
- [ ] API endpoint

## Contributing

This is a personal project, but suggestions and improvements are welcome!

## License

MIT License - feel free to use and modify as needed.

## Credits

- **OpenAI Whisper**: Speech recognition model
- **faster-whisper**: Optimized Whisper implementation
- **pyaudiowpatch**: Windows audio capture
- **Ollama**: Local AI post-processing

## Support

For issues:
1. Check [Troubleshooting](#troubleshooting) section
2. Verify all prerequisites are installed
3. Enable debug mode in `config.py`
4. Check the output file for error messages

## Changelog

### Version 1.0.0 (2025-12-29)
- Initial release
- GPU-accelerated Whisper transcription
- WASAPI loopback audio capture
- Real-time status display
- Ollama integration
- Timestamped text output
- Silence detection

---

**Happy Transcribing! 🎙️**
