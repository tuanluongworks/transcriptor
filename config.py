"""
Configuration settings for Transcriptor application
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Whisper Model Settings
WHISPER_MODEL = "large-v2"  # Maximum accuracy
WHISPER_DEVICE = "cuda"  # Use GPU
WHISPER_COMPUTE_TYPE = "float16"  # Optimize for RTX 3060
WHISPER_LANGUAGE = "en"  # English only

# Audio Settings
SAMPLE_RATE = 16000  # Whisper's native sample rate
CHANNELS = 1  # Mono
CHUNK_DURATION = 60  # 60 seconds for high accuracy
OVERLAP_DURATION = 5  # 5 seconds overlap to prevent word cutting
FRAMES_PER_BUFFER = 1024

# Voice Activity Detection (VAD)
VAD_MODE = 3  # Aggressiveness: 0-3, 3 is most aggressive
SILENCE_THRESHOLD = 0.5  # Minimum silence duration to mark (seconds)
SILENCE_ENERGY_THRESHOLD = 0.01  # Audio energy threshold for silence detection

# Output Settings
OUTPUT_FORMAT = "txt"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
INCLUDE_SILENCE_MARKERS = True
BUFFER_FLUSH_INTERVAL = 5  # Flush file buffer every N seconds

# Ollama Settings (Optional)
OLLAMA_ENABLED = False  # Set to True to enable post-processing
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"  # Or your preferred model
OLLAMA_POST_PROCESS = False  # Enable real-time post-processing

# Performance Settings
MAX_QUEUE_SIZE = 10  # Maximum audio chunks in processing queue
THREAD_WORKERS = 2  # Number of worker threads

# Display Settings
SHOW_REALTIME_STATUS = True
STATUS_UPDATE_INTERVAL = 1  # Update console status every N seconds
SHOW_AUDIO_LEVELS = True

# Error Handling
MAX_RETRIES = 3  # Maximum retries for failed operations
RETRY_DELAY = 2  # Seconds between retries

# Debug Settings
DEBUG_MODE = False
LOG_AUDIO_STATS = False
