"""
Transcription engine using Faster-Whisper with GPU acceleration
"""
import os
import warnings

# Suppress cuDNN library warnings before importing ML libraries
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Disable CUDA if not properly configured to prevent cuDNN errors
if os.environ.get('FORCE_CPU', '').lower() in ('1', 'true', 'yes'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
from faster_whisper import WhisperModel
from typing import List, Tuple, Optional
import threading
import queue
import time
import config
from utils.vad import VoiceActivityDetector


class TranscriptionResult:
    """Container for transcription results"""
    
    def __init__(self, text: str, start_time: float, end_time: float, 
                 is_silence: bool = False, confidence: float = 0.0):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.is_silence = is_silence
        self.confidence = confidence
    
    def __repr__(self):
        return f"TranscriptionResult(text='{self.text[:50]}...', start={self.start_time:.2f}s, end={self.end_time:.2f}s)"


class TranscriptionEngine:
    """GPU-accelerated transcription using Faster-Whisper"""
    
    def __init__(self, 
                 model_name: str = config.WHISPER_MODEL,
                 device: str = config.WHISPER_DEVICE,
                 compute_type: str = config.WHISPER_COMPUTE_TYPE):
        """
        Initialize transcription engine
        
        Args:
            model_name: Whisper model name (e.g., 'large-v2')
            device: 'cuda' for GPU or 'cpu'
            compute_type: 'float16', 'int8', or 'float32'
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.actual_device = device  # Track what device is actually used
        
        print(f"Loading Whisper model '{model_name}' on {device} ({compute_type})...")
        print("This may take a few minutes on first run (downloading model)...")
        
        # Temporarily suppress stderr to hide cuDNN library warnings
        # These are benign - CTranslate2 3.24.0 expects cuDNN 8.x but PyTorch bundles cuDNN 9.x
        import sys
        from io import StringIO
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        
        try:
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=None,  # Use default cache
                local_files_only=False
            )
            print(f"Model loaded successfully on {device}!")
            self.actual_device = device
            
        except Exception as e:
            sys.stderr = original_stderr  # Restore stderr for error messages
            if device == "cuda":
                print(f"WARNING: GPU initialization failed: {e}")
                print("Falling back to CPU mode...")
                try:
                    # Retry with CPU
                    self.model = WhisperModel(
                        model_name,
                        device="cpu",
                        compute_type="int8",  # More efficient for CPU
                        download_root=None,
                        local_files_only=False
                    )
                    self.actual_device = "cpu"
                    print(f"Model loaded successfully on CPU!")
                    print("NOTE: CPU transcription will be slower than GPU")
                except Exception as cpu_error:
                    print(f"ERROR: Failed to load model on both GPU and CPU: {cpu_error}")
                    raise
            else:
                print(f"ERROR loading model: {e}")
                raise
        finally:
            # Always restore stderr
            sys.stderr = original_stderr
        
        # Voice Activity Detector
        self.vad = VoiceActivityDetector()
        
        # Processing queue and thread
        self.input_queue = queue.Queue(maxsize=config.MAX_QUEUE_SIZE)
        self.output_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_processed = 0
        self.total_duration = 0.0
        self.chunk_offset = 0.0  # Running time offset for chunks
    
    def start_processing(self):
        """Start background transcription processing"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        print("Transcription engine started")
    
    def stop_processing(self):
        """Stop background processing"""
        if not self.is_processing:
            return
        
        print("Stopping transcription engine...")
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        print("Transcription engine stopped")
    
    def add_audio_chunk(self, audio_data: np.ndarray) -> bool:
        """
        Add audio chunk for processing
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if added successfully
        """
        try:
            self.input_queue.put(audio_data, block=False)
            return True
        except queue.Full:
            print("Warning: Transcription queue full, dropping chunk")
            return False
    
    def get_transcription(self, timeout: float = 0.1) -> Optional[TranscriptionResult]:
        """
        Get next transcription result
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            TranscriptionResult or None if no results available
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Background processing loop"""
        while self.is_processing:
            try:
                # Get audio chunk from queue
                audio_data = self.input_queue.get(timeout=1.0)
                
                # Process the chunk
                results = self._transcribe_chunk(audio_data)
                
                # Add results to output queue
                for result in results:
                    self.output_queue.put(result)
                
                # Update statistics
                self.total_processed += 1
                self.total_duration += len(audio_data) / config.SAMPLE_RATE
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
                if config.DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
    
    def _transcribe_chunk(self, audio_data: np.ndarray) -> List[TranscriptionResult]:
        """
        Transcribe a single audio chunk
        
        Args:
            audio_data: Audio data as numpy array (float32, normalized)
            
        Returns:
            List of TranscriptionResult objects
        """
        results = []
        chunk_duration = len(audio_data) / config.SAMPLE_RATE
        
        # Check if chunk is silent
        if config.INCLUDE_SILENCE_MARKERS and self.vad.is_silent(audio_data):
            result = TranscriptionResult(
                text=f"<silence: {chunk_duration:.1f}s>",
                start_time=self.chunk_offset,
                end_time=self.chunk_offset + chunk_duration,
                is_silence=True
            )
            results.append(result)
            self.chunk_offset += chunk_duration
            return results
        
        # Transcribe with Whisper
        try:
            start_time = time.time()
            
            # Suppress cuDNN warnings during transcription
            import sys
            from io import StringIO
            original_stderr = sys.stderr
            sys.stderr = StringIO()
            
            try:
                segments, info = self.model.transcribe(
                    audio_data,
                    language=config.WHISPER_LANGUAGE,
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    vad_filter=True,  # Use Whisper's built-in VAD
                    vad_parameters={
                        "threshold": 0.5,
                        "min_speech_duration_ms": 250,
                        "min_silence_duration_ms": 500
                    }
                )
            finally:
                sys.stderr = original_stderr
            
            processing_time = time.time() - start_time
            
            if config.DEBUG_MODE:
                print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
                print(f"Processing time: {processing_time:.2f}s for {chunk_duration:.2f}s audio")
                print(f"Real-time factor: {chunk_duration / processing_time:.2f}x")
            
            # Process segments
            segment_found = False
            for segment in segments:
                segment_found = True
                
                # Adjust timestamps to absolute time
                abs_start = self.chunk_offset + segment.start
                abs_end = self.chunk_offset + segment.end
                
                result = TranscriptionResult(
                    text=segment.text.strip(),
                    start_time=abs_start,
                    end_time=abs_end,
                    is_silence=False,
                    confidence=segment.avg_logprob
                )
                results.append(result)
            
            # If no segments but not silent, mark as unclear audio
            if not segment_found and config.INCLUDE_SILENCE_MARKERS:
                result = TranscriptionResult(
                    text=f"<unclear audio: {chunk_duration:.1f}s>",
                    start_time=self.chunk_offset,
                    end_time=self.chunk_offset + chunk_duration,
                    is_silence=True
                )
                results.append(result)
            
        except Exception as e:
            print(f"Error transcribing chunk: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            
            # Add error marker
            result = TranscriptionResult(
                text=f"<transcription error>",
                start_time=self.chunk_offset,
                end_time=self.chunk_offset + chunk_duration,
                is_silence=True
            )
            results.append(result)
        
        # Update chunk offset
        self.chunk_offset += chunk_duration
        
        return results
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return {
            "total_chunks": self.total_processed,
            "total_duration": self.total_duration,
            "queue_size": self.input_queue.qsize(),
            "pending_results": self.output_queue.qsize()
        }


if __name__ == "__main__":
    # Test transcription engine
    print("Testing transcription engine...")
    
    try:
        engine = TranscriptionEngine()
        engine.start_processing()
        
        # Create test audio (1 second of sine wave at 440 Hz)
        duration = 1.0
        sample_rate = config.SAMPLE_RATE
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        
        print("Adding test audio...")
        engine.add_audio_chunk(test_audio)
        
        print("Waiting for results...")
        time.sleep(5)
        
        result = engine.get_transcription(timeout=1.0)
        if result:
            print(f"Result: {result}")
        else:
            print("No transcription (expected for sine wave)")
        
        stats = engine.get_stats()
        print(f"Stats: {stats}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.stop_processing()
