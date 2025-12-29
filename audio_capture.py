"""
Audio capture module using WASAPI Loopback
Captures system audio output on Windows 11
"""
import pyaudiowpatch as pyaudio
import numpy as np
from typing import Optional, Callable
import threading
import queue
import time
import config


class AudioCapture:
    """Captures system audio using WASAPI loopback"""
    
    def __init__(self, 
                 sample_rate: int = config.SAMPLE_RATE,
                 chunk_duration: int = config.CHUNK_DURATION,
                 overlap_duration: int = config.OVERLAP_DURATION):
        """
        Initialize audio capture
        
        Args:
            sample_rate: Target sample rate in Hz
            chunk_duration: Duration of each audio chunk in seconds
            overlap_duration: Overlap between chunks in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.overlap_samples = int(sample_rate * overlap_duration)
        
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_capturing = False
        self.capture_thread: Optional[threading.Thread] = None
        
        # Audio buffer and queue
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_queue = queue.Queue(maxsize=config.MAX_QUEUE_SIZE)
        
        # Callback for processed chunks
        self.chunk_callback: Optional[Callable] = None
        
        # Statistics
        self.total_chunks = 0
        self.current_audio_level = 0.0
        
    def get_loopback_device(self) -> Optional[dict]:
        """
        Find Windows WASAPI loopback device
        
        Returns:
            Device info dict or None if not found
        """
        try:
            # Get default WASAPI loopback device
            wasapi_info = self.audio.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = self.audio.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )
            
            if not default_speakers["isLoopbackDevice"]:
                # Find the loopback device
                for loopback in self.audio.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        return loopback
            else:
                return default_speakers
                
        except Exception as e:
            print(f"Error finding loopback device: {e}")
            return None
    
    def list_devices(self) -> None:
        """Print all available audio devices"""
        print("\n=== Available Audio Devices ===")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            print(f"[{i}] {info['name']}")
            print(f"    Channels: {info['maxInputChannels']} in, {info['maxOutputChannels']} out")
            print(f"    Sample Rate: {info['defaultSampleRate']}")
            if info.get('isLoopbackDevice', False):
                print(f"    ** LOOPBACK DEVICE **")
            print()
    
    def start_capture(self, callback: Optional[Callable] = None) -> bool:
        """
        Start capturing audio
        
        Args:
            callback: Function to call with each audio chunk (numpy array)
            
        Returns:
            True if capture started successfully
        """
        if self.is_capturing:
            print("Already capturing")
            return False
        
        # Get loopback device
        device = self.get_loopback_device()
        if device is None:
            print("ERROR: Could not find WASAPI loopback device")
            print("Make sure you have audio output enabled on Windows")
            return False
        
        print(f"Using device: {device['name']}")
        print(f"Sample rate: {device['defaultSampleRate']} Hz")
        
        self.chunk_callback = callback
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=device['maxInputChannels'],
                rate=int(device['defaultSampleRate']),
                input=True,
                frames_per_buffer=config.FRAMES_PER_BUFFER,
                input_device_index=device['index'],
                stream_callback=self._audio_callback
            )
            
            self.is_capturing = True
            self.stream.start_stream()
            
            # Start processing thread
            self.capture_thread = threading.Thread(target=self._process_audio_chunks, daemon=True)
            self.capture_thread.start()
            
            print("Audio capture started successfully")
            return True
            
        except Exception as e:
            print(f"Error starting capture: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for incoming audio data"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Convert to float32 and normalize
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Mix down to mono if stereo
        if len(audio_data) > frame_count:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        
        # Calculate audio level (RMS)
        self.current_audio_level = np.sqrt(np.mean(audio_data ** 2))
        
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_chunks(self):
        """Process audio buffer and create chunks"""
        import scipy.signal
        
        while self.is_capturing:
            # Check if we have enough data for a chunk
            if len(self.audio_buffer) >= self.chunk_samples:
                # Extract chunk
                chunk = self.audio_buffer[:self.chunk_samples]
                
                # Resample if necessary
                if self.stream:
                    original_rate = int(self.stream._rate)
                    if original_rate != self.sample_rate:
                        num_samples = int(len(chunk) * self.sample_rate / original_rate)
                        chunk = scipy.signal.resample(chunk, num_samples)
                
                # Add to queue
                try:
                    self.audio_queue.put(chunk, block=False)
                    self.total_chunks += 1
                    
                    # Call callback if provided
                    if self.chunk_callback:
                        self.chunk_callback(chunk.copy())
                        
                except queue.Full:
                    print("Warning: Audio queue full, dropping chunk")
                
                # Remove processed data, keep overlap
                self.audio_buffer = self.audio_buffer[self.chunk_samples - self.overlap_samples:]
            
            # Small sleep to prevent busy waiting
            time.sleep(0.1)
    
    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next audio chunk from queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Audio chunk as numpy array or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_audio_level(self) -> float:
        """Get current audio level (0.0 to 1.0)"""
        return self.current_audio_level
    
    def stop_capture(self):
        """Stop capturing audio"""
        if not self.is_capturing:
            return
        
        print("Stopping audio capture...")
        self.is_capturing = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        print("Audio capture stopped")
    
    def __del__(self):
        """Cleanup"""
        self.stop_capture()
        if self.audio:
            self.audio.terminate()


if __name__ == "__main__":
    # Test audio capture
    print("Testing audio capture...")
    
    capture = AudioCapture()
    capture.list_devices()
    
    def test_callback(chunk):
        energy = np.sqrt(np.mean(chunk ** 2))
        print(f"Received chunk: {len(chunk)} samples, energy: {energy:.4f}")
    
    if capture.start_capture(callback=test_callback):
        print("Capturing for 10 seconds...")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            capture.stop_capture()
