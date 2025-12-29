"""
Voice Activity Detection (VAD) module
Detects speech vs silence in audio chunks
"""
import numpy as np
import webrtcvad
from typing import List, Tuple
import config


class VoiceActivityDetector:
    """Detects voice activity and silence periods in audio"""
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE, 
                 mode: int = config.VAD_MODE):
        """
        Initialize VAD
        
        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            mode: Aggressiveness mode (0-3, higher is more aggressive)
        """
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(mode)
        self.frame_duration_ms = 30  # WebRTC VAD uses 10, 20, or 30ms frames
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
    def detect_speech(self, audio_data: np.ndarray) -> bool:
        """
        Detect if audio contains speech
        
        Args:
            audio_data: Audio data as numpy array (int16)
            
        Returns:
            True if speech detected, False otherwise
        """
        # Convert to bytes if needed
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
            
        audio_bytes = audio_data.tobytes()
        
        # Process in frames
        num_frames = len(audio_data) // self.frame_size
        speech_frames = 0
        
        for i in range(num_frames):
            start = i * self.frame_size * 2  # *2 because int16 is 2 bytes
            end = start + self.frame_size * 2
            frame = audio_bytes[start:end]
            
            if len(frame) == self.frame_size * 2:
                try:
                    if self.vad.is_speech(frame, self.sample_rate):
                        speech_frames += 1
                except Exception:
                    continue
        
        # Consider speech if >30% of frames contain speech
        speech_ratio = speech_frames / num_frames if num_frames > 0 else 0
        return speech_ratio > 0.3
    
    def detect_silence_periods(self, audio_data: np.ndarray, 
                               min_silence_duration: float = config.SILENCE_THRESHOLD
                               ) -> List[Tuple[float, float]]:
        """
        Detect silence periods in audio
        
        Args:
            audio_data: Audio data as numpy array
            min_silence_duration: Minimum silence duration in seconds
            
        Returns:
            List of (start_time, end_time) tuples for silence periods
        """
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        audio_bytes = audio_data.tobytes()
        num_frames = len(audio_data) // self.frame_size
        
        silence_periods = []
        silence_start = None
        
        for i in range(num_frames):
            start = i * self.frame_size * 2
            end = start + self.frame_size * 2
            frame = audio_bytes[start:end]
            
            if len(frame) != self.frame_size * 2:
                continue
            
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                current_time = i * self.frame_duration_ms / 1000.0
                
                if not is_speech:
                    if silence_start is None:
                        silence_start = current_time
                else:
                    if silence_start is not None:
                        silence_duration = current_time - silence_start
                        if silence_duration >= min_silence_duration:
                            silence_periods.append((silence_start, current_time))
                        silence_start = None
                        
            except Exception:
                continue
        
        # Handle silence at end
        if silence_start is not None:
            end_time = num_frames * self.frame_duration_ms / 1000.0
            silence_duration = end_time - silence_start
            if silence_duration >= min_silence_duration:
                silence_periods.append((silence_start, end_time))
        
        return silence_periods
    
    def get_audio_energy(self, audio_data: np.ndarray) -> float:
        """
        Calculate audio energy level
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            RMS energy level (0.0 to 1.0)
        """
        if len(audio_data) == 0:
            return 0.0
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        
        # Normalize to 0-1 range
        return min(rms, 1.0)
    
    def is_silent(self, audio_data: np.ndarray) -> bool:
        """
        Check if audio chunk is mostly silent
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if silent, False otherwise
        """
        energy = self.get_audio_energy(audio_data)
        has_speech = self.detect_speech(audio_data)
        
        return energy < config.SILENCE_ENERGY_THRESHOLD or not has_speech
