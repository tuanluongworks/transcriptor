"""
Output handler for writing transcriptions to file
Thread-safe with timestamp formatting
"""
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
import config
from transcription_engine import TranscriptionResult


class OutputHandler:
    """Handles writing transcriptions to text file"""
    
    def __init__(self, output_dir: Path = config.OUTPUT_DIR):
        """
        Initialize output handler
        
        Args:
            output_dir: Directory to save transcription files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create output file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"transcription_{timestamp}.txt"
        
        # Thread safety
        self.lock = threading.Lock()
        self.file_handle: Optional[object] = None
        
        # Statistics
        self.lines_written = 0
        self.last_flush = datetime.now()
        
        # Initialize file
        self._open_file()
        print(f"Output file: {self.output_file}")
    
    def _open_file(self):
        """Open output file for writing"""
        try:
            self.file_handle = open(self.output_file, 'w', encoding='utf-8', buffering=1)
            
            # Write header
            header = f"""{'='*80}
Transcriptor - Audio Transcription
Session started: {datetime.now().strftime(config.TIMESTAMP_FORMAT)}
Model: {config.WHISPER_MODEL}
{'='*80}

"""
            self.file_handle.write(header)
            self.file_handle.flush()
            
        except Exception as e:
            print(f"Error opening output file: {e}")
            raise
    
    def write_transcription(self, result: TranscriptionResult):
        """
        Write transcription result to file
        
        Args:
            result: TranscriptionResult object
        """
        with self.lock:
            try:
                if not self.file_handle:
                    return
                
                # Format timestamp
                start_dt = datetime.fromtimestamp(
                    datetime.now().timestamp() - (result.end_time - result.start_time)
                )
                timestamp_str = start_dt.strftime(config.TIMESTAMP_FORMAT)
                
                # Format output line
                if result.is_silence:
                    line = f"[{timestamp_str}] {result.text}\n"
                else:
                    # Include confidence if available
                    if hasattr(result, 'confidence') and result.confidence != 0.0:
                        confidence_pct = int(min(max(result.confidence, -5), 0) * 20 + 100)
                        line = f"[{timestamp_str}] {result.text} (confidence: {confidence_pct}%)\n"
                    else:
                        line = f"[{timestamp_str}] {result.text}\n"
                
                # Write to file
                self.file_handle.write(line)
                self.lines_written += 1
                
                # Flush periodically
                now = datetime.now()
                if (now - self.last_flush).total_seconds() >= config.BUFFER_FLUSH_INTERVAL:
                    self.file_handle.flush()
                    self.last_flush = now
                
                # Echo to console if debug mode
                if config.DEBUG_MODE:
                    print(line.rstrip())
                
            except Exception as e:
                print(f"Error writing transcription: {e}")
    
    def write_text(self, text: str):
        """
        Write arbitrary text to file (for markers, errors, etc.)
        
        Args:
            text: Text to write
        """
        with self.lock:
            try:
                if self.file_handle:
                    timestamp_str = datetime.now().strftime(config.TIMESTAMP_FORMAT)
                    line = f"[{timestamp_str}] {text}\n"
                    self.file_handle.write(line)
                    self.file_handle.flush()
            except Exception as e:
                print(f"Error writing text: {e}")
    
    def write_separator(self):
        """Write a separator line"""
        with self.lock:
            try:
                if self.file_handle:
                    self.file_handle.write(f"\n{'-'*80}\n\n")
                    self.file_handle.flush()
            except Exception as e:
                print(f"Error writing separator: {e}")
    
    def write_session_stats(self, stats: dict):
        """
        Write session statistics
        
        Args:
            stats: Dictionary of statistics
        """
        with self.lock:
            try:
                if self.file_handle:
                    self.file_handle.write(f"\n{'='*80}\n")
                    self.file_handle.write("Session Statistics:\n")
                    for key, value in stats.items():
                        self.file_handle.write(f"  {key}: {value}\n")
                    self.file_handle.write(f"{'='*80}\n")
                    self.file_handle.flush()
            except Exception as e:
                print(f"Error writing stats: {e}")
    
    def flush(self):
        """Force flush file buffer"""
        with self.lock:
            try:
                if self.file_handle:
                    self.file_handle.flush()
            except Exception as e:
                print(f"Error flushing file: {e}")
    
    def close(self):
        """Close output file"""
        with self.lock:
            try:
                if self.file_handle:
                    # Write footer
                    footer = f"""
{'='*80}
Session ended: {datetime.now().strftime(config.TIMESTAMP_FORMAT)}
Total lines written: {self.lines_written}
{'='*80}
"""
                    self.file_handle.write(footer)
                    self.file_handle.flush()
                    self.file_handle.close()
                    self.file_handle = None
                    print(f"\nTranscription saved to: {self.output_file}")
            except Exception as e:
                print(f"Error closing file: {e}")
    
    def get_output_path(self) -> Path:
        """Get the path to the output file"""
        return self.output_file
    
    def get_stats(self) -> dict:
        """Get output statistics"""
        return {
            "output_file": str(self.output_file),
            "lines_written": self.lines_written,
            "file_size": self.output_file.stat().st_size if self.output_file.exists() else 0
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


if __name__ == "__main__":
    # Test output handler
    print("Testing output handler...")
    
    with OutputHandler() as handler:
        # Test transcription result
        result = TranscriptionResult(
            text="This is a test transcription",
            start_time=0.0,
            end_time=2.5,
            is_silence=False,
            confidence=-0.5
        )
        handler.write_transcription(result)
        
        # Test silence marker
        silence = TranscriptionResult(
            text="<silence: 5.0s>",
            start_time=2.5,
            end_time=7.5,
            is_silence=True
        )
        handler.write_transcription(silence)
        
        # Test text
        handler.write_text("Test message")
        
        # Test separator
        handler.write_separator()
        
        # Test stats
        handler.write_session_stats({
            "test_stat_1": 100,
            "test_stat_2": "test_value"
        })
        
        print(f"Output stats: {handler.get_stats()}")
    
    print("Test complete!")
