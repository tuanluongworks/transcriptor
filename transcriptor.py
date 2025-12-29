"""
Transcriptor - Real-time Audio Transcription Application
Captures system audio and transcribes using GPU-accelerated Whisper
"""
import argparse
import sys
import time
import signal
from datetime import datetime, timedelta
from typing import Optional
import os

# Ensure colorama works on Windows
os.system("")

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Dummy color codes if colorama not available
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""

import config
from audio_capture import AudioCapture
from transcription_engine import TranscriptionEngine
from output_handler import OutputHandler
from utils.ollama_client import OllamaClient


class Transcriptor:
    """Main transcription application"""
    
    def __init__(self, args):
        """Initialize transcriptor with command line arguments"""
        self.args = args
        self.running = False
        self.start_time = None
        
        # Components
        self.audio_capture: Optional[AudioCapture] = None
        self.transcription_engine: Optional[TranscriptionEngine] = None
        self.output_handler: Optional[OutputHandler] = None
        self.ollama_client: Optional[OllamaClient] = None
        
        # Statistics
        self.chunks_captured = 0
        self.chunks_transcribed = 0
        self.last_status_update = time.time()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n{Fore.YELLOW}Received shutdown signal, stopping gracefully...{Style.RESET_ALL}")
        self.stop()
    
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Transcriptor - Real-time Audio Transcription{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
            
            # Initialize output handler
            print(f"{Fore.YELLOW}[1/4] Initializing output handler...{Style.RESET_ALL}")
            self.output_handler = OutputHandler(output_dir=self.args.output_dir)
            
            # Initialize transcription engine (loads model)
            print(f"{Fore.YELLOW}[2/4] Loading Whisper model '{self.args.model}'...{Style.RESET_ALL}")
            self.transcription_engine = TranscriptionEngine(
                model_name=self.args.model,
                device=self.args.device,
                compute_type=self.args.compute_type
            )
            
            # Initialize audio capture
            print(f"{Fore.YELLOW}[3/4] Initializing audio capture...{Style.RESET_ALL}")
            self.audio_capture = AudioCapture(
                sample_rate=config.SAMPLE_RATE,
                chunk_duration=self.args.chunk_duration,
                overlap_duration=config.OVERLAP_DURATION
            )
            
            # List available devices if requested
            if self.args.list_devices:
                self.audio_capture.list_devices()
                return False
            
            # Initialize Ollama (optional)
            if self.args.ollama:
                print(f"{Fore.YELLOW}[4/4] Connecting to Ollama...{Style.RESET_ALL}")
                self.ollama_client = OllamaClient(
                    base_url=self.args.ollama_url,
                    model=self.args.ollama_model
                )
                if not self.ollama_client.is_available():
                    print(f"{Fore.RED}Warning: Ollama not available, post-processing disabled{Style.RESET_ALL}")
                    self.ollama_client = None
            else:
                print(f"{Fore.YELLOW}[4/4] Ollama post-processing disabled{Style.RESET_ALL}")
            
            print(f"\n{Fore.GREEN}✓ Initialization complete!{Style.RESET_ALL}\n")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}ERROR during initialization: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False
    
    def start(self) -> bool:
        """Start transcription"""
        try:
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Starting Transcription{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"Model: {self.args.model}")
            print(f"Chunk duration: {self.args.chunk_duration}s")
            print(f"Output: {self.output_handler.get_output_path()}")
            print(f"Device: {self.args.device} ({self.args.compute_type})")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
            
            # Start transcription engine
            self.transcription_engine.start_processing()
            
            # Start audio capture
            if not self.audio_capture.start_capture():
                print(f"{Fore.RED}Failed to start audio capture{Style.RESET_ALL}")
                return False
            
            self.running = True
            self.start_time = datetime.now()
            
            print(f"{Fore.GREEN}Recording started! Press Ctrl+C to stop.{Style.RESET_ALL}\n")
            
            # Write start marker
            self.output_handler.write_text("=== Recording Started ===")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}ERROR starting transcription: {e}{Style.RESET_ALL}")
            return False
    
    def run(self):
        """Main processing loop"""
        try:
            while self.running:
                # Get audio chunk
                audio_chunk = self.audio_capture.get_chunk(timeout=0.5)
                if audio_chunk is not None:
                    self.chunks_captured += 1
                    # Send to transcription engine
                    self.transcription_engine.add_audio_chunk(audio_chunk)
                
                # Get transcription results
                result = self.transcription_engine.get_transcription(timeout=0.1)
                if result:
                    self.chunks_transcribed += 1
                    
                    # Post-process with Ollama if enabled
                    if self.ollama_client and not result.is_silence and self.args.ollama_improve:
                        improved = self.ollama_client.improve_transcription(result.text)
                        if improved:
                            result.text = improved
                    
                    # Write to output
                    self.output_handler.write_transcription(result)
                    
                    # Print to console
                    if not result.is_silence:
                        print(f"{Fore.GREEN}{result.text}{Style.RESET_ALL}")
                    elif self.args.verbose:
                        print(f"{Fore.CYAN}{result.text}{Style.RESET_ALL}")
                
                # Update status display
                if config.SHOW_REALTIME_STATUS:
                    current_time = time.time()
                    if current_time - self.last_status_update >= config.STATUS_UPDATE_INTERVAL:
                        self._display_status()
                        self.last_status_update = current_time
                
        except Exception as e:
            print(f"\n{Fore.RED}ERROR in main loop: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def _display_status(self):
        """Display real-time status"""
        if not self.start_time:
            return
        
        # Calculate runtime
        runtime = datetime.now() - self.start_time
        runtime_str = str(runtime).split('.')[0]  # Remove microseconds
        
        # Get audio level
        audio_level = self.audio_capture.get_audio_level()
        level_bars = int(audio_level * 20)
        level_display = "█" * level_bars + "░" * (20 - level_bars)
        
        # Get engine stats
        engine_stats = self.transcription_engine.get_stats()
        
        # Build status line
        status = (
            f"\r{Fore.BLUE}[{runtime_str}]{Style.RESET_ALL} "
            f"Audio: {Fore.GREEN}{level_display}{Style.RESET_ALL} "
            f"Captured: {Fore.YELLOW}{self.chunks_captured}{Style.RESET_ALL} "
            f"Transcribed: {Fore.MAGENTA}{self.chunks_transcribed}{Style.RESET_ALL} "
            f"Queue: {Fore.CYAN}{engine_stats['queue_size']}/{config.MAX_QUEUE_SIZE}{Style.RESET_ALL}"
        )
        
        # Print without newline
        print(status, end='', flush=True)
    
    def stop(self):
        """Stop transcription and cleanup"""
        if not self.running:
            return
        
        self.running = False
        
        print(f"\n\n{Fore.YELLOW}Stopping transcription...{Style.RESET_ALL}")
        
        # Stop audio capture
        if self.audio_capture:
            self.audio_capture.stop_capture()
        
        # Stop transcription engine
        if self.transcription_engine:
            # Process remaining chunks
            print(f"{Fore.YELLOW}Processing remaining audio chunks...{Style.RESET_ALL}")
            time.sleep(2)
            
            while True:
                result = self.transcription_engine.get_transcription(timeout=0.5)
                if result:
                    self.output_handler.write_transcription(result)
                    if not result.is_silence:
                        print(f"{Fore.GREEN}{result.text}{Style.RESET_ALL}")
                else:
                    break
            
            self.transcription_engine.stop_processing()
        
        # Write end marker and stats
        if self.output_handler:
            self.output_handler.write_separator()
            self.output_handler.write_text("=== Recording Stopped ===")
            
            # Calculate session stats
            if self.start_time:
                runtime = datetime.now() - self.start_time
                engine_stats = self.transcription_engine.get_stats()
                output_stats = self.output_handler.get_stats()
                
                stats = {
                    "Runtime": str(runtime).split('.')[0],
                    "Audio chunks captured": self.chunks_captured,
                    "Chunks transcribed": self.chunks_transcribed,
                    "Total audio duration": f"{engine_stats['total_duration']:.1f}s",
                    "Output file": output_stats['output_file'],
                    "Lines written": output_stats['lines_written'],
                    "File size": f"{output_stats['file_size']} bytes"
                }
                
                self.output_handler.write_session_stats(stats)
                
                # Print stats to console
                print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Session Statistics{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
                for key, value in stats.items():
                    print(f"{key}: {Fore.WHITE}{value}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
            
            self.output_handler.close()
        
        print(f"{Fore.GREEN}✓ Transcription stopped successfully{Style.RESET_ALL}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Transcriptor - Real-time audio transcription with GPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Start with default settings
  %(prog)s --model medium.en                  # Use faster medium model
  %(prog)s --chunk-duration 30                # Use 30-second chunks
  %(prog)s --ollama --ollama-improve          # Enable Ollama post-processing
  %(prog)s --list-devices                     # List available audio devices
        """
    )
    
    # Model settings
    parser.add_argument('--model', default=config.WHISPER_MODEL,
                       help=f'Whisper model to use (default: {config.WHISPER_MODEL})')
    parser.add_argument('--device', default=config.WHISPER_DEVICE,
                       choices=['cuda', 'cpu'],
                       help=f'Device to use (default: {config.WHISPER_DEVICE})')
    parser.add_argument('--compute-type', default=config.WHISPER_COMPUTE_TYPE,
                       choices=['float16', 'int8', 'float32'],
                       help=f'Compute type (default: {config.WHISPER_COMPUTE_TYPE})')
    
    # Audio settings
    parser.add_argument('--chunk-duration', type=int, default=config.CHUNK_DURATION,
                       help=f'Audio chunk duration in seconds (default: {config.CHUNK_DURATION})')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices and exit')
    
    # Output settings
    parser.add_argument('--output-dir', default=str(config.OUTPUT_DIR),
                       help=f'Output directory (default: {config.OUTPUT_DIR})')
    parser.add_argument('--verbose', action='store_true',
                       help='Show silence markers in console output')
    
    # Ollama settings
    parser.add_argument('--ollama', action='store_true',
                       help='Enable Ollama post-processing')
    parser.add_argument('--ollama-url', default=config.OLLAMA_BASE_URL,
                       help=f'Ollama server URL (default: {config.OLLAMA_BASE_URL})')
    parser.add_argument('--ollama-model', default=config.OLLAMA_MODEL,
                       help=f'Ollama model to use (default: {config.OLLAMA_MODEL})')
    parser.add_argument('--ollama-improve', action='store_true',
                       help='Use Ollama to improve transcriptions in real-time')
    
    args = parser.parse_args()
    
    # Create and run transcriptor
    app = Transcriptor(args)
    
    if not app.initialize():
        sys.exit(1)
    
    if args.list_devices:
        sys.exit(0)
    
    if not app.start():
        sys.exit(1)
    
    app.run()


if __name__ == "__main__":
    main()
