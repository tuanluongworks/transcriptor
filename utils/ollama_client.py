"""
Ollama client for optional post-processing of transcriptions
"""
import requests
from typing import Optional, List
import config


class OllamaClient:
    """Client for interacting with local Ollama server"""
    
    def __init__(self, 
                 base_url: str = config.OLLAMA_BASE_URL,
                 model: str = config.OLLAMA_MODEL):
        """
        Initialize Ollama client
        
        Args:
            base_url: Base URL of Ollama server
            model: Model name to use
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api"
        
        # Check if server is available
        self.available = self._check_server()
        if self.available:
            print(f"Ollama server connected: {self.base_url}")
            print(f"Using model: {self.model}")
        else:
            print(f"Warning: Ollama server not available at {self.base_url}")
    
    def _check_server(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def is_available(self) -> bool:
        """Check if Ollama client is available"""
        return self.available
    
    def generate(self, prompt: str, system: Optional[str] = None, 
                 temperature: float = 0.7, stream: bool = False) -> Optional[str]:
        """
        Generate text using Ollama
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Temperature for generation (0.0 to 1.0)
            stream: Whether to stream response
            
        Returns:
            Generated text or None if failed
        """
        if not self.available:
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": stream
            }
            
            if system:
                payload["system"] = system
            
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"Ollama error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None
    
    def improve_transcription(self, text: str) -> Optional[str]:
        """
        Improve transcription by fixing punctuation, grammar, and formatting
        
        Args:
            text: Raw transcription text
            
        Returns:
            Improved text or None if failed
        """
        if not self.available or not text or text.startswith("<"):
            return None
        
        system_prompt = """You are a helpful assistant that improves speech transcriptions. 
Your task is to:
1. Add proper punctuation and capitalization
2. Fix obvious transcription errors
3. Improve readability while preserving the original meaning
4. Do NOT add or remove any actual words, only improve formatting

Return ONLY the improved text, nothing else."""
        
        prompt = f"Improve this transcription:\n\n{text}"
        
        return self.generate(prompt, system=system_prompt, temperature=0.3)
    
    def summarize_transcription(self, text: str, max_length: int = 200) -> Optional[str]:
        """
        Summarize transcription
        
        Args:
            text: Transcription text
            max_length: Maximum length of summary
            
        Returns:
            Summary or None if failed
        """
        if not self.available or not text:
            return None
        
        system_prompt = f"""You are a helpful assistant that creates concise summaries.
Create a summary of no more than {max_length} words.
Focus on the main points and key information."""
        
        prompt = f"Summarize this transcription:\n\n{text}"
        
        return self.generate(prompt, system=system_prompt, temperature=0.5)
    
    def extract_action_items(self, text: str) -> Optional[List[str]]:
        """
        Extract action items from transcription
        
        Args:
            text: Transcription text
            
        Returns:
            List of action items or None if failed
        """
        if not self.available or not text:
            return None
        
        system_prompt = """You are a helpful assistant that extracts action items from transcriptions.
Identify tasks, to-dos, and action items mentioned in the text.
Return each action item on a new line, starting with a dash (-)."""
        
        prompt = f"Extract action items from this transcription:\n\n{text}"
        
        result = self.generate(prompt, system=system_prompt, temperature=0.3)
        
        if result:
            # Parse action items
            items = [line.strip('- ').strip() for line in result.split('\n') 
                    if line.strip().startswith('-')]
            return items if items else None
        
        return None
    
    def format_as_meeting_notes(self, text: str) -> Optional[str]:
        """
        Format transcription as structured meeting notes
        
        Args:
            text: Transcription text
            
        Returns:
            Formatted notes or None if failed
        """
        if not self.available or not text:
            return None
        
        system_prompt = """You are a helpful assistant that formats transcriptions into structured meeting notes.
Create well-organized notes with sections like:
- Summary
- Key Points
- Decisions
- Action Items
- Next Steps

Format with markdown headers and bullet points."""
        
        prompt = f"Format this transcription as meeting notes:\n\n{text}"
        
        return self.generate(prompt, system=system_prompt, temperature=0.5)
    
    def chat(self, messages: List[dict]) -> Optional[str]:
        """
        Send chat messages to Ollama
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Response text or None if failed
        """
        if not self.available:
            return None
        
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False
            }
            
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                print(f"Ollama chat error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error in Ollama chat: {e}")
            return None


if __name__ == "__main__":
    # Test Ollama client
    print("Testing Ollama client...")
    
    client = OllamaClient()
    
    if client.is_available():
        print("\nTesting transcription improvement...")
        test_text = "hello this is a test transcription without proper punctuation"
        improved = client.improve_transcription(test_text)
        print(f"Original: {test_text}")
        print(f"Improved: {improved}")
        
        print("\nTesting summarization...")
        long_text = """The meeting discussed the new project requirements. 
        The team agreed to use Python for the backend and React for the frontend. 
        We need to complete the design phase by next week. 
        John will handle the database setup. Sarah will work on the API design."""
        summary = client.summarize_transcription(long_text)
        print(f"Summary: {summary}")
        
        print("\nTesting action item extraction...")
        action_text = """We need to complete the design by Friday. 
        John should set up the database. 
        Sarah will create the API documentation."""
        actions = client.extract_action_items(action_text)
        print(f"Action items: {actions}")
        
    else:
        print("Ollama server not available. Make sure it's running:")
        print("  1. Install Ollama from https://ollama.ai")
        print("  2. Run: ollama serve")
        print("  3. Pull a model: ollama pull llama2")
