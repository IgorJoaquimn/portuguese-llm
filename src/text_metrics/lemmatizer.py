"""
Lemmatization analyzer using UDPipe.
"""

import logging
from typing import Dict, Optional
from .base import BaseAnalyzer
from .udpipe_client import UDPipeClient, extract_lemmas_string

logger = logging.getLogger(__name__)


class LemmatizerAnalyzer(BaseAnalyzer):
    """Handles lemmatization of Portuguese text using UDPipe."""
    
    def __init__(self):
        """Initialize the lemmatizer analyzer."""
        self.udpipe_client = UDPipeClient()
    
    def analyze(self, text: str, udpipe_output: Optional[str] = None) -> Dict:
        """
        Extract lemmas from text.
        
        Args:
            text: Original text (for reference)
            udpipe_output: Optional pre-parsed UDPipe output. If not provided, will call UDPipe API.
            
        Returns:
            Dictionary with lemmas field
        """
        if not text or not text.strip():
            return {"lemmas": ""}
        
        try:
            # If UDPipe output not provided, call the API
            if udpipe_output is None:
                udpipe_output = self.udpipe_client.generate_one_response(text)
            
            # Parse the UDPipe output
            sentences = self.udpipe_client.parse_response(udpipe_output)
            
            # Extract lemmas
            lemmas = extract_lemmas_string(sentences)
            
            return {"lemmas": lemmas}
            
        except Exception as e:
            logger.error(f"Error during lemmatization: {e}")
            return {"lemmas": ""}
