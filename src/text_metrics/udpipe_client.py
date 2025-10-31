"""
Unified UDPipe client with retry logic and error handling.
"""

import requests
import time
import logging
from typing import Optional, List

try:
    from conllu import parse
except ImportError:
    print("Warning: conllu not installed. Install with: pip install conllu")
    parse = None

logger = logging.getLogger(__name__)

URL = 'http://lindat.mff.cuni.cz/services/udpipe/api/process'


class UDPipeClient:
    """Shared UDPipe client with retry logic and error handling."""
    
    def __init__(self, model="portuguese-bosque-ud-2.12-230717", max_retries=3, retry_delay=5):
        self.url = URL
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.data_metadata = {
            'tokenizer': '',
            'tagger': '',
            'parser': '',
            'model': model,
        }
    
    def generate_one_response(self, message: str) -> str:
        """
        Generate UDPipe response with retry logic and error handling.
        
        Args:
            message: Text to parse
            
        Returns:
            UDPipe output in CoNLL-U format
            
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                request_param = self.data_metadata.copy()
                request_param["data"] = message
                response = requests.post(self.url, data=request_param, timeout=30)
                
                # Check if the response is valid
                if response.status_code != 200:
                    raise Exception(f"HTTP Error: {response.status_code}, {response.text}")
                
                # Check if the response contains result
                response_json = response.json()
                if "result" not in response_json:
                    raise Exception(f"No result in response: {response_json}")
                
                udpipe_output = response_json["result"]
                return udpipe_output
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise Exception("Request timed out after all retries")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise Exception(f"Request failed after all retries: {e}")
                    
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise e
        
        raise Exception("All retry attempts failed")
    
    def parse_response(self, udpipe_output: str):
        """
        Parse UDPipe output into CoNLL-U format.
        
        Args:
            udpipe_output: UDPipe CoNLL-U output
            
        Returns:
            List of parsed sentences
        """
        if parse is None:
            raise ImportError("conllu package not available")
        return parse(udpipe_output)


def extract_lemmas_string(sentences) -> str:
    """
    Extract lemmas from a list of sentences and return them as a single string.
    Filters out punctuation tokens (deprel == "punct") and numerical modifiers.
    
    Args:
        sentences: List of parsed sentences from UDPipe
        
    Returns:
        str: Space-separated string of lemmas (excluding punctuation and nummod)
    """
    lemmas = []
    for sentence in sentences:
        for token in sentence:
            if token["deprel"] == "punct":
                continue
            if token["deprel"] == "nummod":
                continue
            lemmas.append(token["lemma"])
    return " ".join(lemmas)
