import asyncio
<<<<<<< HEAD:src/adapters/gemini_adapter.py
import time
import google.generativeai as genai
from src.adapters.default_adapter import GenericClient
=======
from google import genai
from src.adapters.generic_client import GenericClient
>>>>>>> ec9c2280e5837fda8031f4dea2ff568e73148850:src/adapters/gemini_client.py

# This is a wrapper for the Gemini API, which is a generative AI model by Google.
# Should follow the rate limit of 15 requests per minute.
class GeminiClient(GenericClient):
    def __init__(self, api_key):
        """Initialize the Gemini client."""
        client = genai.Client(api_key = api_key)
        super().__init__(api_key, client)


    def create(self, config, messages):
        """Synchronous content generation."""
        prompt = self._format_messages(messages)
<<<<<<< HEAD:src/adapters/gemini_adapter.py
        response = self.client.generate_content(prompt, **config)
        # Follow the rate limit of 15 requests per minute.
        time.sleep(4)
        # Check if the response contains candidates and parts
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return None  # Handle cases where no text is generated.
=======
        response = self.client.models.generate_content(**config, contents=prompt)
        return response.text

>>>>>>> ec9c2280e5837fda8031f4dea2ff568e73148850:src/adapters/gemini_client.py

    async def create_async(self, config, messages):
        """Asynchronous content generation."""
        loop = asyncio.get_running_loop()
        prompt = self._format_messages(messages)
        response = await loop.run_in_executor(None, self.client.generate_content, prompt, **config)
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return None

    async def send_batch_messages(self, config, messages):
        """Asynchronously send a batch of messages."""
        tasks = [self.create_async(config, [msg]) for msg in messages] #gemini messages are not an array like openai.
        responses = await asyncio.gather(*tasks)
        return responses

    def _format_messages(self, message):
        """Formats messages for Gemini."""
        formatted_prompt = ""
        role = message.role
        for c in message.content:
            formatted_prompt += c.text + "\n"
        return formatted_prompt
