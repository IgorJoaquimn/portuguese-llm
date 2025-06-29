import asyncio
import time
from google import genai
from src.adapters.generic_client import GenericClient

# This is a wrapper for the Gemini API, which is a generative AI model by Google.
# Should follow the rate limit of 2000 requests per minute.
class GeminiClient(GenericClient):
    def __init__(self, api_key):
        """Initialize the Gemini client."""
        client = genai.Client(api_key = api_key)
        super().__init__(api_key, client)


    def create(self, config, messages):
        """Synchronous content generation."""
        prompt = self._format_messages(messages)
        response = self.client.models.generate_content(**config, contents=prompt)
        return response.text

    async def create_async(self, config, messages): # Renamed to create_async to distinguish
        """Asynchronous content generation."""
        prompt = self._format_messages(messages)
        # Await the generate_content call, as it performs network I/O
        response = await self.client.aio.models.generate_content(**config, contents=prompt)
        return response.text


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
