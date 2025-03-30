import asyncio
from google import genai
from src.adapters.generic_client import GenericClient

class GeminiClient(GenericClient):
    def __init__(self, api_key):
        """Initialize the Gemini client."""
        client = genai.Client(api_key = api_key)
        super().__init__(api_key, client)

    def create(self, config, messages):
        """Synchronous content generation."""
        prompt = self._format_messages(messages)
        response = self.client.generate_content(prompt, **config)
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return None  # Handle cases where no text is generated.

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

    def _format_messages(self, messages):
        """Formats messages for Gemini."""
        formatted_prompt = ""
        for message in messages:
            if message["role"] == "user":
                formatted_prompt += message["content"] + "\n" #gemini expects plain text.
            elif message["role"] == "assistant":
                formatted_prompt += message["content"] + "\n" #gemini expects plain text.
            elif message["role"] == "system":
                formatted_prompt += message["content"] + "\n" #gemini expects plain text.
        return formatted_prompt
