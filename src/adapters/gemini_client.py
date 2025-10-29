import asyncio
import time
from google import genai
from src.adapters.generic_client import GenericClient
from aiolimiter import AsyncLimiter  # <-- 1. Import the limiter

class GeminiClient(GenericClient):
    def __init__(self, api_key):
        """Initialize the Gemini client."""
        client = genai.Client(api_key=api_key)
        super().__init__(api_key, client)
        
        # Initialize the rate limiter for 150 requests per 60 seconds
        self.limiter = AsyncLimiter(150, 60)

    def create(self, config, messages):
        """
        Synchronous content generation.
        
        NOTE: This method does NOT use the asynchronous rate limiter.
        It is intended for single, low-volume calls. If you call this 
        method in a fast synchronous loop, you will still get
        rate limit errors from the API server.
        """
        prompt = self._format_messages(messages)
        response = self.client.models.generate_content(**config, contents=prompt)
        return response.text

    async def create_async(self, config, messages):
        """Asynchronous content generation with rate limiting."""
        prompt = self._format_messages(messages)
        
        # Asynchronously wait for the limiter before making the API call
        async with self.limiter:
            # This code block will only execute when the
            # limiter has a token available (i.e., within the rate limit)
            response = await self.client.aio.models.generate_content(**config, contents=prompt)
        
        return response.text

    def _format_messages(self, messages):
        """Formats messages for Gemini."""
        formatted_prompt = ""
        for c in messages.content:
            formatted_prompt += c.text + "\n"
        return formatted_prompt
