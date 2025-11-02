import asyncio
from src.adapters.generic_client import GenericClient
from openai import AsyncOpenAI
from openai import OpenAI
from aiolimiter import AsyncLimiter


class MaritacaClient(GenericClient):
    def __init__(self, api_key):
        # Maritaca uses OpenAI-compatible API with custom base URL
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://chat.maritaca.ai/api"
        )
        super().__init__(api_key, client)
        
        # Initialize the rate limiter for 1000 requests per 60 seconds
        self.limiter = AsyncLimiter(1000, 60)

    def _convert_message_format(self, messages):
        """Convert message objects to OpenAI API format"""
        if isinstance(messages, list):
            converted_messages = []
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # If it's already in the right format
                    if isinstance(msg.content, str):
                        converted_messages.append({"role": msg.role, "content": msg.content})
                    elif hasattr(msg.content, '__iter__') and len(msg.content) > 0:
                        # Handle content[0].text format
                        if hasattr(msg.content[0], 'text'):
                            converted_messages.append({"role": msg.role, "content": msg.content[0].text})
                        else:
                            converted_messages.append({"role": msg.role, "content": str(msg.content[0])})
                    else:
                        converted_messages.append({"role": msg.role, "content": str(msg.content)})
                elif isinstance(msg, dict):
                    # If it's already a dict, pass it through
                    converted_messages.append(msg)
                else:
                    # Fallback: try to extract text content
                    if hasattr(msg, 'content') and hasattr(msg.content[0], 'text'):
                        converted_messages.append({"role": "user", "content": msg.content[0].text})
                    else:
                        converted_messages.append({"role": "user", "content": str(msg)})
            return converted_messages
        else:
            # Single message case
            if hasattr(messages, 'content') and hasattr(messages.content[0], 'text'):
                return [{"role": "user", "content": messages.content[0].text}]
            else:
                return [{"role": "user", "content": str(messages)}]

    async def create_async(self, config, messages):
        converted_messages = self._convert_message_format(messages)
        
        # Asynchronously wait for the limiter before making the API call
        async with self.limiter:
            completion = await self.client.chat.completions.create(
                **config, 
                messages=converted_messages
            )
        
        return completion.choices[0].message.content
