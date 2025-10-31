import asyncio
from src.adapters.generic_client import GenericClient
from openai import AsyncOpenAI
from openai import OpenAI

class DeepseekClient(GenericClient):
    """
    Deepseek client that uses the OpenAI-compatible API.
    Deepseek API endpoint: https://api.deepseek.com
    Supports models: deepseek-chat, deepseek-reasoner
    """
    
    def __init__(self, api_key):
        # Use Deepseek's OpenAI-compatible endpoint
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        super().__init__(api_key, client)

    def _convert_message_format(self, messages):
        """Convert message objects to OpenAI API format (same as OpenAI)"""
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
        completion = await self.client.chat.completions.create(
            **config, 
            messages=converted_messages
        )
        return completion.choices[0].message.content
