import asyncio
from src.adapters.generic_client import GenericClient
from openai import AsyncOpenAI
from openai import OpenAI

class OpenAIClient(GenericClient):
    def __init__(self, api_key):
        client = OpenAI(api_key=api_key)
        super().__init__(api_key, client)

    def create(self, config, messages):
        completion = self.client.chat.completions.create(
            **config, 
            messages=messages
        )
        return completion.choices[0].message.content
    
    async def create_async(self, config, messages):
        completion = await self.client.chat.completions.create(
            **config, 
            messages=messages
        )
        return completion.choices[0].message.content
    
    async def send_batch_messages(self,config,messages):
        tasks = [self.create_async(config,msg) for msg in messages]
        responses = await asyncio.gather(*tasks)
        return responses
