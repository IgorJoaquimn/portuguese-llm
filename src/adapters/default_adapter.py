import asyncio

class GenericClient():
    def __init__(self,api_key,client):
        self.api_key = api_key
        self.client = client

    def create(self,config,messages):
        pass

    async def create_async(self,config,messages):
        pass

    async def send_batch_messages(self,config,messages):
        pass
