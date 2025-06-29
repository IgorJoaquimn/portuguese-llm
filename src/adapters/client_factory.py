from src.adapters.generic_client import GenericClient  
from src.adapters.openai_client  import OpenAIClient
from src.adapters.gemini_client  import GeminiClient

models_list = {
    "gpt-3.5-turbo": "openai",
    "gpt-4o": "openai",
    "gpt-4": "openai",
    "sabia-3": "openai",
    "deepseek-chat": "openai",
    "deepseek-reasoner": "openai",
    "gemini-2.0-flash": "gemini",
    "gemini-1.5-flash": "gemini",
    "gemini-2.0": "gemini",
    "gemini-1.5": "gemini",
}

class ClientFactory:
    def __init__(self):
        self.openai_keys = []
        self.gemini_keys = []
        self.sabia_keys = []
        self.deepseek_keys = []

        self.clients = {}

    def _create_client(self, model):
        """Create a client for the specified model."""
        if model == "openai":
            if not self.openai_keys:
                raise ValueError("No OpenAI keys available.")
            return OpenAIClient(api_key=self.openai_keys[0])

        elif model == "gemini":
            if not self.gemini_keys:
                raise ValueError("No Gemini keys available.")
            return GeminiClient(api_key=self.gemini_keys[0])

        else:
            raise ValueError(f"Model {model} is not supported.")

    def get_client(self,model_name):
        """Get the client for the specified model."""
        if(model_name in self.clients):
            return self.clients[model_name]

        model = models_list.get(model_name)
        if model is not None:
            self.clients[model_name] = self._create_client(model)
            return self.clients[model_name]
