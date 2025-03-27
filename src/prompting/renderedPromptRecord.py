import pickle
import pandas as pd
from token_count import TokenCount
from uuid import uuid5,NAMESPACE_DNS

template_suffix = ".tmpl"

class RenderedPromptRecord():
    def __init__(self,original_prompt,prompt_path):
        self.original_prompt = original_prompt
        self.prompt_path = prompt_path
        self.messages  = []
        self.configs   = []
        self.responses = []

        self.message_data = pd.DataFrame()

        self.response_data = pd.DataFrame()

    def add_message(self,original_prompt,config,trait,message):
        message_id = uuid5(NAMESPACE_DNS, 
                           original_prompt + str(trait) + str(config)).int

        message_record = {
            "messageId": message_id,
            "message": message,
            "trait": trait,
            "original_prompt": original_prompt,
            **config  # Unpack configs into columns
        }
        self.message_data = pd.concat(
            [self.message_data, pd.DataFrame([message_record])], ignore_index=True
        )
        self.configs = config
        return self.message_data

    def append_response(self,messageId, response):
        response_record = {
            "messageId": messageId,
            "response": response
        }
        self.response_data = pd.concat(
            [self.response_data, pd.DataFrame([response_record])], ignore_index=True
        )
        return self.response_data


    def save_to_mirror_file(self):
        if template_suffix not in self.prompt_path: raise EnvironmentError(self.prompt_path)
        prefix_index = self.prompt_path.find(template_suffix)
        self.new_path = self.prompt_path[:prefix_index] + "_rendered.pickle"
        pickle.dump(self, open(self.new_path,"wb"))

    @staticmethod
    def load_from_file_static(path):
        return pickle.load(open(path,"rb"))

    def load_from_file(self,path):
        return pickle.load(open(path,"rb"))

    def generate_token_count(self):
        token_counts = []
        
        for _, row in self.message_data.iterrows():  # Iterate over DataFrame rows
            message = row["message"]
            model_name = row.get("model", None)  # Get model name if it exists in configs

            if hasattr(message, "content") and message.content:
                text = message.content[0].text  # Extract the actual text
            else:
                text = str(message)  # Fallback if message is not in expected format
            
            tc = TokenCount(model_name=model_name)  # Initialize TokenCount with model
            token_counts.append(tc.num_tokens_from_string(text))  # Compute token count

        return token_counts

    def __iter__(self):
        """Iterator that yields each row in message_data as a dictionary."""
        for _, row in self.message_data.iterrows():
            yield row.to_dict()  # Convert each row to a dictionary and yield it


    def __str__(self):
        """Returns a readable string representation of the object."""
        # Convert the 'message' column using `.content[0].text`
        if not self.message_data.empty:
            self.message_data["message_text"] = self.message_data["message"].apply(
                lambda msg: msg.content[0].text if hasattr(msg, "content") and msg.content else str(msg)
            )

        return (
            f"RenderedPromptRecord:\n"
            f"Original Prompt: {self.original_prompt}\n"
            f"Prompt Path: {self.prompt_path}\n\n"
            f"Message Data:\n{self.message_data[['messageId','message_text']].to_string(index=False)}\n\n"
            f"Response Data:\n{self.response_data.to_string(index=False)}"
        )

