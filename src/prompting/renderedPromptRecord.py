import pickle
import os
import pandas as pd
from token_count import TokenCount
from uuid import uuid5,NAMESPACE_DNS

template_suffix = ".tmplt"

class RenderedPromptRecord():
    def __init__(self,original_prompt,prompt_path):
        self.original_prompt = original_prompt
        self.prompt_path = prompt_path

        self.config_keys = {}
        self.message_data = pd.DataFrame()

        # Make the default response a empty String
        self.response_data = pd.DataFrame(columns=["messageId","response"])
        self.response_data["response"] = self.response_data["response"].astype(str)

        self.udpipe_data = pd.DataFrame()

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
        self.config_keys = list(config.keys())
        return self.message_data

    def add_response(self,messageId, response):
        response_record = {
            "messageId": messageId,
            "response": response
        }
        self.response_data = pd.concat(
            [self.response_data, pd.DataFrame([response_record])], ignore_index=True
        )
        return self.response_data
    
    def add_udpipe(self,responseId, response,stats):
        response_record = {
            "responseId": responseId,
            "udpipe_result": response,
            **stats
        }

        self.set_udpipe_if_non_existant()
        self.udpipe_data = pd.concat(
            [self.udpipe_data, pd.DataFrame([response_record])], ignore_index=True
        )
        return self.udpipe_data

    def generate_responseId(self):
        # Generate a unique ID for the response
        if self.response_data.empty:
            return
        # For each response, the UUID is the messageId + response
        self.response_data["responseId"] = self.response_data["messageId"].astype(str) + self.response_data["response"].astype(str)
        self.response_data["responseId"] = self.response_data["responseId"].apply(
            lambda x: uuid5(NAMESPACE_DNS, x).int
        )
        return self.response_data

    def set_udpipe_if_non_existant(self):
        # Check if the udpipe data already exists
        try: 
            if not self.udpipe_data.empty:
                return self.udpipe_data
        except AttributeError:
            # If udpipe_data is not defined, initialize it
            self.udpipe_data = pd.DataFrame()
            return self.udpipe_data

    def count_responses(self,messageId):
        # Count the number of responses for a given messageId
        count = self.response_data[self.response_data["messageId"] == messageId].shape[0]
        return count

    def save_to_mirror_file(self):
        if template_suffix not in self.prompt_path:
            raise ValueError(f"Prompt path must contain '{template_suffix}'")
        # Create a new directory for rendered files
        rendered_dir = "/".join(self.prompt_path.split("/")[:-1]) + "/rendered"
        # Create the directory if it doesn't exist
        os.makedirs(rendered_dir, exist_ok=True)
        # Save the rendered prompt to a new file
        prefix = rendered_dir + "/" 
        suffix = self.prompt_path.split("/")[-1].replace(template_suffix, ".pickle")
        self.new_path = prefix + suffix
        pickle.dump(self, open(self.new_path,"wb"))
        assert os.path.exists(self.new_path), f"Failed to save file at {self.new_path}"

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

    def message_iter(self):
        """Iterator that yields each row in message_data as a dictionary."""
        # Iterate over the rows of message_data
        for _, row in self.message_data.iterrows():
            yield row.to_dict()

    def response_iter(self):
        """Iterator that yields each row in response_data as a dictionary."""
        # Iterate over the rows of response_data
        for _, row in self.response_data.iterrows():
            yield row.to_dict()

    def merged_iter(self):
        """Iterator that yields each row in message_data and response_data as a dictionary."""
        # Join message_data and response_data on 'messageId'
        merged_data = pd.merge(self.message_data, self.response_data, on="messageId", how="left")
        # Fill NaN values in 'response' column with empty strings
        merged_data["response"] = merged_data["response"].fillna("")
        # Iterate over the rows of the merged DataFrame
        for _, row in merged_data.iterrows():
            yield row.to_dict()


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
            f"\n\nUDPipe Data:\n{self.udpipe_data.to_string(index=False)}"

        )

