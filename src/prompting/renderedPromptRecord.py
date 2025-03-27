import pickle
import pandas as pd
from token_count import TokenCount

template_suffix = ".tmpl"

class RenderedPromptRecord():
    def __init__(self,
                 original_prompt,
                 prompt_path,
                 messages,
                 configs):
        self.original_prompt = original_prompt
        self.prompt_path = prompt_path
        self.messages = messages
        self.configs = configs
        self.responses = []
        
        self.data = pd.DataFrame(columns = [
            "original_prompt",
            "prompt_path",
            "message",
            "response"
        ] + list(configs.keys()))

    def replace_config(config):
        self.configs = len(self.configs) * [config]

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

    def append_response(self,responses):
        self.responses.append(responses)

    def generate_token_count(self):
        token_counts = []
        for message, config in zip(self.messages,self.configs):
            tc = TokenCount(model_name=config["model"])
            token_counts.append(tc.num_tokens_from_string(message[0].content[0].text))
        return token_counts


    def __str__(self):
        header =  f"""Prompt path: {self.prompt_path}\n"""

        body = ""
        if(self.responses):
            for message, config,responses in zip(self.messages,self.configs, self.responses):
                body += f"\tMessage: {message[0].content[0].text}\n"
                body += f"\tConfig: {config}\n"
                if(responses):
                    body += f"\tFirst response: {responses[0]}\n"
                body += "-"*80 + "\n" 
        else:
            for message, config in zip(self.messages,self.configs):
                body += f"\tMessage: {message[0].content[0].text}\n"
                body += f"\tConfig: {config}\n"
                body += "-"*80 + "\n" 
        return header + body
    
