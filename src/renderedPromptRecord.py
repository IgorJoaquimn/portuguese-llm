import pickle


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

    def replace_config(config):
        self.configs = len(self.configs) * [config]

    def save_to_mirror_file(self):
        if template_suffix not in self.prompt_path: raise FileNotFoundError()
        prefix_index = self.prompt_path.find(template_suffix)
        self.new_path = self.prompt_path[:prefix_index] + "_rendered.pickle"

        pickle.dump(self, open(self.new_path,"wb"))

    def load_from_file(self,path):
        return pickle.load(open(path,"rb"))

    def __str__(self):
        header =  f"""Prompt path: {self.prompt_path}\n"""

        body = ""
        for message, config in zip(self.messages,self.configs):
            body += f"""Message: {message} \t Config: {config}\n"""
        
        return header + body
    
