import os
import json
from promptl_ai import Promptl

import itertools
from tqdm import tqdm

import absl.flags
import absl.app
from src.prompting.renderedPromptRecord import RenderedPromptRecord

FLAGS = absl.flags.FLAGS

# Definição das flags
absl.flags.DEFINE_string("prompt_path", None, "Path that contains the desired template")
absl.flags.DEFINE_string("trait_list_path",None,"Path of the traits JSON file - can be null")
absl.flags.DEFINE_string("model_config_folder","model_config/","Path of the config files")
absl.flags.mark_flag_as_required("prompt_path")
absl.flags.mark_flag_as_required("trait_list_path")


class PromptRenderGenerator():
    def __init__(self,traits, model_config_folder = "model_config/"):
        self.promptl = Promptl()
        self.traits_comb = list(itertools.product(*traits.values()))
        self.traits_keys = traits.keys()
        self.model_config_folder = model_config_folder

    def trait_comb_to_dict(self,trait_list):
        return dict(zip(self.traits_keys,trait_list))

    def enhance_traits(self,trait_dict):
        artigo_mapping = {
            "homem": "o",
            "mulher": "a",
            "não-binária": "a",
        }

        pronome_mapping = {
            "homem": "ele",
            "mulher": "ela",
            "não-binária": "elu",
        }

        trait = trait_dict.get("genero", "")
        trait_dict["artigo"] = artigo_mapping.get(trait, "")
        trait_dict["pronome"] = pronome_mapping.get(trait, "")
        return trait_dict

    @staticmethod
    def read_from_file(path):
        with open(path,"r") as f:
            return f.read()

    def treat_message(self,message):
        text = message[0].content[0].text
        text = " ".join(text.split())
        message[0].content[0].text = text
        return message

    def generate_prompt_from_template(self,prompt_template):
        # Format the prompt using Promptl
        messages_list = []
        config_list = []
        trait_list = []
        for trait_comb in tqdm(self.traits_comb):
            traits = self.trait_comb_to_dict(trait_comb)
            traits = self.enhance_traits(traits)
            messages, config= self.promptl.prompts.render(
                prompt=prompt_template,
                parameters=traits
            )
            messages_list.append(self.treat_message(messages[1]))
            config_list.append(config[1])
            trait_list.append(traits)
        return messages_list, config_list, trait_list
    
    def generate_record(self,prompt_path):
        configs = [
            PromptRenderGenerator.read_from_file(self.model_config_folder + file) 
            for file in os.listdir(self.model_config_folder)
        ]
        print(configs)
        prompt_template = PromptRenderGenerator.read_from_file(prompt_path)
        print("Generating prompts from template",prompt_path)
        render = [
            self.generate_prompt_from_template(config + prompt_template) 
            for config in configs
        ]
        # messages_list, configs_list, trait_list = zip(*render)
        record = RenderedPromptRecord(prompt_template, prompt_path)
        # Iterate directly over the 'render' object
        for message_contents, config_dicts, trait_values in render:
            # Assuming 'message_content' is already the string/content you want,
            # and not a list like message[0] was implying.
            for message_content, config_dict, trait_value in zip(message_contents, config_dicts, trait_values):
                record.add_message(prompt_template, config_dict, trait_value, message_content[0])

        return record

def main(_):
    prompt_path = FLAGS.prompt_path
    trait_list_path = FLAGS.trait_list_path
    model_config_folder = FLAGS.model_config_folder
    traits = json.load(open(trait_list_path))

    promptRenderGenerator = PromptRenderGenerator(traits,model_config_folder)
    record = promptRenderGenerator.generate_record(prompt_path)
    record.save_to_mirror_file()
    record = record.load_from_file(record.new_path)
    print(record)

if __name__ == '__main__':
    absl.app.run(main)
