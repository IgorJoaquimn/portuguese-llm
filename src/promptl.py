import openai
from promptl_ai import Promptl

import itertools

import absl.flags
import absl.app

from envs import openai_keys
from renderedPromptRecord import RenderedPromptRecord

FLAGS = absl.flags.FLAGS

# Definição das flags
absl.flags.DEFINE_string("prompt_path", None, "Path that contains the desired template")
absl.flags.DEFINE_string("trait_list_path",None,"Path of the traits JSON file - can be null")
absl.flags.mark_flag_as_required("prompt_path")


class PromptRenderGenerator():
    def __init__(self,traits):
        self.promptl = Promptl()
        self.traits_comb = list(itertools.product(*traits.values()))
        self.traits_keys = traits.keys()

    def trait_comb_to_dict(self,trait_list):
        return dict(zip(self.traits_keys,trait_list))

    def read_prompt_from_file(self,path):
        with open(path,"r") as f:
            return f.read()

    def generate_response(self,prompt_template):
        # Format the prompt using Promptl
        messages_list = []
        config_list = []
        for traits_list in self.traits_comb:
            traits = self.trait_comb_to_dict(traits_list)
            messages, config= self.promptl.prompts.render(
                prompt=prompt_template,
                parameters=traits
            )
            messages_list.append(messages)
            config_list.append(config)
        return messages_list, config_list
    
    def generate_record(self,prompt_path):
        prompt_template = self.read_prompt_from_file(prompt_path)
        messages_list, config_list = self.generate_response(prompt_template)
        return RenderedPromptRecord(prompt_template, prompt_path, messages_list, config_list)

def main(_):
    prompt_path = FLAGS.prompt_path
    trait_list_path = FLAGS.trait_list_path


    promptRenderGenerator = PromptRenderGenerator({
        "genero": ["m","h"],
        "raca": ["1"],
        "regiao":["r"],
        "unused": ["ee"]
    })
    record = promptRenderGenerator.generate_record(prompt_path)
    print(record)
    record.save_to_mirror_file()
    record = record.load_from_file(record.new_path)
    print(record)


if __name__ == '__main__':
    absl.app.run(main)
