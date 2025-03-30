import json
from openai import OpenAI
import math

import absl.flags
import absl.app
from src.envs import openai_keys, gemini_keys
from src.adapters.client_factory import ClientFactory
from src.prompting.renderedPromptRecord import RenderedPromptRecord
import pickle

FLAGS = absl.flags.FLAGS

# Definição das flags
absl.flags.DEFINE_string("record_path", None, "Path that contains the desired record")
absl.flags.DEFINE_integer("ntimes", 1, "Number of times that the response will be generated")
absl.flags.mark_flag_as_required("record_path")

class LlmCaller():
    def __init__(self,client_factory):
        self.client_factory = client_factory

    def generate_one_response(self,config,message):
        return self.client.create(
            **config, 
            messages=message
        )


    def feed_into_llm(self, record, ntimes=1):
        for row in record.message_iter(): 
            messageId = row["messageId"]
            # Check if the messageId already has responses
            response_count = record.count_responses(messageId)
            if response_count >= ntimes:
                print(f"Already generated {response_count} responses for messageId {messageId}")
                continue

            config = {key: row[key] for key in record.config_keys}  # Extract config from row

            if(config["model"] != "gemini-2.0-flash"): continue
            self.client = self.client_factory.get_client(config["model"])

            message = row["message"]
            # Generate responses for each message 'ntimes' times
            responses = [ str(i)
                # self.generate_one_response(config, message)
                for i in range(ntimes - response_count)
            ]

            # Append all generated responses to the record
            for response in responses:
                record.add_response(messageId, response)
        return record

def main(_):

    record_path = FLAGS.record_path
    ntimes = FLAGS.ntimes
    record = RenderedPromptRecord.load_from_file_static(record_path)
    assert record 

    client_factory = ClientFactory()
    client_factory.openai_keys = openai_keys
    client_factory.gemini_keys = gemini_keys

    caller = LlmCaller(client_factory)
    record = caller.feed_into_llm(record,ntimes)
    record.save_to_mirror_file()
    print(record)


if __name__ == '__main__':
    absl.app.run(main)
