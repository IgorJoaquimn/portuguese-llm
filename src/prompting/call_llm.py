import json
from openai import OpenAI

import absl.flags
import absl.app
from src.envs import openai_keys
from src.prompting.renderedPromptRecord import RenderedPromptRecord
import pickle

FLAGS = absl.flags.FLAGS

# Definição das flags
absl.flags.DEFINE_string("record_path", None, "Path that contains the desired record")
absl.flags.DEFINE_integer("ntimes", 1, "Number of times that the response will be generated")
absl.flags.mark_flag_as_required("record_path")

class LlmCaller():
    def __init__(self,client):
        self.client = client 

    def generate_one_response(self,config,message):
        return self.client.create(
            **config, 
            messages=message
        )


    def feed_into_llm(self, record, ntimes=1):
        for row in record:  # Use the iterator to go through message_data
            messageId = row["messageId"]
            message = row["message"]
            config = {key: row[key] for key in record.configs.keys()}  # Extract config from row

            # Generate responses for each message 'ntimes' times
            responses = [
                self.generate_one_response(config, message)
                for _ in range(ntimes)
            ]

            # Append all generated responses to the record
            for response in responses:
                record.add_response(messageId, response)

def main(_):
    client = OpenAI(
        api_key=openai_keys[0],
    )

    record_path = FLAGS.record_path
    ntimes = FLAGS.ntimes
    record = RenderedPromptRecord.load_from_file_static(record_path)
    assert record 

    caller = LlmCaller(client)
    caller.feed_into_llm(record,ntimes)
    record.save_to_mirror_file()
    print(record)


if __name__ == '__main__':
    absl.app.run(main)
