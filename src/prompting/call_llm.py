import json
from openai import OpenAI
import math

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
        for row in record.message_iter(): 
            messageId = row["messageId"]
            message = row["message"]

            # Check if the messageId already has responses
            response_count = record.count_responses(messageId)

            if response_count >= ntimes:
                print(f"Already generated {response_count} responses for messageId {row['messageId']}.")
                continue

            config = {key: row[key] for key in record.config_keys}  # Extract config from row

            # Generate responses for each message 'ntimes' times
            responses = [
                self.generate_one_response(config, message)
                for _ in range(ntimes - response_count)
            ]

            # Append all generated responses to the record
            for response in responses:
                record.add_response(messageId, response)
        return record

def main(_):
    client = OpenAI(
        api_key=openai_keys[0],
    )

    record_path = FLAGS.record_path
    ntimes = FLAGS.ntimes
    record = RenderedPromptRecord.load_from_file_static(record_path)
    assert record 

    caller = LlmCaller(client)
    record = caller.feed_into_llm(record,ntimes)
    record.save_to_mirror_file()
    print(record)


if __name__ == '__main__':
    absl.app.run(main)
