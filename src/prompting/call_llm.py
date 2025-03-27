import json
from openai import OpenAI

import absl.flags
import absl.app
from envs import openai_keys
from renderedPromptRecord import RenderedPromptRecord
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
            message = row["message"]
            config = {key: row[key] for key in record.configs.keys()}  # Extract config from row

            # Append responses for each message, repeated 'ntimes' times
            record.append_response([
                self.generate_one_response(config, message)
                for _ in range(ntimes)
            ])

                
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
