import sys
import json
import pickle
import signal
import logging
import absl.app
import absl.flags

from src.envs import openai_keys, gemini_keys
from src.adapters.client_factory import ClientFactory
from src.prompting.renderedPromptRecord import RenderedPromptRecord

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

FLAGS = absl.flags.FLAGS

# Definição das flags
absl.flags.DEFINE_string("record_path", None, "Path that contains the desired record")
absl.flags.DEFINE_integer("ntimes", 1, "Number of times that the response will be generated")
absl.flags.mark_flag_as_required("record_path")

class LlmCaller():
    def __init__(self,client_factory):
        self.client_factory = client_factory
        signal.signal(signal.SIGINT, self.handle_sigint)
        signal.signal(signal.SIGTERM, self.handle_sigint)

    def generate_one_response(self,config,message):
        return self.client.create(
            config, message
        )


    def feed_into_llm(self, record, ntimes=1):
        self.record = record
        for row in self.record.message_iter(): 
            print(f"Processing row with messageId:\t\t{row['messageId']}")
            messageId = row["messageId"]
            # Check if the messageId already has responses
            response_count = record.count_responses(messageId)
            if response_count >= ntimes:
                print(f"Already generated {response_count} responses for messageId \t{messageId}\n")
                continue

            config = {key: row[key] for key in record.config_keys}  # Extract config from row
            if(config["model"] != "gemini-2.0-flash"): continue
            self.client = self.client_factory.get_client(config["model"])

            message = row["message"]
            # Generate responses for each message 'ntimes' times
            print(f"Generating {ntimes - response_count} responses for messageId \t{messageId}")
            responses = [ 
                self.generate_one_response(config, message)
                for i in range(ntimes - response_count)
            ]

            # Append all generated responses to the record
            for response in responses:
                self.record.add_response(messageId, response)

            print(f"Done generating responses for messageId \t{messageId}\n")
        return self.record



    def handle_sigint(self,signal_received, frame):
        signal_str = signal.Signals(signal_received).name
        print(f"\n{signal_str} detected! Running cleanup function...")
        # Perform any cleanup here
        self.record.save_to_mirror_file()  # Save the record to a file
        sys.exit(0)  # Exit the program gracefully

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
