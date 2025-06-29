import sys
import json
import pickle
import signal
import logging
import absl.app
import absl.flags
import asyncio

from src.envs import openai_keys, gemini_keys
from src.adapters.client_factory import ClientFactory
from src.prompting.renderedPromptRecord import RenderedPromptRecord

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("record_path", None, "Path that contains the desired record")
absl.flags.DEFINE_integer("ntimes", 1, "Number of times that the response will be generated")
absl.flags.mark_flag_as_required("record_path")

class LlmCaller():
    def __init__(self,client_factory):
        self.client_factory = client_factory
        signal.signal(signal.SIGINT, self.handle_sigint)
        signal.signal(signal.SIGTERM, self.handle_sigint)

    async def generate_one_response(self,config,message):
        return await self.client.create_async(
            config, message
        )

    async def feed_into_llm(self, record, ntimes=1):
        self.record = record
        for i,row in enumerate(self.record.message_iter()):
            print(f"Processing row with messageId:\t\t\t{row['messageId']} from {i+1}/{len(self.record.message_data)}")
            messageId = row["messageId"]
            response_count = record.count_responses(messageId)
            if response_count >= ntimes:
                print(f"Already generated {response_count} responses for messageId \t{messageId}\n")
                continue

            config = {key: row[key] for key in record.config_keys}
            if(config["model"] not in ["gemini-1.5-flash", "gemini-2.0-flash"]):
                print(f"Skipping messageId {messageId} as model is not impl (model is {config['model']})")
                continue
            self.client = self.client_factory.get_client(config["model"])

            message = row["message"]
            print(f"Generating {ntimes - response_count} responses for messageId \t\t{messageId}")

            coroutines = [
                self.generate_one_response(config, message)
                for _ in range(ntimes - response_count)
            ]

            responses = await asyncio.gather(*coroutines)

            for response in responses:
                self.record.add_response(messageId, response)
            
            print(f"Done generating responses for messageId \t{messageId}\n")
        return self.record

    def handle_sigint(self,signal_received, frame):
        signal_str = signal.Signals(signal_received).name
        print(f"\n{signal_str} detected! Running cleanup function...")
        if hasattr(self, 'record') and self.record:
            self.record.save_to_mirror_file()
        sys.exit(0)

# Define a synchronous wrapper for absl.app.run
def sync_main_wrapper(argv):
    # This function will be called by absl.app.run
    # Inside here, we explicitly run our async main
    print("Inside sync_main_wrapper, initiating asyncio.run(async_main_logic)")
    asyncio.run(async_main_logic(argv))

# Define your actual async logic here
async def async_main_logic(argv):
    record_path = FLAGS.record_path
    ntimes = FLAGS.ntimes
    record = RenderedPromptRecord.load_from_file_static(record_path)
    assert record

    client_factory = ClientFactory()
    client_factory.openai_keys = openai_keys
    client_factory.gemini_keys = gemini_keys

    caller = LlmCaller(client_factory)
    record = await caller.feed_into_llm(record, ntimes)
    record.save_to_mirror_file()
    print(f"Record saved to {record.new_path}")


if __name__ == '__main__':
    print("Executing absl.app.run(sync_main_wrapper)")
    absl.app.run(sync_main_wrapper)