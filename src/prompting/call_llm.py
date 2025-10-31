import sys
import signal
import logging
import absl.app
import absl.flags
import asyncio

from src.envs import openai_keys, gemini_keys, deepseek_keys
from src.adapters.client_factory import ClientFactory
from src.prompting.renderedPromptRecord import RenderedPromptRecord

logging.getLogger().setLevel(logging.ERROR)

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("record_path", None, "Path that contains the desired record")
absl.flags.DEFINE_integer("ntimes", 1, "Number of times that the response will be generated")
absl.flags.DEFINE_integer("concurrency", 6, "Number of rows to process concurrently.")
absl.flags.mark_flag_as_required("record_path")


class LlmCaller():
    def __init__(self, client_factory, concurrency_limit):
        self.client_factory = client_factory
        signal.signal(signal.SIGINT, self.handle_sigint)
        signal.signal(signal.SIGTERM, self.handle_sigint)

        # Semaphore limits how many process_row tasks can run at once
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        # Lock ensures only one task writes to the record/file at a time
        self.record_lock = asyncio.Lock()

    async def generate_one_response(self, client, config, message):
        """
        Generates one response using the *provided* client.
        This is critical for concurrency.
        """
        try:
            # We no longer use 'self.client', which was a race condition
            return await client.create_async(config, message)
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    async def process_row(self, i, row, ntimes):
        """
        This function contains the logic to process a SINGLE row.
        It will be run concurrently for many rows, limited by the semaphore.
        """

        # Wait for a "slot" to become available before proceeding
        async with self.semaphore:
            messageId = None
            try:
                messageId = row["messageId"]
                print(f"Processing row with messageId:\t\t\t{messageId} from {i+1}/{len(self.record.message_data)}")

                response_count = self.record.count_responses(messageId)
                if response_count >= ntimes:
                    print(f"Already generated {response_count} responses for messageId \t{messageId}\n")
                    return  # This row is done

                config = {key: row[key] for key in self.record.config_keys}
                if config["model"] not in ["deepseek-chat"]:
                    print(f"Skipping messageId {messageId} as model is not impl (model is {config['model']})")
                    return # This row is done

                client = self.client_factory.get_client(config["model"])
                message = row["message"]

                print(f"Generating {ntimes - response_count} responses for messageId \t\t{messageId} and model {config['model']}")

                coroutines = [
                    # Pass the local client to each coroutine
                    self.generate_one_response(client, config, message)
                    for _ in range(ntimes - response_count)
                ]

                responses = await asyncio.gather(*coroutines, return_exceptions=True)

                # Filter out None responses and exceptions
                valid_responses = []
                for response in responses:
                    if response is not None and not isinstance(response, Exception):
                        valid_responses.append(response)
                    elif isinstance(response, Exception):
                        print(f"Exception in response generation: {response}")

                if not valid_responses:
                    print(f"Failed to generate any responses for messageId \t{messageId}")
                    return # This row is done

                # Acquire the lock *before* modifying the shared record or saving the file
                async with self.record_lock:
                    for response in valid_responses:
                        self.record.add_response(messageId, response)
                    self.record.save_to_mirror_file() # Save is now atomic with add

                print(f"Successfully generated {len(valid_responses)} responses for messageId \t{messageId}")
                print(f"Progress saved for messageId \t{messageId}\n")

            except Exception as e:
                print(f"Error processing messageId {messageId}: {e}")
                # --- 7. USE LOCK for data/file safety (even on error) ---
                async with self.record_lock:
                    self.record.save_to_mirror_file()
                print(f"Progress saved after error for messageId \t{messageId}\n")
                return # Continue with next task

    # --- 8. REFACTOR: feed_into_llm now just creates tasks ---
    async def feed_into_llm(self, record, ntimes=1):
        self.record = record
        tasks = []

        # Create a task for *every* row, but don't await them yet
        for i, row in enumerate(self.record.message_iter()):
            tasks.append(
                self.process_row(i, row, ntimes)
            )

        # Now, run all tasks concurrently.
        # The semaphore inside process_row will limit the *actual*
        # concurrency to FLAGS.concurrency
        print(f"Starting processing for {len(tasks)} rows with concurrency={FLAGS.concurrency}...")
        await asyncio.gather(*tasks)

        # One final save to be safe
        async with self.record_lock:
            self.record.save_to_mirror_file()

        return self.record

    def handle_sigint(self, signal_received, frame):
        signal_str = signal.Signals(signal_received).name
        print(f"\n{signal_str} detected! Running cleanup function...")
        if hasattr(self, 'record') and self.record:
            # Note: This is synchronous, which is OK for a shutdown signal
            self.record.save_to_mirror_file()
        sys.exit(0)

# Define a synchronous wrapper for absl.app.run
def sync_main_wrapper(argv):
    # This function will be called by absl.app.run
    # Inside here, we explicitly run our async main
    print("Inside sync_main_wrapper, initiating asyncio.run(async_main_logic)")
    absl.logging.set_verbosity(absl.logging.WARNING)
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
    client_factory.deepseek_keys = deepseek_keys

    # --- 9. Pass the concurrency limit to the caller ---
    caller = LlmCaller(client_factory, FLAGS.concurrency)
    record = await caller.feed_into_llm(record, ntimes)

    # Final save is already handled by feed_into_llm
    print(f"All processing complete. Record saved to {record.new_path}")


if __name__ == '__main__':
    print("Executing absl.app.run(sync_main_wrapper)")
    absl.app.run(sync_main_wrapper)
