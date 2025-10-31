import sys
import json
import pickle
import signal
import logging
import requests
import absl.app
import absl.flags
import time
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.prompting.renderedPromptRecord import RenderedPromptRecord
from src.text_metrics.linguistic_complexity import LinguisticComplexityAnalyzer
from src.udpipe.udpipe_utils import UDPipeClient

logging.getLogger().setLevel(logging.ERROR)

FLAGS = absl.flags.FLAGS

# Definição das flags
absl.flags.DEFINE_string("record_path", None, "Path that contains the desired record")
absl.flags.mark_flag_as_required("record_path")

URL = 'http://lindat.mff.cuni.cz/services/udpipe/api/process'

class UdpipeCaller():
    def __init__(self,
                 url,
                 stats_generator = None,
                 model="portuguese-bosque-ud-2.12-230717",
                 save_interval=10):
        self.url = url
        self.model = model
        self.stats_generator = stats_generator
        self.save_interval = save_interval  # Save every N processed items
        self.processed_count = 0
        self.failed_items = []  # Track failed items for retry
        self.udpipe_client = UDPipeClient(model=model)
        signal.signal(signal.SIGTERM, self.handle_sigint)
        signal.signal(signal.SIGINT, self.handle_sigint)

    def generate_one_response(self, message, max_retries=3, retry_delay=5):
        """Generate response with retry logic and error handling."""
        return self.udpipe_client.generate_one_response(message)
        
    def save_failed_items(self):
        """Save failed items to a separate file for later retry."""
        if self.failed_items:
            try:
                failed_path = self.record.file_path.replace('.pickle', '_failed_items.json')
                with open(failed_path, 'w') as f:
                    json.dump(self.failed_items, f, indent=2)
                print(f"Failed items saved to: {failed_path}")
            except Exception as e:
                print(f"Failed to save failed items: {e}")
        
    def feed_into_udpipe(self, record, generate_stats=True):
        self.record = record
        self.record.generate_responseId()
        records_all = len(self.record.response_data)
        tabs = "\t" * 2
        print(f"Total records to process:{tabs}{records_all}")
        
        try:
            for i,row in enumerate(self.record.response_iter()): 
                responseId = row["responseId"]
                percentage = (i / records_all * 100) if records_all > 0 else 0
                print(f"[{i:04d}/{records_all}] ({percentage:.1f}%) Processing row with responseId:{tabs}{responseId}")
                
                # Check if the responseId already has udpipe called 
                if self.record.count_udpipe(responseId) > 0:
                    print(f"Already generated udpipe for responseId \t\t{responseId}\n")
                    continue

                if("response" not in row):
                    print(f"Response not found for responseId \t\t{responseId}\n")
                    continue

                message = row["response"]

                if(message is None):
                    print(f"Response is None for responseId \t\t{responseId}\n")
                    continue
                if(message == "" or message == " "):
                    print(f"Response is empty for responseId \t\t{responseId}\n")
                    continue
                
                # Process with error handling
                try:
                    # Call udpipe API
                    response = self.generate_one_response(message)

                    # Call statistics
                    stats = {}
                    if generate_stats:
                        stats = self.stats_generator.generate_statistics(response)

                    self.record.add_udpipe(responseId, response, stats)
                    self.processed_count += 1
                    
                    # Save progress periodically
                    if self.processed_count % self.save_interval == 0:
                        self.record.save_to_mirror_file()
                        print(f"Progress saved: {self.processed_count} items processed")
                    
                    print(f"Done generating responses for responseId \t\t{responseId}\n")
                    
                except Exception as e:
                    error_info = {
                        "responseId": responseId,
                        "index": i,
                        "error": str(e),
                        "message_preview": message[:100] if message else "None"
                    }
                    self.failed_items.append(error_info)
                    print(f"Error processing responseId {responseId}: {e}")
                    print(f"Adding to failed items list. Continuing with next item...\n")
                    
                    # Save current state even on error
                    try:
                        self.record.save_to_mirror_file()
                    except Exception as save_error:
                        print(f"Warning: Failed to save after error: {save_error}")
                    
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Saving current progress...")
            self.handle_sigint(signal.SIGINT, None)
            
        except Exception as e:
            print(f"Critical error in main processing loop: {e}")
            print("Attempting to save current state...")
            try:
                self.record.save_to_mirror_file()
                self.save_failed_items()
            except Exception as save_error:
                print(f"Failed to save during error recovery: {save_error}")
            raise e
        
        finally:
            # Final save and cleanup
            try:
                self.record.save_to_mirror_file()
                self.save_failed_items()
                
                print(f"\n=== Processing Summary ===")
                print(f"Total items: {records_all}")
                print(f"Successfully processed: {self.processed_count}")
                print(f"Failed items: {len(self.failed_items)}")
                if self.failed_items:
                    print("Failed items details saved to failed_items.json")
                    
            except Exception as e:
                print(f"Error during final cleanup: {e}")
        
        return self.record

    def handle_sigint(self, signal_received, frame):
        signal_str = signal.Signals(signal_received).name if signal_received else "UNKNOWN"
        print(f"\n{signal_str} detected! Running cleanup function...")
        
        # Perform cleanup using existing mirror file save
        try:
            if hasattr(self, 'record') and self.record:
                print("Saving current record state...")
                self.record.save_to_mirror_file()
                self.save_failed_items()
                
                print(f"Emergency save completed. Processed {self.processed_count} items.")
                if self.failed_items:
                    print(f"Failed items: {len(self.failed_items)} (saved to failed_items.json)")
                    
        except Exception as e:
            print(f"Error during emergency save: {e}")
            
        print("Cleanup completed. Exiting gracefully...")
        sys.exit(0)  # Exit the program gracefully

def main(_):
    record_path = FLAGS.record_path
    
    print(f"Loading record from: {record_path}")
    try:
        record = RenderedPromptRecord.load_from_file_static(record_path)
        assert record
        print(f"Successfully loaded record with {len(record.response_data)} items")
    except Exception as e:
        print(f"Failed to load record from {record_path}: {e}")
        return
    
    try:
        caller = UdpipeCaller(URL, stats_generator=LinguisticComplexityAnalyzer())
        record = caller.feed_into_udpipe(record, generate_stats=True)
        
        # Final save
        record.save_to_mirror_file()
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Critical error during processing: {e}")
        print("Check mirror file for data recovery.")
        raise e


if __name__ == '__main__':
    absl.app.run(main)
