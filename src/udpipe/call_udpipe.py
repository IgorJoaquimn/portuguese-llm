import sys
import json
import pickle
import signal
import logging
import requests
import absl.app
import absl.flags


from src.prompting.renderedPromptRecord import RenderedPromptRecord
from src.udpipe.LinguisticComplexityAnalyzer import LinguisticComplexityAnalyzer

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
                 model="portuguese-bosque-ud-2.12-230717"):
        self.url = url
        self.model = model
        self.stats_generator = stats_generator
        self.data_metadata = {
            'tokenizer': '',
            'tagger': '',
            'parser': '',
            'model': model,
        }
        signal.signal(signal.SIGTERM, self.handle_sigint)
        signal.signal(signal.SIGINT, self.handle_sigint)

    def generate_one_response(self,message):
        request_param = self.data_metadata.copy()
        request_param["data"] = message
        response = requests.post(self.url, data=request_param)
        # Check if the response is valid
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        # Check if the response is valid
        if "result" not in response.json():
            raise Exception(f"Error: {response.status_code}, {response.text}")
        udpipe_output = response.json()["result"]
        return udpipe_output
        
    def feed_into_udpipe(self, record, generate_stats=True):
        self.record = record
        self.record.generate_responseId()
        records_all = len(self.record.response_data)
        tabs = "\t" * 2
        print(f"Total records to process:{tabs}{records_all}")
        for i,row in enumerate(self.record.response_iter()): 
            responseId = row["responseId"]
            print(f"[{i:04d}/{records_all}]Processing row with responseId:{tabs}{responseId}")
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
            # Call udpipe API
            response = self.generate_one_response(message)

            # Call statistics
            stats = {}
            if generate_stats:
                stats = self.stats_generator.generate_statistics(response)

            self.record.add_udpipe(responseId, response,stats)

            print(f"Done generating responses for responseId \t\t{responseId}\n")
        return self.record

    def handle_sigint(self,signal_received, frame):
        signal_str = signal.Signals(signal_received).name
        print(f"\n{signal_str} detected! Running cleanup function...")
        # Perform any cleanup here
        self.record.save_to_mirror_file()  # Save the record to a file
        sys.exit(0)  # Exit the program gracefully

def main(_):

    record_path = FLAGS.record_path
    record = RenderedPromptRecord.load_from_file_static(record_path)
    assert record 

    caller = UdpipeCaller(URL,
                        stats_generator=LinguisticComplexityAnalyzer())
    record = caller.feed_into_udpipe(record,
                        generate_stats=True)
    record.save_to_mirror_file()


if __name__ == '__main__':
    absl.app.run(main)
