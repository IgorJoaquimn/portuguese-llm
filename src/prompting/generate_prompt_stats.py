import os

import numpy as np
import absl.flags
import absl.app
from src.prompting.renderedPromptRecord import RenderedPromptRecord

FLAGS = absl.flags.FLAGS

# Definição das flags
absl.flags.DEFINE_string("record_folder", None, "Path that contains the desired record")
absl.flags.mark_flag_as_required("record_folder")

def main(_):
    record_folder = FLAGS.record_folder
    records_path = [record_folder + f for f in os.listdir(record_folder) if "_rendered.pickle" in f]
    print(records_path)
    records  = [RenderedPromptRecord.load_from_file_static(file) for file in records_path]

    token_counts = [np.sum(r.generate_token_count()) for r in records]
    token_count_all = np.sum(token_counts)
    token_count_mean = np.mean(token_counts)
    n_messsages = np.sum([len(r.messages) for r in records])
    n_messsages_mean = np.mean([len(r.messages) for r in records])

    print("Token count all",  token_count_all)
    print("Token count mean", token_count_mean)
    print("Messages count all",n_messsages)
    print("Messages count mean",n_messsages_mean)


if __name__ == '__main__':
    absl.app.run(main)
