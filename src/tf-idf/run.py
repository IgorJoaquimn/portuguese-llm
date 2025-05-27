import os

import numpy as np
import absl.flags
import absl.app

from sklearn.feature_extraction.text import TfidfVectorizer
from src.prompting.renderedPromptRecord import RenderedPromptRecord

FLAGS = absl.flags.FLAGS

# Definição das flags
absl.flags.DEFINE_string("record_folder", None, "Path that contains the desired record")
absl.flags.mark_flag_as_required("record_folder")

def main(_):
    record_folder = FLAGS.record_folder
    records_path = [record_folder + f for f in os.listdir(record_folder) if ".pickle" in f]
    records  = [RenderedPromptRecord.load_from_file_static(file) for file in records_path]
    
    one_to_text = lambda x: [row["message"] for row in x.response_iter()]

    # All records to string
    corpus = []

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # Save to each token, the tf-idf value into a csv 


if __name__ == '__main__':
    absl.app.run(main)
