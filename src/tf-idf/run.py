import os

import numpy as np
import absl.flags
import absl.app
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from src.prompting.renderedPromptRecord import RenderedPromptRecord

FLAGS = absl.flags.FLAGS

# Definição das flags
absl.flags.DEFINE_string("record_folder", None, "Path that contains the desired record")
absl.flags.mark_flag_as_required("record_folder")

def flatten_list(records, consumer):
    """
    Flattens a list of lists and applies a consumer function to each item.
    """
    rows = []
    for record in records:
        for row in record.response_iter():
            rows.append(consumer(row))
    return rows

def main(_):
    record_folder = FLAGS.record_folder
    records_path = [record_folder + f for f in os.listdir(record_folder) if ".pickle" in f]
    records  = [RenderedPromptRecord.load_from_file_static(file) for file in records_path]
    print(f"Found {len(records)} records in {record_folder}")

    one_to_text = lambda x: x["response"]
    one_to_id = lambda x: x["responseId"]

    corpus = flatten_list(records, one_to_text)
    corpus_ids = flatten_list(records, one_to_id)

    # Filter None values and empty strings
    corpus_not_none = [i for i,text in enumerate(corpus) if text is not None and text.strip() != '']
    corpus = [corpus[i] for i in corpus_not_none]
    corpus_ids = [corpus_ids[i] for i in corpus_not_none]
    

    vectorizer = TfidfVectorizer()
    result = vectorizer.fit_transform(corpus)
    print(f"TF-IDF matrix shape: {result.shape}")
    # Get feature names (i.e., the words)
    words = vectorizer.get_feature_names_out()

    # Create DataFrame
    rows = []
    for doc_idx, doc in tqdm(enumerate(corpus), total=len(corpus), desc="Processing documents"):
        for word_idx in result[doc_idx].nonzero()[1]:
            tfidf_value = result[doc_idx, word_idx]
            rows.append({
                "ResponseId": corpus_ids[doc_idx],
                "Document": corpus[doc_idx],
                "Word": words[word_idx],
                "Document Index": doc_idx,
                "Word Index": word_idx,
                "tf-idf value": tfidf_value
            })

    df = pd.DataFrame(rows)
    # Save to parquet
    output_path = os.path.join(record_folder, "tfidf_results.parquet")
    df.to_parquet(output_path, index=False)


if __name__ == '__main__':
    absl.app.run(main)
