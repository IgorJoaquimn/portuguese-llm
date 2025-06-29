"""
Running this file obtains the words that distinguish a target group from the corresponding
unmarked ones.

A common task in the quantitative analysis of text is to determine how documents differ from each other concerning word usage. This is usually achieved by identifying words that are particular for one document but not for another.
"""
import logging
import numpy as np
import pandas as pd
import absl.flags
import absl.app
import nltk
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
logging.getLogger().setLevel(logging.ERROR)

avoid_zero_division = 1e-10
FLAGS = absl.flags.FLAGS

# Definir as flags
absl.flags.DEFINE_string("input_file", None, "Caminho do arquivo de entrada (Parquet)")
absl.flags.DEFINE_string("target_col", None, "Nome da coluna com os grupos de interesse")
absl.flags.DEFINE_string("target_name", None, "Nome da query a ser analisada")
absl.flags.DEFINE_string("unmarked_name", None, "Nome da query a ser analisada")
absl.flags.DEFINE_string("text_col", "response", "Nome da coluna com o texto a ser analisado")
absl.flags.DEFINE_string("model_name", "gemini-1.5-flash", "Nome do modelo a ser utilizado")
absl.flags.mark_flag_as_required("input_file")
absl.flags.mark_flag_as_required("target_col")
absl.flags.mark_flag_as_required("target_name")
absl.flags.mark_flag_as_required("unmarked_name")

def generate_counts(corpus, ngrams=2):
    """
    Generate the vocabulary of the corpus
    """
    stop_words = list(set(stopwords.words('portuguese') + stopwords.words('english') + stopwords.words('spanish')))
    cv  = CountVectorizer(
            analyzer='word',
            decode_error='ignore',
            min_df=0.01 , # Ignore words that appear in less than 2% of the documents
            max_df=0.8, # Ignore words that appear in more than 80% of the documents
            stop_words= stop_words,
            ngram_range=(1, ngrams),
            max_features=10000,
    )
    # Generate the count vectors
    count_vector = cv.fit_transform(corpus)
    vocab = cv.get_feature_names_out()
    return count_vector,vocab

def get_log_odds(target_words, unmarked_words, prior_words, ngrams=1):
    corpus = target_words + unmarked_words + prior_words
    count_vector, vocab = generate_counts(corpus, ngrams=ngrams) 
    # Slice the count vector to get the target, unmarked and prior words
    target = count_vector[:len(target_words),:]
    unmarked = count_vector[len(target_words):len(target_words)+len(unmarked_words),:]
    prior = count_vector[len(target_words)+len(unmarked_words):,:]
    # Sum over all sentences
    target = np.sum(target, axis=0)
    unmarked = np.sum(unmarked, axis=0)
    prior = np.sum(prior, axis=0)
    
    assert prior.shape == target.shape == unmarked.shape, "The shape of the count vectors must be the same"
    # Get the sum over all ngrams
    prior_total = np.sum(prior,axis=1).reshape(1,-1)
    target_total = np.sum(target,axis=1).reshape(1,-1)
    unmarked_total = np.sum(unmarked,axis=1).reshape(1,-1)

    assert prior_total.shape == target_total.shape == unmarked_total.shape, "The shape of the count vectors must be the same"
    assert prior_total.shape == (1,1), "The shape of the count vectors must be 1"
    assert target_total.shape == (1,1), "The shape of the count vectors must be 1"
    assert unmarked_total.shape == (1,1), "The shape of the count vectors must be 1"

    target_term = (target + prior) / (target_total + prior_total - (target + prior))
    unmarked_term = (unmarked + prior) / (unmarked_total + prior_total - (unmarked + prior))

    target_term = np.log(target_term + 1e-10) # Avoid log(0)
    unmarked_term = np.log(unmarked_term + 1e-10) # Avoid log(0)

    delta = target_term - unmarked_term
    
    var = 1./(target + prior + avoid_zero_division) + 1./(unmarked + prior + avoid_zero_division) # Avoid division by zero
    z_scores = delta / np.sqrt(var)
    # Return dict<str, float> the ngram and the z_score
    z_scores_dict = {}
    for i, word in enumerate(vocab):
        z_scores_dict[word] = z_scores[0,i]
    # Sort the dictionary by z_score
    z_scores_dict = dict(sorted(z_scores_dict.items(), key=lambda item: item[1], reverse=True))
    return z_scores_dict

def get_top_words(z_scores_dict, n=10):
    """
    Get the top n words from the z_scores_dict
    """
    # Filter z_scores > 1.96
    z_scores_dict = {k: v for k, v in z_scores_dict.items() if v > 1.96}
    # Sort the dictionary by z_score
    z_scores_dict = dict(sorted(z_scores_dict.items(), key=lambda item: item[1], reverse=True))
    # Get the top n words
    z_scores_dict = dict(list(z_scores_dict.items())[:n])
    return z_scores_dict

def main(_):
    input_file = FLAGS.input_file
    target_col = FLAGS.target_col
    target_name = FLAGS.target_name
    unmarked_name = FLAGS.unmarked_name
    text_col = FLAGS.text_col
    model_name = FLAGS.model_name

    # Read the input file
    df = pd.read_parquet(input_file)
    if target_col not in df.columns:
        raise ValueError(f"Column {target_col} not found in the input file")
    if target_name not in df[target_col].unique():
        raise ValueError(f"Target name {target_name} not found in the column {target_col}")
    if unmarked_name not in df[target_col].unique():
        raise ValueError(f"Unmarked name {unmarked_name} not found in the column {target_col}")
    if "model" not in df.columns:
        raise ValueError(f"Column 'model' not found in the input file")
    df = df[df["model"] == model_name]
    target = df[df[target_col] == target_name][text_col].tolist()
    unmarked = df[df[target_col] == unmarked_name][text_col].tolist()
    prior = df[~df[target_col].isin([target_name, unmarked_name])][text_col].tolist()
    
    z_scores = get_log_odds(target, unmarked, prior)
    z_scores = get_top_words(z_scores, n=20)
    # Print the top words
    print(f"Top words for target group [{target_name}] compared to unmarked group [{unmarked_name}]:")
    for word, score in z_scores.items():
        print(f"{word}: {score:.2f}")

if __name__ == '__main__':
    absl.app.run(main)
