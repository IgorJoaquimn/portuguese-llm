"""
Running this file obtains the words that distinguish a target group from the corresponding
unmarked ones.

A common task in the quantitative analysis of text is to determine how documents differ from each other concerning word usage. This is usually achieved by identifying words that are particular for one document but not for another.
"""
import sys
import json
import pickle
import signal
import logging
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
logging.getLogger().setLevel(logging.ERROR)

def get_count_vector(corpus,cv):
    """
    Get the count vector of the corpus
    """
    count_vector = cv.fit_transform(corpus)
    return count_vector

def get_log_odds(target_words, unmarked_words, prior_words, ngrams=2):
    vocab = set(target_words + unmarked_words + prior_words)
    cv  = CountVectorizer(
            vocabulary=vocab,
            decode_error='ignore',
            min_df=10,
            max_df=0.8, # Ignore words that appear in more than 80% of the documents
            ngrams=(1, ngrams), 
            binarize=False,
            max_features=10000,
    )
    # Generate the count vectors
    prior = get_count_vector(prior_words, cv)
    target = get_count_vector(target_words, cv)
    unmarked = get_count_vector(unmarked_words, cv)

    assert prior.shape == target.shape == unmarked.shape, "The shape of the count vectors must be the same"

    # Get the sum over all ngrams
    prior_total = np.sum(prior, axis=0)
    target_total = np.sum(target, axis=0)
    unmarked_total = np.sum(unmarked, axis=0)    

    new_vocab = cv.vocabulary
    vocab_size = len(new_vocab)

    target_term = (target + prior) / (target_total + prior_total - (target + prior))
    unmarked_term = (unmarked + prior) / (unmarked_total + prior_total - (unmarked + prior))

    target_term = np.log(target_term + 1e-10) # Avoid log(0)
    unmarked_term = np.log(unmarked_term + 1e-10) # Avoid log(0)

    delta = target_term - unmarked_term
    
    var = 1./(target + prior) + 1./(unmarked + prior)
    z_scores = delta / np.sqrt(var)
    # Return dict<str, float> the ngram and the z_score
    z_scores_dict = {k: v for k, v in zip(cv.get_features_names_out(), z_scores)}
    return z_scores_dict


def masked_personas(record,target_group, unmarked_group,ngrams=2):
    # Prior = all words
    pass
    

if __name__ == '__main__':
    pass
