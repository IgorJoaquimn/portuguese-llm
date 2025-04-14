"""
Running this file obtains the words that distinguish a target group from the corresponding
unmarked ones.

A common task in the quantitative analysis of text is to determine how documents differ from each other concerning word usage. This is usually achieved by identifying words that are particular for one document but not for another.
"""
import json
import pickle
import signal
import logging
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
logging.getLogger().setLevel(logging.ERROR)

avoid_zero_division = 1e-10

def generate_counts(corpus, ngrams=2):
    """
    Generate the vocabulary of the corpus
    """
    cv  = CountVectorizer(
            analyzer='word',
            decode_error='ignore',
            min_df=0.0,
            max_df=0.8, # Ignore words that appear in more than 80% of the documents
            ngram_range=(1, ngrams),
            max_features=10000,
    )
    # Generate the count vectors
    count_vector = cv.fit_transform(corpus)
    vocab = cv.get_feature_names_out()
    return count_vector,vocab

def get_log_odds(target_words, unmarked_words, prior_words, ngrams=2):
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


def masked_personas(record,target_group, unmarked_group,ngrams=2):
    # Prior = all words
    pass
    

if __name__ == '__main__':
    # Example usage
    target_words   = ["one two", "two", "three", "one", "two", "three", "ten"]
    unmarked_words = ["one two", "two", "three", "one", "two", "three", "five","one"]
    prior_words = ["this", "is"]


    z_scores_dict = get_log_odds(target_words, unmarked_words, prior_words, ngrams=3)
    print(z_scores_dict)
