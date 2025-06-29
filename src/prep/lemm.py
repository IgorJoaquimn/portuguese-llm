# %%
import requests
from conllu import parse

# %%
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

# %%
import pandas as pd
df = pd.read_parquet("data/merged_data.parquet")

# %%
def generate_one_response(message):
    request_param = data_metadata.copy()
    request_param["data"] = message
    response = requests.post(URL, data=request_param)
    # Check if the response is valid
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}, {response.text}")
    # Check if the response is valid
    if "result" not in response.json():
        raise Exception(f"Error: {response.status_code}, {response.text}")
    udpipe_output = parse(response.json()["result"])
    return udpipe_output

# %%
URL = 'http://lindat.mff.cuni.cz/services/udpipe/api/process'
data_metadata = {
'tokenizer': '',
'tagger': '',
'parser': '',
'model': "portuguese-bosque-ud-2.12-230717",
}

# %%
sentences = generate_one_response("oi mulheres 32 anos, tudo bem?")

# %%
def extract_lemmas_string(sentences):
    """
    Extract lemmas from a list of sentences and return them as a single string.
    Filters out punctuation tokens (deprel == "punct").
    
    Args:
        sentences: List of parsed sentences from UDPipe
        
    Returns:
        str: Space-separated string of lemmas (excluding punctuation)
    """
    lemmas = []
    for sentence in sentences:
        for token in sentence:
            if token["deprel"] == "punct":
                continue
            if token["deprel"] == "nummod":
                continue
            lemmas.append(token["lemma"])
    return " ".join(lemmas)

# %%
# Example usage
lemmas_string = extract_lemmas_string(sentences)
print("Lemmas as string:", lemmas_string)

# %%
df['response_lemm'] = df['response'].parallel_apply(lambda x: extract_lemmas_string(generate_one_response(x)))


df.to_parquet("data/merged_data_lemm.parquet", index=False)