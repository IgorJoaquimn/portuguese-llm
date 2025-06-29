# %%
import spacy
import re

# %%
import pandas as pd
df = pd.read_parquet("data/merged_data.parquet")

# %%
import nltk
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# Initialize Portuguese stemmer as fallback
portuguese_stemmer = SnowballStemmer('portuguese')

# Try to load Portuguese spaCy model
nlp = None
try:
    import spacy
    nlp = spacy.load("pt_core_news_sm")
except ImportError:
    print("spaCy not installed. Please install it with: pip install spacy")
    print("Then download Portuguese model with: python -m spacy download pt_core_news_sm")
    print("Falling back to Portuguese stemmer...")
except OSError:
    print("Portuguese spaCy model not found. Please install it with:")
    print("python -m spacy download pt_core_news_sm")
    print("Falling back to Portuguese stemmer...")


# %%

def lemmatize_text(text):
    """
    Lemmatize text using spaCy for Portuguese, with fallback to Portuguese stemmer
    """
    # Handle null or empty text
    if pd.isna(text) or not text.strip():
        return ""
    
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text.lower())
    
    if nlp is not None:
        # Use spaCy for proper Portuguese lemmatization
        doc = nlp(text)
        # Get lemmas for each token, excluding stop words, punctuation, and spaces
        lemmatized_tokens = [token.lemma_ for token in doc 
                           if not token.is_stop and not token.is_punct and not token.is_space 
                           and len(token.text.strip()) > 2]
        return ' '.join(lemmatized_tokens)
    else:
        # Fallback: use Portuguese stemmer from NLTK
        words = text.split()
        # Filter out short words and apply stemming
        stemmed_words = [portuguese_stemmer.stem(word) for word in words if len(word) > 2]
        return ' '.join(stemmed_words)


# %%
# Apply lemmatization to the response column
print("Starting lemmatization of response column...")
df['response_lemmatized'] = df['response'].apply(lemmatize_text)

# %%
# Save the updated dataframe
print("Saving lemmatized data...")
df.to_parquet("data/merged_data_lemmatized.parquet", index=False)
print("Lemmatization complete! Saved to merged_data_lemmatized.parquet")

# %%
# Display sample results
print("\nSample of original vs lemmatized text:")
for i in range(min(3, len(df))):
    print(f"\nOriginal: {df.iloc[i]['response'][:100]}...")
    print(f"Lemmatized: {df.iloc[i]['response_lemmatized'][:100]}...")