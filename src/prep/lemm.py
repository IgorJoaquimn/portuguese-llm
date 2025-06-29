# %%
import re

# %%
import pandas as pd
df = pd.read_parquet("data/merged_data.parquet")

from pt_lemmatizer import Lemmatizer

# %%
import nltk
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

# Initialize Portuguese lemmatizer
lemmatizer = Lemmatizer()


# %%

def lemmatize_text(text):
    """
    Lemmatize text using pt_lemmatizer
    """
    # Handle null or empty text
    if pd.isna(text) or not text.strip():
        return ""
    
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text.lower())
    
    # Split into words and lemmatize each word
    words = text.split()
    # Filter out short words and apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if len(word) > 2]
    return ' '.join(lemmatized_words)


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