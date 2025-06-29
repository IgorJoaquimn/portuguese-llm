#!/bin/bash

# Download NLTK data
echo "Downloading NLTK data..."
python -c "
import nltk
nltk.download('stopwords')
nltk.download('punkt')
print('NLTK data downloaded successfully')
"

# Try to install spaCy Portuguese model
echo "Installing spaCy Portuguese model..."
python -m spacy download pt_core_news_sm || echo "Failed to download spaCy Portuguese model. The script will use stemming as fallback."

echo "Setup completed!"
