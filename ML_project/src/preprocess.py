# src/preprocess.py

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Load once (faster)
STOPWORDS = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', ' ', text)

    words = text.split()

    # remove stopwords
    words = [w for w in words if w not in STOPWORDS]

    # lemmatize
    words = [lemmatizer.lemmatize(w) for w in words]

    # stem
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)