# src/preprocess.py

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Load once (faster)
ENGLISH_STOPWORDS = set(stopwords.words('english'))

# Common Roman Urdu stopwords
ROMAN_URDU_STOPWORDS = {
    'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'se', 'mein', 'aur', 'yeh',
    'jo', 'kya', 'nahi', 'tha', 'thi', 'the', 'bhi', 'par', 'ek', 'ho',
    'na', 'koi', 'sab', 'ap', 'main', 'tum', 'woh', 'hum', 'ye', 'wo'
}

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def detect_language(text):
    """Detect language based on script."""
    if not isinstance(text, str):
        return 'english'

    # Urdu unicode range: 0600–06FF
    urdu_chars = re.findall(r'[\u0600-\u06FF]', text)
    latin_chars = re.findall(r'[a-zA-Z]', text)

    if len(urdu_chars) > len(latin_chars):
        return 'urdu'
    elif len(latin_chars) > 0:
        # Could be English or Roman Urdu — use stopwords to differentiate
        words = text.lower().split()
        roman_hits = sum(1 for w in words if w in ROMAN_URDU_STOPWORDS)
        english_hits = sum(1 for w in words if w in ENGLISH_STOPWORDS)
        return 'roman_urdu' if roman_hits >= english_hits else 'english'

    return 'english'


def clean_english(text):
    """Clean English text with lemmatization and stemming."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Keep only Latin chars

    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOPWORDS]
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)


def clean_roman_urdu(text):
    """Clean Roman Urdu — keep Latin chars, remove Roman Urdu stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Keep only Latin chars

    words = text.split()
    words = [w for w in words if w not in ROMAN_URDU_STOPWORDS]
    words = [w for w in words if w not in ENGLISH_STOPWORDS]  # Remove English stopwords too

    # Light stemming only (Urdu words don't respond well to English stemmer)
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)


def clean_urdu(text):
    """Clean Urdu script — preserve Arabic/Urdu characters, remove noise."""
    # Keep Urdu script + spaces only
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # No stemming/lemmatization — not supported for Urdu script
    return text


def clean_text(text, lang=None):
    """
    Main entry point. Auto-detects language if not provided.

    Args:
        text : input text string
        lang : 'english', 'roman_urdu', 'urdu', or None (auto-detect)

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()

    if lang is None:
        lang = detect_language(text)

    if lang == 'urdu':
        return clean_urdu(text)
    elif lang == 'roman_urdu':
        return clean_roman_urdu(text)
    else:
        return clean_english(text)