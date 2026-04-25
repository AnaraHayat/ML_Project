# src/shared_features.py
import scipy.sparse as sp

def build_features(texts, tfidf, bow):
    """
    Unified feature pipeline used by BOTH training and inference.
    """
    tfidf_features = tfidf.transform(texts)
    bow_features = bow.transform(texts)

    return sp.hstack([tfidf_features, bow_features])