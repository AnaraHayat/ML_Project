# src/features.py

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_hybrid_features(X_train, X_test, max_features=5000):
    """
    Returns combined TF-IDF + BoW features for train and test.
    Optimized for accuracy and speed.
    """

    # -------------------------
    # TF-IDF Features
    # -------------------------
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # unigrams + bigrams (important)
        min_df=2              # ignore rare words
    )

    tfidf_train = tfidf.fit_transform(X_train)
    tfidf_test  = tfidf.transform(X_test)

    # -------------------------
    # Bag of Words Features
    # -------------------------
    bow = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2
    )

    bow_train = bow.fit_transform(X_train)
    bow_test  = bow.transform(X_test)

    # -------------------------
    # Combine Features
    # -------------------------
    X_train_hf = sp.hstack([tfidf_train, bow_train]).tocsr()
    X_test_hf  = sp.hstack([tfidf_test,  bow_test]).tocsr()

    return X_train_hf, X_test_hf, tfidf, bow