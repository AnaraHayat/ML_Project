from sklearn.feature_extraction.text import TfidfVectorizer

def train_vectorizer(X_train):
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        min_df=2
    )

    X_train_vec = tfidf.fit_transform(X_train)
    return tfidf, X_train_vec