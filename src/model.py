"""Baseline TF-IDF + Logistic Regression text classifier."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_vectorizer(ngram_range=(1, 1), min_df=2):
    """Build a TF-IDF vectorizer with the given n-gram range."""
    return TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)


def train_logistic_regression(X, y, C=1.0, seed=42):
    """Fit a logistic regression classifier with balanced class weights."""
    model = LogisticRegression(
        C=C, class_weight="balanced", max_iter=1000, random_state=seed,
    )
    model.fit(X, y)
    return model
