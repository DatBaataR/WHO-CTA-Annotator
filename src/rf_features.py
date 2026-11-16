"""
Random Forest Feature Extractors and Model Utilities for WHO Epidemiological Column Type Annotation.
Uses the optimal Pruned Feature Set (10,004 dimensions):
  1. Word TF-IDF (max_features=5000)
  2. Character n-gram TF-IDF (char_wb, 2-4g, max_features=5000)
  3. Character Ratio Features (digit, alpha, space, punctuation ratios: 4 dims)
"""

import os
import string
import joblib
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline


def space_tokenizer(text):
    return str(text).split(" ")


class CharRatioExtractor(BaseEstimator, TransformerMixin):
    """Extracts 4 character-type ratios: digits, alphabetical, whitespace, punctuation."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            if pd.isna(text) or not str(text).strip():
                features.append([0.0, 0.0, 0.0, 0.0])
                continue
            text = str(text)
            length = len(text)
            n_digits = sum(c.isdigit() for c in text)
            n_alpha = sum(c.isalpha() for c in text)
            n_space = sum(c.isspace() for c in text)
            n_punct = sum(c in string.punctuation for c in text)
            features.append([
                n_digits / length,
                n_alpha / length,
                n_space / length,
                n_punct / length,
            ])
        return np.array(features, dtype=np.float32)


class NumericalStatsExtractor(BaseEstimator, TransformerMixin):
    """Retained for backward-compatible unpickling of legacy checkpoints."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.zeros((len(X), 8), dtype=np.float32)


class DateStatsExtractor(BaseEstimator, TransformerMixin):
    """Retained for backward-compatible unpickling of legacy checkpoints."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.zeros((len(X), 1), dtype=np.float32)


# Per-fold optimal Random Forest hyperparameters found via GridSearchCV during testing
FOLD_OPTIMAL_RF_PARAMS = {
    0: {"n_estimators": 300, "max_depth": 20, "min_samples_split": 2, "random_state": 42, "n_jobs": -1},
    1: {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "random_state": 42, "n_jobs": -1},
    2: {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2, "random_state": 42, "n_jobs": -1},
    3: {"n_estimators": 100, "max_depth": None, "min_samples_split": 5, "random_state": 42, "n_jobs": -1},
    4: {"n_estimators": 200, "max_depth": None, "min_samples_split": 5, "random_state": 42, "n_jobs": -1},
}


def build_pruned_rf_pipeline(fold_idx=0):
    """Constructs the pruned 10,004-dim Random Forest pipeline with optimal fold hyperparameters."""
    clf_params = FOLD_OPTIMAL_RF_PARAMS.get(fold_idx, FOLD_OPTIMAL_RF_PARAMS[0]).copy()

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=space_tokenizer,
        token_pattern=None,
        max_features=5000,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=5000,
    )
    char_ratio_extractor = CharRatioExtractor()

    feature_union = FeatureUnion([
        ("word_tf_idf", word_vectorizer),
        ("char_tf_idf", char_vectorizer),
        ("char_ratios", char_ratio_extractor),
    ])

    pipeline = Pipeline([
        ("features", feature_union),
        ("clf", RandomForestClassifier(**clf_params)),
    ])
    return pipeline
