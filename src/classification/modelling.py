# In src/modelling.py

import numpy as np
from mne.decoding import CSP, SlidingEstimator, cross_val_multiscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

def get_csp_lda_pipeline():
    """Returns a standard CSP+LDA classification pipeline."""
    return make_pipeline(
        CSP(n_components=6, reg='ledoit_wolf'),
        LinearDiscriminantAnalysis()
    )

def evaluate_model(epochs):
    """Performs a standard cross-validation for overall accuracy."""
    X = epochs.copy().crop(tmin=1.0, tmax=4.0).get_data()
    y = epochs.events[:, -1]
    clf = get_csp_lda_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
    return np.mean(scores)

def get_scores_over_time(epochs):
    """Performs the sliding window analysis."""
    X = epochs.get_data()
    y = epochs.events[:, -1]
    clf = get_csp_lda_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    time_decoder = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=-1)
    return np.mean(scores, axis=0)