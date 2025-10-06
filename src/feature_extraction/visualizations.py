# In src/visualizations.py

import matplotlib.pyplot as plt
import numpy as np
from mne.decoding import CSP

def plot_psd_and_tfr(epochs):
    """Plots PSD and TFR for exploratory analysis."""
    print("\n--- Displaying exploratory visualizations ---")
    epochs.plot_psd()
    plt.show()
    tfr = epochs.compute_tfr(method='multitaper', freqs=np.arange(8., 31., 2.), average=True)
    tfr.apply_baseline(mode='logratio', baseline=(None, 0.0))
    tfr.plot(picks=['C3', 'Cz', 'C4'], title='Time-Frequency Power', combine='mean')
    plt.show()

def plot_csp_patterns(epochs):
    """Fits CSP and plots the spatial patterns."""
    X = epochs.copy().crop(tmin=1.0, tmax=4.0).get_data()
    y = epochs.events[:, -1]
    csp = CSP(n_components=6, reg='ledoit_wolf')
    csp.fit(X, y)
    epochs.set_montage('standard_1020', on_error='warn')
    csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    plt.suptitle("CSP Spatial Patterns")
    plt.show()

def plot_scores_over_time(epochs, scores):
    """Plots classification accuracy over time."""
    chance_level = 1 / len(epochs.event_id)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs.times, scores, label='Score')
    ax.axvline(0, linestyle="--", color="k", label="Cue Onset")
    ax.axhline(chance_level, linestyle="-", color="k", label=f"Chance ({chance_level*100:.0f}%)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Classification Score Over Time")
    ax.legend(loc="lower right")
    plt.show()