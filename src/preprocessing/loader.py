
# loader.py

import mne
import matplotlib.pyplot as plt
import numpy as np

def load_physionet_edf(file_path, tmin=-0.2, tmax=0.8, plot_epochs_flag=True):
    """
    Loader for PhysioNet EEG EDF dataset.
    - Selects motor channels (C3, Cz, C4) even if EDF uses dots or uppercase letters.
    - Extracts epochs (X) and labels (y).
    - Optionally plots epochs.
    
    Parameters:
    -----------
    file_path : str
        Path to EDF file
    tmin, tmax : float
        Epoch window (seconds)
    plot_epochs_flag : bool
        If True, plot epoch waveforms per channel
    
    Returns:
    --------
    X : ndarray
        Epoch features (n_epochs, n_channels, n_times)
    y : ndarray
        Epoch labels
    epochs : mne.Epochs
        Epoch object
    motor_channels : list
        Final channel names used
    """
    
    # 1. Load EDF
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.filter(1., 40., fir_design='firwin')  # 1–40 Hz bandpass
    
    # 2. Select motor channels, flexible to uppercase/dots
    motor_channels = [ch for ch in raw.ch_names if ch.upper().startswith(('C3', 'C4', 'CZ'))]
    raw.pick_channels(motor_channels)
    print("Picked channels:", raw.ch_names)
    
    # 3. Extract events
    events, event_id = mne.events_from_annotations(raw)
    print("Event IDs:", event_id)
    
    # 4. Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(None, 0),
        preload=True
    )
    
    # 5. Extract features and labels
    X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]
    
    print("Features shape (X):", X.shape)
    print("Labels shape (y):", y.shape)
    
    # 6. Optional plotting
    if plot_epochs_flag:
        n_epochs, n_channels, n_times = X.shape
        time = np.arange(n_times)  # sample indices
        plt.figure(figsize=(12, 6))
        
        for ch in range(n_channels):
            plt.subplot(n_channels, 1, ch+1)
            for ep in range(n_epochs):
                plt.plot(time, X[ep, ch, :], color='gray', alpha=0.5)
            mean_epoch = X[:, ch, :].mean(axis=0)
            plt.plot(time, mean_epoch, color='red', linewidth=2, label='Mean Epoch')
            plt.title(f"Channel: {motor_channels[ch]}")
            plt.ylabel("Amplitude")
            if ch == 0:
                plt.legend()
        plt.xlabel("Time points (samples)")
        plt.tight_layout()
        plt.show()
    
    return X, y, epochs, motor_channels


# Example usage
if __name__ == "__main__":
    file_path = "S004R04.EDF"  # adjust filename/path (case-insensitive)
    X, y, epochs, motor_channels = load_physionet_edf(file_path)
