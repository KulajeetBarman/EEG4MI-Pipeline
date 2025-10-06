# In src/preprocessing.py
import mne
from typing import List, Optional

def load_and_preprocess(
    file_path: str,
    bad_channels: Optional[List[str]] = None  # <-- NEW PARAMETER ADDED
) -> Optional[mne.Epochs]:
    """
    Takes a raw Physionet EDF file and performs all preprocessing steps.
    """
    print(f"--- Loading and preprocessing {file_path} ---")
    
    # --- 1. Load Data, Excluding Known Bad Channels ---
    if bad_channels is None:
        bad_channels = []
    # The 'exclude' parameter now uses the list you provide
    raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto', 
                              exclude=bad_channels)

    # --- 2. Clean Channel Names & Set Montage ---
    try:
        mapping = {ch_name: ch_name.strip('.').upper() for ch_name in raw.ch_names}
        raw.rename_channels(mapping)
        specific_mapping = {
            'FCZ': 'FCz', 'CZ': 'Cz', 'CPZ': 'CPz', 'FP1': 'Fp1', 'FPZ': 'Fpz',
            'FP2': 'Fp2', 'AFZ': 'AFz', 'FZ': 'Fz', 'PZ': 'Pz', 'POZ': 'POz',
            'OZ': 'Oz', 'IZ': 'Iz'
        }
        raw.rename_channels(specific_mapping)
        raw.set_montage('standard_1020')
    except Exception as e:
        print(f"Warning: Could not set montage: {e}")

    # --- 3. Apply Filters and References ---
    raw.annotations.rename({'T0': 'rest', 'T1': 'left_fist', 'T2': 'right_fist'})
    raw.set_eeg_reference(projection=True)
    raw.filter(l_freq=8.0, h_freq=30.0, fir_design="firwin")

    # --- 4. Create Epochs with a reasonable rejection threshold ---
    event_names = {'rest': 1, 'left_fist': 2, 'right_fist': 3}
    events, event_id = mne.events_from_annotations(raw, event_id=event_names)
    
    reject_criteria = dict(eeg=150e-6)  # Use a stricter 150µV threshold now

    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=-1.0, tmax=4.0, proj=True,
        picks='eeg', baseline=None, preload=True, reject=reject_criteria
    )
    epochs.drop_bad()
    
    if len(epochs) == 0:
        print(f"Error: All epochs were rejected. Check data quality or rejection threshold.")
        return None
        
    print(f"Preprocessing complete. Found {len(epochs)} clean epochs.")
    return epochs