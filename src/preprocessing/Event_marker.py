import pandas as pd
import mne
import os


# load files and set dirs 

filepath = ".\\S001R04.edf"

os.path.join(os.path.dirname(__file__), 'filepath')  # donno syntax for this [updated]

raw = mne.io.read_raw_edf(filepath, preload = True) # for  preloading into memory 

print(raw.info)


# detect annotations

annots = raw.annotations

print("Annotations : ", annots)


events, event_id = mne.events_from_annotations(raw)

print("\n Events id mappings: ",event_id)
print("Events shape: ",events.shape)
print("First 10 events: ",events[:10])




label_map = {
             'T0' : 'Rest',
             'T1' : 'Left_fist',
             'T2' : 'Right_fist'
            }



mapped_events = [(e[0],e[2], label_map[list(event_id.keys())[list(event_id.values()).index(e[2])]])for e in events]

print("\nMapped events(sample):  \n")

for i in mapped_events[:10]:
    print(i)


def epoch_creator(raw, events, event_id, tmin=0.5, tmax = 3.5):
    
     """
    Create epochs from raw EEG data around event markers.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The preprocessed raw EEG dataset.
    events : np.ndarray
        Events array (from mne.events_from_annotations).
    event_id : dict
        Mapping of event labels to integer IDs.
    tmin : float
        Start time before the event (seconds).
    tmax : float
        End time after the event (seconds).
    
    Returns
    -------
    epochs : mne.Epochs
        Epoched EEG data (n_epochs, n_channels, n_times).
    """


     print(f"epoching dataset from {tmin}s to {tmax}s")

     epochs = mne.Epochs(

             raw,
             events,
             event_id = event_id,
             tmin = tmin,
             tmax = tmax,
             baseline  = None,
             preload = False
            )

     print("epochs created", epochs)
     print("Epochs data shape: ",epochs.get_data().shape)

     return epochs
    

def epoch_plot(epochs):

    """

            Plot summaries of epoch with automatic scaling
    """

    print("\n=====Epochs Summary=====")
    print(epochs)
    print(f"Number of epochs : {len(epochs)}")
    print(f"Data shape: {epochs.get_data().shape}")
    print("Event ID mappning:", epochs.event_id)


# n_epochs = number of epochs to display , channels , scalings = auto lets mne handle per channel scaling 
    epochs.plot(scalings='auto', n_epochs=4, n_channels=20, block=True)


# block = True => lets plot to remain until closed by user








      

# epoch meter

epochs = epoch_creator(raw, events, event_id, tmin=0.5, tmax = 3.5)

#plot epochs
epoch_plot(epochs)

# Psuudo Code : 

'''
# ============================================
# EEG Motor Imagery Pipeline (Simulation Pseudocode)
# ============================================



# MAIN PIPELINE
def process_primary_dataset(dataset):
    dataset = filter_bandpass(dataset)             # Step 1
    dataset = artifact_removal(dataset)            # Step 2
    dataset = channel_selector(dataset)            # Step 3
    dataset, events = event_marker(dataset)        # Step 4
    epochs = epoch_creator(dataset, events)        # Step 5
    features, labels = feature_extractor(epochs)   # Step 6
    model = classifier_trainer(features, labels)   # Step 7
    return model

# PREPROCESSING
def filter_bandpass(dataset):
    print("Bandpass filter [8-30 Hz] applied...")
    return dataset

def artifact_removal(dataset):
    print("ICA artifact removal running...")
    return dataset

def channel_selector(dataset):
    print("Selecting C3, C4, Cz channels...")
    return dataset

# EVENT HANDLING
def event_marker(dataset):
    print("Extracting T1 (Left) and T2 (Right) event markers...")
    events = {"T1": "Left", "T2": "Right"}
    return dataset, events



# FEATURE EXTRACTION
def feature_extractor(epochs):
    print("Extracting CSP / band-power features...")
    features = "feature_matrix"
    labels = "labels_vector"
    return features, labels

# CLASSIFIER
def classifier_trainer(features, labels):
    print("Training SVM/LDA classifier...")
    model = "trained_model"
    return model

# UTILITIES
def epoch_meter():
    print("Epoch window: start=-1s, end=+4s")
    return -1, 4

def label_engine(events):
    print("Mapping T1->1 (Left), T2->2 (Right)")
    labels = {"T1": 1, "T2": 2}
    return labels

# ENTRY POINT
def main():
    raw_dataset = "raw_eeg_data.edf"
    print("Loading raw EEG dataset...")
    model = process_primary_dataset(raw_dataset)
    print("Pipeline complete. Model ready.")

# Run
main()
'''




