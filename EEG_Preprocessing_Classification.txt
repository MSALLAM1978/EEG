# EEG Preprocessing, Feature Extraction, and Classification with Graphics

import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from mne.preprocessing import ICA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import welch
from mne.connectivity import spectral_connectivity
from sklearn.metrics import confusion_matrix, classification_report

# Load EEG Data
def load_eeg_data(file_path, montage='standard_1020', low_freq=1, high_freq=40):
    raw = mne.io.read_raw_fif(file_path, preload=True)
    raw.set_montage(montage)
    raw.filter(l_freq=low_freq, h_freq=high_freq)
    return raw

# Apply ICA for artifact removal
def apply_ica(raw, n_components=20):
    ica = ICA(n_components=n_components, random_state=42)
    ica.fit(raw)
    raw_clean = ica.apply(raw)
    return raw_clean

# Plot Raw EEG Signal
def plot_raw_eeg(raw):
    raw.plot(duration=5, n_channels=30, scalings='auto')
    plt.show()

# Segment EEG data into epochs
def segment_data(raw, event_id, tmin=-0.2, tmax=0.5):
    events, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True)
    return epochs

# Compute Power Spectral Density (PSD) Features
def compute_psd(epochs, fmin=1, fmax=40):
    psd_features = []
    for epoch in epochs:
        psd, freqs = welch(epoch, fs=epochs.info['sfreq'], nperseg=256)
        psd_mean = np.mean(psd[(freqs >= fmin) & (freqs <= fmax)], axis=0)
        psd_features.append(psd_mean)
    return np.array(psd_features), freqs

# Plot PSD
def plot_psd(epochs):
    psd, freqs = compute_psd(epochs)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, np.mean(psd, axis=0), label='Average PSD', color='b')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('EEG Power Spectral Density (PSD)')
    plt.legend()
    plt.show()

# Compute Coherence and PLI Features
def compute_connectivity(epochs, method='coh', fmin=8, fmax=30):
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']
    con, _, _, _, _ = spectral_connectivity(data, method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True)
    return con.mean(axis=0)

# Plot Coherence
def plot_coherence(epochs):
    coherence_values = compute_connectivity(epochs)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(coherence_values)), coherence_values, color='g')
    plt.xlabel('Electrode Pairs')
    plt.ylabel('Coherence')
    plt.title('EEG Coherence Across Channels')
    plt.show()

# Prepare Data for Classification
def prepare_features(raw, event_id):
    raw_clean = apply_ica(raw)
    epochs = segment_data(raw_clean, event_id)
    psd_features, _ = compute_psd(epochs)
    connectivity_features = compute_connectivity(epochs)
    features = np.hstack((psd_features, connectivity_features))
    return features

# Load dataset
file_path = "path/to/eeg_data.fif"  # Replace with actual file path
raw_data = load_eeg_data(file_path)

# Generate features and labels
event_id = {'target': 1, 'non-target': 2}  # Modify based on dataset
X = prepare_features(raw_data, event_id)
y = np.random.randint(0, 2, X.shape[0])  # Dummy labels, replace with actual labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate classifiers
def train_and_evaluate(classifier, X_train, X_test, y_train, y_test):
    clf = classifier.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {classifier.__class__.__name__}')
    plt.show()
    
    # Print classification report
    print(f"\n{classifier.__class__.__name__} Performance:")
    print(classification_report(y_test, y_pred))

# Initialize classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='linear', random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# Train and evaluate models
train_and_evaluate(rf, X_train, X_test, y_train, y_test)
train_and_evaluate(svm, X_train, X_test, y_train, y_test)
train_and_evaluate(knn, X_train, X_test, y_train, y_test)

# Plot EEG Data and Feature Distributions
plot_raw_eeg(raw_data)
plot_psd(segment_data(raw_data, event_id))
plot_coherence(segment_data(raw_data, event_id))
