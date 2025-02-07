"""
EEG Analysis Script

This script performs the following:
1. Channel Selection: Filters EEG data to retain specific channels.
2. Coherence Calculation: Computes coherence between EEG channels.
3. Phase Lag Index (PLI) Calculation: Computes the PLI between EEG channels.

Author: [Your Name]
Date: [Today's Date]

Dependencies:
- NumPy
- SciPy
"""

import numpy as np
from scipy.signal import hilbert, welch

# Function for Channel Selection
def select_channels(eeg_data, channel_names, selected_channels):
    """
    Filters EEG data to retain only selected channels.
    
    Parameters:
        eeg_data (np.ndarray): EEG data [n_channels x n_timepoints]
        channel_names (list): List of all channel names
        selected_channels (list): List of desired channels
    
    Returns:
        np.ndarray: Filtered EEG data [n_selected_channels x n_timepoints]
    """
    indices = [channel_names.index(ch) for ch in selected_channels if ch in channel_names]
    filtered_data = eeg_data[indices, :]
    return filtered_data

# Function for Coherence Calculation
def compute_coherence(eeg_data, sampling_rate):
    """
    Computes coherence between all pairs of EEG channels.
    
    Parameters:
        eeg_data (np.ndarray): EEG data [n_channels x n_timepoints]
        sampling_rate (float): Sampling rate of the EEG data
    
    Returns:
        np.ndarray: Coherence matrix [n_channels x n_channels]
    """
    n_channels = eeg_data.shape[0]
    coherence_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                coherence_matrix[i, j] = 1.0  # Coherence with itself is 1
                continue
            
            f, Pxx = welch(eeg_data[i], fs=sampling_rate)
            _, Pyy = welch(eeg_data[j], fs=sampling_rate)
            _, Pxy = welch(eeg_data[i] * eeg_data[j], fs=sampling_rate)
            
            coherence_matrix[i, j] = (np.abs(Pxy)**2) / (Pxx * Pyy)
    
    return coherence_matrix

# Function for Phase Lag Index (PLI)
def compute_pli(eeg_data):
    """
    Computes the Phase Lag Index (PLI) between all pairs of EEG channels.
    
    Parameters:
        eeg_data (np.ndarray): EEG data [n_channels x n_timepoints]
    
    Returns:
        np.ndarray: PLI matrix [n_channels x n_channels]
    """
    n_channels = eeg_data.shape[0]
    pli_matrix = np.zeros((n_channels, n_channels))

    analytic_signal = hilbert(eeg_data, axis=1)
    phases = np.angle(analytic_signal)

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                pli_matrix[i, j] = 0.0  # PLI with itself is undefined
                continue
            
            phase_diff = phases[i] - phases[j]
            pli_matrix[i, j] = np.abs(np.mean(np.sign(np.sin(phase_diff))))
    
    return pli_matrix

if __name__ == "__main__":
    # Example EEG data (32 channels x 1000 time points)
    eeg_data = np.random.rand(32, 1000)  # Replace with actual EEG data
    channel_names = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                     'O1', 'O2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                     'T1', 'T2', 'Cz', 'Pz', 'Oz', 'Fz', 'FC1', 'FC2',
                     'CP1', 'CP2', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2']
    selected_channels = ['Fp2', 'F8', 'T4', 'T6', 'O2', 'Fp1', 'F7', 'T3',
                         'T5', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3']
    sampling_rate = 256  # Hz

    # Step 1: Channel Selection
    filtered_eeg_data = select_channels(eeg_data, channel_names, selected_channels)
    print(f"Filtered EEG data shape: {filtered_eeg_data.shape}")

    # Step 2: Compute Coherence
    coherence_matrix = compute_coherence(filtered_eeg_data, sampling_rate)
    print(f"Coherence Matrix Shape: {coherence_matrix.shape}")

    # Step 3: Compute PLI
    pli_matrix = compute_pli(filtered_eeg_data)
    print(f"PLI Matrix Shape: {pli_matrix.shape}")
