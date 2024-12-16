#pre-processing code
import mne
import numpy as np
from mne.preprocessing import ICA

# Loop over subjects 11 to 50
#for subject_id in range(30, 50):
for subject_id in range(14, 50): 
#if i != 13
    # Create file paths for raw and preprocessed data
    #file_path = f'/Users/tanisha/Desktop/TEST EEG/21679035/edffile/sub-{subject_id}/eeg/sub-{subject_id}_task-motor-imagery_eeg.edf'
    #output_file = f'/Users/tanisha/Desktop/eeg/preprocessed_eeg_raw_{subject_id}.fif'
    s=str(subject_id).rjust(2, '0')
    #file_path = f'/Users/tcs/Titir Code/EEG Motor/edffile/sub-{subject_id}/eeg/sub-{subject_id}_task-motor-imagery_eeg.edf'
    #output_file = f'/Users/tcs/Titir Code/EEG Motor/newEEG/preprocessed_eeg_raw_{subject_id}.fif'
    # file_path = f'/Users/tcs/Titir Code/EEG Motor/edffile/sub-{s}/eeg/sub-{s}_task-motor-imagery_eeg.edf'
    # output_file = f'/Users/tcs/Titir Code/EEG Motor/newEEG/preprocessed_eeg_raw_{s}.fif'
    file_path = f'/Users/tanisha/Desktop/TEST EEG/21679035/edffile/sub-{s}/eeg/sub-{s}_task-motor-imagery_eeg.edf'
    output_file = f'/Users/tanisha/Desktop/eeg2/preprocessed_eeg_raw_{s}.fif'
    # Step 1: Load the raw EEG data
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Print raw data info
    print(f"\n--- Raw Data Info for subject {subject_id} ---")
    print(raw.info)

    # Plot the raw data
    raw.plot(title=f"Raw EEG Data - Subject {subject_id}", scalings="auto")

    # Step 2: Set EEG reference (average reference)
    raw.set_eeg_reference('average', projection=True)
    raw.plot(title=f"After Setting Average Reference - Subject {subject_id}")

    # Step 3: High-pass filter (to remove slow drifts)
    raw.filter(l_freq=65.0, h_freq=35.0)
    raw.filter(l_freq=4.0, h_freq=None)  # High-pass filter at 1 Hz
    raw.plot(title=f"After High-pass Filtering - Subject {subject_id}")

    # Step 4: Low-pass filter (to remove high-frequency noise)
    raw.filter(l_freq=None, h_freq=30.0)  # Low-pass filter at 40 Hz
    raw.plot(title=f"After Low-pass Filtering - Subject {subject_id}")

    # Step 5: Detect and mark bad channels
    raw.plot(title=f"Inspect for Bad Channels - Subject {subject_id}")
    # Manual step: Mark channels as bad based on visual inspection
    # Uncomment the line below if you identify bad channels manually
    # raw.info['bads'] = ['CHANNEL_NAME']  # Example: Replace CHANNEL_NAME with bad channel name

    # Step 6: Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)
    raw.plot(title=f"After Interpolating Bad Channels - Subject {subject_id}")

    # Step 7: Remove large artifacts (by amplitude thresholding)
    reject_criteria = dict(eeg=100e-6)  # 100 µV
    events = mne.make_fixed_length_events(raw, duration=2.0)  # Events for epoching
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2, baseline=None, reject=reject_criteria, preload=True)
    epochs.plot(title=f"Epochs with Artifact Rejection - Subject {subject_id}")

    # Step 8: Perform ICA
    ica = ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)

    # Plot ICA component time series
    ica.plot_sources(raw, title=f'ICA Sources - Subject {subject_id}')

    # Optionally plot individual ICA components (to inspect topographies, if montage is set)
    # If no digitization, skip this step or set a standard montage
    # montage = mne.channels.make_standard_montage('standard_1020')
    # raw.set_montage(montage)
    # ica.plot_components()

    # Identify and exclude components manually based on the source plots
    ica.exclude = []  # Add indices of components to exclude based on visual inspection

    # Apply ICA to remove artifacts
    raw_cleaned = ica.apply(raw)

    # Step 9: Save the preprocessed data
    raw_cleaned.save(output_file, overwrite=True)
    print(f"Preprocessed data for subject {subject_id} saved to {output_file}")

    # Step 10: Plot the preprocessed data
    raw_cleaned.plot(title=f"Preprocessed EEG Data - Subject {subject_id}")

