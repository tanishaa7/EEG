#svm+fbcsp
import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from mne.decoding import CSP
from scipy.signal import butter, lfilter

# Define bandpass filter function
def bandpass_filter(data, low, high, sfreq, order=4):
    nyquist = 0.5 * sfreq
    low = low / nyquist
    high = high / nyquist
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data, axis=-1)

# Define the base path where the files are located
base_path = '/Users/tanisha/Desktop/eeg/'

# Generate the file paths dynamically
file_paths = [f'{base_path}preprocessed_eeg_raw_{i}.fif' for i in range(1, 40) if i != 13]

# Initialize lists to hold data and labels
all_X = []
all_y = []

# Process each file using a for loop
for file_path in file_paths:
    # Load the EEG data
    raw = mne.io.read_raw_fif(file_path, preload=True)

    # Extract relevant channels: 1:17 and 19:30 (0-based index)
    selected_channels = list(range(0, 17)) + list(range(18, 30))
    raw.pick_channels([raw.ch_names[i] for i in selected_channels])

    # Step 1: Detect or create events
    try:
        events = mne.find_events(raw, stim_channel='STI 014')  # Adjust stim_channel if needed
        event_id = {'task1': 1, 'task2': 2}  # Update event IDs based on detected events
    except ValueError:
        print(f"No events found in {file_path}, creating synthetic events.")
        events = mne.make_fixed_length_events(raw, id=1, duration=2.0)
        events[1::2, 2] = 2  # Assign alternating labels for synthetic events
        event_id = {'task1': 1, 'task2': 2}

    # Step 2: Create epochs
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=2, baseline=None, preload=True)

    # Step 3: Extract data and labels
    X = epochs.get_data()
    y = events[:, -1]

    # Truncate y to match the number of epochs in X
    y = y[:len(X)]

    # Append to the lists
    all_X.append(X)
    all_y.append(y)

# Convert lists into numpy arrays
all_X = np.concatenate(all_X, axis=0)
all_y = np.concatenate(all_y, axis=0)

print(f"Shape of all_X: {all_X.shape}")
print(f"Length of all_y: {len(all_y)}")

# Step 4: Split the data into training and testing sets (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(
    all_X, all_y, test_size=0.4, random_state=42, stratify=all_y
)

# Step 5: Define frequency bands for FBCSP
freq_bands = [(10, 13), (13, 16), (16, 19), (19, 22), (22, 25)]
sfreq = 250  # Sampling frequency
n_components = 2  # Number of CSP components per band

# Step 6: Apply FBCSP
train_features = []
test_features = []

for low, high in freq_bands:
    # Bandpass filter the data
    X_train_filt = np.array([bandpass_filter(epoch, low, high, sfreq) for epoch in X_train])
    X_test_filt = np.array([bandpass_filter(epoch, low, high, sfreq) for epoch in X_test])

    # Apply CSP
    csp = CSP(n_components=n_components, reg=0.1, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train_filt, y_train)
    X_test_csp = csp.transform(X_test_filt)

    # Append the features
    train_features.append(X_train_csp)
    test_features.append(X_test_csp)

# Concatenate features across all bands
X_train_fbcsp = np.concatenate(train_features, axis=1)
X_test_fbcsp = np.concatenate(test_features, axis=1)

print(f"Shape of X_train_fbcsp: {X_train_fbcsp.shape}")
print(f"Shape of X_test_fbcsp: {X_test_fbcsp.shape}")

# Step 7: Train an SVM classifier
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train_fbcsp, y_train)

# Step 8: Make predictions
y_pred = svm.predict(X_test_fbcsp)

# Step 9: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# Step 10: Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=list(event_id.keys()))
disp.plot(cmap='viridis')
disp.ax_.set_title("Confusion Matrix")
plt.show()
