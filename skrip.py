import numpy as np
import librosa
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Import joblib untuk menyimpan model
import os
import noisereduce as nr
from scipy.signal import butter, sosfilt
from os import listdir
from os.path import isfile, join
import pathlib
import tensorflow as tf

# Fungsi untuk memuat file audio dan label
def load_audio_files_and_labels(mypath):
    filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
    filepaths = [join(mypath, f) for f in filenames]
    audio_list = []
    sr_list = []
    labels = []
    for filepath in filepaths:
        try:
            y, sr = librosa.load(filepath, sr=48000)
            audio_list.append(y)
            sr_list.append(sr)
            # Ambil label dari nama file
            if 'Normal' in filepath:
                labels.append('Normal')
            elif 'Asthma' in filepath:
                labels.append('Asthma')
            elif 'PPOK' in filepath:
                labels.append('PPOK')
            elif 'Pneumonia' in filepath:
                labels.append('Pneumonia')
            else:
                raise ValueError(f"Unexpected class label found in the file name: {filepath}")
        except Exception as e:  # Tangani semua jenis error
            print(f"Error loading {filepath}: {e}")
            continue
    return audio_list, sr_list, labels

# Fungsi untuk normalisasi audio dengan peak value normalization
def normalize_audio(audio):
    peak_val = np.max(np.abs(audio))
    return audio / peak_val

# Fungsi untuk segmentasi audio
def segment_audio(audio, sr, segment_length=2.0):
    segment_samples = int(segment_length * sr)
    segments = []
    for start in range(0, len(audio), segment_samples):
        end = start + segment_samples
        segment = audio[start:end]
        if len(segment) == segment_samples:
            segments.append(segment)
    return segments

# Fungsi untuk filtering audio
def filter_audio(audio, sr, low_freq=250, high_freq=2000):
    sos = butter(10, [low_freq, high_freq], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, audio)

# Fungsi untuk preprocessing: Filtering, Normalisasi, dan Segmentasi
def preprocess_audio(audio_list, sr_list):
    processed_audio_list = []
    processed_sr_list = []
    segment_counts = []  # Tambahkan ini untuk melacak jumlah segmen per audio
    for audio, sr in zip(audio_list, sr_list):
        normalized_audio = normalize_audio(audio)
        filtered_audio = filter_audio(normalized_audio, sr)
        segments = segment_audio(filtered_audio, sr)
        processed_audio_list.extend(segments)
        processed_sr_list.extend([sr] * len(segments))
        segment_counts.append(len(segments))  # Simpan jumlah segmen
    return processed_audio_list, processed_sr_list, segment_counts

def calculate_spectral_centroid(audio, sr):
    S = np.abs(librosa.stft(audio)) # Menghitung STFT dari audio
    freqs = np.linspace(0, sr / 2, S.shape[0]) # Mendapatkan frekuensi untuk setiap bin
    magnitudes = np.abs(S) # Menghitung magnitudo spektrum
    # Menghitung spectral centroid
    spectral_centroid = np.sum(freqs[:, np.newaxis] * magnitudes, axis=0) / np.sum(magnitudes, axis=0)
    # Mengembalikan nilai rata-rata spectral centroid
    return np.mean(spectral_centroid)

# Fungsi untuk menghitung spectral spread
def calculate_spectral_spread(audio, sr):
    S = np.abs(librosa.stft(audio))  # Menghitung STFT dari audio
    freqs = np.linspace(0, sr / 2, S.shape[0])  # Mendapatkan frekuensi untuk setiap bin
    magnitudes = np.abs(S)  # Menghitung magnitudo spektrum
    spectral_centroid = np.sum(freqs[:, np.newaxis] * magnitudes, axis=0) / np.sum(magnitudes, axis=0)  # Menghitung spectral centroid
    spectral_spread = np.sqrt(np.sum(((freqs[:, np.newaxis] - spectral_centroid) ** 2) * magnitudes, axis=0) / np.sum(magnitudes, axis=0))  # Menghitung spectral spread
    return np.mean(spectral_spread)  # Mengembalikan nilai rata-rata spectral spread

def calculate_spectral_entropy(audio, sr):
    S = np.abs(librosa.stft(audio))
    S_db = librosa.amplitude_to_db(S)
    S_db[S_db <= 0] = np.finfo(float).eps
    spectral_entropy = -np.sum(S_db * np.log2(S_db))
    return spectral_entropy

# Fungsi untuk ekstraksi fitur spektral
def extract_features(audio_list, sr_list):
    features = []
    for audio, sr in zip(audio_list, sr_list):
        spectral_centroid = calculate_spectral_centroid(audio, sr)
        spectral_spread = calculate_spectral_spread(audio, sr)
        spectral_entropy = calculate_spectral_entropy(audio, sr)
        
        features.append([spectral_centroid, spectral_entropy, spectral_spread])
    return np.array(features)

# Fungsi untuk normalisasi fitur
def normalize_features(features):
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

# Fungsi untuk melatih model SVM dengan Grid Search
def train_svm_with_grid_search(X_train, y_train):
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    return grid_search.best_estimator_

# Fungsi untuk evaluasi model
def evaluate_model(clf, X_train, y_train, X_valid, y_valid):
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_valid)
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_valid, y_pred_val)
    print("Training accuracy: %.4f%%" % (train_acc * 100))
    print("Validation accuracy: %.4f%%" % (val_acc * 100))
    return y_pred_val

# Fungsi untuk menampilkan confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Fungsi utama
def main():
    mypath = "/home/anton/Documents/SKRIPSI_ANTON/SKRIPSI/dataset-Copy"
    audio_list, sr_list, label = load_audio_files_and_labels(mypath)
    processed_audio_list, processed_sr_list, segment_counts = preprocess_audio(audio_list, sr_list)
    features = extract_features(processed_audio_list, processed_sr_list)
    normalized_features = normalize_features(features)
    
    # One-hot encode labels
    le = LabelEncoder()
    i_labels = le.fit_transform(label)
    #oh_labels = to_categorical(i_labels)
    
    # Perluas label untuk mencocokkan jumlah segmen
    expanded_labels = []
    for label, count in zip(i_labels, segment_counts):
        expanded_labels.extend([label] * count)
    
    # Debugging: Print jumlah fitur dan label
    print(f"Number of features: {len(normalized_features)}")
    print(f"Number of labels: {len(expanded_labels)}")
    
    # Pastikan jumlah label sesuai dengan jumlah fitur
    if len(expanded_labels) != len(normalized_features):
        raise ValueError(f"Number of labels ({len(expanded_labels)}) does not match number of features ({len(normalized_features)})")
    
    oh_labels = to_categorical(expanded_labels)
    
    # Split data into training and validation sets using StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=6)
    for train_index, valid_index in sss.split(normalized_features, expanded_labels):
        X_train, X_valid = normalized_features[train_index], normalized_features[valid_index]
        y_train, y_valid = oh_labels[train_index], oh_labels[valid_index]
    
    print('\nTraining Samples: {}\nValidation Samples: {}\n'.format(len(X_train), len(X_valid)))
    
    # Konversi label one-hot encoded menjadi label integer setelah split
    y_train = np.argmax(y_train, axis=1)
    y_valid = np.argmax(y_valid, axis=1)
    
    # Latih model SVM dengan Grid Search
    clf = train_svm_with_grid_search(normalized_features, expanded_labels)
    
    # Evaluasi model
    y_pred_val = evaluate_model(clf, X_train, y_train, X_valid, y_valid)
    
    # Menampilkan confusion matrix
    class_names = le.classes_
    plot_confusion_matrix(y_valid, y_pred_val, class_names)
    
    # Simpan model yang telah dilatih
    model_dir = "/home/anton/Documents/SKRIPSI_ANTON/SKRIPSI/models/"
    os.makedirs(model_dir, exist_ok=True)  # Buat direktori jika belum ada
    model_filename = os.path.join(model_dir, "svm_model.pkl")
    joblib.dump(clf, model_filename)
    print(f"Model saved to {model_filename}")

    # Simpan label kelas klasifikasi
    np.save('/home/anton/Documents/SKRIPSI_ANTON/SKRIPSI/models/class.npy', le.classes_)

if __name__ == "__main__":
    main()