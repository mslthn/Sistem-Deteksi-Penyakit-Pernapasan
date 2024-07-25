import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import librosa
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Memuat model SVM
model_filename = '/home/anton/Documents/SKRIPSI_ANTON/SKRIPSI/models/svm_model.pkl'
clf = joblib.load(model_filename)

# Memuat LabelEncoder
le = LabelEncoder()
le.classes_ = np.load('/home/anton/Documents/SKRIPSI_ANTON/SKRIPSI/models/class.npy')

# Fungsi untuk membuat folder baru jika belum ada
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Fungsi untuk membuat nama fail
def get_next_filename(directory, base_filename, extension):
    i = 1
    while os.path.exists(os.path.join(directory, f"{base_filename}_{i}.{extension}")):
        i += 1
    return os.path.join(directory, f"{base_filename}_{i}.{extension}")

# Fungsi untuk merekam audio
def record_audio(duration=15, sr=48000):
    clear_display()  # Menghapus tampilan sebelumnya
    status_label.config(text="Sedang merekam...")
    root.update_idletasks()  # Update GUI untuk menampilkan pesan
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float64')
    sd.wait()
    audio = audio.flatten()
    
    # Membuat folder baru
    directory = '/home/anton/Documents/SKRIPSI_ANTON/recorded_audio'
    create_directory(directory)
    
    # Mendapatkan nama fail berikutnya
    filename = get_next_filename(directory, 'recorded_audio', 'wav')
    
    # Menyimpan audio perekaman
    write(filename, sr, audio)
    
    status_label.config(text="Perekaman selesai!")
    root.update_idletasks()  # Update GUI untuk menampilkan pesan
    return audio, sr

# Fungsi untuk plot audio waveform
def plot_waveform(audio, sr, width, height):
    # Menghitung ukuran grafik untuk 1/4 ukuran jendela
    fig_width = width / 2 / 96  # Konversi piksel ke inci (dengan asumsi 96 DPI)
    fig_height = height / 2 / 96  # Konversi piksel ke inci (dengan asumsi 96 DPI)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(np.linspace(0, len(audio) / sr, num=len(audio)), audio)
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Amplitude', fontsize=8)
    ax.set_title('Audio Waveform', fontsize=8)
    return fig

def normalize_audio(audio):
    peak_val = np.max(np.abs(audio))
    return audio / peak_val

# Fungsi untuk filtering audio
def filter_audio(audio, sr, low_freq=250, high_freq=2000):
    sos = butter(10, [low_freq, high_freq], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, audio)

# Fungsi untuk preprocessing: Normalisasi dan Filtering
def preprocess_audio(audio_list, sr_list):
    processed_audio_list = []
    processed_sr_list = []
    for audio, sr in zip(audio_list, sr_list):
        audio = normalize_audio(audio)
        filtered_audio = filter_audio(audio, sr)
        processed_audio_list.append(filtered_audio)
        processed_sr_list.append(sr)
    return processed_audio_list, processed_sr_list

def calculate_spectral_centroid(audio, sr):
    S = np.abs(librosa.stft(audio)) # Menghitung STFT dari audio
    freqs = np.linspace(0, sr / 2, S.shape[0]) # Mendapatkan frekuensi untuk setiap bin
    magnitudes = np.abs(S) # Menghitung magnitudo spektrum
    # Menghitung spectral centroid
    spectral_centroid = np.sum(freqs[:, np.newaxis] * magnitudes, axis=0) / np.sum(magnitudes, axis=0)    
    # Mengembalikan nilai rata-rata spectral centroid
    return np.mean(spectral_centroid)

def calculate_spectral_entropy(audio, sr):
    S = np.abs(librosa.stft(audio))
    S_db = librosa.amplitude_to_db(S)
    S_db[S_db <= 0] = np.finfo(float).eps
    spectral_entropy = -np.sum(S_db * np.log2(S_db))
    return spectral_entropy

def calculate_spectral_spread(audio, sr):
    S = np.abs(librosa.stft(audio))  # Menghitung STFT dari audio
    freqs = np.linspace(0, sr / 2, S.shape[0])  # Mendapatkan frekuensi untuk setiap bin
    magnitudes = np.abs(S)  # Menghitung magnitudo spektrum
    spectral_centroid = np.sum(freqs[:, np.newaxis] * magnitudes, axis=0) / np.sum(magnitudes, axis=0)  # Menghitung spectral centroid
    spectral_spread = np.sqrt(np.sum(((freqs[:, np.newaxis] - spectral_centroid) ** 2) * magnitudes, axis=0) / np.sum(magnitudes, axis=0))  # Menghitung spectral spread
    return np.mean(spectral_spread)  # Mengembalikan nilai rata-rata spectral spread

# Fungsi untuk ekstraksi fitur spektral
def extract_features(audio_list, sr_list):
    features = []
    for audio, sr in zip(audio_list, sr_list):
        spectral_centroid = calculate_spectral_centroid(audio, sr)
        spectral_entropy = calculate_spectral_entropy(audio, sr)
        spectral_spread = calculate_spectral_spread(audio, sr)
        
        features.append([spectral_centroid, spectral_entropy, spectral_spread])
    return np.array(features)

# Fungsi untuk normalisasi fitur
def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # Menghindari pembagian dengan nol
    return (features - mean) / std

# Fungsi untuk klasifikasi audio
def classify_audio(audio, sr):
    filtered_audio = filter_audio(audio, sr)
    audio = normalize_audio(audio)
    features = extract_features([filtered_audio], [sr])
    #normalized_features = normalize_features(features)
    predictions = clf.predict(features)
    return le.inverse_transform([predictions[0]])

# Fungsi untuk menyisipkan dan memuat fail audio
def browse_file():
    clear_display()  # Menghapus tampilan sebelumnya
    global recorded_audio, recorded_sr
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        status_label.config(text="Memuat fail audio...")
        root.update_idletasks()  # Update GUI untuk menampilkan pesan
        recorded_audio, recorded_sr = librosa.load(file_path, sr=48000)
        fig = plot_waveform(recorded_audio, recorded_sr, width, height)
        canvas = FigureCanvasTkAgg(fig, master=waveform_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)
        
        # Menambahkan label status di bawah grafik
        status_label.config(text="Audio berhasil dimuat!")
        status_label.grid(row=2, column=1, pady=10)
        
        root.update_idletasks()  # Update GUI untuk menampilkan pesan

# Fungsi untuk memulai perekaman
def start_recording():
    clear_display()  # Menghapus tampilan sebelumnya
    global recorded_audio, recorded_sr
    recorded_audio, recorded_sr = record_audio()
    fig = plot_waveform(recorded_audio, recorded_sr, width, height)
    canvas = FigureCanvasTkAgg(fig, master=waveform_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

# Fungsi untuk klasifikasi audio yang terekam atau dimuat
def classify_recorded_audio():
    if recorded_audio is not None and recorded_sr is not None:
        result = classify_audio(recorded_audio, recorded_sr)
        classification_result.config(text=f"Hasil: {result}", font=("Arial", 12))
    else:
        messagebox.showwarning("Warning", "Belum ada audio yang terekam ataupun termuat!")

# Fungsi untuk menghapus tampilan grafik
def clear_display():
    for widget in waveform_frame.winfo_children():
        widget.destroy()
    classification_result.config(text="Hasil: -")
    status_label.config(text="")

# Fungsi untuk keluar program
def exit_program():
    root.quit()

# Fungsi untuk memainkan audio
def play_audio():
    if recorded_audio is not None and recorded_sr is not None:
        sd.play(recorded_audio, recorded_sr)
        sd.wait()
    else:
        messagebox.showwarning("Warning", "Belum ada audio yang terekam ataupun termuat!")

# GUI setup
root = tk.Tk()
root.title("Sistem Deteksi Penyakit Pernapasan")

# Mengatur ukuran window menjadi 12x8 inch
dpi = 96
width = 800 #5 * dpi
height = 480 #3 * dpi
root.geometry(f"{width}x{height}")

recorded_audio = None
recorded_sr = None

# Mengatur frame untuk layout
waveform_frame = tk.Frame(root, width=width * 0.3, height=height * 0.3, bd=2, relief="solid")
waveform_frame.grid(row=0, column=1, padx=10, pady=10)

left_frame = tk.Frame(root, width=width * 0.25)
left_frame.grid(row=0, column=0, sticky="n")

right_frame = tk.Frame(root, width=width * 0.25)
right_frame.grid(row=0, column=2, sticky="n")

# Mengatur kolom grid untuk memiliki lebar yang sama
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)

# Status label
status_label = tk.Label(root, text="", font=("Arial", 12))
status_label.grid(row=1, column=1, pady=10)

# Tombol di kiri
left_inner_frame = tk.Frame(left_frame)
left_inner_frame.pack(expand=True)

browse_button = tk.Button(left_inner_frame, text="Sisipkan Fail", command=browse_file)
browse_button.pack(pady=10)

record_button = tk.Button(left_inner_frame, text="Rekam Suara", command=start_recording)
record_button.pack(pady=10)

# Hasil klasifikasi dan tombol dikanan
right_inner_frame = tk.Frame(right_frame)
right_inner_frame.pack(expand=True)

classification_result = tk.Label(right_inner_frame, text="Hasil: -", font=("Arial", 12))
classification_result.pack(pady=10)

classify_button = tk.Button(right_inner_frame, text="Mulai Klasifikasi", command=classify_recorded_audio)
classify_button.pack(pady=10)

# Tombol untuk memainkan audio
play_button = tk.Button(right_inner_frame, text="Putar Audio", command=play_audio)
play_button.pack(pady=10)

# Tombol untuk keluar program
exit_button = tk.Button(right_inner_frame, text="Keluar", command=exit_program)
exit_button.pack(pady=10)

root.mainloop()