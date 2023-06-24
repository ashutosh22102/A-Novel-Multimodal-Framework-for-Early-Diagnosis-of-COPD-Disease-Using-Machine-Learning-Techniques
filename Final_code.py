import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn import svm
from sklearn.preprocessing import StandardScaler

model = joblib.load('Trained_data.pkl')  # Replace 'Trained_data.pkl' with the actual model file name

scaler = joblib.load('scaler.pkl')  # Replace 'scaler.pkl' with the actual scaler file name

feature_names = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff',
                 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6',
                 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14',
                 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']

def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, mono=True)
    features = []
    for feature_name in feature_names:
        if feature_name == 'mfcc1':
            mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=1)
            features.append(np.mean(mfcc))
        else:
            if feature_name == 'chroma_stft':
                feature = librosa.feature.chroma_stft(audio, sr=sr)
            else:
                feature = getattr(librosa.feature, feature_name)(audio, sr=sr)
            features.append(np.mean(feature))
    return np.array(features).reshape(1, -1)




def predict_cough():
    audio_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav")])
    if audio_path:
        features = extract_features(audio_path)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        result_label.configure(text="Prediction: " + str(prediction[0]))
    else:
        result_label.configure(text="No audio file selected.")

window = tk.Tk()
window.title("Cough Classifier")

label = ttk.Label(window, text="Select an audio file:")
label.pack()

browse_button = ttk.Button(window, text="Browse", command=predict_cough)
browse_button.pack()

result_label = ttk.Label(window, text="Prediction: ")
result_label.pack()

window.mainloop()
