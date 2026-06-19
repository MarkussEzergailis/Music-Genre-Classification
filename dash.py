import pandas as pd
import numpy as np
import sklearn

import librosa as lr
import joblib
import streamlit as st

from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

st.title("Music Genre Classification")


#Model + Sclaer + Min Max Scaler
st.spinner("Loading model...")
xgb = joblib.load('xgb_model.joblib')
min_max_scaler = joblib.load('min_max_scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')


#Function: Extract Features
def extract_features(input_file):
    y,sr = lr.load(input_file)

    features = pd.DataFrame()

    #Lenght of the song in seconds
    features['length'] = [len(y)]

    #chroma_stft_mean + chroma_stft_var
    features['chroma_stft_mean'] = [np.mean(lr.feature.chroma_stft(y=y, sr=sr))]
    features['chroma_stft_var'] = [np.var(lr.feature.chroma_stft(y=y, sr=sr))]

    #rms_mean + rms_var
    features['rms_mean'] = [np.mean(lr.feature.rms(y=y))]
    features['rms_var'] = [np.var(lr.feature.rms(y=y))]

    #spectral_centroid_mean + spectral_centroid_var
    features['spectral_centroid_mean'] = [np.mean(lr.feature.spectral_centroid(y=y, sr=sr))]
    features['spectral_centroid_var'] = [np.var(lr.feature.spectral_centroid(y=y, sr=sr))]

    #spectral_bandwidth_mean + spectral_bandwidth_var
    features['spectral_bandwidth_mean'] = [np.mean(lr.feature.spectral_bandwidth(y=y, sr=sr))]
    features['spectral_bandwidth_var'] = [np.var(lr.feature.spectral_bandwidth(y=y, sr=sr))]

    #rolloff_mean + rolloff_var
    features['rolloff_mean'] = [np.mean(lr.feature.spectral_rolloff(y=y, sr=sr))]
    features['rolloff_var'] = [np.var(lr.feature.spectral_rolloff(y=y, sr=sr))]

    #zero_crossing_rate_mean + zero_crossing_rate_var
    features['zero_crossing_rate_mean'] = [np.mean(lr.feature.zero_crossing_rate(y))]
    features['zero_crossing_rate_var'] = [np.var(lr.feature.zero_crossing_rate(y))]

    #harmony_mean + harmony_var
    features['harmony_mean'] = [np.mean(lr.effects.harmonic(y))]
    features['harmony_var'] = [np.var(lr.effects.harmonic(y))]

    #perceptr_mean + perceptr_var
    features['perceptr_mean'] = [np.mean(lr.effects.percussive(y))]
    features['perceptr_var'] = [np.var(lr.effects.percussive(y))]

    #tempo
    features['tempo'] = [lr.beat.tempo(y=y, sr=sr)[0]]

    #mfcc1_mean + mfcc1_var + ... + mfcc20_mean + mfcc20_var
    mfccs = lr.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = [np.mean(mfccs[i])]
        features[f'mfcc{i+1}_var'] = [np.var(mfccs[i])]
    



    return features

#Function: Predict Genre
def predict_genre(input_file):
    features = extract_features(input_file)
    features_scaled = min_max_scaler.transform(features)
    predicted_label_encoded = xgb.predict(features_scaled)
    predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
    return predicted_label[0]

#File Uploader
uploaded_file = st.file_uploader("Choose a music file", type=["wav", "mp3", "ogg"])
if uploaded_file is not None:
    with st.spinner("Extracting features and predicting genre..."):
        predicted_genre = predict_genre(uploaded_file)
    st.success(f"Predicted Genre: {predicted_genre}")
