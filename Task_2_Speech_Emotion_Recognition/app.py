import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tempfile

# Load trained model
model = load_model("models/emotion_model.h5")

# Emotion labels
emotions = [
    'Angry',
    'Calm',
    'Disgust',
    'Fearful',
    'Happy',
    'Neutral',
    'Sad',
    'Surprised'
]

# MFCC extraction function
def extract_mfcc(file_path, n_mfcc=40):
    
    audio, sample_rate = librosa.load(file_path, sr=None)
    
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )
    
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    return mfccs_scaled

# Streamlit UI
st.title("Speech Emotion Recognition")

st.write("Upload a WAV audio file to detect emotion.")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav"]
)

if uploaded_file is not None:
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    # Extract features
    features = extract_mfcc(temp_audio_path)

    # Reshape for model
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)

    # Predict
    prediction = model.predict(features)

    predicted_label = np.argmax(prediction)

    predicted_emotion = emotions[predicted_label]

    # Display result
    st.success(f"Predicted Emotion: {predicted_emotion}")