import pandas as pd
import streamlit as st
import sounddevice as sd
import librosa
import joblib
import soundfile as sf
import resampy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import requests
from bs4 import BeautifulSoup
import geocoder

# Function to set background image
def set_bg_hack_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://plus.unsplash.com/premium_photo-1681400786117-df414eb06089?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mzd8fG11c2ljJTIwYmFja2dyb3VuZHxlbnwwfHwwfHx8MA%3D%3D");
             background-size: cover;
         }}
         .stButton > button {{
             color: white;
             background-color: #1DB954;
             border-radius: 10px;
             padding: 10px;
             font-size: 16px;
             font-weight: bold;
             border: none;
             box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
             transition: 0.3s;
         }}
         .stButton > button:hover {{
             background-color: #1ED760;
             box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.3);
         }}
         .stMarkdown h1 {{
             color: #1DB954;
             text-align: center;
             font-weight: bold;
         }}
         .stMarkdown h2 {{
             color: #1ED760;
             text-align: center;
             font-weight: bold;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack_url()

st.title('🎵 Welcome to Song World 🎵')

# Load and preprocess the dataset
df = pd.read_csv('genres_v2.csv')
df = df.drop(['Unnamed: 0', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'song_name', 'title', 'key', 'mode', 'time_signature'], axis=1)
df = df.dropna(axis=0)

# Encode the genre column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['genre'] = le.fit_transform(df['genre'])

# KMeans clustering
x = df.iloc[:, [0]].values
km = KMeans(n_clusters=3)
y_hat1 = km.fit_predict(x)

# Create a DataFrame for high danceability songs
df1 = pd.read_csv('genres_v2.csv')
cx = df1.iloc[:, :].values
s2 = pd.DataFrame()
s2['high danceability'] = pd.DataFrame(cx[y_hat1 == 1, 13])
s2['danceability'] = df1[['danceability']]
s2 = s2.sort_values(by='danceability', ascending=False)
s2 = s2.iloc[:5, :]

# Function to reset session state
def reset_state():
    st.session_state.q3 = 'Select Your Mode'
    st.session_state.open_playlist = False
    st.session_state.show_dance = False
    st.session_state.show_rap = False
    st.session_state.user_input = ''

# Home button to reset state
if st.button('🏠 Home'):
    reset_state()

# Weather-based song recommendation
if st.button('☁️ Click To Listen Songs Based On Weather'):
    g = geocoder.ip('me')
    city = g.city
    url = f"https://www.google.com/search?q=weather+{city}"
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text[:2]
    temp = int(temp)
    
    def recommend_song(temperature):
        if temperature > 30:
            genre = "Chillout, Ambient, Smooth Jazz, Reggae, Tropical House"
            message = "It's hot today! Cool down with these chill and relaxing songs."
        elif 25 < temperature <= 30:
            genre = "Pop, Dance, Electronic, Latin, Hip-Hop"
            message = "It's a warm day! Enjoy these upbeat and energetic tunes."
        elif 15 < temperature <= 25:
            genre = "Indie Pop, Rock, Folk, Acoustic, Singer-Songwriter"
            message = "It's a nice day! Perfect for some feel-good music."
        elif 0 < temperature <= 15:
            genre = "Acoustic, Classical, Jazz, Soul, Blues"
            message = "It's cool outside. Warm up with these cozy and comforting songs."
        else:
            genre = "Classical, Ambient, Indie Folk, Soft Rock, Chillhop"
            message = "It's freezing! Stay warm with these soulful and mellow tracks."
        return message, genre

    message, genre = recommend_song(temp)
    st.write(f"**Your Current Live City is:** {city}")
    st.write(f"**Fetched Weather is:** {temp}°C")
    st.write(f"**Recommendation:** {message}")
    st.write(f"**Suggested Genres:** {genre}")

# Emotion recognition model
model = joblib.load('emotion_recognition_model5.pkl')

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def record_audio(duration, fs):
    st.write("🎙️ Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.write("🎤 Recording completed.")
    return audio

if st.button("🎤 Click To Prefer Songs Based on Your Voice Mode"):
    fs = 22050  # Sample rate
    duration = 2  # Duration of recording
    audio = record_audio(duration, fs)
    sf.write("recorded.wav", audio, fs)
    st.audio("recorded.wav")
    features = extract_features("recorded.wav")
    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    st.write(f"**Predicted Your Mode as:** {prediction}")
    st.write(f"Then listen to these {prediction} Songs For You")

# Initialize session state
if 'q3' not in st.session_state:
    st.session_state.q3 = 'Select Your Mode'
if 'open_playlist' not in st.session_state:
    st.session_state.open_playlist = False
if 'show_dance' not in st.session_state:
    st.session_state.show_dance = False
if 'show_rap' not in st.session_state:
    st.session_state.show_rap = False

# Select mode
q3 = st.selectbox("🎵 Listen Songs of Your Mood", options=('Select Your Mode', 'happy', 'sad'), index=('Select Your Mode', 'happy', 'sad').index(st.session_state.q3))    

if q3 == 'happy':
    st.session_state.q3 = 'happy'
    st.write('🎉 Chill by listening to these happy songs')

# Main button to open the playlist
if st.button('🎧 Open Playlist'):
    st.session_state.open_playlist = True

# Display the playlist options if the playlist is open
if st.session_state.open_playlist:
    st.header("🎶 Playlist")
    col1, col2 = st.columns(2)
    with col1:
        if st.button('💃 For Dance'):
            st.session_state.show_dance = True
            st.session_state.show_rap = False  # Reset the other button state
    with col2:
        if st.button('🎤 For Rap'):
            st.session_state.show_dance = False  # Reset the other button state
            st.session_state.show_rap = True

    # Display the corresponding playlist based on the button clicked
    if st.session_state.show_dance:
        st.write("💃 **High Danceability Songs** 💃")
        for index, row in s2.iterrows():
            st.write(f'{row[0]}')
            st.write(f'[Click to Listen]({row[0]})')
    if st.session_state.show_rap:
        st.write("🎤 **High Rap Songs** 🎤")
        # Add your code for rap songs here
