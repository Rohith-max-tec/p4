import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import resampy
data_dir = "C:\\Users\\Rohith\\Desktop\\datasets\\archive (5)\\audio_speech_actors_01-24"
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

features = []
labels = []

for subdir, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(subdir, file)
            emotion_label = file.split('-')[2]
            
            if emotion_label in emotions:
                emotion = emotions[emotion_label]
                if emotion != 'disgust':  # Skip files with 'disgust' emotion
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(emotion)

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# Train the model
model = RandomForestClassifier(n_estimators=500, random_state=0)
model.fit(X_train,y_train)

# Evaluate the model
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'emotion_recognition_model5.pkl')

