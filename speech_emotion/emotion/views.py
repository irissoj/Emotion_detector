# emotion/views.py
import os
import logging
import numpy as np
import librosa
import torch
import torch.nn as nn
from pathlib import Path
from django.conf import settings
from django.core.files.storage import default_storage
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import google.generativeai as genai
from transformers import pipeline

# Initialize logger
logger = logging.getLogger(__name__)

# --- 1. Define the Same Model Architecture as Training Code ---
class AudioEmotionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. Model Configuration ---
EMOTIONS = [
    'neutral', 'calm', 'happy', 'sad',
    'angry', 'fearful', 'disgust', 'surprised'
]

# Model paths
MODEL_DIR = Path(settings.BASE_DIR) / 'emotion' / 'models'
MODEL_PATH = MODEL_DIR / 'local_emotion.h5'  # Your trained model file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Load Trained Model ---
try:
    # Initialize model with correct dimensions
    model = AudioEmotionModel(input_dim=180, num_classes=8)  # 40+12+128=180 features
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info("Audio emotion model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError("Failed to initialize audio model")

# --- 4. Feature Extraction (Same as Training) ---
def extract_features(file_path, duration=3, offset=0.5):
    """Match the feature extraction from training"""
    try:
        y, sr = librosa.load(file_path, duration=duration, offset=offset)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        features = np.hstack((mfcc, chroma, mel))
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        features = np.zeros(180)  # 40+12+128
    
    # Reshape for model input (seq_length, 1)
    return torch.tensor(features).reshape(-1, 1).float()

# --- 5. Audio Processing View ---
@api_view(['POST'])
def analyze_audio(request):
    if 'audio' not in request.FILES:
        return Response(
            {'error': 'No audio file provided'}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        # Save and process audio
        audio_file = request.FILES['audio']
        file_name = default_storage.save(audio_file.name, audio_file)
        file_path = default_storage.path(file_name)
        
        # Extract features
        features = extract_features(file_path)
        
        # Add batch dimension and predict
        with torch.no_grad():
            output = model(features.unsqueeze(0).to(device))
            emotion_id = torch.argmax(output).item()
        
        return Response({'emotion': EMOTIONS[emotion_id]})

    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        return Response(
            {'error': 'Audio analysis failed'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        if 'file_name' in locals():
            default_storage.delete(file_name)

# --- 6. Text Processing Setup ---
text_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# --- 7. Chat Integration with Gemini ---
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

@api_view(['POST'])
def chat(request):
    try:
        text = request.data.get('text', '')
        audio_emotion = request.data.get('audio_emotion', 'unknown')
        
        if not text:
            return Response(
                {'error': 'No text provided'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Text emotion detection
        text_results = text_classifier(text)[0]
        text_emotion = max(text_results, key=lambda x: x['score'])['label']

        # Gemini integration
        prompt = f"""User's emotional state:
        - Voice analysis: {audio_emotion}
        - Text analysis: {text_emotion}
        Respond appropriately to: "{text}"
        """
        
        response = gemini_model.generate_content(prompt)
        
        return Response({
            'text_emotion': text_emotion,
            'response': response.text
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return Response(
            {'error': 'Chat processing failed'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )