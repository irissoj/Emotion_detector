from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import FileSystemStorage
from .models import AudioFile
from .utils.audio_processor import process_audio
from .utils.emotion_predictor import audio_model, text_emotion_classifier
import google.generativeai as genai
import torch

genai.configure(api_key='GEMINI_API_KEY')
model = genai.GenerativeModel('gemini-pro')

class ProcessInput(APIView):
    def post(self, request):
        # Handle audio file
        audio_emotion = None
        if 'audio' in request.FILES:
            audio_file = request.FILES['audio']
            fs = FileSystemStorage()
            filename = fs.save(audio_file.name, audio_file)
            
            # Process audio
            features = process_audio(fs.path(filename))
            with torch.no_grad():
                output = audio_model(features)
                emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'sad']
                audio_emotion = emotions[torch.argmax(output).item()]

        # Handle text emotion
        text = request.data.get('text', '')
        text_emotion = max(
            text_emotion_classifier(text)[0],
            key=lambda x: x['score']
        )['label']

        # Generate Gemini response
        prompt = f"""User is feeling {text_emotion} through text and {
            audio_emotion if audio_emotion else 'no detected'} emotion through voice. 
            Respond appropriately to their message: {text}"""
        
        response = model.generate_content(prompt)
        return Response({'response': response.text})