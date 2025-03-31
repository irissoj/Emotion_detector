from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import EmotionPrediction
from .serializers import EmotionPredictionSerializer
from .ml_model import predict_text_emotion, predict_audio_emotion, predict_video_emotion

@api_view(['POST'])
def predict_emotion(request):
    text = request.data.get('text', None)
    audio = request.FILES.get('audio', None)
    video = request.FILES.get('video', None)

    emotion = "neutral"
    if text:
        emotion = predict_text_emotion(text)
    elif audio:
        emotion = predict_audio_emotion(audio.temporary_file_path())
    elif video:
        emotion = predict_video_emotion(video.temporary_file_path())

    prediction = EmotionPrediction.objects.create(text=text, audio=audio, video=video, predicted_emotion=emotion)
    return Response(EmotionPredictionSerializer(prediction).data)       