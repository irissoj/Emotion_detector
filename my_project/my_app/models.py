from django.db import models

class EmotionPrediction(models.Model):
    text = models.TextField(blank=True, null=True)
    audio = models.FileField(upload_to='audios/', blank=True, null=True)
    video = models.FileField(upload_to='videos/', blank=True, null=True)
    predicted_emotion = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)