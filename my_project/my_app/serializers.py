from rest_framework import serializers
from .models import EmotionPrediction

class EmotionPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmotionPrediction
        fields = '_all_'