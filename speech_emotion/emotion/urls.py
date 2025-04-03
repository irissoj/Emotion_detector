from django.urls import path
from . import views

urlpatterns = [
    path('analyze-audio/', views.analyze_audio, name='analyze-audio'),
    path('chat/', views.chat, name='chat'),
]