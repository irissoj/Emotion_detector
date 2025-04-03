# speech_emotion/urls.py (project-level)
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('emotion.urls')),  # Replace with your app name
]