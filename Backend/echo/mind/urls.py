from django.urls import path
from .views import ProcessInput

urlpatterns = [
    path('api/process/', ProcessInput.as_view(), name='process-input'),
]