from django.urls import path
from cybersecurity_ai.views import detector_view

urlpatterns = [
    path('', detector_view, name='detector'),
]