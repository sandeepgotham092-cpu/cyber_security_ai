from django.shortcuts import render
from .ml_model.phishing_classifier import detect_threat
import mysql.connector

def detector_view(request):
    result = None
    confidence = None
    mode = 'url'  # Default mode for GET requests
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')
        mode = request.POST.get('mode', 'url')  # Update mode from POST
        if input_text:
            try:
                prediction, conf = detect_threat(input_text, mode)
                result = 'Threat' if prediction == 1 else 'Legitimate'
                confidence = f"{conf:.2%}"
            except ValueError as e:
                result = f"Input Error: {str(e)}"
            except mysql.connector.Error as e:
                result = f"Database Error: {str(e)}. Please ensure MySQL is configured and models are trained."
            except FileNotFoundError as e:
                result = f"Model Error: {str(e)}. Please run phishing_detector/ml_model/phishing_classifier.py to train models."
            except Exception as e:
                result = f"Error: {str(e)}"
        else:
            result = "Input Error: Please provide a non-empty input"
    
    return render(request, 'detector.html', {
        'result': result,
        'confidence': confidence,
        'mode': mode,
    })