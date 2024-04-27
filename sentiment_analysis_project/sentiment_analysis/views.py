from django.shortcuts import render
from django.http import HttpResponse , HttpResponseBadRequest
from .models import SentimentAnalysisModel 
import logging

logger = logging.getLogger(__name__)
def analyze_sentiment(request):
    if request.method == 'POST':
        text = request.POST.get('text') 
        sentiment = SentimentAnalysisModel.analyze(text)
        return HttpResponse(sentiment)
    else:
        return HttpResponseBadRequest('Invalid request method')

def sentiment_analysis(request):
    return render(request, 'index.html')
