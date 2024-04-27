from django.contrib import admin
from django.urls import path
from sentiment_analysis.views import sentiment_analysis , analyze_sentiment

urlpatterns = [
    path('admin/', admin.site.urls),
    path('analyze/', analyze_sentiment),
    path('', sentiment_analysis, name='sentiment_analysis'),
]
