from django.apps import AppConfig


from numpy import vectorize

class SentimentAnalysisConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sentiment_analysis'

