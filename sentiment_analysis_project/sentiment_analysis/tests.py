from django.test import TestCase
from .models import SentimentAnalysisModel

class SentimentAnalysisModelTestCase(TestCase):
    def setUp(self):
        self.sentiment_model = SentimentAnalysisModel()

    def test_negative_sentiment(self):
        text = "I hate this product."  
        expected_sentiment = 0  
        predicted_sentiment = self.sentiment_model.analyze(text)
        self.assertEqual(predicted_sentiment, expected_sentiment)

    def test_positive_sentiment(self):
        text = "I love this product!" 
        expected_sentiment = 2  
        predicted_sentiment = self.sentiment_model.analyze(text)
        self.assertEqual(predicted_sentiment, expected_sentiment)

    def test_neutral_sentiment(self):
        text = "This product is okay." 
        expected_sentiment = 1  
        predicted_sentiment = self.sentiment_model.analyze(text)
        self.assertEqual(predicted_sentiment, expected_sentiment)

    def test_very_positive_sentiment(self):
        text = "This product is amazing!" 
        expected_sentiment = 2  
        predicted_sentiment = self.sentiment_model.analyze(text)
        self.assertEqual(predicted_sentiment, expected_sentiment)

    def test_mixed_sentiment(self):
        text = "I like some features but dislike others." 
        expected_sentiment = 1  
        predicted_sentiment = self.sentiment_model.analyze(text)
        self.assertEqual(predicted_sentiment, expected_sentiment)

    def test_special_characters(self):
        text = "This product is $%#@^&" 
        expected_sentiment = 2
        predicted_sentiment = self.sentiment_model.analyze(text)
        self.assertEqual(predicted_sentiment, expected_sentiment)
