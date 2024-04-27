from django.db import models
import os
import pickle
import numpy as np
import re
from nltk import PorterStemmer
from nltk.corpus import stopwords

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the absolute paths to the model files
clf_path = os.path.join(current_dir, 'clf.pkl')
tfidf_path = os.path.join(current_dir, 'tfidf.pkl')

class SentimentAnalysisModel:
    @staticmethod
    def analyze(text):
        

        with open(clf_path, 'rb') as f: 
            model = pickle.load(f)

        with open(tfidf_path, 'rb') as tf:
            victorizer = pickle.load(tf)
        print("Received text:", text)
        # Transform the input text using TF-IDF vectorizer
        comment_cleaned = datacleaning(text)
        TextVictorized = victorizer.transform([comment_cleaned])

        
         

     
        sentiment = model.predict(TextVictorized)

        return sentiment
    
stopwords_set = set(stopwords.words('english'))
emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251\U0001F910-\U0001F9FF]+', flags=re.UNICODE)
def datacleaning(text):
    text = re.sub('<[^>]*>', ' ', text)  # Removing HTML tags
    text = emoji_pattern.sub('', text)  # Removing emojis
    text = re.sub(r'\W', ' ', text.lower())  # Removing non-word characters and converting to lowercase
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]  # Stemming and removing stopwords
    return " ".join(text)


    