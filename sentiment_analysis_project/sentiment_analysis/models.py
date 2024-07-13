from django.db import models
import os
import pickle
import re
from nltk.corpus import stopwords
import re
import string
import nltk
# Download the WordNet resource
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
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
        print("Received text:", text.lower())
       
        # Transform the input text using TF-IDF vectorizer
        comment_cleaned = preprocess_text(text)
        print(comment_cleaned)
        TextVictorized = victorizer.transform([comment_cleaned])

        print(TextVictorized)
     
         
         
     
        sentiment = model.predict(TextVictorized)
        return sentiment


def remove_emojis(text):

    """Remove emojis from the text."""
    if isinstance(text, str):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    else:
        return text
def preprocess_text(text):
        # Normalization
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Remove emojis
        tokens = [remove_emojis(word) for word in tokens]
        print(tokens)
        return ' '.join(tokens)