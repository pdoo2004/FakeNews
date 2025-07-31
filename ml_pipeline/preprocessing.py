import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

class TextPreprocessor:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean text by removing HTML, special characters, etc."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_remove_stopwords(self, text):
        """Tokenize text and remove stopwords"""
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def preprocess(self, texts):
        """Full preprocessing pipeline"""
        processed_texts = []
        for text in texts:
            cleaned = self.clean_text(text)
            no_stopwords = self.tokenize_and_remove_stopwords(cleaned)
            processed_texts.append(no_stopwords)
        return processed_texts