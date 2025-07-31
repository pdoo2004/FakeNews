from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class BaselineModel:
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
        self.is_trained = False
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Train the baseline model"""
        print("Training baseline TF-IDF + Logistic Regression model...")
        
        self.pipeline.fit(train_texts, train_labels)
        self.is_trained = True
        
        # Print training accuracy
        train_pred = self.pipeline.predict(train_texts)
        train_acc = accuracy_score(train_labels, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
        
        # Print validation accuracy if provided
        if val_texts is not None and val_labels is not None:
            val_pred = self.pipeline.predict(val_texts)
            val_acc = accuracy_score(val_labels, val_pred)
            print(f"Validation accuracy: {val_acc:.4f}")
            print("\nValidation Classification Report:")
            print(classification_report(val_labels, val_pred, target_names=['Real', 'Fake']))
    
    def predict(self, texts):
        """Make predictions on new texts"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.pipeline.predict_proba(texts)
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.pipeline = joblib.load(filepath)
            self.is_trained = True
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"No model found at {filepath}")