import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests
import os

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        
    def download_fakenews_dataset(self):
        """Download FakeNewsNet dataset"""
        print("Creating sample dataset...")
        # Note: This is a simplified version - actual dataset would need proper API access
        # For demo purposes, we'll create a larger sample dataset
        fake_news = [
            "Breaking: Scientists discover aliens living among us",
            "Government hiding truth about flat earth",
            "Miracle cure for all diseases found in backyard herb",
            "5G towers causing coronavirus outbreak",
            "Celebrity dies in mysterious accident - conspiracy theorists claim cover-up",
            "Ancient pyramids built by aliens, not humans",
            "Secret cure for cancer hidden by big pharma",
            "Moon landing was completely faked",
            "Vaccines contain microchips for population control",
            "Time travel technology discovered but kept secret",
            "Dinosaurs still exist in hidden locations",
            "Secret society controls world governments",
            "Chemtrails being sprayed to control population",
            "Lizard people rule the world",
            "Earth is actually flat and NASA lies about everything"
        ]
        real_news = [
            "Stock market closes up 2% following Federal Reserve announcement",
            "New study shows benefits of regular exercise for mental health",
            "Local university receives $10 million grant for research",
            "Weather forecast predicts rain for the weekend",
            "City council approves new park construction",
            "Scientists discover new species of butterfly in Amazon",
            "Electric vehicle sales increase by 30% this year",
            "Local restaurant wins award for best pizza",
            "New library opens in downtown area",
            "Study finds link between diet and heart health",
            "Construction begins on new highway bridge",
            "Local school district hires 50 new teachers",
            "Research shows benefits of reading to children",
            "City announces plans for recycling program",
            "New museum exhibit opens featuring local artists"
        ]
        
        data = []
        labels = []
        
        for text in fake_news:
            data.append(text)
            labels.append(1)  # 1 for fake
            
        for text in real_news:
            data.append(text)
            labels.append(0)  # 0 for real
            
        df = pd.DataFrame({'text': data, 'label': labels})
        df.to_csv(os.path.join(self.data_dir, 'sample_dataset.csv'), index=False)
        return df
    
    def load_and_split_data(self, test_size=0.2, val_size=0.25):
        """Load data and create train/val/test splits"""
        if not os.path.exists(os.path.join(self.data_dir, 'sample_dataset.csv')):
            df = self.download_fakenews_dataset()
        else:
            df = pd.read_csv(os.path.join(self.data_dir, 'sample_dataset.csv'))
        
        # For small datasets, use simple random splitting instead of stratified
        if len(df) < 20:
            # Split into train and temp (test + validation)
            X_train, X_temp, y_train, y_temp = train_test_split(
                df['text'], df['label'], test_size=test_size, random_state=42
            )
            
            # Split temp into validation and test
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42
            )
        else:
            # Use stratified splitting for larger datasets
            X_train, X_temp, y_train, y_temp = train_test_split(
                df['text'], df['label'], test_size=test_size, stratify=df['label'], random_state=42
            )
            
            # Split temp into validation and test
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42
            )
        
        return {
            'train': (X_train.tolist(), y_train.tolist()),
            'val': (X_val.tolist(), y_val.tolist()),
            'test': (X_test.tolist(), y_test.tolist())
        }