import os
import sys
sys.path.append('.')

from data_loader import DataLoader
from preprocessing import TextPreprocessor
from baseline_model import BaselineModel

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load and split data
    loader = DataLoader('data')
    data_splits = loader.load_and_split_data()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess all splits
    train_texts = preprocessor.preprocess(data_splits['train'][0])
    train_labels = data_splits['train'][1]
    
    val_texts = preprocessor.preprocess(data_splits['val'][0])
    val_labels = data_splits['val'][1]
    
    test_texts = preprocessor.preprocess(data_splits['test'][0])
    test_labels = data_splits['test'][1]
    
    # Train baseline model
    model = BaselineModel()
    model.train(train_texts, train_labels, val_texts, val_labels)
    
    # Test the model
    test_pred = model.predict(test_texts)
    test_acc = sum(p == l for p, l in zip(test_pred, test_labels)) / len(test_labels)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save the model
    model.save_model('models/baseline_model.pkl')
    
    # Test prediction on sample text
    sample_text = ["This is breaking news about a miracle cure"]
    sample_processed = preprocessor.preprocess(sample_text)
    prediction = model.predict(sample_processed)
    prob = model.predict_proba(sample_processed)
    
    print(f"\nSample prediction:")
    print(f"Text: {sample_text[0]}")
    print(f"Prediction: {'Fake' if prediction[0] == 1 else 'Real'}")
    print(f"Probability: Real={prob[0][0]:.3f}, Fake={prob[0][1]:.3f}")

if __name__ == "__main__":
    main()