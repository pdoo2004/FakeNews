import os
import sys
import logging
sys.path.append('.')

from data_loader import EnhancedDataLoader
from preprocessing import TextPreprocessor
from baseline_model import BaselineModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Train enhanced fake news detection model (baseline only)"""
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load and split data
    logger.info("Loading enhanced dataset...")
    loader = EnhancedDataLoader('data')
    
    # Use comprehensive dataset if possible, fallback to sample
    try:
        data_splits = loader.load_and_split_data(use_comprehensive=True)
    except Exception as e:
        logger.warning(f"Failed to load comprehensive dataset: {e}")
        logger.info("Falling back to enhanced sample dataset...")
        data_splits = loader.load_and_split_data(use_comprehensive=False)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess all splits
    train_texts = preprocessor.preprocess(data_splits['train'][0])
    train_labels = data_splits['train'][1]
    
    val_texts = preprocessor.preprocess(data_splits['val'][0])
    val_labels = data_splits['val'][1]
    
    test_texts = preprocessor.preprocess(data_splits['test'][0])
    test_labels = data_splits['test'][1]
    
    # Train enhanced baseline model
    logger.info("Training enhanced baseline model...")
    model = BaselineModel(max_features=15000, ngram_range=(1, 3))
    model.train(train_texts, train_labels, val_texts, val_labels)
    
    # Test the model
    test_pred = model.predict(test_texts)
    test_acc = sum(p == l for p, l in zip(test_pred, test_labels)) / len(test_labels)
    logger.info(f"Enhanced Baseline Test accuracy: {test_acc:.4f}")
    
    # Save the model
    model.save_model('models/enhanced_baseline_model.pkl')
    
    # Show feature importance
    try:
        importance = model.get_feature_importance()
        logger.info("\nTop fake-indicating features:")
        for feature, score in importance['fake_indicators'][:10]:
            logger.info(f"  {feature}: {score:.4f}")
        
        logger.info("\nTop real-indicating features:")
        for feature, score in importance['real_indicators'][:10]:
            logger.info(f"  {feature}: {score:.4f}")
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
    
    # Test prediction on sample text
    sample_texts = [
        "Breaking news: Scientists discover miracle cure hidden by big pharma",
        "Stock market closes up following Federal Reserve announcement"
    ]
    sample_processed = preprocessor.preprocess(sample_texts)
    predictions = model.predict(sample_processed)
    probabilities = model.predict_proba(sample_processed)
    
    logger.info("\nSample predictions:")
    for i, text in enumerate(sample_texts):
        pred_label = 'Fake' if predictions[i] == 1 else 'Real'
        real_prob, fake_prob = probabilities[i]
        logger.info(f"Text: {text}")
        logger.info(f"Prediction: {pred_label} (Real={real_prob:.3f}, Fake={fake_prob:.3f})")
    
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)
    logger.info(f"Enhanced Baseline Model Test Accuracy: {test_acc:.4f}")
    logger.info("Model saved to: models/enhanced_baseline_model.pkl")
    logger.info("Ready for Chrome extension integration!")

if __name__ == "__main__":
    main() 