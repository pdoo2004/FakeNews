import os
import sys
import argparse
import logging
sys.path.append('.')

from data_loader import EnhancedDataLoader
from preprocessing import TextPreprocessor
from baseline_model import BaselineModel
from bert_model import BertModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_baseline_model(data_splits, preprocessor):
    """Train the enhanced baseline model"""
    logger.info("Training enhanced baseline model...")
    
    # Preprocess all splits
    train_texts = preprocessor.preprocess(data_splits['train'][0])
    train_labels = data_splits['train'][1]
    
    val_texts = preprocessor.preprocess(data_splits['val'][0])
    val_labels = data_splits['val'][1]
    
    test_texts = preprocessor.preprocess(data_splits['test'][0])
    test_labels = data_splits['test'][1]
    
    # Train baseline model
    model = BaselineModel(max_features=15000, ngram_range=(1, 3))
    model.train(train_texts, train_labels, val_texts, val_labels)
    
    # Test the model
    test_pred = model.predict(test_texts)
    test_acc = sum(p == l for p, l in zip(test_pred, test_labels)) / len(test_labels)
    logger.info(f"Baseline Test accuracy: {test_acc:.4f}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
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
    
    return model, test_acc

def train_bert_model(data_splits):
    """Train BERT model"""
    logger.info("Training BERT model...")
    
    # Use raw texts for BERT (no preprocessing needed)
    train_texts = data_splits['train'][0]
    train_labels = data_splits['train'][1]
    
    val_texts = data_splits['val'][0]
    val_labels = data_splits['val'][1]
    
    test_texts = data_splits['test'][0]
    test_labels = data_splits['test'][1]
    
    # Initialize and train BERT model
    bert_model = BertModel(
        model_name='bert-base-uncased',
        max_length=256,  # Reduced for faster training
        output_dir='./bert_output'
    )
    
    # Train with smaller batch size and fewer epochs for demo
    eval_results = bert_model.train(
        train_texts, train_labels, 
        val_texts, val_labels,
        num_epochs=2,  # Reduced for faster training
        batch_size=8,  # Reduced for memory efficiency
        learning_rate=2e-5
    )
    
    # Test the model
    test_pred = bert_model.predict(test_texts)
    test_acc = sum(p == l for p, l in zip(test_pred, test_labels)) / len(test_labels)
    logger.info(f"BERT Test accuracy: {test_acc:.4f}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    bert_model.save_model('models/bert_model')
    
    return bert_model, test_acc

def main():
    parser = argparse.ArgumentParser(description='Train enhanced fake news detection models')
    parser.add_argument('--model', choices=['baseline', 'bert', 'both'], default='both',
                       help='Which model to train')
    parser.add_argument('--use-sample', action='store_true',
                       help='Use sample data instead of comprehensive dataset')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load and split data
    logger.info("Loading enhanced dataset...")
    loader = EnhancedDataLoader('data')
    data_splits = loader.load_and_split_data(use_comprehensive=not args.use_sample)
    
    results = {}
    
    if args.model in ['baseline', 'both']:
        # Initialize preprocessor for baseline model
        preprocessor = TextPreprocessor()
        
        try:
            baseline_model, baseline_acc = train_baseline_model(data_splits, preprocessor)
            results['baseline'] = baseline_acc
            
            # Test prediction on sample text
            sample_texts = [
                "Breaking news: Scientists discover miracle cure hidden by big pharma",
                "Stock market closes up following Federal Reserve announcement"
            ]
            sample_processed = preprocessor.preprocess(sample_texts)
            predictions = baseline_model.predict(sample_processed)
            probabilities = baseline_model.predict_proba(sample_processed)
            
            logger.info("\nBaseline Sample predictions:")
            for i, text in enumerate(sample_texts):
                pred_label = 'Fake' if predictions[i] == 1 else 'Real'
                real_prob, fake_prob = probabilities[i]
                logger.info(f"Text: {text}")
                logger.info(f"Prediction: {pred_label} (Real={real_prob:.3f}, Fake={fake_prob:.3f})")
                
        except Exception as e:
            logger.error(f"Failed to train baseline model: {e}")
    
    if args.model in ['bert', 'both']:
        try:
            # Check if transformers is available
            import torch
            from transformers import BertTokenizer
            
            bert_model, bert_acc = train_bert_model(data_splits)
            results['bert'] = bert_acc
            
            # Test prediction on sample text
            sample_texts = [
                "Breaking news: Scientists discover miracle cure hidden by big pharma",
                "Stock market closes up following Federal Reserve announcement"
            ]
            predictions = bert_model.predict(sample_texts)
            probabilities = bert_model.predict_proba(sample_texts)
            
            logger.info("\nBERT Sample predictions:")
            for i, text in enumerate(sample_texts):
                pred_label = 'Fake' if predictions[i] == 1 else 'Real'
                real_prob, fake_prob = probabilities[i]
                logger.info(f"Text: {text}")
                logger.info(f"Prediction: {pred_label} (Real={real_prob:.3f}, Fake={fake_prob:.3f})")
                
        except ImportError:
            logger.warning("PyTorch or transformers not available. Skipping BERT training.")
            logger.info("Install with: pip install torch transformers")
        except Exception as e:
            logger.error(f"Failed to train BERT model: {e}")
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    for model_name, accuracy in results.items():
        logger.info(f"{model_name.upper()} Model Test Accuracy: {accuracy:.4f}")
    
    logger.info("\nModels saved in 'models/' directory")
    logger.info("Ready for Chrome extension integration!")

if __name__ == "__main__":
    main() 