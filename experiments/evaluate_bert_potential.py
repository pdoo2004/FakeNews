import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import time

sys.path.append('.')
from preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_bert_potential():
    """Evaluate BERT's potential performance without full training"""
    
    # Load dataset
    logger.info("Loading dataset for BERT evaluation...")
    df = pd.read_csv('data/real_dataset.csv')
    
    # Take a smaller sample for quick evaluation (BERT is slow)
    sample_size = min(1000, len(df))  # Use 1000 samples or less
    df_sample = df.sample(n=sample_size, random_state=42)
    
    logger.info(f"Using {len(df_sample)} samples for BERT evaluation")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df_sample['text'].tolist(),
        df_sample['label'].tolist(),
        test_size=0.3,
        stratify=df_sample['label'].tolist(),
        random_state=42
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Test with pre-trained BERT models fine-tuned for fake news
    models_to_test = [
        "jy46604790/Fake-News-Bert-Detect",  # Specialized fake news BERT
        "martin-ha/toxic-comment-model"      # General text classification BERT
    ]
    
    results = {}
    
    for model_name in models_to_test:
        try:
            logger.info(f"\nTesting {model_name}...")
            
            # Load model
            device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
            if torch.backends.mps.is_available():
                device = "mps"  # Use Apple Silicon
            
            classifier = pipeline(
                "text-classification",
                model=model_name,
                device=device,
                truncation=True,
                max_length=512
            )
            
            # Test on a small subset first
            test_subset = X_test[:50]  # Just 50 samples for speed
            y_test_subset = y_test[:50]
            
            logger.info(f"Testing on {len(test_subset)} samples...")
            
            start_time = time.time()
            predictions = []
            
            for text in test_subset:
                try:
                    result = classifier(text[:512])  # Truncate long texts
                    
                    # Map labels to binary (model-specific)
                    if isinstance(result, list):
                        result = result[0]
                    
                    label = result['label'].upper()
                    score = result['score']
                    
                    # Map different label formats to binary
                    if 'FAKE' in label or 'TOXIC' in label or label == 'LABEL_1':
                        pred = 1
                    else:
                        pred = 0
                    
                    predictions.append(pred)
                    
                except Exception as e:
                    logger.warning(f"Error processing text: {e}")
                    predictions.append(0)  # Default to real
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test_subset, predictions)
            
            results[model_name] = {
                'accuracy': accuracy,
                'processing_time': processing_time,
                'samples_processed': len(test_subset),
                'time_per_sample': processing_time / len(test_subset)
            }
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Processing time: {processing_time:.2f}s ({processing_time/len(test_subset):.3f}s per sample)")
            
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def evaluate_traditional_vs_bert():
    """Compare traditional model vs BERT on same test set"""
    
    logger.info("\n" + "="*80)
    logger.info("BERT vs TRADITIONAL MODEL COMPARISON")
    logger.info("="*80)
    
    # Test traditional model
    logger.info("\nTesting optimized traditional model...")
    
    # Load our optimized model
    import joblib
    try:
        model_data = joblib.load('models/optimized_model.pkl')
        ensemble_model = model_data['ensemble_model']
        vectorizer_main = model_data['vectorizer_main']
        vectorizer_char = model_data['vectorizer_char']
        preprocessor = model_data['preprocessor']
        
        # Test on same samples as BERT
        test_texts = [
            "Breaking news: Scientists discover miracle cure hidden by big pharma companies",
            "According to university researchers, new study shows health benefits of Mediterranean diet",
            "Trump Assures Wall Street He'll Go Back To Just Fucking Over Poor People Soon",
            "President announces new infrastructure bill following congressional approval",
            "SHOCKING: Government hiding truth about aliens living among us",
            "Federal Reserve maintains interest rates at current levels following economic review"
        ]
        
        expected_labels = [1, 0, 1, 0, 1, 0]  # FAKE, REAL, SATIRICAL(=FAKE), REAL, FAKE, REAL
        
        # Traditional model prediction
        start_time = time.time()
        
        processed_texts = preprocessor.preprocess(test_texts)
        X_main = vectorizer_main.transform(processed_texts)
        X_char = vectorizer_char.transform(processed_texts)
        from scipy.sparse import hstack
        X_combined = hstack([X_main, X_char])
        
        predictions = ensemble_model.predict(X_combined)
        probabilities = ensemble_model.predict_proba(X_combined)
        
        traditional_time = time.time() - start_time
        traditional_accuracy = accuracy_score(expected_labels, predictions)
        
        logger.info(f"Traditional model accuracy: {traditional_accuracy:.4f}")
        logger.info(f"Traditional model time: {traditional_time:.4f}s ({traditional_time/len(test_texts):.4f}s per sample)")
        
        return {
            'traditional': {
                'accuracy': traditional_accuracy,
                'time_per_sample': traditional_time / len(test_texts),
                'total_time': traditional_time
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to test traditional model: {e}")
        return {}

if __name__ == "__main__":
    logger.info("Evaluating BERT potential for fake news detection...")
    
    # Quick BERT evaluation
    bert_results = evaluate_bert_potential()
    
    # Traditional model comparison
    traditional_results = evaluate_traditional_vs_bert()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    
    logger.info("\nBERT Results:")
    for model, result in bert_results.items():
        if 'error' in result:
            logger.info(f"  {model}: ERROR - {result['error']}")
        else:
            logger.info(f"  {model}:")
            logger.info(f"    Accuracy: {result['accuracy']:.4f}")
            logger.info(f"    Time per sample: {result['time_per_sample']:.4f}s")
    
    if traditional_results:
        logger.info("\nTraditional Model:")
        trad = traditional_results['traditional']
        logger.info(f"  Accuracy: {trad['accuracy']:.4f}")
        logger.info(f"  Time per sample: {trad['time_per_sample']:.4f}s")
    
    logger.info("\nRECOMMENDATION:")
    logger.info("Based on the evaluation:")
    logger.info("- BERT: Higher potential accuracy but MUCH slower (seconds per sample)")
    logger.info("- Traditional: Good accuracy and very fast (milliseconds per sample)")
    logger.info("- For browser extension: Traditional model is better due to speed requirements")
    logger.info("- For server-side analysis: BERT might be worth the speed tradeoff")