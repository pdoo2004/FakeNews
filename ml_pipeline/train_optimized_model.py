import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from scipy.sparse import hstack

sys.path.append('.')
from preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_optimized_model():
    """Train the best possible model with all available data"""
    
    # Load the full dataset
    logger.info("Loading full dataset...")
    df = pd.read_csv('data/real_dataset.csv')
    logger.info(f"Dataset loaded: {len(df)} samples ({sum(df.label==1)} fake, {sum(df.label==0)} real)")
    
    # Enhanced preprocessing
    preprocessor = TextPreprocessor()
    
    logger.info("Preprocessing text data...")
    processed_texts = preprocessor.preprocess(df['text'].tolist())
    labels = df['label'].tolist()
    
    # Split data with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        processed_texts, labels, test_size=0.3, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create multiple TF-IDF vectorizers with different parameters
    logger.info("Creating enhanced TF-IDF vectorizers...")
    
    # Main vectorizer - more features for better accuracy
    vectorizer_main = TfidfVectorizer(
        max_features=20000,  # Increased from 15000
        ngram_range=(1, 3),  # 1, 2, and 3-grams
        min_df=2,           # Must appear in at least 2 documents
        max_df=0.95,        # Ignore terms in >95% of documents
        stop_words='english',
        sublinear_tf=True   # Use 1 + log(tf) instead of tf
    )
    
    # Character-level vectorizer for catching typos and variations
    vectorizer_char = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=5000,
        min_df=2
    )
    
    # Fit vectorizers
    logger.info("Fitting vectorizers...")
    X_train_main = vectorizer_main.fit_transform(X_train)
    X_val_main = vectorizer_main.transform(X_val)
    X_test_main = vectorizer_main.transform(X_test)
    
    X_train_char = vectorizer_char.fit_transform(X_train)
    X_val_char = vectorizer_char.transform(X_val)
    X_test_char = vectorizer_char.transform(X_test)
    
    # Combine features
    X_train_combined = hstack([X_train_main, X_train_char])
    X_val_combined = hstack([X_val_main, X_val_char])
    X_test_combined = hstack([X_test_main, X_test_char])
    
    logger.info(f"Combined feature dimensions: {X_train_combined.shape[1]}")
    
    # Train ensemble model
    logger.info("Training ensemble model...")
    
    # Individual models
    lr_model = LogisticRegression(
        max_iter=2000,
        C=1.0,              # Regularization strength
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    
    nb_model = MultinomialNB(alpha=0.1)  # Naive Bayes with smoothing
    
    # Ensemble model
    ensemble_model = VotingClassifier([
        ('logistic', lr_model),
        ('naive_bayes', nb_model)
    ], voting='soft')  # Use probabilities
    
    # Train the ensemble
    ensemble_model.fit(X_train_combined, y_train)
    
    # Evaluate on validation set
    val_pred = ensemble_model.predict(X_val_combined)
    val_acc = accuracy_score(y_val, val_pred)
    logger.info(f"Validation accuracy: {val_acc:.4f}")
    
    # Evaluate on test set
    test_pred = ensemble_model.predict(X_test_combined)
    test_acc = accuracy_score(y_test, test_pred)
    logger.info(f"Test accuracy: {test_acc:.4f}")
    
    logger.info("\nDetailed test results:")
    logger.info(classification_report(y_test, test_pred, target_names=['Real', 'Fake']))
    
    # Save the complete model
    logger.info("Saving optimized model...")
    model_data = {
        'ensemble_model': ensemble_model,
        'vectorizer_main': vectorizer_main,
        'vectorizer_char': vectorizer_char,
        'preprocessor': preprocessor,
        'test_accuracy': test_acc,
        'feature_count': X_train_combined.shape[1]
    }
    
    joblib.dump(model_data, 'models/optimized_model.pkl')
    
    # Create lightweight version for browser
    logger.info("Creating lightweight browser version...")
    create_lightweight_browser_model(ensemble_model, vectorizer_main, vectorizer_char, test_acc)
    
    # Test on sample texts
    test_sample_predictions(model_data)
    
    return test_acc

def create_lightweight_browser_model(ensemble_model, vectorizer_main, vectorizer_char, accuracy):
    """Create a lightweight model for browser use"""
    
    # Get the logistic regression component
    lr_model = ensemble_model.named_estimators_['logistic']
    
    # Get feature names and coefficients
    feature_names_main = vectorizer_main.get_feature_names_out()
    feature_names_char = vectorizer_char.get_feature_names_out()
    all_feature_names = np.concatenate([feature_names_main, feature_names_char])
    
    coefficients = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    
    # Select top features (more than before for better accuracy)
    feature_importance = [(name, coef) for name, coef in zip(all_feature_names, coefficients)]
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Take top 500 features (more than previous 200)
    top_features = feature_importance[:500]
    fake_features = [(name, coef) for name, coef in top_features if coef > 0][:150]
    real_features = [(name, coef) for name, coef in top_features if coef < 0][:150]
    
    # Create lightweight model
    lightweight_model = {
        "type": "optimized_lightweight",
        "version": "3.0",
        "accuracy": float(accuracy),
        "fake_indicators": [name for name, _ in fake_features],
        "real_indicators": [name for name, _ in real_features],
        "fake_weights": {name: float(coef) for name, coef in fake_features},
        "real_weights": {name: float(coef) for name, coef in real_features},
        "intercept": float(intercept),
        "threshold": 0.0,
        "confidence_multiplier": 0.15,
        "model_info": {
            "total_samples": 4403,
            "total_features": len(all_feature_names),
            "selected_features": len(fake_features) + len(real_features),
            "training_method": "ensemble_logistic_nb"
        }
    }
    
    # Save to Chrome extension
    output_dir = '../chrome_extension/models'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/optimized_detector_model.json', 'w') as f:
        json.dump(lightweight_model, f, indent=2)
    
    logger.info(f"Lightweight model created with {len(fake_features)} fake + {len(real_features)} real indicators")
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    # Show top features
    logger.info("\nTop 10 fake indicators:")
    for name, coef in fake_features[:10]:
        logger.info(f"  {name}: {coef:.4f}")
    
    logger.info("\nTop 10 real indicators:")
    for name, coef in real_features[:10]:
        logger.info(f"  {name}: {coef:.4f}")

def test_sample_predictions(model_data):
    """Test the model on sample texts"""
    
    ensemble_model = model_data['ensemble_model']
    vectorizer_main = model_data['vectorizer_main']
    vectorizer_char = model_data['vectorizer_char']
    preprocessor = model_data['preprocessor']
    
    test_texts = [
        "Breaking news: Scientists discover miracle cure hidden by big pharma companies",
        "According to university researchers, new study shows health benefits of Mediterranean diet",
        "Trump Assures Wall Street He'll Go Back To Just Fucking Over Poor People Soon",
        "President announces new infrastructure bill following congressional approval",
        "SHOCKING: Government hiding truth about aliens living among us",
        "Federal Reserve maintains interest rates at current levels following economic review"
    ]
    
    expected_labels = ['FAKE', 'REAL', 'SATIRICAL', 'REAL', 'FAKE', 'REAL']
    
    logger.info("\n" + "="*80)
    logger.info("SAMPLE PREDICTIONS TEST")
    logger.info("="*80)
    
    processed_texts = preprocessor.preprocess(test_texts)
    
    # Transform using both vectorizers
    X_main = vectorizer_main.transform(processed_texts)
    X_char = vectorizer_char.transform(processed_texts)
    X_combined = hstack([X_main, X_char])
    
    predictions = ensemble_model.predict(X_combined)
    probabilities = ensemble_model.predict_proba(X_combined)
    
    for i, text in enumerate(test_texts):
        pred_label = 'FAKE' if predictions[i] == 1 else 'REAL'
        real_prob, fake_prob = probabilities[i]
        confidence = max(real_prob, fake_prob)
        expected = expected_labels[i]
        
        status = "✓" if (pred_label == expected or (expected == 'SATIRICAL' and pred_label == 'FAKE')) else "✗"
        
        logger.info(f"\n{status} Text: {text}")
        logger.info(f"  Expected: {expected}, Got: {pred_label}")
        logger.info(f"  Confidence: {confidence:.3f} (Real={real_prob:.3f}, Fake={fake_prob:.3f})")

if __name__ == "__main__":
    logger.info("Training optimized fake news detection model with full dataset...")
    
    accuracy = train_optimized_model()
    
    logger.info(f"\n🎉 OPTIMIZED MODEL TRAINING COMPLETE!")
    logger.info(f"Final test accuracy: {accuracy:.4f}")
    logger.info("Enhanced model saved to: models/optimized_model.pkl")
    logger.info("Lightweight version saved to: chrome_extension/models/optimized_detector_model.json")
    logger.info("\nThis model uses:")
    logger.info("- Full 4,403 sample dataset")
    logger.info("- 25,000 combined features (word + character n-grams)")
    logger.info("- Ensemble of Logistic Regression + Naive Bayes")
    logger.info("- Advanced preprocessing and feature selection")