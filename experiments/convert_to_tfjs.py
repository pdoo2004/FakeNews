import os
import sys
import json
import logging
import joblib
import pickle
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_baseline_to_simple_format():
    """Convert baseline model to a simple format for JavaScript"""
    try:
        # Load the trained baseline model
        model_path = 'models/enhanced_baseline_model.pkl'
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False
        
        pipeline = joblib.load(model_path)
        vectorizer = pipeline['tfidf']
        classifier = pipeline['clf']
        
        # Extract model parameters
        vocabulary = vectorizer.vocabulary_
        idf_values = vectorizer.idf_
        coefficients = classifier.coef_[0]
        intercept = classifier.intercept_[0]
        
        # Create a simplified model representation
        # Convert vocabulary to have string keys and int values
        vocab_converted = {k: int(v) for k, v in vocabulary.items()}
        
        simple_model = {
            'type': 'tfidf_logistic',
            'vocabulary': vocab_converted,
            'idf_values': [float(x) for x in idf_values],
            'coefficients': [float(x) for x in coefficients],
            'intercept': float(intercept),
            'max_features': int(vectorizer.max_features) if vectorizer.max_features else None,
            'ngram_range': [int(x) for x in vectorizer.ngram_range],
            'stop_words': list(vectorizer.stop_words) if vectorizer.stop_words else None
        }
        
        # Save to JSON
        output_dir = '../chrome_extension/models'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/baseline_model.json', 'w') as f:
            json.dump(simple_model, f, indent=2)
        
        logger.info("Baseline model converted to JSON format for Chrome extension")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert baseline model: {e}")
        return False

def create_keyword_model():
    """Create an enhanced keyword-based model for the Chrome extension"""
    
    # Enhanced keyword lists based on common fake news patterns
    fake_indicators = {
        'sensational': ['breaking', 'shocking', 'unbelievable', 'amazing', 'incredible', 'stunning'],
        'conspiracy': ['hidden', 'secret', 'cover-up', 'conspiracy', 'exposed', 'revealed', 'leaked'],
        'medical_claims': ['miracle', 'cure', 'doctors hate', 'big pharma', 'natural remedy'],
        'authority_doubt': ['government lies', 'media won\'t tell you', 'experts don\'t want', 'they don\'t want'],
        'urgency': ['urgent', 'immediately', 'before it\'s too late', 'limited time'],
        'absolutes': ['never', 'always', 'completely', 'totally', 'absolutely', 'definitely'],
        'emotional': ['outrageous', 'disgusting', 'terrifying', 'heartbreaking', 'infuriating']
    }
    
    real_indicators = {
        'sources': ['according to', 'study shows', 'research indicates', 'data suggests'],
        'institutions': ['university', 'hospital', 'government', 'agency', 'department'],
        'formal': ['announced', 'reported', 'confirmed', 'stated', 'published'],
        'measured': ['approximately', 'about', 'around', 'estimated', 'roughly'],
        'attribution': ['said', 'told', 'explained', 'noted', 'mentioned']
    }
    
    keyword_model = {
        'type': 'keyword_enhanced',
        'fake_indicators': fake_indicators,
        'real_indicators': real_indicators,
        'weights': {
            'fake_base_weight': 0.15,
            'real_base_weight': 0.1,
            'length_penalty': 0.05,  # Penalty for very short articles
            'caps_penalty': 0.02,    # Penalty for excessive caps
            'exclamation_penalty': 0.03  # Penalty for multiple exclamation marks
        }
    }
    
    # Save enhanced keyword model
    output_dir = '../chrome_extension/models'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/keyword_model.json', 'w') as f:
        json.dump(keyword_model, f, indent=2)
    
    logger.info("Enhanced keyword model created for Chrome extension")
    return True

def main():
    """Convert models for Chrome extension use"""
    logger.info("Converting models for Chrome extension...")
    
    success = True
    
    # Convert baseline model if it exists
    if convert_baseline_to_simple_format():
        logger.info("✓ Baseline model converted successfully")
    else:
        logger.warning("⚠ Baseline model conversion failed")
        success = False
    
    # Create enhanced keyword model
    if create_keyword_model():
        logger.info("✓ Enhanced keyword model created successfully")
    else:
        logger.warning("⚠ Keyword model creation failed")
        success = False
    
    if success:
        logger.info("\n🎉 Models ready for Chrome extension!")
        logger.info("Models saved in chrome_extension/models/")
    else:
        logger.error("❌ Some model conversions failed")

if __name__ == "__main__":
    main() 