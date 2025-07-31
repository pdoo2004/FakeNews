import os
import sys
import json
import joblib
import numpy as np
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_lightweight_model():
    """Extract most important features for a lightweight browser model"""
    
    # Load the enhanced baseline model
    model_path = 'models/enhanced_baseline_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"Enhanced model not found: {model_path}")
        return False
    
    pipeline = joblib.load(model_path)
    vectorizer = pipeline['tfidf']
    classifier = pipeline['clf']
    
    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    intercept = classifier.intercept_[0]
    
    # Find most important features (highest absolute coefficients)
    feature_importance = [(name, coef) for name, coef in zip(feature_names, coefficients)]
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Extract top features for fake (positive coefficients) and real (negative coefficients)
    fake_features = [(name, coef) for name, coef in feature_importance if coef > 0][:100]
    real_features = [(name, coef) for name, coef in feature_importance if coef < 0][:100]
    
    logger.info(f"Extracted {len(fake_features)} fake indicators and {len(real_features)} real indicators")
    
    # Create lightweight model
    lightweight_model = {
        "type": "lightweight_trained",
        "fake_indicators": [name for name, _ in fake_features],
        "real_indicators": [name for name, _ in real_features],
        "fake_weights": {name: float(coef) for name, coef in fake_features},
        "real_weights": {name: float(coef) for name, coef in real_features},
        "intercept": float(intercept),
        "threshold": 0.0,
        "confidence_multiplier": 0.1,
        "model_info": {
            "total_features": len(feature_names),
            "selected_features": len(fake_features) + len(real_features),
            "accuracy_note": "Lightweight version of 91.4% accuracy model"
        }
    }
    
    # Save to Chrome extension
    output_dir = '../chrome_extension/models'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/detector_model.json', 'w') as f:
        json.dump(lightweight_model, f, indent=2)
    
    logger.info("Lightweight model saved to chrome_extension/models/detector_model.json")
    
    # Show top features
    logger.info("\nTop 10 fake indicators:")
    for name, coef in fake_features[:10]:
        logger.info(f"  {name}: {coef:.4f}")
    
    logger.info("\nTop 10 real indicators:")
    for name, coef in real_features[:10]:
        logger.info(f"  {name}: {coef:.4f}")
    
    return True

def create_fast_keyword_model():
    """Create an optimized keyword model for speed"""
    
    # Curated high-impact keywords based on common fake news patterns
    fake_keywords = [
        # Sensational language
        "breaking", "shocking", "unbelievable", "amazing", "incredible", "stunning",
        "exclusive", "revealed", "exposed", "leaked", "secret", "hidden",
        
        # Medical/health misinformation
        "miracle", "cure", "doctors hate", "big pharma", "natural remedy",
        "toxic", "poisonous", "deadly", "dangerous chemicals",
        
        # Conspiracy theories
        "conspiracy", "cover up", "government lies", "media won't tell",
        "they don't want you to know", "wake up", "sheeple",
        
        # Emotional manipulation
        "outrageous", "disgusting", "terrifying", "heartbreaking", "infuriating",
        "you won't believe", "this will shock you", "must see", "going viral",
        
        # Authority undermining
        "experts are wrong", "science is fake", "mainstream media", "fake news",
        "alternative facts", "do your own research"
    ]
    
    real_keywords = [
        # Formal reporting
        "according to", "study shows", "research indicates", "data suggests",
        "scientists found", "researchers discovered", "analysis revealed",
        
        # Institutional sources
        "university", "hospital", "government", "agency", "department",
        "official", "spokesperson", "statement", "announcement",
        
        # Measured language
        "approximately", "about", "around", "estimated", "roughly",
        "may", "could", "might", "appears", "seems", "likely",
        
        # Attribution
        "said", "told", "explained", "noted", "mentioned", "stated",
        "reported", "confirmed", "published", "peer reviewed"
    ]
    
    fast_model = {
        "type": "fast_keyword",
        "fake_keywords": fake_keywords,
        "real_keywords": real_keywords,
        "fake_weight": 0.4,
        "real_weight": -0.3,
        "threshold": 0.2,
        "confidence_multiplier": 0.2,
        "penalties": {
            "short_text": 0.1,      # Text under 100 chars
            "excessive_caps": 0.15,   # >15% capital letters
            "many_exclamations": 0.1, # >3 exclamation marks
            "clickbait_numbers": 0.1  # Numbers in suspicious contexts
        }
    }
    
    output_dir = '../chrome_extension/models'
    with open(f'{output_dir}/fast_keyword_model.json', 'w') as f:
        json.dump(fast_model, f, indent=2)
    
    logger.info("Fast keyword model created")
    return True

if __name__ == "__main__":
    logger.info("Creating lightweight models for fast Chrome extension performance...")
    
    success = True
    
    if extract_lightweight_model():
        logger.info("✓ Lightweight trained model created")
    else:
        logger.error("✗ Failed to create lightweight trained model")
        success = False
    
    if create_fast_keyword_model():
        logger.info("✓ Fast keyword model created")
    else:
        logger.error("✗ Failed to create fast keyword model")
        success = False
    
    if success:
        logger.info("\n🚀 Fast models ready! Extension should be much faster now.")
    else:
        logger.error("❌ Some models failed to create")