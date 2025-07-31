import os
import json
import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def create_simple_model():
    """Create a simple model that definitely works"""
    logger.info("Creating simple keyword-based model...")
    
    simple_model = {
        "type": "simple_detector",
        "fake_keywords": [
            "breaking", "shocking", "unbelievable", "secret", "hidden",
            "conspiracy", "miracle", "cure", "exposed", "revealed",
            "government lies", "big pharma", "cover-up", "leaked"
        ],
        "real_keywords": [
            "according to", "study shows", "research indicates", "university",
            "published", "announced", "reported", "data suggests",
            "scientists", "researchers", "official", "confirmed"
        ],
        "fake_weight": 1.0,
        "real_weight": -0.8,
        "threshold": 0.0,
        "confidence_multiplier": 0.1
    }
    
    return simple_model

def convert_trained_model():
    """Try to convert the trained model, with fallback to simple model"""
    try:
        model_path = 'models/enhanced_baseline_model.pkl'
        if not os.path.exists(model_path):
            logger.warning("No trained model found, using simple model")
            return create_simple_model()
        
        logger.info("Loading trained model...")
        pipeline = joblib.load(model_path)
        
        if not hasattr(pipeline, '__getitem__'):
            logger.error("Model format not recognized")
            return create_simple_model()
        
        vectorizer = pipeline['tfidf']
        classifier = pipeline['clf']
        
        # Get the most important features (top fake and real indicators)
        vocab = vectorizer.vocabulary_
        coefficients = classifier.coef_[0]
        
        # Create feature importance list
        features = []
        for word, idx in vocab.items():
            if idx < len(coefficients):
                features.append({
                    'word': word,
                    'score': float(coefficients[idx])
                })
        
        # Sort by score
        features.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top fake indicators (high positive scores)
        top_fake = [f['word'] for f in features[:20]]
        
        # Get top real indicators (low negative scores)
        features.sort(key=lambda x: x['score'])
        top_real = [f['word'] for f in features[:20]]
        
        # Create lightweight model
        lightweight_model = {
            "type": "trained_lightweight",
            "fake_indicators": top_fake,
            "real_indicators": top_real,
            "intercept": float(classifier.intercept_[0]),
            "fake_weight": 0.3,
            "real_weight": -0.3,
            "threshold": 0.5,
            "confidence_multiplier": 0.15
        }
        
        logger.info(f"Converted model with {len(top_fake)} fake and {len(top_real)} real indicators")
        return lightweight_model
        
    except Exception as e:
        logger.error(f"Failed to convert trained model: {e}")
        logger.info("Falling back to simple model")
        return create_simple_model()

def main():
    """Convert models for Chrome extension"""
    logger.info("Converting models for Chrome extension...")
    
    # Create output directory
    output_dir = '../chrome_extension/models'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert main model
    model = convert_trained_model()
    
    # Save main model
    try:
        with open(f'{output_dir}/detector_model.json', 'w') as f:
            json.dump(model, f, indent=2)
        logger.info("✓ Main detection model saved")
    except Exception as e:
        logger.error(f"Failed to save main model: {e}")
        return False
    
    # Create enhanced keyword model
    keyword_model = {
        "type": "keyword_enhanced",
        "patterns": {
            "fake_patterns": [
                "breaking.*news",
                "shocking.*truth",
                "doctors.*hate",
                "government.*hiding",
                "secret.*revealed",
                "miracle.*cure",
                "you.*won.*believe"
            ],
            "real_patterns": [
                "according.*to.*study",
                "research.*shows",
                "university.*researchers",
                "published.*in.*journal",
                "data.*indicates"
            ]
        },
        "keywords": {
            "fake": ["breaking", "shocking", "secret", "miracle", "exposed"],
            "real": ["study", "research", "university", "published", "data"]
        },
        "weights": {
            "pattern_weight": 2.0,
            "keyword_weight": 1.0,
            "length_bonus": 0.1
        }
    }
    
    try:
        with open(f'{output_dir}/keyword_model.json', 'w') as f:
            json.dump(keyword_model, f, indent=2)
        logger.info("✓ Keyword model saved")
    except Exception as e:
        logger.error(f"Failed to save keyword model: {e}")
    
    logger.info("\n🎉 Model conversion completed!")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("Chrome extension is ready to use!")
    
    return True

if __name__ == "__main__":
    main()