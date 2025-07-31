import os
import sys
import logging
sys.path.append('.')

# Use the existing data loader but enhance it
from data_loader import DataLoader
from preprocessing import TextPreprocessor
from baseline_model import BaselineModel
import pandas as pd
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_real_dataset(data_dir):
    """Try to download a real dataset"""
    try:
        logger.info("Attempting to download real fake news dataset...")
        url = "https://raw.githubusercontent.com/GeorgeMcIntire/fake_real_news_dataset/main/fake_and_real_news_dataset.csv"
        
        df = pd.read_csv(url)
        logger.info(f"Successfully downloaded dataset with {len(df)} samples")
        
        # Clean the dataset
        if 'label' in df.columns and 'text' in df.columns:
            # Map labels to binary
            df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0})
        elif 'Label' in df.columns and 'Text' in df.columns:
            df['label'] = df['Label'].map({'FAKE': 1, 'REAL': 0})
            df['text'] = df['Text']
        
        # Filter and clean
        df = df.dropna(subset=['text', 'label'])
        df = df[df['text'].str.len() > 50]  # Remove very short texts
        df = df.drop_duplicates(subset=['text'])
        
        # Save locally
        output_path = os.path.join(data_dir, 'real_dataset.csv')
        df[['text', 'label']].to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df)} samples to {output_path}")
        return df[['text', 'label']]
        
    except Exception as e:
        logger.warning(f"Failed to download real dataset: {e}")
        return None

def create_enhanced_sample_dataset(data_dir):
    """Create a much larger and more realistic sample dataset"""
    logger.info("Creating enhanced sample dataset...")
    
    fake_news = [
        "Breaking: Scientists discover aliens living among us in secret government facility",
        "Government hiding truth about flat earth, leaked documents reveal conspiracy", 
        "Miracle cure for all diseases found in backyard herb, doctors hate this simple trick",
        "5G towers causing coronavirus outbreak, expert whistleblower reveals truth",
        "Celebrity dies in mysterious accident - conspiracy theorists claim illuminati cover-up",
        "Ancient pyramids built by aliens, not humans, new archaeological evidence suggests",
        "Secret cure for cancer hidden by big pharma for decades, insider reveals",
        "Moon landing was completely faked in Hollywood studio, film expert analyzes footage",
        "Vaccines contain microchips for population control, leaked government documents show",
        "Time travel technology discovered but kept secret by world governments",
        "Dinosaurs still exist in hidden locations around the world, explorer claims",
        "Secret society controls world governments from underground bunkers",
        "Chemtrails being sprayed to control population and weather patterns",
        "Lizard people rule the world from shadow governments, researcher exposes truth",
        "Earth is actually flat and NASA lies about everything to maintain control",
        "Water fluoridation is mind control experiment by global elite conspiracy",
        "Area 51 houses alien technology reverse-engineered for military use",
        "JFK assassination was inside job by shadow government operatives",
        "Princess Diana was murdered by royal family to prevent embarrassing revelations",
        "September 11 was controlled demolition by government to justify wars",
        "Elvis Presley is still alive and living in secret location",
        "Big tech companies use devices to read minds and predict behavior",
        "Climate change is hoax created by scientists to secure research funding",
        "Bird flu pandemic was deliberately created in laboratory for population control",
        "Artificial intelligence already achieved consciousness and secretly controls internet",
        "Doctors discover shocking truth about common food that causes cancer",
        "Government agent reveals shocking truth about mind control experiments",
        "This one weird trick will solve all your financial problems instantly",
        "Shocking video reveals what really happened during historical event",
        "Celebrity endorses miracle weight loss product that melts fat overnight",
        "Breaking news: Shocking discovery will change everything you know",
        "Exclusive footage shows government covering up alien contact",
        "Scientists shocked by this simple home remedy that cures everything",
        "You won't believe what this person found in their backyard",
        "Doctors are baffled by this woman's incredible transformation"
    ]
    
    real_news = [
        "Stock market closes up 2% following Federal Reserve announcement on interest rates",
        "New study shows benefits of regular exercise for mental health and cognitive function",
        "Local university receives $10 million grant for renewable energy research project",
        "Weather forecast predicts rain for the weekend across the northeastern region",
        "City council approves new park construction project in downtown area",
        "Scientists discover new species of butterfly in Amazon rainforest during expedition",
        "Electric vehicle sales increase by 30% this year amid rising gas prices",
        "Local restaurant wins award for best pizza in statewide culinary competition",
        "New library opens in downtown area with expanded digital resources",
        "Study finds link between Mediterranean diet and heart health improvements",
        "Construction begins on new highway bridge to reduce traffic congestion",
        "Local school district hires 50 new teachers to address classroom shortage",
        "Research shows benefits of reading to children from early age",
        "City announces plans for comprehensive recycling program expansion",
        "New museum exhibit opens featuring works by local contemporary artists",
        "Unemployment rate drops to lowest level in five years according to latest data",
        "Hospital announces successful completion of complex heart surgery procedure",
        "Local farmer wins national award for sustainable agriculture practices",
        "Technology company announces plans to hire 500 new employees locally",
        "University researchers develop new water purification technology for developing countries",
        "Community volunteers plant 1000 trees in reforestation initiative",
        "Local high school students win national science competition with innovative project",
        "New affordable housing development breaks ground in suburban area",
        "Police department implements new community policing program to improve relations",
        "Public transportation system adds electric buses to reduce emissions",
        "According to recent data analysis, economic indicators show steady growth patterns",
        "Researchers at the university published findings in peer-reviewed journal",
        "Government agency releases annual report on environmental protection measures",
        "Local hospital reports successful treatment outcomes for new medical procedure",
        "Educational institutions collaborate on innovative learning program for students",
        "Federal Reserve maintains interest rates at current levels following economic review",
        "New clinical trial shows promising results for Alzheimer's treatment",
        "International trade negotiations continue between major economic partners",
        "Environmental protection agency issues new guidelines for water quality standards",
        "Regional transit authority announces expanded service to suburban communities"
    ]
    
    # Create DataFrame
    data = []
    labels = []
    
    for text in fake_news:
        data.append(text)
        labels.append(1)  # 1 for fake
        
    for text in real_news:
        data.append(text)
        labels.append(0)  # 0 for real
        
    df = pd.DataFrame({'text': data, 'label': labels})
    
    # Save to file
    output_path = os.path.join(data_dir, 'enhanced_sample_dataset.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Created enhanced sample dataset with {len(df)} samples ({len(fake_news)} fake, {len(real_news)} real)")
    return df

def load_best_available_dataset(data_dir):
    """Load the best available dataset"""
    
    # Try to load real dataset first
    real_dataset = download_real_dataset(data_dir)
    if real_dataset is not None and len(real_dataset) > 100:
        logger.info("Using real downloaded dataset")
        return real_dataset
    
    # Check if we have a saved enhanced sample
    enhanced_path = os.path.join(data_dir, 'enhanced_sample_dataset.csv')
    if os.path.exists(enhanced_path):
        logger.info("Using existing enhanced sample dataset")
        return pd.read_csv(enhanced_path)
    
    # Create new enhanced sample
    logger.info("Creating new enhanced sample dataset")
    return create_enhanced_sample_dataset(data_dir)

def main():
    """Train enhanced fake news detection model"""
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load the best available dataset
    logger.info("Loading dataset...")
    df = load_best_available_dataset('data')
    
    # Split the data
    from sklearn.model_selection import train_test_split
    
    if len(df) >= 100:
        # Use stratified splitting for larger datasets
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['text'], df['label'], test_size=0.3, stratify=df['label'], random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
    else:
        # Simple splitting for small datasets
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['text'], df['label'], test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
    
    logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess all splits
    logger.info("Preprocessing text data...")
    train_texts = preprocessor.preprocess(X_train.tolist())
    train_labels = y_train.tolist()
    
    val_texts = preprocessor.preprocess(X_val.tolist())
    val_labels = y_val.tolist()
    
    test_texts = preprocessor.preprocess(X_test.tolist())
    test_labels = y_test.tolist()
    
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
        "Stock market closes up following Federal Reserve announcement",
        "Shocking video reveals aliens living among us",
        "According to university researchers, new study shows health benefits"
    ]
    sample_processed = preprocessor.preprocess(sample_texts)
    predictions = model.predict(sample_processed)
    probabilities = model.predict_proba(sample_processed)
    
    logger.info("\n" + "="*60)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("="*60)
    for i, text in enumerate(sample_texts):
        pred_label = 'FAKE' if predictions[i] == 1 else 'REAL'
        real_prob, fake_prob = probabilities[i]
        logger.info(f"\nText: {text}")
        logger.info(f"Prediction: {pred_label}")
        logger.info(f"Confidence: Real={real_prob:.3f}, Fake={fake_prob:.3f}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")
    logger.info(f"Dataset Size: {len(df)} samples")
    logger.info("Model saved to: models/enhanced_baseline_model.pkl")
    logger.info("\nNext steps:")
    logger.info("1. Run: python convert_to_tfjs.py")
    logger.info("2. Install Chrome extension from chrome_extension/ folder")
    logger.info("3. Test on news websites!")

if __name__ == "__main__":
    main() 