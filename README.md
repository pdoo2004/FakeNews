# Fake News Detector

A machine learning-powered Chrome extension that detects potentially fake or misleading news articles.

## Project Structure

```
FakeNews/
├── ml_pipeline/           # Machine learning components
│   ├── data_loader.py     # Dataset loading and preparation
│   ├── preprocessing.py   # Text preprocessing pipeline
│   ├── baseline_model.py  # TF-IDF + Logistic Regression model
│   └── train_baseline.py  # Training script
├── chrome_extension/      # Chrome extension files
│   ├── manifest.json      # Extension manifest
│   ├── background.js      # Background service worker
│   ├── contentScript.js   # Content script for article analysis
│   ├── popup.html         # Extension popup UI
│   ├── popup.js           # Popup functionality
│   └── tfjs_model/        # TensorFlow.js model files (to be added)
└── requirements.txt       # Python dependencies
```

## Setup Instructions

### 1. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd ml_pipeline
python train_baseline.py
```

This will:
- Create sample training data
- Train a baseline TF-IDF + Logistic Regression model
- Save the trained model to `models/baseline_model.pkl`

### 3. Install Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" and select the `chrome_extension` folder
4. The extension should now appear in your extensions list

## How It Works

### Machine Learning Pipeline

1. **Data Collection**: Currently uses sample data, but can be extended to use real datasets like FakeNewsNet or LIAR
2. **Preprocessing**: Cleans text, removes stopwords, and tokenizes
3. **Model Training**: Uses TF-IDF features with Logistic Regression as baseline
4. **Evaluation**: Provides accuracy metrics and classification reports

### Chrome Extension

1. **Content Script**: Automatically extracts article text from web pages
2. **Background Script**: Performs fake news analysis (currently keyword-based)
3. **Popup Interface**: Shows analysis results and allows user feedback
4. **Visual Indicators**: Displays warning banners for potentially fake content

## Current Features

- ✅ Basic project structure
- ✅ Text preprocessing pipeline
- ✅ Baseline ML model (TF-IDF + Logistic Regression)
- ✅ Chrome extension with popup UI
- ✅ Article text extraction from web pages
- ✅ Simple keyword-based detection
- ✅ User feedback collection

## TODO

- [ ] Implement real dataset integration (FakeNewsNet, LIAR)
- [ ] Add BERT transformer fine-tuning
- [ ] Convert model to TensorFlow.js format
- [ ] Replace keyword detection with actual ML model
- [ ] Add model performance metrics
- [ ] Implement active learning from user feedback
- [ ] Add explainability features (highlight suspicious text)

## Usage

1. **Training**: Run `python ml_pipeline/train_baseline.py` to train the model
2. **Extension**: Load the extension in Chrome and browse news websites
3. **Analysis**: The extension will automatically analyze articles and show warnings for potentially fake content
4. **Feedback**: Use the popup to provide feedback on predictions

## Development Notes

- The current implementation uses a simple keyword-based approach for demonstration
- Real ML model integration requires converting the trained model to TensorFlow.js
- User feedback is stored locally and can be used for model improvement
- The extension respects privacy by not sending data to external servers

## Next Steps

1. Integrate real datasets for better training data
2. Implement BERT model for improved accuracy  
3. Convert trained models to browser-compatible format
4. Add comprehensive testing and evaluation metrics
5. Implement server-side inference option for larger models# FakeNews
