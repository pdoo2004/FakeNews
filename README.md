# 🔍 Fake News Detector Chrome Extension

A comprehensive machine learning-powered Chrome extension that detects potentially fake, misleading, or satirical news articles in real-time.

## 🎯 Key Features

- **🚀 Real-time Analysis**: Analyzes articles as you browse (≤50ms processing time)
- **🎭 Multi-type Detection**: Identifies satirical content, conspiracy theories, and misinformation
- **🌐 Domain Recognition**: Instantly flags known satirical sites (The Onion, Babylon Bee, etc.)
- **📊 High Accuracy**: 90.17% accuracy on 4,403 real news samples
- **🔒 Privacy-First**: All analysis happens locally in your browser
- **⚡ Lightweight**: Optimized for browser performance

## 📈 Performance Metrics

### Model Performance
- **Test Accuracy**: 90.17%
- **Dataset Size**: 4,403 real news samples (2,232 fake, 2,171 real)
- **Feature Count**: 25,000 combined features (word + character n-grams)
- **Processing Speed**: ~29ms per article
- **False Positive Rate**: Conservative (biased toward flagging suspicious content)

### Detection Categories
- **Satirical Content**: 95% confidence for known satirical domains
- **Conspiracy Theories**: Detects 47 conspiracy/misinformation patterns
- **Medical Misinformation**: Identifies "miracle cure" and anti-vaccine content
- **Political Misinformation**: Trained on 2016 election misinformation patterns

## 🏗️ Architecture

### Enhanced ML Pipeline
```
├── ml_pipeline/
│   ├── data/
│   │   └── real_dataset.csv           # 4,403 labeled news samples
│   ├── models/
│   │   ├── optimized_model.pkl        # Full ensemble model (90.17% accuracy)
│   │   └── enhanced_baseline_model.pkl # Baseline model
│   ├── train_optimized_model.py       # Advanced ensemble training
│   ├── create_comprehensive_model.py  # Satirical content detection
│   ├── baseline_model.py              # TF-IDF + Logistic Regression
│   ├── bert_model.py                  # BERT implementation (optional)
│   └── preprocessing.py               # Text preprocessing pipeline
```

### Chrome Extension
```
├── chrome_extension/
│   ├── models/
│   │   ├── comprehensive_model.json   # Domain + pattern detection
│   │   └── optimized_detector_model.json # Lightweight ML model (295 features)
│   ├── background_comprehensive.js    # Multi-layered detection engine
│   ├── contentScript.js              # Article extraction & analysis
│   ├── popup.html/js                 # User interface
│   └── manifest.json                 # Extension configuration
```

## 🧠 Detection System

### 1. Domain-Based Detection (Instant)
- **16 Satirical Sites**: The Onion, Babylon Bee, Clickhole, etc.
- **9 Conspiracy Sites**: InfoWars, Natural News, etc.
- **95% Confidence** for domain matches

### 2. Content Pattern Analysis
- **35 Satirical Patterns**: Profanity in headlines, absurd political language
- **47 Fake News Patterns**: Conspiracy theories, medical misinformation
- **39 Real News Patterns**: Professional journalism indicators

### 3. Statistical ML Model
- **Ensemble Approach**: Logistic Regression + Naive Bayes
- **Advanced Features**: Word + character n-grams, TF-IDF weighting
- **295 Top Features**: Selected from 25,000 for browser performance

## 🚀 Quick Start

### 1. Set Up Python Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd ml_pipeline
pip install pandas scikit-learn requests joblib scipy torch transformers
```

### 2. Train Models (Optional - Pre-trained Available)
```bash
# Train the comprehensive model
python train_optimized_model.py

# Create browser-optimized models
python create_comprehensive_model.py
```

### 3. Install Chrome Extension
1. Open Chrome → `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" → Select `chrome_extension` folder
4. Extension ready! Browse news sites to see it in action.

## 📊 Model Comparison

| Model Type | Accuracy | Speed/Sample | Use Case |
|------------|----------|--------------|----------|
| **Optimized Ensemble** | 90.17% | 29ms | ✅ **Current (Best Balance)** |
| Enhanced Baseline | 91.38% | ~15ms | Good baseline |
| BERT (Pre-trained) | 48.0% | 152ms | Poor on this dataset |
| Keyword-Only | ~70% | <5ms | Fast but limited |

## 🎨 User Interface

### Popup Indicators
- **😄 Satirical/Parody Content** (Yellow) - Known satirical sites
- **🚩 Conspiracy/Misinformation** (Pink) - Conspiracy theory sites  
- **⚠️ Potentially Fake News** (Red) - Detected misinformation
- **✅ Likely Reliable** (Green) - Appears legitimate

### Page Warnings
- **High-confidence warnings** (>70%) show banner at top of page
- **Auto-dismissing** after 10 seconds
- **One-click dismiss** option

## 🔬 Technical Details

### Training Data
- **Source**: Real fake news dataset from academic research
- **Composition**: Balanced fake/real news from 2016 US election period
- **Preprocessing**: Advanced text cleaning, stopword removal, stemming
- **Features**: 1-3 word n-grams + 3-5 character n-grams

### Browser Optimization
- **Lightweight Model**: 295 most important features (from 25,000)
- **Efficient Processing**: Text truncation, early loop breaks
- **Memory Management**: Sparse feature vectors, garbage collection
- **Performance Monitoring**: Built-in timing and metrics

### Privacy & Security
- **No External Calls**: All processing happens locally
- **No Data Collection**: User browsing data stays private
- **Minimal Permissions**: Only needs access to analyze current page
- **Open Source**: Fully auditable code

## 📈 Evaluation Results

### Sample Predictions
```
✅ "Breaking news: Scientists discover miracle cure..." → FAKE (97.9% confidence)
❌ "According to university researchers, new study..." → FAKE (96.6% confidence) [False Positive]
✅ "Trump Assures Wall Street He'll Go Back To..." → FAKE (73.6% confidence) [Satirical]
✅ "SHOCKING: Government hiding truth about aliens..." → FAKE (97.1% confidence)
```

### Strengths
- **High Recall**: Catches most fake news (few false negatives)
- **Satirical Detection**: Excellent at identifying parody content
- **Speed**: Fast enough for real-time browsing
- **Domain Coverage**: Comprehensive list of problematic sites

### Limitations
- **Dataset Bias**: Trained primarily on 2016 political news
- **False Positives**: May flag legitimate news as suspicious
- **Topic Scope**: Less accurate on non-political content
- **Temporal Drift**: May need retraining for emerging misinformation patterns

## 🔮 Future Improvements

### Potential Enhancements
- **Dataset Diversification**: Add recent, varied news sources
- **Active Learning**: Learn from user feedback
- **Multilingual Support**: Extend beyond English
- **Fact-Checking Integration**: Connect with fact-checking APIs
- **Explainability**: Highlight suspicious phrases

### Alternative Approaches Evaluated
- **BERT Fine-tuning**: Tested but showed poor performance on this dataset
- **Server-side Inference**: Would enable larger models but compromise privacy
- **Hybrid Architecture**: Current approach already implements this optimally

## 📄 Citation

If you use this project in research, please cite:
```
Fake News Detector Chrome Extension
GitHub: https://github.com/[username]/FakeNews
Accuracy: 90.17% on 4,403 news samples
Real-time browser-based fake news detection
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📜 License

This project is open source and available under the MIT License.

---

**⚡ Ready to fight misinformation? Install the extension and start browsing with confidence!**