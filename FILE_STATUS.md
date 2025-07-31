# 📋 File Status Guide

This document shows which files are actively used vs archived for your fake news detection system.

## 🚀 **ACTIVE FILES** (Currently Used)

### Chrome Extension (Production)
```
chrome_extension/
├── background_comprehensive.js          ✅ Current detection engine
├── contentScript.js                     ✅ Article extraction
├── popup.html                          ✅ User interface  
├── popup.js                            ✅ UI functionality
├── manifest.json                       ✅ Extension config
└── models/
    ├── comprehensive_model.json         ✅ Domain & pattern detection
    └── optimized_detector_model.json    ✅ ML model (90.17% accuracy)
```

### ML Pipeline (Training & Development)
```
ml_pipeline/
├── train_optimized_model.py            ✅ Current training script
├── create_comprehensive_model.py       ✅ Creates domain/pattern models
├── baseline_model.py                   ✅ Core ML components
├── bert_model.py                       ✅ BERT implementation (optional)
├── data_loader.py                      ✅ Dataset utilities
├── preprocessing.py                    ✅ Text preprocessing
├── data/real_dataset.csv               ✅ Training data (4,403 samples)
└── models/optimized_model.pkl          ✅ Full trained model
```

### Documentation
```
README.md                               ✅ Main project documentation
FILE_STATUS.md                          ✅ This file
```

## 📦 **ARCHIVED FILES** (Legacy/Experimental)

### Legacy Chrome Extension Versions
```
legacy/chrome_extension_versions/
├── background.js                       ❌ Original keyword-based
├── background_simple.js                ❌ Simple keyword model
├── background_fast.js                  ❌ Performance-optimized (superseded)
└── background_enhanced.js              ❌ TF-IDF heavy (too slow)
```

### Legacy ML Training Scripts
```
legacy/ml_training_versions/
├── train_baseline.py                   ❌ Original basic training
├── train_enhanced.py                   ❌ Enhanced with larger dataset
├── train_simple_enhanced.py            ❌ Simplified enhanced
└── train_working_enhanced.py           ❌ Working version (basis for optimized)
```

### Legacy Models & Data
```
legacy/models/
├── baseline_model.pkl                  ❌ Original baseline model
├── enhanced_baseline_model.pkl         ❌ Enhanced baseline (91.4% accuracy)
├── baseline_model.json                 ❌ Browser version of baseline
├── detector_model.json                 ❌ Lightweight (200 features)
├── fast_keyword_model.json             ❌ Fast keyword-only
├── keyword_model.json                  ❌ Enhanced keywords
└── sample_dataset.csv                  ❌ Original 30-sample test data
```

### Experimental Research
```
experiments/
├── evaluate_bert_potential.py          🧪 BERT vs traditional ML study
├── convert_to_tfjs.py                  🧪 TensorFlow.js experiments
├── create_lightweight_model.py         🧪 Browser optimization tests
└── convert_simple.py                   🧪 Simple model conversion
```

## 🎯 **Quick Start - Active Files Only**

If you want to run the current system, you only need:

1. **For Extension Users**:
   - `chrome_extension/` folder (load in Chrome)

2. **For Developers**:
   - `ml_pipeline/train_optimized_model.py` (retrain models)
   - `ml_pipeline/create_comprehensive_model.py` (update patterns)
   - `ml_pipeline/data/real_dataset.csv` (training data)

3. **For Documentation**:
   - `README.md` (complete guide)

## ⚠️ **Safe to Delete**

You can safely delete these folders if you want to save space:
- `legacy/` (kept for reference only)
- `experiments/` (research completed)
- `venv/` (can be recreated with `python -m venv venv`)

## 🔄 **Evolution Summary**

1. **v1.0**: Simple keyword matching → `legacy/chrome_extension_versions/background.js`
2. **v2.0**: Basic ML model (85% accuracy) → `legacy/ml_training_versions/train_baseline.py`
3. **v3.0**: Enhanced dataset (91.4% accuracy) → `legacy/ml_training_versions/train_working_enhanced.py`
4. **v4.0**: **CURRENT** - Comprehensive system (90.17% accuracy + satirical detection) → `chrome_extension/background_comprehensive.js`

## 📊 **Current Performance**
- **Accuracy**: 90.17% on test data
- **Speed**: ~29ms per article
- **Features**: 295 optimized features
- **Detection Types**: Satirical, conspiracy, misinformation
- **Domain Coverage**: 25 known problematic sites