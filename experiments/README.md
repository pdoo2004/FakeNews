# 🧪 Experiments

This folder contains experimental scripts and research code used during development.

## 📋 Files

### `convert_simple.py`
- Converts sklearn models to simple JSON format
- Used for creating lightweight browser models
- **Status**: Superseded by optimized conversion in main training scripts

### `convert_to_tfjs.py`
- Experimental TensorFlow.js conversion attempts
- Would enable full model running in browser
- **Status**: Abandoned due to size/performance constraints

### `create_lightweight_model.py`
- Creates performance-optimized models for browser use
- Extracts top N features from full models
- **Status**: Integrated into main training pipeline

### `evaluate_bert_potential.py`
- Evaluates BERT vs traditional ML approaches
- Tests pre-trained BERT models on fake news data
- **Results**: BERT showed 48% accuracy vs 90% for traditional ML
- **Status**: Research complete - traditional ML chosen

## 🔬 Research Findings

### BERT Evaluation Results
```
Model                          | Accuracy | Speed/Sample
------------------------------|----------|-------------
jy46604790/Fake-News-Bert-Detect | 48.0%   | 152ms
martin-ha/toxic-comment-model    | 42.0%   | 30ms
Optimized Traditional ML          | 90.17%  | 29ms
```

### Key Insights
1. **BERT underperformed** on this specific dataset (2016 political news)
2. **Traditional ML** achieved better accuracy with 5x speed improvement
3. **Browser constraints** favor lightweight models over transformer-based approaches
4. **Domain-specific training** matters more than model sophistication

## 🎯 Conclusions

- **Traditional ML won** for this use case due to dataset characteristics
- **Performance constraints** in browser favor lightweight approaches
- **Ensemble methods** provide good accuracy-speed balance
- **Domain knowledge** (satirical sites) more valuable than complex models

## 🔄 Future Research

Potential areas for further experimentation:
- **Fine-tuned BERT** on fake news data (would require training infrastructure)
- **Lightweight transformers** (DistilBERT, MobileBERT)
- **Federated learning** from user feedback
- **Multi-modal detection** (text + images + metadata)