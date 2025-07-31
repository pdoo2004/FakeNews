# 📁 Legacy Files

This folder contains previous versions and experimental files that are no longer actively used but kept for reference.

## 📂 Folder Structure

### `chrome_extension_versions/`
Previous versions of the Chrome extension background scripts:
- `background.js` - Original keyword-based detector
- `background_simple.js` - Simple keyword model version
- `background_fast.js` - Performance-optimized version (superseded by comprehensive)
- `background_enhanced.js` - TF-IDF heavy version (too slow for browser)

### `ml_training_versions/`
Previous training scripts and approaches:
- `train_baseline.py` - Original basic TF-IDF training
- `train_enhanced.py` - Enhanced version with larger dataset
- `train_simple_enhanced.py` - Simplified enhanced training
- `train_working_enhanced.py` - Working version (basis for optimized)

### `models/`
Legacy trained models and datasets:
- `baseline_model.pkl` - Original baseline model
- `enhanced_baseline_model.pkl` - Enhanced baseline (91.4% accuracy)
- `baseline_model.json` - Browser version of baseline
- `detector_model.json` - Lightweight detector (200 features)
- `fast_keyword_model.json` - Fast keyword-only model
- `keyword_model.json` - Enhanced keywords
- `sample_dataset.csv` - Original 30-sample test dataset

## 🔄 Evolution Path

1. **Simple Keywords** → Basic fake news patterns
2. **Baseline TF-IDF** → Statistical text analysis (85% accuracy)
3. **Enhanced Baseline** → Larger dataset (91.4% accuracy)
4. **Optimized Ensemble** → Current version (90.17% accuracy with speed)

## ⚠️ Usage Notes

- These files are **not used** by the current system
- Kept for historical reference and potential future experiments
- May contain outdated dependencies or APIs
- Use current versions in the main directories instead

## 🧪 Research Value

These files document the iterative development process and can be useful for:
- Understanding design decisions
- Comparing different approaches
- Academic research on fake news detection
- Learning from development iterations