// Simple, robust fake news detector
class SimpleFakeNewsDetector {
  constructor() {
    this.detectorModel = null;
    this.keywordModel = null;
    this.isInitialized = false;
    this.initializeModels();
  }

  async initializeModels() {
    try {
      // Load main detector model
      const detectorResponse = await fetch(chrome.runtime.getURL('models/detector_model.json'));
      if (detectorResponse.ok) {
        this.detectorModel = await detectorResponse.json();
        console.log('Detector model loaded:', this.detectorModel.type);
      }
    } catch (error) {
      console.warn('Failed to load detector model:', error);
    }

    try {
      // Load keyword model
      const keywordResponse = await fetch(chrome.runtime.getURL('models/keyword_model.json'));
      if (keywordResponse.ok) {
        this.keywordModel = await keywordResponse.json();
        console.log('Keyword model loaded');
      }
    } catch (error) {
      console.warn('Failed to load keyword model:', error);
    }

    this.isInitialized = true;
    console.log('Simple fake news detector initialized');
  }

  // Main detection method
  detectFakeNews(text) {
    if (!text || text.length < 20) {
      return {
        prediction: 0,
        confidence: 0.5,
        method: 'insufficient_data',
        message: 'Text too short for analysis'
      };
    }

    const lowerText = text.toLowerCase();
    let score = 0;
    let matchedFeatures = [];

    // Use detector model if available
    if (this.detectorModel) {
      if (this.detectorModel.type === 'trained_lightweight') {
        // Check fake indicators
        for (const word of this.detectorModel.fake_indicators) {
          if (lowerText.includes(word)) {
            score += this.detectorModel.fake_weight;
            matchedFeatures.push({ type: 'fake', word });
          }
        }

        // Check real indicators
        for (const word of this.detectorModel.real_indicators) {
          if (lowerText.includes(word)) {
            score += this.detectorModel.real_weight;
            matchedFeatures.push({ type: 'real', word });
          }
        }

        score += this.detectorModel.intercept || 0;

      } else if (this.detectorModel.type === 'simple_detector') {
        // Simple keyword detection
        for (const word of this.detectorModel.fake_keywords) {
          if (lowerText.includes(word)) {
            score += this.detectorModel.fake_weight;
            matchedFeatures.push({ type: 'fake', word });
          }
        }

        for (const word of this.detectorModel.real_keywords) {
          if (lowerText.includes(word)) {
            score += this.detectorModel.real_weight;
            matchedFeatures.push({ type: 'real', word });
          }
        }
      }
    }

    // Calculate confidence
    const rawConfidence = Math.abs(score) * (this.detectorModel?.confidence_multiplier || 0.1);
    const confidence = Math.min(0.95, Math.max(0.05, 0.5 + rawConfidence));
    
    // Make prediction
    const prediction = score > (this.detectorModel?.threshold || 0) ? 1 : 0;
    
    // Adjust confidence based on prediction
    const finalConfidence = prediction === 1 ? confidence : (1 - confidence);

    return {
      prediction,
      confidence: finalConfidence,
      method: this.detectorModel?.type || 'fallback',
      score,
      matchedFeatures: matchedFeatures.slice(0, 5),
      message: prediction === 1 ? 
        `This content may be misleading (${Math.round(finalConfidence * 100)}% confidence)` : 
        `This content appears legitimate (${Math.round(finalConfidence * 100)}% confidence)`
    };
  }
}

// Initialize detector
const detector = new SimpleFakeNewsDetector();

// Message handling
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'analyzeText') {
    const result = detector.detectFakeNews(message.text);
    
    // Store result for popup
    chrome.storage.local.set({
      lastAnalysis: {
        ...result,
        text: message.text.substring(0, 200) + '...',
        timestamp: Date.now(),
        url: sender.tab?.url || 'Unknown'
      }
    });

    sendResponse(result);
    return true;
  }

  if (message.action === 'getLastAnalysis') {
    chrome.storage.local.get(['lastAnalysis']).then(result => {
      sendResponse(result.lastAnalysis || null);
    });
    return true;
  }
});

console.log('Simple fake news detector background script loaded'); 