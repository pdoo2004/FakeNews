// Fast fake news detector optimized for real-time performance
class FastFakeNewsDetector {
  constructor() {
    this.detectorModel = null;
    this.keywordModel = null;
    this.isInitialized = false;
    this.initializeModels();
  }

  async initializeModels() {
    try {
      // Load lightweight trained model
      const detectorResponse = await fetch(chrome.runtime.getURL('models/detector_model.json'));
      if (detectorResponse.ok) {
        this.detectorModel = await detectorResponse.json();
        console.log('Fast detector model loaded:', this.detectorModel.type);
      }
    } catch (error) {
      console.warn('Failed to load detector model:', error);
    }

    try {
      // Load fast keyword model  
      const keywordResponse = await fetch(chrome.runtime.getURL('models/fast_keyword_model.json'));
      if (keywordResponse.ok) {
        this.keywordModel = await keywordResponse.json();
        console.log('Fast keyword model loaded');
      }
    } catch (error) {
      console.warn('Failed to load keyword model:', error);
    }

    this.isInitialized = true;
    console.log('Fast fake news detector initialized');
  }

  // Lightning fast trained model prediction
  predictWithTrainedModel(text) {
    if (!this.detectorModel || this.detectorModel.type !== 'lightweight_trained') {
      return null;
    }

    const lowerText = text.toLowerCase();
    let score = this.detectorModel.intercept;
    let matchedFeatures = [];

    // Check fake indicators (optimized loop)
    for (const indicator of this.detectorModel.fake_indicators) {
      if (lowerText.includes(indicator)) {
        score += this.detectorModel.fake_weights[indicator];
        matchedFeatures.push({ type: 'fake', term: indicator });
        if (matchedFeatures.length > 8) break; // Limit for performance
      }
    }

    // Check real indicators (optimized loop)
    for (const indicator of this.detectorModel.real_indicators) {
      if (lowerText.includes(indicator)) {
        score += this.detectorModel.real_weights[indicator];
        matchedFeatures.push({ type: 'real', term: indicator });
        if (matchedFeatures.length > 8) break; // Limit for performance
      }
    }

    // Convert to probability using sigmoid function
    const probability = 1 / (1 + Math.exp(-score));
    
    // Clamp probability to reasonable bounds
    const clampedProbability = Math.max(0.05, Math.min(0.95, probability));

    const prediction = clampedProbability > 0.5 ? 1 : 0;

    return {
      prediction,
      probability: clampedProbability,
      score,
      matchedFeatures: matchedFeatures.slice(0, 5),
      method: 'lightweight_trained'
    };
  }

  // Ultra-fast keyword prediction
  predictWithKeywords(text) {
    if (!this.keywordModel) return null;

    const lowerText = text.toLowerCase();
    let score = 0;
    let matchedFeatures = [];

    // Fast fake keyword check
    for (const keyword of this.keywordModel.fake_keywords) {
      if (lowerText.includes(keyword)) {
        score += this.keywordModel.fake_weight;
        matchedFeatures.push({ type: 'fake', keyword });
        if (matchedFeatures.length > 6) break; // Performance limit
      }
    }

    // Fast real keyword check
    for (const keyword of this.keywordModel.real_keywords) {
      if (lowerText.includes(keyword)) {
        score += this.keywordModel.real_weight;
        matchedFeatures.push({ type: 'real', keyword });
        if (matchedFeatures.length > 6) break; // Performance limit
      }
    }

    // Apply fast penalties
    const penalties = this.keywordModel.penalties;
    if (text.length < 100) score += penalties.short_text;
    
    const capsRatio = (text.match(/[A-Z]/g) || []).length / text.length;
    if (capsRatio > 0.15) score += penalties.excessive_caps;
    
    const exclamations = (text.match(/!/g) || []).length;
    if (exclamations > 3) score += penalties.many_exclamations;

    // Check for clickbait numbers
    if (/\b(shocking|amazing|incredible)\s+\d+|this\s+\d+|\d+\s+(tricks?|secrets?|ways?)\b/i.test(text)) {
      score += penalties.clickbait_numbers;
    }

    const probability = Math.max(0.05, Math.min(0.95, 0.5 + score * this.keywordModel.confidence_multiplier));
    const prediction = probability > 0.5 ? 1 : 0;

    return {
      prediction,
      probability,
      score,
      matchedFeatures: matchedFeatures.slice(0, 4),
      method: 'fast_keyword'
    };
  }

  // Main detection method - optimized for speed
  detectFakeNews(text) {
    // Early exit for very short text
    if (!text || text.length < 20) {
      return {
        prediction: 0,
        confidence: 0.5,
        method: 'insufficient_data',
        message: 'Text too short for analysis'
      };
    }

    // Truncate very long text for performance
    const analysisText = text.length > 2000 ? text.substring(0, 2000) : text;
    
    let primaryResult = null;
    let backupResult = null;

    // Try trained model first (most accurate)
    primaryResult = this.predictWithTrainedModel(analysisText);
    
    // Fallback to keyword model
    if (!primaryResult) {
      primaryResult = this.predictWithKeywords(analysisText);
    } else {
      // Use keyword model as backup for confidence boost
      backupResult = this.predictWithKeywords(analysisText);
    }

    if (!primaryResult) {
      return {
        prediction: 0,
        confidence: 0.5,
        method: 'no_models',
        message: 'Analysis unavailable'
      };
    }

    let finalPrediction = primaryResult.prediction;
    let finalProbability = primaryResult.probability;
    let methods = [primaryResult.method];

    // Combine with backup if available and predictions agree
    if (backupResult && backupResult.prediction === primaryResult.prediction) {
      // Average the probabilities when they agree
      finalProbability = (primaryResult.probability + backupResult.probability) / 2;
      methods.push(backupResult.method);
    }

    // Calculate confidence based on how far probability is from 0.5 (uncertainty)
    const finalConfidence = Math.abs(finalProbability - 0.5) * 2;
    
    // Get features (prefer trained model features)
    const features = primaryResult.matchedFeatures || [];

    return {
      prediction: finalPrediction,
      confidence: Math.max(0.05, Math.min(0.95, finalConfidence)),
      probability: finalProbability,
      method: methods.join(' + '),
      matchedFeatures: features,
      message: finalPrediction === 1 ? 
        `This content may be misleading (${Math.round(finalConfidence * 100)}% confidence)` : 
        `This content appears legitimate (${Math.round(finalConfidence * 100)}% confidence)`
    };
  }
}

// Initialize fast detector
const detector = new FastFakeNewsDetector();

// Optimized message handling
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'analyzeText') {
    // Validate input
    if (!message.text || typeof message.text !== 'string') {
      const errorResult = {
        prediction: 0,
        confidence: 0.5,
        method: 'error',
        message: 'No text provided for analysis'
      };
      sendResponse(errorResult);
      return true;
    }
    
    // Performance monitoring
    const startTime = performance.now();
    
    const result = detector.detectFakeNews(message.text);
    
    const endTime = performance.now();
    const processingTime = Math.round(endTime - startTime);
    
    // Add performance info
    result.processingTime = processingTime;
    
    // Store result for popup
    const textPreview = message.text.length > 200 ? 
      message.text.substring(0, 200) + '...' : 
      message.text;
      
    chrome.storage.local.set({
      lastAnalysis: {
        ...result,
        text: textPreview,
        timestamp: Date.now(),
        url: sender.tab?.url || 'Unknown'
      }
    });

    console.log(`Analysis completed in ${processingTime}ms`);
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

console.log('Fast fake news detector loaded - optimized for real-time performance');