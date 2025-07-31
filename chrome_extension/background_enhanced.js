// Enhanced fake news detector with TF-IDF model support
class EnhancedFakeNewsDetector {
  constructor() {
    this.baselineModel = null;
    this.keywordModel = null;
    this.isInitialized = false;
    this.initializeModels();
  }

  async initializeModels() {
    try {
      // Load TF-IDF baseline model
      const baselineResponse = await fetch(chrome.runtime.getURL('models/baseline_model.json'));
      if (baselineResponse.ok) {
        this.baselineModel = await baselineResponse.json();
        console.log('Enhanced baseline model loaded:', this.baselineModel.type);
      }
    } catch (error) {
      console.warn('Failed to load baseline model:', error);
    }

    try {
      // Load enhanced keyword model
      const keywordResponse = await fetch(chrome.runtime.getURL('models/keyword_model.json'));
      if (keywordResponse.ok) {
        this.keywordModel = await keywordResponse.json();
        console.log('Enhanced keyword model loaded');
      }
    } catch (error) {
      console.warn('Failed to load keyword model:', error);
    }

    this.isInitialized = true;
    console.log('Enhanced fake news detector initialized');
  }

  // Text preprocessing for TF-IDF
  preprocessText(text) {
    return text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')  // Remove punctuation
      .replace(/\s+/g, ' ')      // Normalize whitespace
      .trim();
  }

  // Generate n-grams from text
  generateNgrams(tokens, n) {
    const ngrams = [];
    for (let i = 0; i <= tokens.length - n; i++) {
      ngrams.push(tokens.slice(i, i + n).join(' '));
    }
    return ngrams;
  }

  // TF-IDF prediction
  predictWithTFIDF(text) {
    if (!this.baselineModel || this.baselineModel.type !== 'tfidf_logistic') {
      return null;
    }

    const preprocessed = this.preprocessText(text);
    const tokens = preprocessed.split(' ').filter(token => token.length > 0);
    
    // Generate n-grams based on model configuration
    const [minN, maxN] = this.baselineModel.ngram_range;
    let allFeatures = [];
    
    for (let n = minN; n <= maxN; n++) {
      allFeatures = allFeatures.concat(this.generateNgrams(tokens, n));
    }

    // Calculate TF-IDF vector
    const featureVector = new Array(this.baselineModel.coefficients.length).fill(0);
    const termCounts = {};
    
    // Count term frequencies
    allFeatures.forEach(feature => {
      termCounts[feature] = (termCounts[feature] || 0) + 1;
    });

    // Calculate TF-IDF values
    const totalTerms = allFeatures.length;
    for (const [term, count] of Object.entries(termCounts)) {
      if (term in this.baselineModel.vocabulary) {
        const vocabIndex = this.baselineModel.vocabulary[term];
        if (vocabIndex < this.baselineModel.idf_values.length) {
          const tf = count / totalTerms;
          const idf = this.baselineModel.idf_values[vocabIndex];
          featureVector[vocabIndex] = tf * idf;
        }
      }
    }

    // Calculate logistic regression prediction
    let score = this.baselineModel.intercept;
    for (let i = 0; i < this.baselineModel.coefficients.length; i++) {
      score += featureVector[i] * this.baselineModel.coefficients[i];
    }

    // Convert to probability using sigmoid function
    const probability = 1 / (1 + Math.exp(-score));
    const prediction = probability > 0.5 ? 1 : 0;

    return {
      prediction,
      probability,
      score,
      method: 'tfidf_logistic'
    };
  }

  // Enhanced keyword-based prediction
  predictWithKeywords(text) {
    if (!this.keywordModel) {
      return null;
    }

    const lowerText = text.toLowerCase();
    let fakeScore = 0;
    let realScore = 0;
    let matchedFeatures = [];

    // Check fake indicators
    for (const [category, keywords] of Object.entries(this.keywordModel.fake_indicators)) {
      for (const keyword of keywords) {
        if (lowerText.includes(keyword)) {
          fakeScore += this.keywordModel.weights.fake_base_weight;
          matchedFeatures.push({ type: 'fake', keyword, category });
        }
      }
    }

    // Check real indicators
    for (const [category, keywords] of Object.entries(this.keywordModel.real_indicators)) {
      for (const keyword of keywords) {
        if (lowerText.includes(keyword)) {
          realScore += this.keywordModel.weights.real_base_weight;
          matchedFeatures.push({ type: 'real', keyword, category });
        }
      }
    }

    // Apply penalties
    if (text.length < 100) {
      fakeScore += this.keywordModel.weights.length_penalty;
    }

    const capsCount = (text.match(/[A-Z]/g) || []).length;
    const capsRatio = capsCount / text.length;
    if (capsRatio > 0.1) {
      fakeScore += this.keywordModel.weights.caps_penalty * capsRatio;
    }

    const exclamationCount = (text.match(/!/g) || []).length;
    if (exclamationCount > 2) {
      fakeScore += this.keywordModel.weights.exclamation_penalty * exclamationCount;
    }

    const totalScore = fakeScore - realScore;
    const probability = Math.max(0.1, Math.min(0.9, 0.5 + totalScore));
    const prediction = probability > 0.5 ? 1 : 0;

    return {
      prediction,
      probability,
      score: totalScore,
      matchedFeatures: matchedFeatures.slice(0, 5),
      method: 'keyword_enhanced'
    };
  }

  // Main detection method with ensemble approach
  detectFakeNews(text) {
    if (!text || text.length < 20) {
      return {
        prediction: 0,
        confidence: 0.5,
        method: 'insufficient_data',
        message: 'Text too short for analysis'
      };
    }

    const results = [];
    
    // Try TF-IDF model first (most accurate)
    const tfidfResult = this.predictWithTFIDF(text);
    if (tfidfResult) {
      results.push({
        ...tfidfResult,
        weight: 0.7  // Higher weight for the more sophisticated model
      });
    }

    // Try keyword model
    const keywordResult = this.predictWithKeywords(text);
    if (keywordResult) {
      results.push({
        ...keywordResult,
        weight: 0.3
      });
    }

    if (results.length === 0) {
      return {
        prediction: 0,
        confidence: 0.5,
        method: 'fallback',
        message: 'No models available for analysis'
      };
    }

    // Ensemble prediction - weighted average
    let weightedScore = 0;
    let totalWeight = 0;
    let methods = [];

    for (const result of results) {
      weightedScore += result.probability * result.weight;
      totalWeight += result.weight;
      methods.push(result.method);
    }

    const finalProbability = weightedScore / totalWeight;
    const prediction = finalProbability > 0.5 ? 1 : 0;
    const confidence = Math.abs(finalProbability - 0.5) * 2; // Convert to 0-1 scale

    // Get features from keyword model if available
    const features = keywordResult ? keywordResult.matchedFeatures : [];

    return {
      prediction,
      confidence: Math.max(0.05, Math.min(0.95, confidence)),
      probability: finalProbability,
      method: methods.join(' + '),
      matchedFeatures: features,
      message: prediction === 1 ? 
        `This content may be misleading (${Math.round(confidence * 100)}% confidence)` : 
        `This content appears legitimate (${Math.round(confidence * 100)}% confidence)`
    };
  }
}

// Initialize enhanced detector
const detector = new EnhancedFakeNewsDetector();

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

console.log('Enhanced fake news detector background script loaded');