// Comprehensive fake news detector with satirical content detection
class ComprehensiveFakeNewsDetector {
  constructor() {
    this.comprehensiveModel = null;
    this.trainedModel = null;
    this.isInitialized = false;
    this.initializeModels();
  }

  async initializeModels() {
    try {
      // Load comprehensive model
      const comprehensiveResponse = await fetch(chrome.runtime.getURL('models/comprehensive_model.json'));
      if (comprehensiveResponse.ok) {
        this.comprehensiveModel = await comprehensiveResponse.json();
        console.log('Comprehensive model loaded:', this.comprehensiveModel.version);
      }
    } catch (error) {
      console.warn('Failed to load comprehensive model:', error);
    }

    try {
      // Load optimized trained model as backup
      const trainedResponse = await fetch(chrome.runtime.getURL('models/optimized_detector_model.json'));
      if (trainedResponse.ok) {
        this.trainedModel = await trainedResponse.json();
        console.log('Optimized trained model loaded:', this.trainedModel.version);
      }
    } catch (error) {
      console.warn('Failed to load trained model:', error);
    }

    this.isInitialized = true;
    console.log('Comprehensive fake news detector initialized');
  }

  // Extract domain from URL
  extractDomain(url) {
    try {
      const domain = new URL(url).hostname.toLowerCase();
      return domain.startsWith('www.') ? domain.substring(4) : domain;
    } catch {
      return '';
    }
  }

  // Check if domain is known satirical site
  checkSatiricalDomain(url) {
    if (!this.comprehensiveModel || !url) return null;
    
    const domain = this.extractDomain(url);
    
    if (this.comprehensiveModel.domain_checks.satirical_domains.includes(domain)) {
      return {
        type: 'satirical',
        confidence: 0.95,
        reason: `${domain} is a known satirical/parody website`
      };
    }
    
    if (this.comprehensiveModel.domain_checks.conspiracy_domains.includes(domain)) {
      return {
        type: 'conspiracy',
        confidence: 0.85,
        reason: `${domain} is known for conspiracy theories and misinformation`
      };
    }
    
    return null;
  }

  // Analyze content for satirical patterns
  analyzeSatiricalContent(text, headline = '') {
    if (!this.comprehensiveModel) return null;

    const lowerText = (headline + ' ' + text).toLowerCase();
    const content = this.comprehensiveModel.content_analysis;
    
    let satiricalScore = 0;
    let matchedPatterns = [];

    // Check satirical indicators
    for (const indicator of content.satirical_indicators) {
      if (lowerText.includes(indicator)) {
        satiricalScore += content.weights.satirical_weight;
        matchedPatterns.push({type: 'satirical', pattern: indicator});
        if (matchedPatterns.length > 6) break;
      }
    }

    // Special patterns for satirical content
    const patterns = this.comprehensiveModel.pattern_detection;
    
    // Profanity in headlines (common in satirical news)
    if (/\b(fuck|shit|damn|ass|crap)\b/i.test(headline)) {
      satiricalScore += patterns.profanity_in_headlines;
      matchedPatterns.push({type: 'satirical', pattern: 'profanity_in_headline'});
    }

    // Absurd political combinations
    if (/\b(assures|promises|vows|pledges).{1,20}(wall street|billionaires|donors|wealthy|rich)\b/i.test(lowerText)) {
      satiricalScore += patterns.absurd_combinations;
      matchedPatterns.push({type: 'satirical', pattern: 'absurd_political_language'});
    }

    // The Onion style "Area Man" patterns
    if (/\b(area|local|nation's)\s+(man|woman|person|resident)\b/i.test(lowerText)) {
      satiricalScore += patterns.absurd_combinations;
      matchedPatterns.push({type: 'satirical', pattern: 'onion_style_headlines'});
    }

    return {
      score: satiricalScore,
      patterns: matchedPatterns,
      probability: Math.min(0.95, Math.max(0.05, 0.5 + satiricalScore))
    };
  }

  // Analyze content for general fake news patterns
  analyzeFakeNewsContent(text, headline = '') {
    if (!this.comprehensiveModel) return null;

    const lowerText = (headline + ' ' + text).toLowerCase();
    const content = this.comprehensiveModel.content_analysis;
    
    let fakeScore = 0;
    let realScore = 0;
    let matchedPatterns = [];

    // Check fake indicators
    for (const indicator of content.fake_indicators) {
      if (lowerText.includes(indicator)) {
        fakeScore += content.weights.fake_weight;
        matchedPatterns.push({type: 'fake', pattern: indicator});
        if (matchedPatterns.length > 5) break;
      }
    }

    // Check real indicators
    for (const indicator of content.real_indicators) {
      if (lowerText.includes(indicator)) {
        realScore += Math.abs(content.weights.real_weight);
        matchedPatterns.push({type: 'real', pattern: indicator});
        if (matchedPatterns.length > 5) break;
      }
    }

    const totalScore = fakeScore - realScore;
    
    return {
      score: totalScore,
      patterns: matchedPatterns,
      probability: Math.min(0.95, Math.max(0.05, 0.5 + totalScore * 0.3))
    };
  }

  // Use trained model as backup
  analyzeWithTrainedModel(text) {
    if (!this.trainedModel) return null;

    const lowerText = text.toLowerCase();
    let score = this.trainedModel.intercept;

    // Quick check of top indicators only for performance
    const topFakeIndicators = this.trainedModel.fake_indicators.slice(0, 20);
    const topRealIndicators = this.trainedModel.real_indicators.slice(0, 20);

    for (const indicator of topFakeIndicators) {
      if (lowerText.includes(indicator)) {
        score += this.trainedModel.fake_weights[indicator];
      }
    }

    for (const indicator of topRealIndicators) {
      if (lowerText.includes(indicator)) {
        score += this.trainedModel.real_weights[indicator];
      }
    }

    const probability = 1 / (1 + Math.exp(-score));
    return {
      prediction: probability > 0.5 ? 1 : 0,
      probability: Math.max(0.05, Math.min(0.95, probability)),
      method: 'trained_backup'
    };
  }

  // Main detection method
  detectFakeNews(text, url = '', headline = '') {
    if (!text || text.length < 20) {
      return {
        prediction: 0,
        confidence: 0.5,
        type: 'insufficient_data',
        method: 'insufficient_data',
        message: 'Text too short for analysis'
      };
    }

    const startTime = performance.now();

    // 1. Check domain first (fastest and most reliable)
    const domainCheck = this.checkSatiricalDomain(url);
    if (domainCheck) {
      return {
        prediction: 1,
        confidence: domainCheck.confidence,
        type: domainCheck.type,
        method: 'domain_detection',
        reason: domainCheck.reason,
        processingTime: Math.round(performance.now() - startTime),
        message: domainCheck.type === 'satirical' ? 
          `This is satirical content from ${this.extractDomain(url)}` :
          `This is from a site known for misinformation`
      };
    }

    // 2. Content analysis
    let results = [];
    
    // Satirical content analysis
    const satiricalAnalysis = this.analyzeSatiricalContent(text, headline);
    if (satiricalAnalysis && satiricalAnalysis.score > 0.7) {
      results.push({
        type: 'satirical',
        probability: satiricalAnalysis.probability,
        patterns: satiricalAnalysis.patterns,
        weight: 0.8
      });
    }

    // General fake news analysis
    const fakeAnalysis = this.analyzeFakeNewsContent(text, headline);
    if (fakeAnalysis) {
      results.push({
        type: 'misinformation',
        probability: fakeAnalysis.probability,
        patterns: fakeAnalysis.patterns,
        weight: 0.6
      });
    }

    // Trained model backup
    const trainedAnalysis = this.analyzeWithTrainedModel(text);
    if (trainedAnalysis) {
      results.push({
        type: 'statistical',
        probability: trainedAnalysis.probability,
        patterns: [],
        weight: 0.4
      });
    }

    // Combine results
    if (results.length === 0) {
      return {
        prediction: 0,
        confidence: 0.5,
        type: 'unknown',
        method: 'no_analysis',
        message: 'Unable to analyze content'
      };
    }

    // Weighted average of probabilities
    let weightedProbability = 0;
    let totalWeight = 0;
    let allPatterns = [];
    let methods = [];

    for (const result of results) {
      weightedProbability += result.probability * result.weight;
      totalWeight += result.weight;
      allPatterns = allPatterns.concat(result.patterns);
      methods.push(result.type);
    }

    const finalProbability = weightedProbability / totalWeight;
    const prediction = finalProbability > 0.5 ? 1 : 0;
    const confidence = Math.abs(finalProbability - 0.5) * 2;

    // Determine type based on highest scoring analysis
    const primaryResult = results.reduce((max, current) => 
      current.probability * current.weight > max.probability * max.weight ? current : max
    );

    const processingTime = Math.round(performance.now() - startTime);

    return {
      prediction,
      confidence: Math.max(0.05, Math.min(0.95, confidence)),
      probability: finalProbability,
      type: primaryResult.type,
      method: methods.join(' + '),
      matchedFeatures: allPatterns.slice(0, 5),
      processingTime,
      message: this.generateMessage(prediction, confidence, primaryResult.type)
    };
  }

  generateMessage(prediction, confidence, type) {
    const confidencePercent = Math.round(confidence * 100);
    
    if (prediction === 1) {
      if (type === 'satirical') {
        return `This appears to be satirical/parody content (${confidencePercent}% confidence)`;
      } else if (type === 'misinformation') {
        return `This content may contain misinformation (${confidencePercent}% confidence)`;
      } else {
        return `This content may be misleading (${confidencePercent}% confidence)`;
      }
    } else {
      return `This content appears legitimate (${confidencePercent}% confidence)`;
    }
  }
}

// Initialize comprehensive detector
const detector = new ComprehensiveFakeNewsDetector();

// Message handling
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'analyzeText') {
    if (!message.text || typeof message.text !== 'string') {
      sendResponse({
        prediction: 0,
        confidence: 0.5,
        method: 'error',
        message: 'No text provided for analysis'
      });
      return true;
    }
    
    const url = sender.tab?.url || message.url || '';
    const headline = message.headline || '';
    
    const result = detector.detectFakeNews(message.text, url, headline);
    
    // Store result for popup
    chrome.storage.local.set({
      lastAnalysis: {
        ...result,
        text: message.text.length > 200 ? 
          message.text.substring(0, 200) + '...' : 
          message.text,
        timestamp: Date.now(),
        url: url
      }
    });

    console.log(`Analysis completed in ${result.processingTime || 0}ms:`, {
      prediction: result.prediction === 1 ? 'FAKE' : 'REAL',
      type: result.type,
      confidence: Math.round(result.confidence * 100) + '%'
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

console.log('Comprehensive fake news detector loaded - now detects satirical content!');