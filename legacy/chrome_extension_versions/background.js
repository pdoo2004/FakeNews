// Background script for fake news detection
class FakeNewsDetector {
    constructor() {
        this.model = null;
        this.analysisCache = new Map();
        this.init();
    }
    
    async init() {
        console.log('Fake News Detector: Initializing background script...');
        
        // For now, we'll use a simple keyword-based detection
        // In a real implementation, this would load the TensorFlow.js model
        this.initializeSimpleModel();
        
        // Listen for messages from content script and popup
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Keep channel open for async response
        });
        
        console.log('Fake News Detector: Background script initialized');
    }
    
    initializeSimpleModel() {
        // Simple keyword-based model for demonstration
        // In production, this would load the actual TF.js model
        this.fakeNewsKeywords = [
            'breaking', 'shocking', 'unbelievable', 'miracle', 'secret',
            'doctors hate', 'they don\'t want you to know', 'banned',
            'conspiracy', 'government hiding', 'big pharma', 'cure',
            'exposed', 'leaked', 'insider reveals', 'urgent',
            'must read', 'going viral', 'share before deleted'
        ];
        
        this.reliableIndicators = [
            'according to', 'study shows', 'research indicates',
            'expert says', 'data suggests', 'evidence shows',
            'peer reviewed', 'published in', 'university',
            'institute', 'professor', 'dr.', 'phd'
        ];
    }
    
    async handleMessage(request, sender, sendResponse) {
        try {
            switch (request.action) {
                case 'analyzeText':
                    const analysis = await this.analyzeText(request.data);
                    
                    // Cache the analysis
                    if (sender.tab) {
                        this.analysisCache.set(sender.tab.id, analysis);
                    }
                    
                    sendResponse({ analysis });
                    break;
                    
                case 'getAnalysis':
                    const cachedAnalysis = this.analysisCache.get(request.tabId);
                    sendResponse({ analysis: cachedAnalysis });
                    break;
                    
                default:
                    sendResponse({ error: 'Unknown action' });
            }
        } catch (error) {
            console.error('Fake News Detector Error:', error);
            sendResponse({ error: error.message });
        }
    }
    
    async analyzeText(data) {
        const text = (data.headline + ' ' + data.content).toLowerCase();
        
        // Simple scoring system
        let fakeScore = 0;
        let realScore = 0;
        
        // Check for fake news indicators
        this.fakeNewsKeywords.forEach(keyword => {
            const count = (text.match(new RegExp(keyword, 'g')) || []).length;
            fakeScore += count * 2;
        });
        
        // Check for reliable indicators
        this.reliableIndicators.forEach(indicator => {
            const count = (text.match(new RegExp(indicator, 'g')) || []).length;
            realScore += count * 1.5;
        });
        
        // Additional heuristics
        if (text.includes('click here') || text.includes('you won\'t believe')) {
            fakeScore += 3;
        }
        
        if (text.match(/\d{4}/) && text.includes('study')) { // Contains year and study
            realScore += 2;
        }
        
        // Normalize scores
        const totalScore = fakeScore + realScore;
        const confidence = totalScore > 0 ? Math.max(fakeScore, realScore) / totalScore : 0.5;
        const isFake = fakeScore > realScore;
        
        // Adjust confidence based on content length and quality
        let adjustedConfidence = confidence;
        if (data.content.length < 200) {
            adjustedConfidence *= 0.7; // Lower confidence for short content
        }
        
        const analysis = {
            isFake: isFake,
            confidence: Math.min(adjustedConfidence, 0.95), // Cap at 95%
            fakeScore: fakeScore,
            realScore: realScore,
            url: data.url,
            timestamp: Date.now()
        };
        
        console.log('Fake News Detector: Analysis complete', {
            url: data.url,
            isFake: analysis.isFake,
            confidence: Math.round(analysis.confidence * 100) + '%',
            fakeScore,
            realScore
        });
        
        return analysis;
    }
    
    // Future method for loading TensorFlow.js model
    async loadTensorFlowModel() {
        try {
            // This would be implemented when we have the converted model
            // const model = await tf.loadGraphModel(chrome.runtime.getURL('tfjs_model/model.json'));
            // this.model = model;
            console.log('TensorFlow.js model would be loaded here');
        } catch (error) {
            console.error('Failed to load TensorFlow.js model:', error);
        }
    }
    
    // Future method for TensorFlow.js inference
    async predictWithTensorFlow(text) {
        if (!this.model) {
            throw new Error('Model not loaded');
        }
        
        // This would implement the actual TF.js prediction
        // const input = this.preprocessText(text);
        // const prediction = this.model.predict(input);
        // return prediction;
    }
}

// Initialize the detector
new FakeNewsDetector();