document.addEventListener('DOMContentLoaded', function() {
    const statusDiv = document.getElementById('status');
    const confidenceDiv = document.getElementById('confidence');
    const confidenceValue = document.getElementById('confidence-value');
    const recheckBtn = document.getElementById('recheck-btn');
    const feedbackSection = document.getElementById('feedback-section');
    const correctBtn = document.getElementById('correct-btn');
    const incorrectBtn = document.getElementById('incorrect-btn');
    
    let currentPrediction = null;
    
    // Get current tab and check for analysis
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        checkCurrentPage(tabs[0].id);
    });
    
    // Recheck button handler
    recheckBtn.addEventListener('click', function() {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            statusDiv.textContent = 'Analyzing current page...';
            statusDiv.className = 'status analyzing';
            confidenceDiv.style.display = 'none';
            feedbackSection.style.display = 'none';
            
            // Send message to content script to re-analyze
            chrome.tabs.sendMessage(tabs[0].id, {action: 'reanalyze'}, function(response) {
                if (chrome.runtime.lastError) {
                    showError('Could not analyze this page');
                } else {
                    checkCurrentPage(tabs[0].id);
                }
            });
        });
    });
    
    // Feedback handlers
    correctBtn.addEventListener('click', function() {
        saveFeedback(true);
    });
    
    incorrectBtn.addEventListener('click', function() {
        saveFeedback(false);
    });
    
    function checkCurrentPage(tabId) {
        // Request analysis from background script
        chrome.runtime.sendMessage({action: 'getAnalysis', tabId: tabId}, function(response) {
            if (response && response.analysis) {
                displayResult(response.analysis);
            } else {
                // Try to trigger analysis
                chrome.tabs.sendMessage(tabId, {action: 'analyze'}, function(contentResponse) {
                    if (chrome.runtime.lastError) {
                        showError('Could not analyze this page');
                    } else {
                        setTimeout(() => checkCurrentPage(tabId), 1000); // Wait and check again
                    }
                });
            }
        });
    }
    
    function displayResult(analysis) {
        currentPrediction = analysis;
        
        if (analysis.isFake) {
            statusDiv.textContent = '⚠️ Potentially Fake News';
            statusDiv.className = 'status fake';
        } else {
            statusDiv.textContent = '✅ Likely Reliable';
            statusDiv.className = 'status real';
        }
        
        if (analysis.confidence) {
            confidenceValue.textContent = Math.round(analysis.confidence * 100);
            confidenceDiv.style.display = 'block';
        }
        
        feedbackSection.style.display = 'block';
    }
    
    function showError(message) {
        statusDiv.textContent = message;
        statusDiv.className = 'status error';
        confidenceDiv.style.display = 'none';
        feedbackSection.style.display = 'none';
    }
    
    function saveFeedback(isCorrect) {
        if (currentPrediction) {
            chrome.storage.local.get(['feedback'], function(result) {
                const feedback = result.feedback || [];
                feedback.push({
                    prediction: currentPrediction,
                    isCorrect: isCorrect,
                    timestamp: Date.now()
                });
                
                chrome.storage.local.set({feedback: feedback}, function() {
                    // Show thank you message
                    const originalText = isCorrect ? correctBtn.textContent : incorrectBtn.textContent;
                    const button = isCorrect ? correctBtn : incorrectBtn;
                    button.textContent = 'Thanks!';
                    setTimeout(() => {
                        button.textContent = originalText;
                    }, 2000);
                });
            });
        }
    }
});