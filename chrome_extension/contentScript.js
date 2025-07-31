// Content script for fake news detection
class ArticleExtractor {
    constructor() {
        this.analyzed = false;
        this.init();
    }
    
    init() {
        // Auto-analyze when page loads
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.analyzeArticle());
        } else {
            this.analyzeArticle();
        }
        
        // Listen for messages from popup
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'analyze' || request.action === 'reanalyze') {
                this.analyzeArticle();
                sendResponse({success: true});
            }
            return true;
        });
    }
    
    extractArticleText() {
        // Try multiple selectors to find article content
        const selectors = [
            'article',
            '[role="main"]',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '.story-body',
            '.article-body'
        ];
        
        let articleText = '';
        
        // Try structured selectors first
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                const paragraphs = element.querySelectorAll('p');
                if (paragraphs.length > 2) {
                    articleText = Array.from(paragraphs)
                        .map(p => p.innerText.trim())
                        .filter(text => text.length > 20)
                        .join(' ');
                    break;
                }
            }
        }
        
        // Fallback: extract all paragraphs from the page
        if (!articleText || articleText.length < 100) {
            const allParagraphs = document.querySelectorAll('p');
            const paragraphTexts = Array.from(allParagraphs)
                .map(p => p.innerText.trim())
                .filter(text => text.length > 30 && !this.isNavigationText(text));
            
            // Take the longest continuous section of paragraphs
            if (paragraphTexts.length > 0) {
                articleText = paragraphTexts.slice(0, Math.min(10, paragraphTexts.length)).join(' ');
            }
        }
        
        // Extract headline
        const headlineSelectors = ['h1', '.headline', '.title', '.article-title', '.post-title'];
        let headline = '';
        
        for (const selector of headlineSelectors) {
            const element = document.querySelector(selector);
            if (element && element.innerText.trim().length > 5) {
                headline = element.innerText.trim();
                break;
            }
        }
        
        return {
            headline: headline,
            content: articleText,
            url: window.location.href,
            title: document.title
        };
    }
    
    isNavigationText(text) {
        const navKeywords = [
            'subscribe', 'login', 'sign up', 'menu', 'navigation', 'footer',
            'copyright', 'privacy policy', 'terms of service', 'cookie',
            'advertisement', 'sponsored', 'follow us', 'social media'
        ];
        
        const lowerText = text.toLowerCase();
        return navKeywords.some(keyword => lowerText.includes(keyword));
    }
    
    analyzeArticle() {
        if (this.analyzed) return;
        
        const articleData = this.extractArticleText();
        
        // Only analyze if we have substantial content
        if (!articleData.content || articleData.content.length < 100) {
            console.log('Fake News Detector: Not enough content to analyze');
            return;
        }
        
        console.log('Fake News Detector: Analyzing article...', {
            headline: articleData.headline,
            contentLength: articleData.content.length
        });
        
        // Send to background script for analysis
        chrome.runtime.sendMessage({
            action: 'analyzeText',
            data: articleData
        }, (response) => {
            if (response && response.analysis) {
                this.displayResults(response.analysis);
                this.analyzed = true;
            }
        });
    }
    
    displayResults(analysis) {
        // Remove existing banner
        const existingBanner = document.getElementById('fake-news-detector-banner');
        if (existingBanner) {
            existingBanner.remove();
        }
        
        // Only show banner for high-confidence fake news predictions
        if (analysis.isFake && analysis.confidence > 0.7) {
            this.showWarningBanner(analysis);
        }
    }
    
    showWarningBanner(analysis) {
        const banner = document.createElement('div');
        banner.id = 'fake-news-detector-banner';
        banner.innerHTML = `
            <div style="
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: linear-gradient(135deg, #ff6b6b, #ee5a52);
                color: white;
                padding: 12px 20px;
                text-align: center;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
                font-weight: 500;
                z-index: 999999;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-bottom: 3px solid #d63031;
            ">
                <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                    <span style="font-size: 18px;">⚠️</span>
                    <span>This article may contain misleading or false information</span>
                    <span style="
                        background: rgba(255,255,255,0.2);
                        padding: 4px 8px;
                        border-radius: 12px;
                        font-size: 12px;
                    ">
                        ${Math.round(analysis.confidence * 100)}% confidence
                    </span>
                    <button onclick="this.parentElement.parentElement.parentElement.remove()" style="
                        background: none;
                        border: none;
                        color: white;
                        font-size: 16px;
                        cursor: pointer;
                        padding: 0 5px;
                        margin-left: 10px;
                    ">✕</button>
                </div>
            </div>
        `;
        
        document.body.insertBefore(banner, document.body.firstChild);
        
        // Adjust page content to account for banner
        document.body.style.paddingTop = '60px';
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            if (banner.parentNode) {
                banner.remove();
                document.body.style.paddingTop = '';
            }
        }, 10000);
    }
}

// Initialize the article extractor
new ArticleExtractor();