import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_comprehensive_fake_news_model():
    """Create a comprehensive model that detects various types of fake news"""
    
    # Known satirical/parody websites
    satirical_domains = [
        'theonion.com', 'babylonbee.com', 'clickhole.com', 'reductress.com',
        'thebeaverton.com', 'newsthump.com', 'waterfordwhispersnews.com',
        'thedailymash.co.uk', 'satiretimes.com', 'worldnewsdailyreport.com',
        'empirenews.net', 'newsbiscuit.com', 'rockcitytimes.com',
        'thespoof.com', 'gomerblog.com', 'duffelblog.com'
    ]
    
    # Conspiracy/misinformation sites (known problematic domains)
    conspiracy_domains = [
        'infowars.com', 'naturalnews.com', 'beforeitsnews.com',
        'worldtruth.tv', 'activistpost.com', 'veteranstoday.com',
        'globalresearch.ca', 'zerohedge.com', 'principia-scientific.org'
    ]
    
    # Strong fake indicators (satirical language patterns)
    satirical_indicators = [
        # Absurd/satirical language
        'fucking over', 'shitshow', 'clusterfuck', 'assures', 'admits to being',
        'confesses', 'reveals he', 'announces plan to', 'promises to continue',
        'vows to keep', 'commits to screwing', 'pledges to destroy',
        
        # The Onion style patterns
        'area man', 'local man', 'nation\'s', 'american people', 'sources confirm',
        'reports indicate', 'breaking: area', 'study finds', 'experts say',
        'scientists baffled', 'doctors shocked',
        
        # Satirical political language
        'assures wall street', 'promises donors', 'tells billionaires',
        'reassures corporate', 'guarantees wealthy', 'commits to rich',
        
        # Absurd scenarios
        'reveals plan to', 'announces initiative to', 'proposes new way to',
        'unveils strategy to', 'introduces program to', 'launches campaign to'
    ]
    
    # Fake news indicators (conspiracy/misinformation)
    fake_indicators = [
        # Conspiracy language
        'deep state', 'globalist agenda', 'new world order', 'false flag',
        'wake up sheeple', 'mainstream media lies', 'they don\'t want you to know',
        'cover up', 'conspiracy', 'hidden truth', 'secret agenda',
        
        # Medical misinformation
        'miracle cure', 'doctors hate this', 'big pharma', 'natural remedy',
        'government hiding', 'suppressed treatment', 'banned by fda',
        'ancient secret', 'pharmaceutical companies', 'medical establishment',
        
        # Sensational claims
        'shocking truth', 'explosive revelation', 'bombshell report',
        'you won\'t believe', 'must see', 'this changes everything',
        'scientists stunned', 'experts baffled', 'incredible discovery',
        
        # Anti-establishment
        'mainstream media', 'fake news media', 'corrupt politicians',
        'government lies', 'official story', 'what they\'re hiding',
        'establishment doesn\'t want', 'powers that be', 'shadow government',
        
        # Clickbait patterns
        'number will shock you', 'what happened next', 'you won\'t believe what',
        'doctors are speechless', 'this simple trick', 'one weird trick',
        'industry doesn\'t want', 'they tried to silence'
    ]
    
    # Real news indicators
    real_indicators = [
        # Professional journalism
        'according to', 'sources say', 'reported by', 'confirmed by',
        'statement from', 'press release', 'official announcement',
        'spokesperson said', 'in a statement', 'told reporters',
        
        # Measured language
        'approximately', 'estimated', 'preliminary', 'initial reports',
        'ongoing investigation', 'developing story', 'more details',
        'expected to', 'scheduled to', 'planned to',
        
        # Institution references
        'university study', 'published in', 'peer reviewed',
        'research shows', 'data indicates', 'analysis reveals',
        'survey found', 'poll shows', 'statistics show',
        
        # Formal attribution
        'department of', 'ministry of', 'office of', 'agency said',
        'committee announced', 'board decided', 'council voted',
        'court ruled', 'judge said', 'jury found'
    ]
    
    # Create comprehensive model
    comprehensive_model = {
        "type": "comprehensive_detector",
        "version": "2.0",
        "domain_checks": {
            "satirical_domains": satirical_domains,
            "conspiracy_domains": conspiracy_domains,
            "domain_weight": 0.8  # Strong weight for domain-based detection
        },
        "content_analysis": {
            "satirical_indicators": satirical_indicators,
            "fake_indicators": fake_indicators,
            "real_indicators": real_indicators,
            "weights": {
                "satirical_weight": 0.7,
                "fake_weight": 0.5,
                "real_weight": -0.4
            }
        },
        "pattern_detection": {
            "profanity_in_headlines": 0.6,  # Headlines with swearing often satirical
            "absurd_combinations": 0.5,     # Detecting absurd word combinations
            "clickbait_numbers": 0.4,       # "This one trick", "7 secrets", etc.
            "all_caps_excessive": 0.3       # Excessive capitalization
        },
        "thresholds": {
            "satirical_threshold": 0.7,     # High confidence for satire
            "fake_threshold": 0.6,          # Medium-high for misinformation
            "real_threshold": 0.4           # Conservative for real news
        }
    }
    
    # Save the model
    output_dir = '../chrome_extension/models'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/comprehensive_model.json', 'w') as f:
        json.dump(comprehensive_model, f, indent=2)
    
    logger.info("Comprehensive fake news detection model created!")
    logger.info(f"- {len(satirical_domains)} satirical domains")
    logger.info(f"- {len(conspiracy_domains)} conspiracy domains") 
    logger.info(f"- {len(satirical_indicators)} satirical patterns")
    logger.info(f"- {len(fake_indicators)} fake news patterns")
    logger.info(f"- {len(real_indicators)} real news patterns")
    
    return True

if __name__ == "__main__":
    create_comprehensive_fake_news_model()