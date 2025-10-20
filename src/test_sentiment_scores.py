#!/usr/bin/env python3
"""
Example script showing how the updated sentiment analysis preserves all three scores.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sentiment.roberta_sentiment import setup_roberta_sentiment, analyze_sentiment_batch

def test_sentiment_scores():
    """Test the sentiment analysis to show all three scores are preserved."""
    
    # Sample texts in Portuguese
    test_texts = [
        "Eu estou muito feliz hoje!",           # Should be positive
        "Estou muito triste e deprimido.",      # Should be negative  
        "O tempo está normal hoje.",            # Should be neutral
        "Adorei este filme, é incrível!",       # Should be positive
        "Odeio quando isso acontece."           # Should be negative
    ]
    
    print("🔍 Testing sentiment analysis with all scores...")
    print("="*60)
    
    # Setup sentiment pipeline
    try:
        sentiment_pipeline = setup_roberta_sentiment()
        
        # Analyze sentiment
        results = analyze_sentiment_batch(test_texts, sentiment_pipeline, batch_size=5)
        
        # Display results
        for i, (text, result) in enumerate(zip(test_texts, results)):
            print(f"\n📝 Text {i+1}: '{text}'")
            
            if isinstance(result, list):  # Multiple scores returned
                print("   All Sentiment Scores:")
                for score_info in result:
                    label = score_info['label']
                    score = score_info['score']
                    print(f"   - {label}: {score:.4f}")
                
                # Find highest scoring sentiment
                max_sentiment = max(result, key=lambda x: x['score'])
                print(f"   🎯 Predicted: {max_sentiment['label']} (confidence: {max_sentiment['score']:.4f})")
                
            else:  # Single score returned (fallback)
                print(f"   🎯 Predicted: {result['label']} (confidence: {result['score']:.4f})")
                print("   ⚠️  Only single score available (not all scores returned)")
    
        print("\n" + "="*60)
        print("✅ Test completed! The updated code now preserves all sentiment scores.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Note: This requires the transformers library to be installed.")
        print("Run: pip install transformers torch")

if __name__ == "__main__":
    test_sentiment_scores()