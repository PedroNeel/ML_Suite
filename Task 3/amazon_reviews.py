# amazon_reviews.py (with auto-install)
import json
import subprocess
import sys

# Try to import spacy, install if missing
try:
    import spacy
    from spacy.matcher import Matcher
except ImportError:
    print("spaCy not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    import spacy
    from spacy.matcher import Matcher

# Try to load English model, download if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("English model not found. Downloading now...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Sample reviews data
reviews = [
    "The Apple iPhone 13 camera is spectacular but battery drains fast.",
    "Samsung Galaxy Buds: Comfortable fit with average noise cancellation.",
    "Amazon Echo Dot stopped working after 2 months. Disappointed!",
    "Sony WH-1000XM4 headphones have amazing noise cancellation.",
    "Microsoft Surface Pro keyboard stopped functioning after 3 weeks."
]

# Custom NER patterns
matcher = Matcher(nlp.vocab)

# Brand patterns (case-insensitive)
brand_patterns = [
    [{"LOWER": {"IN": ["apple", "iphone"]}}],
    [{"LOWER": "samsung"}],
    [{"LOWER": "amazon"}],
    [{"LOWER": "sony"}],
    [{"LOWER": "microsoft"}]
]

# Product patterns (sequence of proper nouns followed by optional noun)
product_patterns = [
    [{"POS": "PROPN"}, {"POS": "PROPN", "OP": "?"}, {"POS": "NOUN", "OP": "?"}]
]

matcher.add("BRAND", brand_patterns)
matcher.add("PRODUCT", product_patterns)

# Sentiment lexicon with more examples
POSITIVE_WORDS = ["great", "excellent", "spectacular", "amazing", "comfortable", 
                  "good", "awesome", "perfect", "love", "recommend"]
NEGATIVE_WORDS = ["defective", "disappointed", "drains", "stopped", "broken", 
                  "poor", "bad", "terrible", "awful", "fail"]

# Process reviews
results = []
for review in reviews:
    doc = nlp(review)
    matches = matcher(doc)
    
    # Extract entities
    entities = {"brands": set(), "products": set()}
    for match_id, start, end in matches:
        span = doc[start:end]
        if nlp.vocab.strings[match_id] == "BRAND":
            entities["brands"].add(span.text)
        elif nlp.vocab.strings[match_id] == "PRODUCT":
            # Only add if it's not a brand and has at least 2 tokens
            if span.text not in [b for b in entities["brands"]] and len(span) > 1:
                entities["products"].add(span.text)
    
    # Analyze sentiment
    positive_count = sum(1 for token in doc if token.text.lower() in POSITIVE_WORDS)
    negative_count = sum(1 for token in doc if token.text.lower() in NEGATIVE_WORDS)
    
    sentiment_score = positive_count - negative_count
    if sentiment_score > 0:
        sentiment = "positive"
    elif sentiment_score < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    results.append({
        "review": review,
        "brands": list(entities["brands"]),
        "products": list(entities["products"]),
        "sentiment": sentiment,
        "positive_words": positive_count,
        "negative_words": negative_count
    })

# Print results
print("Analysis Results:")
print(json.dumps(results, indent=2))

# Save results to file
with open('amazon_reviews_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to 'amazon_reviews_analysis.json'")