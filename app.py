import os
import pickle
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify
import fake_news_detector

app = Flask(__name__)

# Model loading state
vectorizer = None
models = {}

def init_models():
    """Loads all ML models into memory if trained."""
    global vectorizer, models
    
    # Check if models are trained, else force training once
    if not os.path.exists(fake_news_detector.LR_MODEL_PATH):
        print("[INFO] Serialized models not found in Web UI. Initiating training...")
        fake_news_detector.train_and_save_models()
        
    try:
        with open(fake_news_detector.VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(fake_news_detector.LR_MODEL_PATH, 'rb') as f:
            models['Logistic Regression'] = pickle.load(f)
        with open(fake_news_detector.DT_MODEL_PATH, 'rb') as f:
            models['Decision Tree'] = pickle.load(f)
        with open(fake_news_detector.GB_MODEL_PATH, 'rb') as f:
            models['Gradient Boosting'] = pickle.load(f)
        with open(fake_news_detector.RF_MODEL_PATH, 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        print("[SUCCESS] All serialized models successfully loaded into Flask Web UI server memory.")
    except Exception as e:
        print(f"[ERROR] Failed to load serialized models: {e}")

def search_live_google_news(clean_text):
    """
    Formulates a search query from clean article text and queries
    the Google News RSS feed for real-time coverage.
    """
    words = clean_text.split()
    if not words:
        return []
        
    # Extract the first 7-9 words as the main query phrase for high precision
    query_phrase = " ".join(words[:8])
    
    # URL encode query
    encoded_query = urllib.parse.quote(query_phrase)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        r = requests.get(rss_url, headers=headers, timeout=3.5) # Short timeout to keep prediction fast
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            items = []
            
            # Extract top 4 matching articles
            for item in root.findall('.//item')[:4]:
                title = item.find('title').text
                link = item.find('link').text
                pub_date = item.find('pubDate').text
                
                # Try to get news source name
                source = "Verified News Outlet"
                source_elem = item.find('source')
                if source_elem is not None and source_elem.text:
                    source = source_elem.text
                elif " - " in title:
                    parts = title.split(" - ")
                    if len(parts) > 1:
                        source = parts[-1]
                        title = " - ".join(parts[:-1]) # Clean publisher name out of title
                
                items.append({
                    'title': title,
                    'link': link,
                    'pub_date': pub_date,
                    'source': source
                })
            return items
    except Exception as e:
        print(f"[WARNING] Live RSS Google News search failed: {e}")
        
    return []

@app.route('/')
def home():
    """Renders the main glassmorphic dashboard."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts text input, runs it through all four classifiers,
    cross-references Google News RSS, and computes TF-IDF weights.
    """
    global vectorizer, models
    
    if not vectorizer or not models:
        return jsonify({'error': 'Models are not loaded on server. Please retrain.'}), 500
        
    data = request.get_json()
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({'error': 'No input text provided.'}), 400
        
    news_text = data['text'].strip()
    
    # Preprocess the text
    cleaned_text = fake_news_detector.wordopt(news_text)
    if not cleaned_text.strip():
        return jsonify({
            'error': 'Text is empty after preprocessing (contains only numeric or punctuation chars).'
        }), 400
        
    # Vectorize the text
    xv = vectorizer.transform([cleaned_text])
    
    # Store predictions
    predictions = {}
    fake_count = 0
    
    # Predict with all models
    for model_name, clf in models.items():
        pred = int(clf.predict(xv)[0])
        prob = 0.5
        try:
            prob_arr = clf.predict_proba(xv)[0]
            prob = float(prob_arr[pred])
        except Exception:
            pass
            
        predictions[model_name] = {
            'label': 'Fake News' if pred == 0 else 'True News',
            'is_fake': pred == 0,
            'confidence': round(prob * 100, 2)
        }
        if pred == 0:
            fake_count += 1
            
    # Consensus calculation
    fake_percentage = round((fake_count / len(models)) * 100, 2)
    
    # Perform Live Google News Cross-Referencing Search
    print(f"[INFO] Querying Google News RSS for live coverage...")
    live_coverage = search_live_google_news(cleaned_text)
    live_coverage_found = len(live_coverage) > 0
    
    # Dynamically synthesize final verdict
    # If fake according to ML, but we found matching real-time coverage on Google News,
    # we override the verdict to True/Real and set fake_percentage to 0 to prevent confusion!
    original_ml_percentage = fake_percentage
    if live_coverage_found:
        consensus_label = "True News"
        consensus_confidence = 100.0
        verdict_str = "VERIFIED REAL (LIVE NEWS MATCH)"
        fake_percentage = 0.0
    else:
        consensus_label = "Fake News" if fake_percentage >= 50 else "True News"
        consensus_confidence = fake_percentage if fake_percentage >= 50 else (100 - fake_percentage)
        verdict_str = "UNRELIABLE / FABRICATED" if fake_percentage >= 50 else "VERIFIED / REAL"
    
    # Extract significant words using TF-IDF weights
    feature_names = np.array(vectorizer.get_feature_names_out())
    row, cols = xv.nonzero()
    tfidf_scores = xv.data
    
    important_words = []
    if len(cols) > 0:
        sorted_indices = np.argsort(tfidf_scores)[::-1]
        for idx in sorted_indices[:8]:
            word_idx = cols[idx]
            word = feature_names[word_idx]
            score = float(tfidf_scores[idx])
            important_words.append({
                'word': word,
                'weight': round(score * 100, 2)
            })
            
    return jsonify({
        'consensus': {
            'label': consensus_label,
            'fake_percentage': fake_percentage,
            'confidence': consensus_confidence,
            'verdict': verdict_str,
            'live_coverage_found': live_coverage_found,
            'original_ml_percentage': original_ml_percentage
        },
        'models': predictions,
        'important_words': important_words,
        'live_coverage': live_coverage,
        'metadata': {
            'character_count': len(news_text),
            'word_count': len(news_text.split()),
            'cleaned_word_count': len(cleaned_text.split())
        }
    })

if __name__ == '__main__':
    # Initialize models before server starts
    init_models()
    # Run server locally on standard port, disabling reloader to ensure Windows stability
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
