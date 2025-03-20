import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import re
import pickle
from flask import Flask, request, render_template
from textblob import TextBlob

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('jobguard_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Best parameters from grid search
best_scale_pos_weight = 1
best_threshold = 0.17

# Feature extraction function
def extract_features(title, description):
    data = {'title': title, 'description': description}
    df = pd.DataFrame([data])
    
    # TF-IDF
    tfidf_matrix = vectorizer.transform(df['description'].fillna(''))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    
    # Custom features
    df['money_mentions'] = df['description'].apply(
        lambda x: len(re.findall(r'\$|usd|pay|salary|cash|earn|quick|unlimited|rich|registrations|investment|bank', str(x).lower()))
    )
    df['urgency'] = df['description'].apply(
        lambda x: len(re.findall(r'urgent|now|immediate|asap|today|fast|quick|hiring|telegram', str(x).lower()))
    )
    df['email_in_desc'] = df['description'].str.contains(r'@[\w\.-]+', regex=True, na=False).astype(int)
    df['desc_length'] = df['description'].str.len().fillna(0)
    df['suspicious_keywords'] = df['description'].apply(
        lambda x: len(re.findall(r'scam|fake|fraud|phish|urgent|immediate|asap|start today|limited spots|quick money|easy money|guaranteed|no risk|free training', str(x).lower()))
    ) + df['title'].apply(
        lambda x: len(re.findall(r'scam|fake|fraud|phish|urgent|immediate|asap|start today|limited spots|quick money|easy money|guaranteed|no risk|free training', str(x).lower()))
    )
    df['link_in_desc'] = df['description'].str.contains(r'http[s]?://|www\.|\.co|\.link', regex=True, na=False).astype(int)
    df['unprofessional_email'] = df['description'].str.contains(r'@gmail\.com|@yahoo\.com|@hotmail\.com', regex=True, na=False).astype(int)
    df['pay_to_work'] = df['description'].apply(
        lambda x: len(re.findall(r'fee|pay|payment|deposit|background check cost|training cost', str(x).lower()))
    ).astype(int)
    df['title_vagueness'] = df['title'].apply(
        lambda x: len(re.findall(r'work from home|earn money|data entry|easy job|quick cash', str(x).lower()))
    )
    df['excessive_caps'] = df['description'].apply(
        lambda x: len(re.findall(r'\b[A-Z]{3,}\b', str(x)))
    ) + df['title'].apply(
        lambda x: len(re.findall(r'\b[A-Z]{3,}\b', str(x)))
    )
    df['scam_phrase_combo'] = (
        (df['description'].str.contains(r'no experience', regex=True, na=False) & 
         df['description'].str.contains(r'high pay|\$\d+', regex=True, na=False)) |
        (df['description'].str.contains(r'work from home', regex=True, na=False) & 
         df['description'].str.contains(r'start today|immediate', regex=True, na=False))
    ).astype(int)
    df['title_desc_mismatch'] = 0
    for idx, row in df.iterrows():
        title = str(row['title']).lower()
        desc = str(row['description']).lower()
        if 'data entry' in title and 'data entry' not in desc:
            df.at[idx, 'title_desc_mismatch'] = 1
        elif 'customer service' in title and 'customer service' not in desc:
            df.at[idx, 'title_desc_mismatch'] = 1
        elif 'executive assistant' in title and 'executive assistant' not in desc:
            df.at[idx, 'title_desc_mismatch'] = 1
    df['exclamation_count'] = df['description'].apply(
        lambda x: str(x).count('!')
    ) + df['title'].apply(
        lambda x: str(x).count('!')
    )
    df['sentiment_score'] = df['description'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    df['keyword_density'] = df['suspicious_keywords'] / (df['desc_length'] + 1)
    df['suspicious_urgency_interaction'] = df['suspicious_keywords'] * df['urgency']
    df['money_urgency_interaction'] = df['money_mentions'] * df['urgency']
    
    # Drop non-feature columns
    drop_cols = ['title', 'description']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return X

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        
        # Extract features
        X = extract_features(title, description)
        
        # Predict
        proba = model.predict_proba(X)[:, 1][0]
        prediction = 'Fake' if proba >= best_threshold else 'Legit'
        confidence = round(proba * 100, 2) if prediction == 'Fake' else round((1 - proba) * 100, 2)
        
        return render_template('index.html', prediction=prediction, confidence=confidence)
    return render_template('index.html', prediction=None, confidence=None)

if __name__ == '__main__':
    app.run(debug=True)