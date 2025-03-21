import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import re
import pickle
from flask import Flask, request, render_template
from textblob import TextBlob
import os

app = Flask(__name__)

# Load the model and vectorizer
model = xgb.Booster()
model.load_model("jobguard_model.json")
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Best parameters from grid search
best_scale_pos_weight = 4
best_threshold = 0.23

# Feature extraction function
def extract_features(job_text):
    data = {'job_text': job_text}
    df = pd.DataFrame([data])
    
    # TF-IDF (using job_text)
    tfidf_matrix = vectorizer.transform(df['job_text'].fillna(''))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Reset indices to ensure alignment
    df = df.reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)
    
    # Concatenate while ensuring indices align
    df = pd.concat([df, tfidf_df], axis=1)
    
    # Compute all features as a dictionary to avoid fragmentation
    features = {}
    
    features['money_mentions'] = df['job_text'].apply(
        lambda x: len(re.findall(r'\$|usd|pay|salary|cash|earn|quick|unlimited|rich|registrations|investment|bank', str(x).lower()))
    )
    features['urgency'] = df['job_text'].apply(
        lambda x: len(re.findall(r'urgent|now|immediate|asap|today|fast|quick|hiring|telegram', str(x).lower()))
    )
    features['email_in_desc'] = df['job_text'].str.contains(r'@[\w\.-]+', regex=True, na=False).astype(int)
    features['desc_length'] = df['job_text'].str.len().fillna(0)
    
    features['has_logo'] = pd.Series(0, index=df.index)
    features['has_questions'] = pd.Series(0, index=df.index)
    features['has_profile'] = pd.Series(1 if job_text else 0, index=df.index)
    features['profile_length'] = pd.Series(0, index=df.index)
    features['location_and_logo'] = pd.Series(0, index=df.index)
    
    features['unexpected_contact'] = features['email_in_desc']
    features['urgency_pressure'] = df['job_text'].apply(
        lambda x: len(re.findall(r'now|immediate|asap', str(x).lower()))
    )
    features['too_good_to_be_true'] = pd.Series(0, index=df.index)
    features['data_entry_flag'] = df['job_text'].str.contains(r'data\s*entry', regex=True, na=False).astype(int)
    features['vague_company'] = pd.Series(0, index=df.index)
    features['casual_language'] = df['job_text'].apply(
        lambda x: len(re.findall(r'telegram|asap|cool|quick|yo|easy|chill|free|gifts|urgent|rich|limited|daily', str(x).lower()))
    )
    features['profile_ratio'] = pd.Series(0, index=df.index)
    
    fraud_words = ['urgent', 'cash', 'immediate', 'asap', 'earn']
    fraud_cols = [col for col in tfidf_df.columns if any(word in col for word in fraud_words)]
    features['tfidf_fraud_score'] = tfidf_df[fraud_cols].sum(axis=1)
    
    dummy_cols = [col for col in model.feature_names_in_ if col.startswith(('employment_type_', 'industry_', 'required_education_'))]
    for col in dummy_cols:
        features[col] = pd.Series(0, index=df.index)
    features['dummy_density'] = pd.Series(0, index=df.index)
    
    job_text_keywords = df['job_text'].apply(
        lambda x: len(re.findall(r'scam|fake|fraud|phish|urgent|immediate|asap|start today|limited spots|quick money|easy money|guaranteed|no risk|free training', str(x).lower()))
    )
    features['suspicious_keywords'] = job_text_keywords
    
    features['link_in_desc'] = df['job_text'].str.contains(r'http[s]?://|www\.|\.co|\.link', regex=True, na=False).astype(int)
    features['unprofessional_email'] = df['job_text'].str.contains(r'@gmail\.com|@yahoo\.com|@hotmail\.com', regex=True, na=False).astype(int)
    features['pay_to_work'] = df['job_text'].apply(
        lambda x: len(re.findall(r'fee|pay|payment|deposit|background check cost|training cost', str(x).lower()))
    ).astype(int)
    
    features['salary_outlier'] = pd.Series(0, index=df.index)
    avg_salaries = {'data entry': 30000, 'customer service': 35000, 'executive assistant': 45000}
    for idx, text in df['job_text'].items():
        text = str(text).lower()
        salary_match = re.search(r'\$(\d+,\d+|\d+)(?:/year|/month|/hour)', text)
        if salary_match:
            salary = int(salary_match.group(1).replace(',', ''))
            if '/month' in text:
                salary *= 12
            elif '/hour' in text:
                salary *= 2000
            for role, avg in avg_salaries.items():
                if role in text:
                    features['salary_outlier'].iloc[idx] = 1 if salary > avg * 1.5 else 0
                    break
    
    features['title_vagueness'] = df['job_text'].apply(
        lambda x: len(re.findall(r'work from home|earn money|data entry|easy job|quick cash', str(x).lower()))
    )
    
    features['excessive_caps'] = df['job_text'].apply(
        lambda x: len(re.findall(r'\b[A-Z]{3,}\b', str(x)))
    )
    
    features['scam_phrase_combo'] = (
        (df['job_text'].str.contains(r'no experience', regex=True, na=False).fillna(False) & 
         df['job_text'].str.contains(r'high pay|\$\d+', regex=True, na=False).fillna(False)) |
        (df['job_text'].str.contains(r'work from home', regex=True, na=False).fillna(False) & 
         df['job_text'].str.contains(r'start today|immediate', regex=True, na=False).fillna(False))
    ).astype(int)
    
    features['title_desc_mismatch'] = pd.Series(0, index=df.index)
    
    features['exclamation_count'] = df['job_text'].apply(
        lambda x: str(x).count('!')
    )
    
    features['required_experience'] = pd.Series(0, index=df.index)
    features['sentiment_score'] = df['job_text'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    features['keyword_density'] = features['suspicious_keywords'] / (features['desc_length'] + 1)
    features['suspicious_urgency_interaction'] = features['suspicious_keywords'] * features['urgency']
    features['money_urgency_interaction'] = features['money_mentions'] * features['urgency']
    
    features_df = pd.DataFrame(features)
    df = pd.concat([df, features_df], axis=1)
    
    drop_cols = ['job_text']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else vectorizer.get_feature_names_out().tolist() + [
        'required_experience', 'money_mentions', 'urgency', 'email_in_desc', 'desc_length', 'has_logo',
        'has_questions', 'has_profile', 'profile_length', 'location_and_logo', 'unexpected_contact',
        'urgency_pressure', 'too_good_to_be_true', 'data_entry_flag', 'vague_company', 'casual_language',
        'profile_ratio', 'tfidf_fraud_score', 'dummy_density', 'suspicious_keywords', 'link_in_desc',
        'unprofessional_email', 'pay_to_work', 'salary_outlier', 'title_vagueness', 'excessive_caps',
        'scam_phrase_combo', 'title_desc_mismatch', 'exclamation_count', 'sentiment_score', 'keyword_density',
        'suspicious_urgency_interaction', 'money_urgency_interaction'
    ]
    X = X.reindex(columns=expected_features, fill_value=0)
    
    return X

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        job_text = request.form['job_text']
        
        # Extract features
        X = extract_features(job_text)
        
        # Predict
        proba = model.predict_proba(X)[:, 1][0]
        prediction = 'Fake' if proba >= best_threshold else 'Legit'
        confidence = round(proba * 100, 2) if prediction == 'Fake' else round((1 - proba) * 100, 2)
        
        # Show the feedback form after prediction
        return render_template('index.html', prediction=prediction, confidence=confidence, job_text=job_text, show_feedback=True)
    return render_template('index.html', prediction=None, confidence=None, show_feedback=False)

@app.route('/feedback', methods=['POST'])
def feedback():
    job_text = request.form['job_text']
    prediction = request.form['prediction']
    confidence = float(request.form['confidence'])
    feedback = request.form['feedback']
    
    # Determine the correct label
    if feedback == 'correct':
        correct_label = prediction  # If prediction was correct, use the predicted label
    else:
        correct_label = request.form['correct_label']  # If incorrect, use the user-provided label
    
    # Prepare feedback data
    feedback_data = {
        'job_text': job_text,
        'predicted_label': prediction,
        'confidence': confidence,
        'correct_label': correct_label,
        'feedback': feedback
    }
    
    # Save feedback to CSV
    feedback_file = 'feedback_data.csv'
    feedback_df = pd.DataFrame([feedback_data])
    
    # Check if file exists, and append or create accordingly
    if os.path.exists(feedback_file):
        feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        feedback_df.to_csv(feedback_file, mode='w', header=True, index=False)
    
    # Re-render the page with the same prediction and a feedback message
    return render_template('index.html', prediction=prediction, confidence=confidence, job_text=job_text, show_feedback=True, feedback_message="Thank you for your feedback!")

if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', 5000))  # Use PORT env var, default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=False)