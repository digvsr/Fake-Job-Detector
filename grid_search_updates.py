import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import re
import pickle
from textblob import TextBlob

# Load + Clean
df = pd.read_csv('C:/Users/digvi/Fake-Job-Detector/data/fake_job_postings/fake_job_postings.csv')
df.drop(columns=['department', 'salary_range', 'job_id'], inplace=True, errors='ignore')
df.fillna({
    'location': 'unknown', 'industry': 'unknown', 'function': 'unknown',
    'company_profile': 'Not Provided', 'requirements': 'Not Provided',
    'benefits': 'Not Provided', 'required_experience': 'No Experience Required',
    'required_education': 'Not Specified', 'employment_type': 'Not Specified'
}, inplace=True)
df.dropna(subset=['description', 'title'], inplace=True)
df = df.reset_index(drop=True)

# Features
exp_order = {'Not Specified': 0, 'Entry level': 1, 'Mid-Senior level': 2, 'Director': 3, 'Executive': 4}
df['required_experience'] = df['required_experience'].map(exp_order).fillna(0)
df = pd.get_dummies(df, columns=['employment_type', 'industry', 'required_education'], drop_first=True)

vectorizer = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['description'].fillna(''))
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1).reset_index(drop=True)
df.index = range(len(df))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

df['money_mentions'] = df['description'].apply(
    lambda x: len(re.findall(r'\$|usd|pay|salary|cash|earn|quick|unlimited|rich|registrations|investment|bank', str(x).lower()))
)
df['urgency'] = df['description'].apply(
    lambda x: len(re.findall(r'urgent|now|immediate|asap|today|fast|quick|hiring|telegram', str(x).lower()))
)
df['email_in_desc'] = df['description'].str.contains(r'@[\w\.-]+', regex=True, na=False).astype(int)
df['desc_length'] = df['description'].str.len().fillna(0)
df['has_logo'] = df['has_company_logo'].fillna(0).astype(int)
df['has_questions'] = df['has_questions'].astype(int)
df['has_profile'] = df['company_profile'].notna().astype(int)
df['profile_length'] = df['company_profile'].str.len().fillna(0)
df['location'] = df['location'].fillna('unknown')
df = df.loc[:, ~df.columns.duplicated()]
df = df.reset_index(drop=True)
df['location_and_logo'] = ((df['location'] != 'unknown') & (df['has_company_logo'] == 1)).astype(int)

# Existing scam-busting features
df['unexpected_contact'] = df['email_in_desc']
df['urgency_pressure'] = df['description'].apply(
    lambda x: len(re.findall(r'now|immediate|asap', str(x).lower()))
)
df['too_good_to_be_true'] = ((df['required_experience'] <= 1) & (df['money_mentions'] > 2)).astype(int)
df['data_entry_flag'] = df['description'].str.contains(r'data\s*entry', regex=True, na=False).astype(int) | \
                        df['title'].str.contains(r'data\s*entry', regex=True, na=False).astype(int)
df['vague_company'] = ((df['company_profile'] == 'Not Provided') & (df['has_logo'] == 0)).astype(int)
df['casual_language'] = df['description'].apply(
    lambda x: len(re.findall(r'telegram|asap|cool|quick|yo|easy|chill|free|gifts|urgent|rich|limited|daily', str(x).lower()))
)

df['profile_ratio'] = df['company_profile'].str.len().fillna(0) / df['company_profile'].str.len().max()
fraud_words = ['urgent', 'cash', 'immediate', 'asap', 'earn']
fraud_cols = [col for col in tfidf_df.columns if any(word in col for word in fraud_words)]
df['tfidf_fraud_score'] = tfidf_df[fraud_cols].sum(axis=1)
dummy_cols = [col for col in df.columns if col.startswith(('employment_type_', 'industry_', 'required_education_'))]
df['dummy_density'] = df[dummy_cols].mean(axis=1)

# Updated features with new keywords from main.py
df['suspicious_keywords'] = df['description'].apply(
    lambda x: len(re.findall(r'scam|fake|fraud|phish|urgent|immediate|asap|start today|limited spots|quick money|easy money|guaranteed|no risk|free training|hot|abroad|large|housing provided|stay provided|fare|student loan|loan|credit cards|any role|feel free|government', str(x).lower()))
) + df['title'].apply(
    lambda x: len(re.findall(r'scam|fake|fraud|phish|urgent|immediate|asap|start today|limited spots|quick money|easy money|guaranteed|no risk|free training|hot|abroad|large|housing provided|stay provided|fare|student loan|loan|credit cards|any role|feel free|government', str(x).lower()))
)
df['link_in_desc'] = df['description'].str.contains(r'http[s]?://|www\.|\.co|\.link', regex=True, na=False).astype(int)
df['unprofessional_email'] = df['description'].str.contains(r'@gmail\.com|@yahoo\.com|@hotmail\.com', regex=True, na=False).astype(int)
df['pay_to_work'] = df['description'].apply(
    lambda x: len(re.findall(r'fee|pay|payment|deposit|background check cost|training cost|fare|loans', str(x).lower()))
).astype(int)
avg_salaries = {'data entry': 30000, 'customer service': 35000, 'executive assistant': 45000}
df['salary_outlier'] = 0
for idx, row in df.iterrows():
    title = str(row['title']).lower()
    desc = str(row['description']).lower()
    salary_match = re.search(r'\$(\d+,\d+|\d+)(?:/year|/month|/hour)', desc)
    if salary_match:
        salary = int(salary_match.group(1).replace(',', ''))
        if '/month' in desc:
            salary *= 12
        elif '/hour' in desc:
            salary *= 2000
        for role, avg in avg_salaries.items():
            if role in title or role in desc:
                df.at[idx, 'salary_outlier'] = 1 if salary > avg * 1.5 else 0
                break
df['title_vagueness'] = df['title'].apply(
    lambda x: len(re.findall(r'work from home|earn money|data entry|easy job|quick cash|only students|only housewives|only on mobile', str(x).lower()))
)
df['excessive_caps'] = df['description'].apply(
    lambda x: len(re.findall(r'\b[A-Z]{3,}\b', str(x)))
) + df['title'].apply(
    lambda x: len(re.findall(r'\b[A-Z]{3,}\b', str(x)))
)
df['scam_phrase_combo'] = (
    (df['description'].str.contains(r'no experience|fun|play', regex=True, na=False) & 
     df['description'].str.contains(r'high pay|\$\d+|USD', regex=True, na=False)) |
    (df['description'].str.contains(r'work from home', regex=True, na=False) & 
     df['description'].str.contains(r'start today|immediate|with in a year', regex=True, na=False))
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

# New features (without spaCy)
df['sentiment_score'] = df['description'].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)
df['keyword_density'] = df['suspicious_keywords'] / (df['desc_length'] + 1)
df['suspicious_urgency_interaction'] = df['suspicious_keywords'] * df['urgency']
df['money_urgency_interaction'] = df['money_mentions'] * df['urgency']

# X and y
drop_cols = ['fraudulent', 'title', 'description', 'company_profile', 'requirements', 
             'benefits', 'location', 'function']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['fraudulent']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Grid Search for scale_pos_weight and threshold
scale_pos_weights = list(range(30, 41, 2))  # 30 to 40, step 2
thresholds = [round(x, 2) for x in np.arange(0.10, 0.21, 0.01)]  # 0.10 to 0.20, step 0.01
results = []

for scale_pos_weight in scale_pos_weights:
    print(f"\nTraining with Scale Pos Weight: {scale_pos_weight}")
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=8,
        learning_rate=0.03,
        n_estimators=200,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    pickle.dump(model, open('jobguard_model.pkl', 'wb'))
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        results.append({
            'Scale Pos Weight': scale_pos_weight,
            'Threshold': threshold,
            'Recall': round(recall, 3),
            'Precision': round(precision, 3),
            'Accuracy': round(accuracy, 3),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        })

# Create a DataFrame for the results
results_df = pd.DataFrame(results)

# Save the full results to a CSV
results_df.to_csv('model_performance_grid_updated_keywords.csv', index=False)
print("\nFull results saved to 'model_performance_grid_updated_keywords.csv'")

# Filter for combos that hit the new goals
target_recall = 0.94
target_precision = 0.70
target_accuracy = 0.98

# Exact matches
exact_matches = results_df[
    (results_df['Recall'] >= target_recall) &
    (results_df['Precision'] >= target_precision) &
    (results_df['Accuracy'] >= target_accuracy)
]

print("\nExact Matches (Recall >= 0.94, Precision >= 0.70, Accuracy >= 0.98):")
if not exact_matches.empty:
    print(exact_matches[['Scale Pos Weight', 'Threshold', 'Recall', 'Precision', 'Accuracy', 'TP', 'FP', 'FN', 'TN']])
else:
    print("No exact matches found.")

# Find the closest combo if no exact matches
if exact_matches.empty:
    target_precision = 0.70
    target_accuracy = 0.98
    results_df['Distance'] = (
        (results_df['Recall'] - target_recall) ** 2 +
        (results_df['Precision'] - target_precision) ** 2 +
        (results_df['Accuracy'] - target_accuracy) ** 2
    ) ** 0.5
    closest_matches = results_df.sort_values(by='Distance').head(5)
    print("\nTop 5 Closest Matches (Distance to Target: Recall >= 0.94, Precision 0.70, Accuracy 0.98):")
    print(closest_matches[['Scale Pos Weight', 'Threshold', 'Recall', 'Precision', 'Accuracy', 'TP', 'FP', 'FN', 'TN', 'Distance']])