JobGuard AI - Fake Job Detector

JobGuard AI is a machine learning-based web application that detects fake job postings. It uses an XGBoost classifier with engineered features to identify scam job listings, achieving 82.7% recall, 71.1% precision, and 97.5% accuracy.

Features
- Detects fake job postings with high accuracy.
- Web interface built with Flask for real-time predictions.
- Engineered 678+ features, including TF-IDF, sentiment analysis, and keyword density.

## Tech Stack
- Python
- XGBoost
- Flask
- HTML/CSS
- Pandas, Scikit-learn, TextBlob

## Installation
1. Clone the repository: 
2. Install dependencies: pip install -r requirements.txt
3. Run the app: python main.py
4. Open `http://127.0.0.1:5000` in your browser.

## Note
The model files (`jobguard_model.pkl`, `tfidf_vectorizer.pkl`) and dataset are not included due to size constraints. To train the model, use the provided grid search script.

## Metrics
- Recall: 82.7% (143/173 frauds detected)
- Precision: 71.1%
- Accuracy: 97.5%