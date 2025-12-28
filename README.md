# IMDB Movie Review Sentiment Analysis

## Overview

This project predicts the sentiment of IMDB movie reviews (positive or negative) using machine learning.  
It demonstrates the full NLP pipeline, including:

- Text preprocessing and cleaning
- Bag-of-Words feature extraction
- Model training using XGBoost
- Evaluation via accuracy, F1 score, cross-validation, and feature importance
- Testing on custom reviews

The goal is to combine predictive performance with interpretability, showing which words most influence sentiment.

## Dataset

- **Source:** IMDB movie reviews dataset : https://www.kaggle.com/datasets/mahmoudshaheen1134/imdp-data
- **Features:** Raw text reviews  
- **Target:** Sentiment label (`positive` or `negative`)  
- **Subset:** A smaller subset (50,000 rows) is used for training due to computational cost  

## Workflow

### 1. Text Preprocessing

- Remove HTML tags and non-alphabetic characters  
- Convert to lowercase  
- Remove stopwords (keeping negations)  
- Lemmatize words to their dictionary form  

### 2. Feature Engineering

- Bag-of-Words representation using `CountVectorizer`  
- Maximum features selected through cross-validation (`max_features=1400`)  

### 3. Encoding Labels

- Map `negative` → 0, `positive` → 1  

### 4. Train-Test Split

- 80/20 split using `train_test_split`  

### 5. Model Training

- **Algorithm:** XGBoost Classifier (`XGBClassifier`)  
- Initial hyperparameters set manually, followed by fine-tuning via `GridSearchCV`  

### 6. Hyperparameter Tuning

- **Tree complexity:** `max_depth`, `min_child_weight`  
- **Learning rate & estimators:** `learning_rate`, `n_estimators`  
- **Sampling parameters:** `subsample`, `colsample_bytree`  

### 7. Model Evaluation

- **Metrics:** Accuracy, F1 score  
- **Cross-validation:** 10-fold for generalization  
- **Feature importance:** Identifies most influential words  

### 8. Testing Custom Reviews

- Users can input their own movie review  
- Review is preprocessed the same way as training data  
- Model predicts sentiment with probability scores  

## Results

- Achieves ~85% accuracy on test data  
- Feature importance shows top words driving sentiment polarity  
- The model automatically filters irrelevant words, focusing on key expressions  

## Exporting Model

Models are saved for reuse in applications:

```python
import joblib

joblib.dump(classifier, "sentiment_model.pkl")
joblib.dump(cv, "countvectorizer.pkl")
