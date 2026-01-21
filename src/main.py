# Customer Purchase Prediction Script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000

# Features
time_on_site = np.random.exponential(5, n_samples)  # exponential distribution
pages_viewed = np.random.poisson(3, n_samples) + 1  # poisson +1 to avoid 0
device = np.random.choice(['mobile', 'desktop', 'tablet'], n_samples)
referral_source = np.random.choice(['google', 'facebook', 'direct', 'email'], n_samples)

# Target: purchase probability depends on features
purchase_prob = 0.1 + 0.01 * time_on_site + 0.05 * pages_viewed
purchase_prob += np.where(device == 'desktop', 0.1, 0)
purchase_prob += np.where(referral_source == 'google', 0.05, 0)
purchase_prob = np.clip(purchase_prob, 0, 1)
purchase = np.random.binomial(1, purchase_prob)

# Create DataFrame
data = pd.DataFrame({
    'time_on_site': time_on_site,
    'pages_viewed': pages_viewed,
    'device': device,
    'referral_source': referral_source,
    'purchase': purchase
})

print("Data generated:")
print(data.head())
print(data.describe())

# Preprocessing
# Encode categorical variables
le_device = LabelEncoder()
le_referral = LabelEncoder()

data['device_encoded'] = le_device.fit_transform(data['device'])
data['referral_encoded'] = le_referral.fit_transform(data['referral_source'])

# Features and target
X = data[['time_on_site', 'pages_viewed', 'device_encoded', 'referral_encoded']]
y = data['purchase']

# Scale numerical features
scaler = StandardScaler()
X[['time_on_site', 'pages_viewed']] = scaler.fit_transform(X[['time_on_site', 'pages_viewed']])

print("\nPreprocessed features:")
print(X.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

print("\nModels trained.")

# Predictions
y_pred_log = log_reg.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Metrics
acc_log = accuracy_score(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)

print(f"\nLogistic Regression - Accuracy: {acc_log:.2f}, Precision: {prec_log:.2f}")
print(f"Random Forest - Accuracy: {acc_rf:.2f}, Precision: {prec_rf:.2f}")

print("\nProject completed successfully!")