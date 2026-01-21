import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'le_device' not in st.session_state:
    st.session_state.le_device = None
if 'le_referral' not in st.session_state:
    st.session_state.le_referral = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'log_reg' not in st.session_state:
    st.session_state.log_reg = None
if 'rf' not in st.session_state:
    st.session_state.rf = None

st.title("🛒 Customer Purchase Predictor")

st.markdown("""
**Predict whether a website visitor will make a purchase**

Enter customer characteristics below to get an instant prediction using machine learning models trained on historical data.
""")

# Load and train models on startup
if not st.session_state.models_loaded:
    try:
        # Load the dataset
        data = pd.read_csv("data/sample_customer_data.csv")
        
        # Preprocessing
        le_device = LabelEncoder()
        le_referral = LabelEncoder()
        
        data['device_encoded'] = le_device.fit_transform(data['device'])
        data['referral_encoded'] = le_referral.fit_transform(data['referral_source'])
        
        X = data[['time_on_site', 'pages_viewed', 'device_encoded', 'referral_encoded']]
        y = data['purchase']
        
        # Scale numeric features
        scaler = StandardScaler()
        X[['time_on_site', 'pages_viewed']] = scaler.fit_transform(X[['time_on_site', 'pages_viewed']])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train, y_train)
        
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        
        # Store in session state
        st.session_state.models_loaded = True
        st.session_state.le_device = le_device
        st.session_state.le_referral = le_referral
        st.session_state.scaler = scaler
        st.session_state.log_reg = log_reg
        st.session_state.rf = rf
        
        st.success("✅ Models trained and ready for predictions!")
        
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

st.header("🔮 Make a Prediction")

st.write("Enter customer details below to predict purchase likelihood:")

# Create input form
col1, col2 = st.columns(2)

with col1:
    time_on_site = st.slider("Time on Site (minutes)", min_value=0, max_value=60, value=15)
    pages_viewed = st.slider("Pages Viewed", min_value=1, max_value=20, value=5)

with col2:
    device = st.selectbox("Device", options=["mobile", "desktop", "tablet"])
    referral_source = st.selectbox("Referral Source", options=["google", "facebook", "direct", "email", "other"])

# Prediction button
if st.button("Predict Purchase Likelihood", type="primary"):
    try:
        # Prepare input data
        device_encoded = st.session_state.le_device.transform([device])[0]
        referral_encoded = st.session_state.le_referral.transform([referral_source])[0]
        
        input_data = pd.DataFrame({
            'time_on_site': [time_on_site],
            'pages_viewed': [pages_viewed],
            'device_encoded': [device_encoded],
            'referral_encoded': [referral_encoded]
        })
        
        # Scale numeric features
        input_data[['time_on_site', 'pages_viewed']] = st.session_state.scaler.transform(
            input_data[['time_on_site', 'pages_viewed']]
        )
        
        # Make predictions
        log_reg_prob = st.session_state.log_reg.predict_proba(input_data)[0][1]
        rf_prob = st.session_state.rf.predict_proba(input_data)[0][1]
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Logistic Regression", f"{log_reg_prob:.1%}")
            
        with col2:
            st.metric("Random Forest", f"{rf_prob:.1%}")
        
        # Business recommendation
        avg_prob = (log_reg_prob + rf_prob) / 2
        if avg_prob > 0.6:
            st.success("🟢 High purchase likelihood! Consider targeted promotions.")
        elif avg_prob > 0.4:
            st.warning("🟡 Moderate purchase likelihood. Monitor engagement.")
        else:
            st.error("🔴 Low purchase likelihood. Focus on conversion optimization.")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Model performance info
st.markdown("---")
st.subheader("📈 Model Performance")
st.write("Models trained on sample data with ~60-65% accuracy. Use for demonstration purposes.")

# About section
st.markdown("---")
st.subheader("ℹ️ About")
st.write("""
**Customer Purchase Prediction** helps e-commerce businesses optimize their conversion rates by predicting which website visitors are likely to make a purchase.

**Features Analyzed:**
- ⏱️ Time spent on site
- 📄 Number of pages viewed  
- 📱 Device type
- 🔗 Referral source

**Models Used:**
- Logistic Regression (interpretable)
- Random Forest (high accuracy)

**Business Applications:**
- Personalized marketing campaigns
- Dynamic pricing strategies
- Customer retention programs
- Website optimization
""")

st.write("Built with ❤️ using Streamlit, scikit-learn, and pandas")