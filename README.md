# 🛒 Customer Purchase Prediction

A complete machine learning project that predicts customer purchase likelihood based on website behavior using Logistic Regression and Random Forest models. Features an interactive Streamlit web app for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This project demonstrates a complete machine learning workflow for predicting customer purchases based on their website interaction patterns. It includes data preprocessing, model training, evaluation, and deployment through an interactive web interface.

## ✨ Features

### 📊 **Core Functionality**
- **Real-time Predictions**: Interactive web app with sliders and dropdowns
- **Dual ML Models**: Logistic Regression and Random Forest for comparison
- **Business Insights**: Color-coded recommendations based on purchase likelihood
- **Synthetic Dataset**: 100 customer records with realistic behavior patterns

### 🔍 **Customer Behavior Analysis**
- **Time on Site**: How long customers spend browsing (0-60 minutes)
- **Pages Viewed**: Number of pages visited (1-20 pages)
- **Device Type**: Mobile, desktop, or tablet usage
- **Referral Source**: How customers found the website (Google, Facebook, Direct, Email, Other)

### 🤖 **Machine Learning Pipeline**
- **Data Preprocessing**: Label encoding for categorical variables, feature scaling
- **Model Training**: Automated training on startup with 80/20 train-test split
- **Model Evaluation**: Accuracy and precision metrics (~60-65% performance)
- **Feature Engineering**: Encoded categorical variables and scaled numeric features

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (optional, for cloning)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd customer-purchase-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate environment**
   ```bash
   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Option 1: Interactive Web App (Recommended)
Launch the Streamlit application for instant predictions:

```bash
streamlit run app.py
```

Then open `http://localhost:8502` in your browser.

**Features:**
- Pre-trained models load automatically on startup
- Adjust customer characteristics with interactive controls
- Get real-time purchase probability predictions
- View business recommendations based on likelihood scores

### Option 2: Command Line Script
Run the complete ML pipeline programmatically:

```bash
python src/main.py
```

**Output:**
- Generates synthetic customer data
- Trains both ML models
- Displays model performance metrics
- Shows data statistics and preprocessing results

### Option 3: Jupyter Notebook
Explore the data and models interactively:

```bash
jupyter notebook notebooks/customer_purchase_prediction.ipynb
```

**Includes:**
- Step-by-step data analysis
- Model training walkthrough
- Visualization of results
- Feature importance analysis

## 📁 Project Structure

```
customer-purchase-prediction/
│
├── 📊 data/
│   └── sample_customer_data.csv      # Synthetic training dataset
│
├── 📓 notebooks/
│   └── customer_purchase_prediction.ipynb  # Exploratory analysis
│
├── 🐍 src/
│   └── main.py                       # Command-line ML pipeline
│
├── 🌐 app.py                         # Streamlit web application
├── 📋 requirements.txt               # Python dependencies
├── 📖 README.md                      # Project documentation
└── 📄 LICENSE                        # MIT License
```

## 🧠 Machine Learning Details

### Models Used
- **Logistic Regression**: Interpretable linear model for probability estimation
- **Random Forest**: Ensemble method for higher accuracy through decision trees

### Performance Metrics
- **Accuracy**: ~60-65% on test data
- **Precision**: ~50-60% for positive purchase predictions
- **Training Size**: 80 samples (80% of dataset)
- **Test Size**: 20 samples (20% of dataset)

### Feature Importance (Random Forest)
1. Time on site (most important)
2. Pages viewed
3. Device type
4. Referral source

## 🎨 Web App Interface

### Input Controls
- **Time on Site**: Slider (0-60 minutes)
- **Pages Viewed**: Slider (1-20 pages)
- **Device**: Dropdown (Mobile, Desktop, Tablet)
- **Referral Source**: Dropdown (Google, Facebook, Direct, Email, Other)

### Output Display
- **Probability Scores**: From both Logistic Regression and Random Forest
- **Business Recommendations**:
  - 🟢 **High Likelihood** (>60%): Consider targeted promotions
  - 🟡 **Moderate Likelihood** (40-60%): Monitor engagement
  - 🔴 **Low Likelihood** (<40%): Focus on conversion optimization

## 📊 Dataset Information

The synthetic dataset includes 100 customer records with:
- **Balanced Classes**: ~40% purchase rate
- **Realistic Distributions**: Time follows exponential decay, pages are normally distributed
- **Categorical Balance**: Even distribution across device types and referral sources

### Sample Data Structure
```csv
time_on_site,pages_viewed,device,referral_source,purchase
15.2,5,desktop,google,1
3.1,2,mobile,direct,0
8.7,7,tablet,facebook,1
```

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Environment**: Jupyter Notebook

## 📈 Learning Outcomes

This project demonstrates:
- **Data Preprocessing**: Handling categorical and numerical features
- **Model Selection**: Comparing interpretable vs. complex models
- **Model Deployment**: Creating production-ready ML applications
- **Web Development**: Building interactive data science apps
- **Business Application**: Translating ML predictions to actionable insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ using Streamlit and scikit-learn
- Inspired by real-world e-commerce analytics
- Designed for learning and demonstration purposes

## 🚀 Deployment

### ❌ Why Not Netlify?
Netlify is designed for **static websites** and doesn't support Python applications. Your Streamlit app requires:
- Python runtime server
- ML model loading (scikit-learn)
- Interactive Python execution

### ✅ Recommended Deployment Options

#### 1. **Streamlit Cloud** (Easiest - Recommended)
**Perfect for Streamlit apps!**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Select `app.py` as the main file
4. Click Deploy!
5. **Free tier available!**

#### 2. **Heroku** (Most Popular)
**Supports Python applications**
1. **Install Heroku CLI**:
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   heroku login
   ```

2. **Create and deploy**:
   ```bash
   heroku create your-customer-predictor-app
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   heroku open
   ```

#### 3. **Railway** (Modern Alternative)
**Easy GitHub integration**
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Railway auto-detects Python and deploys
4. **Free tier available!**

#### 4. **Vercel** (Advanced)
Requires converting to API routes - more complex setup needed.

### 🌐 Live Demo URLs
Your app will be available at:
- **Streamlit Cloud**: `https://your-app.streamlit.app`
- **Heroku**: `https://your-app-name.herokuapp.com`
- **Railway**: `https://your-app.up.railway.app`