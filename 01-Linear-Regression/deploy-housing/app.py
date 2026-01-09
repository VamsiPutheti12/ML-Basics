"""
🏠 House Price Predictor - Streamlit App
=========================================
A web application to predict house prices using
Multiple Linear Regression trained on 2015-2024 data.
"""

import streamlit as st
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
    }
    .feature-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'housing_model.pkl')
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

# Header
st.markdown('<p class="main-header">🏠 House Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multiple Linear Regression Model • 2015-2024 Housing Data</p>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found! Please run `python save_model.py` first to generate the model file.")
    st.code("python save_model.py", language="bash")
    st.stop()

# Display model info
with st.expander("ℹ️ About this Model"):
    st.write(f"""
    **Model Type:** Multiple Linear Regression (from scratch)  
    **Training Data:** 5,000 houses (2015-2024 synthetic data)  
    **Test R² Score:** {model['r2_score']:.4f}  
    **Features Used:** {len(model['feature_names'])}
    """)

st.divider()

# Input features
st.subheader("📝 Enter House Details")

col1, col2 = st.columns(2)

with col1:
    square_feet = st.slider(
        "🏠 Square Feet",
        min_value=800,
        max_value=5000,
        value=2000,
        step=100,
        help="Total living area in square feet"
    )
    
    bedrooms = st.selectbox(
        "🛏️ Bedrooms",
        options=[1, 2, 3, 4, 5],
        index=2,
        help="Number of bedrooms"
    )
    
    bathrooms = st.slider(
        "🚿 Bathrooms",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.5,
        help="Number of bathrooms"
    )
    
    year_built = st.slider(
        "📅 Year Built",
        min_value=1960,
        max_value=2024,
        value=2000,
        step=1,
        help="Year the house was built"
    )

with col2:
    lot_size = st.slider(
        "🌳 Lot Size (sq ft)",
        min_value=2000,
        max_value=30000,
        value=8000,
        step=500,
        help="Total lot size in square feet"
    )
    
    garage_spaces = st.selectbox(
        "🚗 Garage Spaces",
        options=[0, 1, 2, 3],
        index=2,
        help="Number of garage spaces"
    )
    
    school_rating = st.slider(
        "🎓 School Rating",
        min_value=3.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        help="Local school district rating (3-10)"
    )
    
    crime_rate = st.slider(
        "🚨 Crime Rate",
        min_value=0.5,
        max_value=8.0,
        value=3.0,
        step=0.5,
        help="Area crime rate (lower is better)"
    )

st.divider()

# Prediction
def predict_price(features, model):
    """Make prediction using saved model parameters."""
    # Scale features using saved scaler parameters
    scaled = (features - model['scaler_mean']) / model['scaler_scale']
    # Calculate prediction
    price = np.dot(scaled, model['weights']) + model['bias']
    return max(150000, min(1500000, price))  # Clip to valid range

# Create feature array
features = np.array([
    square_feet,
    bedrooms,
    bathrooms,
    year_built,
    lot_size,
    garage_spaces,
    school_rating,
    crime_rate
])

# Make prediction
predicted_price = predict_price(features, model)

# Display prediction
st.markdown(f"""
<div class="prediction-box">
    <div class="prediction-label">Estimated House Price</div>
    <div class="prediction-value">${predicted_price:,.0f}</div>
</div>
""", unsafe_allow_html=True)

# Feature breakdown
st.subheader("📊 Feature Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Square Feet",
        f"{square_feet:,}",
        f"${square_feet * 150:,} impact"
    )

with col2:
    st.metric(
        "Bedrooms",
        bedrooms,
        f"${bedrooms * 25000:,} impact"
    )

with col3:
    st.metric(
        "School Rating",
        f"{school_rating}/10",
        f"${school_rating * 15000:,.0f} impact"
    )

with col4:
    age = 2024 - year_built
    st.metric(
        "House Age",
        f"{age} years",
        f"-${age * 800:,}" if age > 0 else "New!"
    )

# Footer
st.divider()
st.caption("""
**Note:** This is a demonstration model using synthetic data based on 2015-2024 housing market trends.
Predictions are for educational purposes only.

Built with ❤️ using Linear Regression from scratch | [View Source Code](https://github.com/VamsiPutheti12/ML-Basics)
""")
