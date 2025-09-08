import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load('../models/best_model.joblib')

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Add title and description
st.title("💻 Laptop Price Prediction App")
st.markdown("Enter the specifications of the laptop to get its predicted price.")

# Create the input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_speed = st.number_input("CPU Speed (GHz)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
        memory_size = st.number_input("Memory Size (GB)", min_value=0, max_value=2048, value=256, step=8)
        screen_resolution = st.selectbox(
            "Screen Resolution",
            ["HD (1366x768)", "Full HD (1920x1080)", "4K (3840x2160)"]
        )
        
    with col2:
        gpu_type = st.selectbox(
            "GPU Type",
            ["Integrated", "Dedicated"]
        )
        memory_type = st.selectbox(
            "Storage Type",
            ["SSD", "HDD", "Flash Storage", "Hybrid"]
        )
        is_gaming = st.checkbox("Is it a Gaming Laptop?")
        
    submit_button = st.form_submit_button(label="Predict Price")

# Make prediction when form is submitted
if submit_button:
    # Prepare input data
    input_data = {
        'cpu_speed': cpu_speed,
        'memory_size_gb': memory_size,
        'ScreenResolution_HD': 1 if "HD" in screen_resolution else 0,
        'ScreenResolution_FHD': 1 if "Full HD" in screen_resolution else 0,
        'ScreenResolution_4K': 1 if "4K" in screen_resolution else 0,
        'GPUType_Integrated': 1 if gpu_type == "Integrated" else 0,
        'GPUType_Dedicated': 1 if gpu_type == "Dedicated" else 0,
        'MemoryType_SSD': 1 if memory_type == "SSD" else 0,
        'MemoryType_HDD': 1 if memory_type == "HDD" else 0,
        'MemoryType_Flash': 1 if memory_type == "Flash Storage" else 0,
        'MemoryType_Hybrid': 1 if memory_type == "Hybrid" else 0,
        'is_gaming': 1 if is_gaming else 0
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display prediction
    st.success(f"Predicted Laptop Price: €{prediction:,.2f}")
    
    # Add confidence note
    st.info("""
    Note: This prediction is based on historical data and should be used as a reference only.
    Actual prices may vary based on brand, market conditions, and other factors.
    """)

# Add sidebar with additional information
with st.sidebar:
    st.header("About")
    st.write("""
    This app predicts laptop prices based on their specifications using machine learning.
    The model was trained on historical laptop price data.
    """)
    
    st.header("Instructions")
    st.write("""
    1. Enter the laptop specifications in the form
    2. Click 'Predict Price' button
    3. Get the estimated price in Euros
    """)

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")