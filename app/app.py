import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model_path = 'models/best_model.joblib'
model = joblib.load(model_path)

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Laptop Price Prediction App")
st.markdown("Enter the specifications of the laptop to get its predicted price.")

# Create the input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_speed = st.number_input("CPU Speed (GHz)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
        memory_size = st.number_input("Memory Size (GB)", min_value=0, max_value=2048, value=256, step=8)
        memory_ssd = st.checkbox("SSD Storage")
        memory_hdd = st.checkbox("HDD Storage")
        
    with col2:
        gpu_company = st.selectbox(
            "GPU Company",
            ["Intel", "AMD", "NVIDIA"]  # Add more based on your training data
        )
        screen_retina = st.checkbox("Retina Display")
        screen_hd = st.checkbox("HD Display")
        
    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    # Initialize all GPU company features to 0
    gpu_features = {f'GPUCompany_{i}': 0 for i in range(150)}  # Adjust range based on your data
    
    # Create input data dictionary with all required features
    input_data = {
        'cpu_speed': cpu_speed,
        'Memory_ssd': 1 if memory_ssd else 0,
        'Memory_hdd': 1 if memory_hdd else 0,
        'Memory_size': memory_size,
        'memory_size_gb': memory_size,
        'ScreenResolution_Retina': 1 if screen_retina else 0,
        'ScreenResolution_HD': 1 if screen_hd else 0,
        'ScreenResolution_Other': 0 if (screen_retina or screen_hd) else 1,
    }
    
    # Add GPU company features
    input_data.update(gpu_features)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure column order matches training data
    expected_columns = model.feature_names_in_  # Get feature names used during training
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Laptop Price: €{prediction:,.2f}")
        
        st.info("""
        Note: This prediction is based on historical data and should be used as a reference only.
        Actual prices may vary based on brand, market conditions, and other factors.
        """)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This app predicts laptop prices based on their specifications using machine learning.
    The model was trained on historical laptop price data.
    """)
    
    st.header("Instructions")
    st.write("""
    1. Enter the laptop specifications
    2. Click 'Predict Price' button
    3. Get the estimated price in Euros
    """)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")