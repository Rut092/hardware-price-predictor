import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# 1. Page Configuration (Set to 'wide' for a dashboard feel)
st.set_page_config(page_title="Laptop Price Predictor | Rutvik Mangrole", page_icon="💻", layout="wide")

# 2. Load the Model and Data
@st.cache_resource
def load_model():
    return joblib.load('laptop_price_predictor.pkl')

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/laptop_data.csv') 
        return df
    except FileNotFoundError:
        return pd.DataFrame()

model = load_model()
df = load_data()


with st.sidebar:
    st.image("data/another_pic.jpg", width=200)
    st.header("👨‍💻 Rutvik Mangrole")
    st.write("**Aspiring ML Engineer**")
    
    # "Hire Me" Contact Card
    with st.container(border=True):
        st.subheader("📬 Contact Details")
        st.write("📧 **Email:** rutvikmangrole99@gmail.com")
        st.write("🔗 **LinkedIn:** [Rutvik Mangrole](https://www.linkedin.com/in/rutvikmangrole/)")
        st.write("🐙 **GitHub:** [rut-ai-portfolio](https://github.com/Rut092)") 
    
    st.divider()
    
    # Model Specs Card
    with st.container(border=True):
        st.subheader("🧠 Model Architecture")
        st.write("**Algorithm:** Random Forest Regressor")
        st.write("**Target Variable:** Laptop Price (INR)")
        st.metric(label="Cross-Validated Accuracy (R²)", value="81.3%")
        st.caption("Best Parameters: n_estimators=300, max_depth=15, min_samples_split=2")


st.title("💻 AI Hardware Price Estimator")
st.markdown("Developed by **Rutvik Mangrole** | End-to-end Machine Learning Pipeline")
st.divider()


tab1, tab2 = st.tabs(["🚀 Price Predictor", "📊 Dataset Reference & Project Info"])


with tab1:
    # We wrap the inputs in a beautiful container box
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Brand & Core Specs")
            company = st.selectbox("Brand", ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'MSI', 'Other'])
            
            # Dynamic OS and CPU logic
            if company == 'Apple':
                os_options = ['macOS']
                cpu_options = ['Apple', 'Intel']
            else:
                os_options = ['Windows 10', 'No OS', 'Linux']
                cpu_options = ['Intel', 'AMD']

            type_name = st.selectbox("Laptop Type", ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation'])
            opsys = st.selectbox("Operating System", os_options)
            cpu_brand = st.selectbox("CPU Brand", cpu_options)
            gpu_brand = st.selectbox("GPU Brand", ['Intel', 'AMD', 'Nvidia'])
            cpu_freq = st.slider("CPU Speed (GHz)", 1.0, 4.0, 2.5, step=0.1)

        with col2:
            st.subheader("Hardware Power")
            ram = st.slider("RAM (GB)", 2, 64, 16)
            weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5)
            inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=18.0, value=15.6)
            
            st.markdown("**Storage Details**")
            flash = st.selectbox("Flash Storage (GB)", [0, 64, 128, 256, 512])
            is_flash_active = flash > 0
            
            ssd = st.select_slider("SSD Capacity", options=[0, 128, 256, 512, 1000, 2000], value=0 if is_flash_active else 512, disabled=is_flash_active)
            hdd = st.select_slider("HDD Capacity", options=[0, 500, 1000, 2000], value=0, disabled=is_flash_active)

    with st.container(border=True):
        st.subheader("Display Tech")
        col3, col4 = st.columns(2)
        with col3:
            ips = st.radio("IPS Panel?", ["Yes", "No"], horizontal=True)
        with col4:
            touchscreen = st.radio("Touchscreen?", ["Yes", "No"], horizontal=True)

    # The Prediction Button
    if st.button("Calculate Market Value", type="primary", use_container_width=True):
        with st.spinner('Analyzing hardware specifications...'):
            time.sleep(2) 
            
            has_ssd = 1 if ssd > 0 else 0
            has_hdd = 1 if hdd > 0 else 0
            has_flash = 1 if flash > 0 else 0
            
            input_data = pd.DataFrame({
                'Company': [company],
                'TypeName': [type_name],
                'Inches': [inches],
                'Ram': [ram],
                'Weight': [weight],
                'OpSys': [opsys],
                'Cpu_Brand': [cpu_brand],
                'Gpu_Brand': [gpu_brand],
                'Cpu_frequency': [cpu_freq], 
                'Touchscreen': [1 if touchscreen == "Yes" else 0],
                'Ips': [1 if ips == "Yes" else 0], 
                'Has_SSD': [has_ssd],
                'Has_HDD': [has_hdd],
                'Has_Flash_Storage': [has_flash],
                'SSD_GB': [ssd],
                'HDD_GB': [hdd],
                'Flash_GB': [flash]
            })
            
            prediction = model.predict(input_data)[0]
            
            st.success(f"### 🏷️ Estimated Market Value: ₹ {prediction:,.2f}####")
            st.balloons() 


with tab2:
    st.header("Project Overview")
    st.write("""
    This project is an end-to-end Machine Learning pipeline designed to predict laptop prices based on hardware specifications. 
    It demonstrates data cleaning, complex feature engineering ,extracting specific hardware metrics from text strings, 
    hyperparameter tuning, and model deployment.
    """)
    
    st.subheader("Data Reference")
    if not df.empty:
        st.write(f"The model was trained on a dataset of **{len(df)}** laptops. Here is a sample of the raw data used:")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Dataset not found in the current directory. To view the data, place the CSV file in the same folder as this app.")