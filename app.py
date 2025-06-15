import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load("mobile_price_model.pkl")

# Manually create scaler based on training logic (you can also save/load it if needed)
# For demo, we'll create it again using training ranges (or save it during training if you prefer)
def load_scaler():
    X = pd.read_csv("X_train_processed.csv")
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

scaler = load_scaler()

# Define feature input fields
st.title("📱 Mobile Price Range Predictor")
st.markdown("Fill in the details below to predict the mobile price range (0 to 3).")

# Feature Inputs (excluding dropped features)
battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=2000, value=1200)
clock_speed = st.slider("Clock Speed (GHz)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
fc = st.slider("Front Camera (MP)", 0, 20, 5)
int_memory = st.slider("Internal Memory (GB)", 2, 64, 32)
m_dep = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.01)
n_cores = st.slider("Number of Cores", 1, 8, 4)
pc = st.slider("Primary Camera (MP)", 0, 20, 10)
px_height = st.slider("Pixel Height", 0, 2000, 800)
px_width = st.slider("Pixel Width", 500, 2000, 1200)
ram = st.slider("RAM (MB)", 256, 4000, 1500)
sc_h = st.slider("Screen Height (cm)", 5, 20, 10)
sc_w = st.slider("Screen Width (cm)", 0, 20, 5)
talk_time = st.slider("Talk Time (hours)", 2, 20, 10)
mobile_wt = st.slider("Mobile Weight (g)", 80, 200, 120)

# Collect into a DataFrame
input_data = pd.DataFrame([[
    battery_power, clock_speed, fc, int_memory, m_dep, mobile_wt,
    n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time
]], columns=[
    'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt',
    'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'
])

# Scale the input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("🔍 Predict Price Range"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"📦 Predicted Price Range: **{prediction}** (0 = Low, 3 = High)")
