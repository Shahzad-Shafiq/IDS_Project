# mobile_price_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')  # Ensure this file is in your working directory
    return df

# Train model
@st.cache_resource
def train_model(df):
    X = df.drop('price_range', axis=1)
    y = df['price_range']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return model, scaler, accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True)

# App Layout
#st.set_page_config(layout='wide')

df = load_data()
model, scaler, acc, report = train_model(df)

# Sidebar Navigation
page = st.sidebar.radio("Select a page:", ["Home", "EDA", "Predict", "Model Evaluation","Conclusion"])

# ---------------- HOME ----------------
if page == "Home":
    st.title("📱 Mobile Price Range Prediction")

    # Show image at the top
    st.image("Mobile_price_prediction_image.png", use_container_width=True)

    # Welcome and description section
    st.markdown("""
    ## 🎯 Welcome!
    This interactive app helps you **predict the price range of mobile phones** based on their specifications using a machine learning model.

    ### 📌 Key Features:
    - Perform **real-time predictions**
    - Explore **in-depth visual EDA**
    - Understand model performance through **metrics and charts**
    - Built using **Random Forest Classifier** and **Streamlit**

    ### 💰 Price Range Categories:
    - `0`: Low Cost  
    - `1`: Medium Cost  
    - `2`: High Cost  
    - `3`: Very High Cost

    ---

    ✅ **Start by exploring the sidebar to navigate through the app!**
    """)



# ---------------- EDA ----------------
elif page == "EDA":
    st.header("📊 Exploratory Data Analysis")
    st.dataframe(df.head())

    st.subheader("🔣 Basic Info")
    buffer = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes,
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(buffer)

    st.subheader("📐 Shape & Summary Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset Shape**")
        st.write(df.shape)
    with col2:
        st.markdown("**Summary Statistics**")
        st.write(df.describe())

    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    st.subheader("📈 Feature Distributions")
    fig, ax = plt.subplots(figsize=(16, 12))
    df.hist(ax=ax, bins=20, color='skyblue', edgecolor='black', layout=(7, 3))
    plt.tight_layout()
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.subplots_adjust(top=0.93)
    st.pyplot(fig)

    # Boxplots by target
    st.subheader("📦 Key Features vs Price Range (Boxplots)")
    important_features = ['ram', 'battery_power', 'int_memory', 'mobile_wt', 'px_height', 'px_width']
    for feature in important_features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='price_range', y=feature, data=df, palette='Set2', ax=ax)
        ax.set_title(f"{feature} vs Price Range")
        st.pyplot(fig)

    st.subheader("🔢 Categorical Binary Feature Distribution")
    cat_features = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
    for feature in cat_features:
        fig, ax = plt.subplots()
        sns.countplot(x=feature, data=df, hue='price_range', palette='pastel', ax=ax)
        ax.set_title(f"{feature} Distribution by Price Range")
        st.pyplot(fig)

    st.subheader("🧠 Feature Importance from Random Forest")
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': df.drop('price_range', axis=1).columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title("Top Important Features")
    st.pyplot(fig)

    st.subheader("🎯 Scatter: RAM vs Battery Power Colored by Price Range")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='ram', y='battery_power', hue='price_range', palette='Set1', ax=ax)
    ax.set_title("RAM vs Battery Power by Price Range")
    st.pyplot(fig)



# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":
    st.header("🧪 Model Performance")

    # Show Accuracy
    st.subheader("✅ Accuracy Score")
    st.metric("Model Accuracy", f"{acc:.2f}")

    # Classification Report
    st.subheader("📋 Classification Report")
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")

    # Get predictions on full dataset
    X = df.drop('price_range', axis=1)
    y_true = df['price_range']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Generate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Low", "Medium", "High", "Very High"],
                yticklabels=["Low", "Medium", "High", "Very High"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


# ---------------- PREDICT ----------------
elif page == "Predict":
    st.header("🔍 Predict Mobile Price Range")

    st.markdown("""
    <style>
    .predict-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #d3d3d3;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.05);
    }
    .prediction-result {
        background-color: #e0f7fa;
        padding: 15px;
        margin-top: 20px;
        border-radius: 10px;
        border: 2px solid #00acc1;
        font-size: 18px;
        font-weight: bold;
        color: #004d40;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("#### 📲 Enter mobile specifications below:")
    with st.container():
        with st.expander("🛠️ Customize Specifications", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                battery_power = st.slider("🔋 Battery Power (mAh)", 500, 2000, 1000)
                blue = st.selectbox("📶 Bluetooth", [0, 1])
                clock_speed = st.slider("🕓 Clock Speed (GHz)", 0.5, 3.0, 1.5)
                dual_sim = st.selectbox("📱 Dual SIM", [0, 1])
                fc = st.slider("🤳 Front Camera (MP)", 0, 20, 5)
                four_g = st.selectbox("📡 4G Support", [0, 1])
                int_memory = st.slider("💾 Internal Memory (GB)", 2, 64, 16)
                m_dep = st.slider("📏 Mobile Depth (cm)", 0.1, 1.0, 0.5)

            with col2:
                mobile_wt = st.slider("⚖️ Mobile Weight (g)", 80, 250, 150)
                n_cores = st.selectbox("🧠 Number of Cores", list(range(1, 9)))
                pc = st.slider("📷 Primary Camera (MP)", 0, 20, 10)
                px_height = st.slider("🔳 Pixel Height", 0, 1960, 800)
                px_width = st.slider("🔲 Pixel Width", 500, 2000, 1000)
                ram = st.slider("🧠 RAM (MB)", 256, 4096, 2048)
                sc_h = st.slider("📐 Screen Height (cm)", 5, 20, 10)
                sc_w = st.slider("📐 Screen Width (cm)", 0, 18, 10)
                talk_time = st.slider("📞 Talk Time (hrs)", 2, 20, 10)
                three_g = st.selectbox("📶 3G Support", [0, 1])
                touch_screen = st.selectbox("👆 Touch Screen", [0, 1])
                wifi = st.selectbox("📡 WiFi", [0, 1])

    user_input = pd.DataFrame([{
        'battery_power': battery_power,
        'blue': blue,
        'clock_speed': clock_speed,
        'dual_sim': dual_sim,
        'fc': fc,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'three_g': three_g,
        'touch_screen': touch_screen,
        'wifi': wifi
    }])

    user_input = user_input[df.drop("price_range", axis=1).columns]

    if st.button("🚀 Predict Price Range"):
        input_scaled = scaler.transform(user_input)
        prediction = model.predict(input_scaled)[0]
        price_label = ["Low", "Medium", "High", "Very High"]
        st.markdown(
            f'<div class="prediction-result">📱 Predicted Price Range: <strong>{price_label[prediction]}</strong></div>',
            unsafe_allow_html=True
        )
# ---------------- CONCLUSION ----------------
elif page == "Conclusion":
    st.header("📌 Conclusion")
    
    st.markdown("""
    ### 📱 Mobile Price Range Prediction Summary
    This app demonstrates how machine learning can be used to classify mobile phones into price ranges based on technical specifications.

    - ✅ We used a **Random Forest Classifier**, which achieved an accuracy of **{:.2f}** on test data.
    - 📊 Features such as **RAM**, **Battery Power**, and **Pixel Resolution** were found to be most influential.
    - 🔍 The app provides tools for **exploratory data analysis**, **model evaluation**, and **real-time predictions**.

    ---
    
    ### 🧭 Future Enhancements
    - Implement other models like **XGBoost** or **SVM** for comparison.
    - Add support for **real mobile device data scraping**.
    - Incorporate **user feedback loop** to improve model predictions over time.
    - Improve UI/UX with **animated transitions** and **dark/light themes**.
    
    ---
    
    ### 🙌 Thank You for Using This App!
    Built with ❤️ using **Streamlit** & **scikit-learn**.
    """.format(acc))
