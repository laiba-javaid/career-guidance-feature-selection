import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Load model and metadata
model = joblib.load("model/saved_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
selected_features = joblib.load("model/selected_features.pkl")

# UI Config
st.set_page_config(page_title="Career Guidance System", layout="centered", page_icon="🎓")
st.title("🎓 Career Guidance System")
st.markdown("#### Get your recommended career path based on your interests and strengths.")

# Sidebar
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio("Go to", ["Career Prediction", "Performance Report", "About"])

# Career Prediction Section
if section == "Career Prediction":
    st.subheader("🧠 Answer the following questions:")

    user_input = {}
    for feature in selected_features:
        le = label_encoders[feature]
        options = le.classes_.tolist()
        choice = st.selectbox(f"{feature}:", options)
        user_input[feature] = le.transform([choice])[0]

    # Predict
    if st.button("🎯 Suggest My Career"):
        input_df = pd.DataFrame([user_input])
        prediction_encoded = model.predict(input_df)[0]

        # Decode prediction
        target_encoder = label_encoders['Recommended Career']
        prediction = target_encoder.inverse_transform([prediction_encoded])[0]

        st.success(f"💼 Based on your responses, a suitable career path is: **{prediction}**")

# Performance Report Section
elif section == "Performance Report":
    st.subheader("📊 Model Performance Analysis")

    plot_dir = "plots"
    metrics = ["accuracy", "precision", "recall"]

    for metric in metrics:
        plot_path = os.path.join(plot_dir, f"{metric}_vs_threshold.png")
        if os.path.exists(plot_path):
            st.markdown(f"##### {metric.capitalize()} vs. Information Gain Threshold")
            st.image(plot_path, use_column_width=True)
        else:
            st.warning(f"Plot for {metric} not found.")

    st.markdown("##### ✅ Features used in the final model:")
    for feat in selected_features:
        st.write(f"- {feat}")

# About Section
elif section == "About":
    st.subheader("📘 About This App")
    st.markdown("""
    This career guidance system was built as part of an **Information Gain & Model Performance Evaluation** project.  
    It compares Decision Tree, k-NN, and Naive Bayes classifiers using different feature selection thresholds.  

    - ✅ Feature Selection: Information Gain  
    - ✅ Models: Decision Tree, k-NN, Naive Bayes  
    - ✅ Evaluation: Accuracy, Precision, Recall  
    - ✅ UI: Built with Streamlit  
    - 📁 Dataset: Career Recommendation System

    Designed by: **Laiba Javaid**  
    """)

    st.info("For best results, answer questions honestly based on your interests!")

# Footer
st.markdown("""
---
Developed by [Laiba Javaid] using **Streamlit** | 2025  
""")