import streamlit as st
import pandas as pd
import joblib
import os

# App title
st.title(" AI House Price Predictor")

st.write("Predict house prices using a trained Machine Learning model.")

# Load model
model_path = os.path.join("models", "house_price_model.pkl")

if not os.path.exists(model_path):
    st.error("Model file not found. Train the model first.")
    st.stop()

model = joblib.load(model_path)

# Load data (to get column structure)
data_path = os.path.join("data", "clean_house_prices.csv")
df = pd.read_csv(data_path)

# Detect target safely
target = "Price" if "Price" in df.columns else "price"

X = df.drop(columns=[target])

st.subheader("Enter House Details")

user_input = {}

for col in X.columns:
    value = float(X[col].mean())
    user_input[col] = st.number_input(col, value=value)

input_df = pd.DataFrame([user_input])

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f" Predicted House Price: {prediction:,.2f}")
