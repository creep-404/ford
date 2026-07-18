import streamlit as st
# streamlit: used to build and run the interactive web app UI

import pandas as pd
# pandas: used for data manipulation and analysis

import joblib
# joblib: used for saving and loading machine learning models

# show the browse tab title and keep layout centered
st.set_page_config(page_title="Browse", page_icon="🔍", layout="centered")

# load the trained model, scaler, and encoded columns
# wrapped in try/except so the app shows a clear message instead of crashing
# if the .pkl files are missing or misplaced
try:
    model = joblib.load('LR_ford_car.pkl')
    scaler = joblib.load('scaler.pkl')
    encoded_columns = joblib.load('columns.pkl')
except FileNotFoundError as e:
    st.error(f"Required model file not found: {e}")
    st.stop()

# show the app title and the short discription
st.title("Ford Car Price Prediction App")
st.write("enter the car details to predict its price")

st.subheader("Car Details")

# numerical inputs, arranged in two columns for a cleaner layout
col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Manufacturing Year", min_value=1996, max_value=2026, value=2017)
    mileage = st.number_input("Mileage", min_value=0, max_value=200000, value=18000)
    tax = st.number_input("Road Tax", min_value=0, max_value=600, value=145)
with col2:
    mpg = st.number_input("MPG", min_value=1.0, max_value=200.0, value=58.9)
    engineSize = st.number_input("Engine Size", min_value=0.0, max_value=6.0, value=1.2, step=0.1)

# selectbox restricts input to fixed valid options
transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Semi-Auto"])
fuelType = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])

model_name = st.text_input("Car Model")

st.divider()
predict_clicked = st.button("Predict Price")

if predict_clicked:
    # basic validation before predicting
    if not model_name.strip():
        st.warning("Please enter a car model name.")
    else:
        try:
            # Build a single-row DataFrame from the user's inputs.
            input_df = pd.DataFrame({
                "model": [model_name],
                "year": [year],
                "transmission": [transmission],
                "mileage": [mileage],
                "fuelType": [fuelType],
                "tax": [tax],
                "mpg": [mpg],
                "engineSize": [engineSize],
            })

            # One-hot encode the categorical columns, same as during training.
            input_encoded = pd.get_dummies(input_df)

            # Align with the training columns so every expected column exists
            # (missing ones are filled with 0), and drop any extra/unseen columns.
            input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

            # scale only the numerical coloumns
            numerical_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
            input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

            # predict the price and display it
            predicted_price = model.predict(input_encoded)[0]
            st.success(f"Predicted Selling Price: £{predicted_price:,.2f}")

        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")










