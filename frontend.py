import streamlit as st
import pandas as pd
import joblib

# page config
st.set_page_config(page_title="Ford Car Price Predictor", layout="centered")

# load the trained model, scaler and columns
model = joblib.load('LR_ford_car.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.title("Ford Car Price Predictor")
st.write("enter the car details to predict its price")

# this model was trained using LabelEncoder for the categorical columns
# (model, transmission, fuelType) instead of one hot encoding
# so the categories need to be converted to the same numbers here

model_map = {'B-MAX': 0, 'C-MAX': 1, 'EcoSport': 2, 'Edge': 3, 'Escort': 4, 'Fiesta': 5,
             'Focus': 6, 'Fusion': 7, 'Galaxy': 8, 'Grand C-MAX': 9, 'Grand Tourneo Connect': 10,
             'KA': 11, 'Ka+': 12, 'Kuga': 13, 'Mondeo': 14, 'Mustang': 15, 'Puma': 16,
             'Ranger': 17, 'S-MAX': 18, 'Streetka': 19, 'Tourneo Connect': 20,
             'Tourneo Custom': 21, 'Transit Tourneo': 22}

transmission_map = {'Automatic': 0, 'Manual': 1, 'Semi-Auto': 2}

fuelType_map = {'Diesel': 0, 'Electric': 1, 'Hybrid': 2, 'Other': 3, 'Petrol': 4}

# numerical inputs
year = st.number_input("Manufacturing Year", min_value=1996, max_value=2026, value=2017)
mileage = st.number_input("Mileage", min_value=0, max_value=200000, value=18000)
tax = st.number_input("Road Tax", min_value=0, max_value=600, value=145)
mpg = st.number_input("MPG", min_value=1.0, max_value=200.0, value=58.9)
engineSize = st.number_input("Engine Size", min_value=0.0, max_value=6.0, value=1.2, step=0.1)

# dropdowns for categorical columns
model_name = st.selectbox("Car Model", sorted(model_map.keys()))
transmission = st.selectbox("Transmission", list(transmission_map.keys()))
fuelType = st.selectbox("Fuel Type", list(fuelType_map.keys()))

predict_clicked = st.button("Predict Price")

if predict_clicked:
    # convert the selected categories to their label encoded numbers
    input_dict = {
        'model': model_map[model_name],
        'year': year,
        'transmission': transmission_map[transmission],
        'mileage': mileage,
        'fuelType': fuelType_map[fuelType],
        'tax': tax,
        'mpg': mpg,
        'engineSize': engineSize,
    }

    input_df = pd.DataFrame([input_dict])

    # keep the same column order as during training
    input_df = input_df[columns]

    # scale the numerical columns
    num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # predict the price
    predicted_price = model.predict(input_df)[0]
    st.success(f"Predicted Selling Price: £{predicted_price:,.2f}")