import streamlit as st
import pickle
import numpy as np
import json

# Load the trained model
with open("LRmodel.pickle", "rb") as file:
    model = pickle.load(file)

# Load location names from columns.json
with open("columns.json", "r") as file:
    data = json.load(file)
    locations = data["data_columns"][3:]  # Extract only location names


st.title("Real Estate Price Prediction")
st.write("Enter property details to predict the price.")

# Input fields
area = st.number_input("Area (sq ft)", min_value=100, step=10)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)
location = st.selectbox("Location", locations)

# Encoding categorical feature
location_encoded = [0] * len(locations)
location_index = locations.index(location) if location in locations else -1
if location_index != -1:
    location_encoded[location_index] = 1

# Predict button
if st.button("Predict Price"):
    input_features = np.array([[area, bathrooms, bedrooms] + location_encoded])
    predicted_price = model.predict(input_features)[0]
    st.success(f"Estimated Price: ${predicted_price:,.2f}")
