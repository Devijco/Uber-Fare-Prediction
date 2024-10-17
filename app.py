import streamlit as st
import pickle
import numpy as np

# Load LGBM model di awal sebelum digunakan
with open('LinearRegression.pkl', 'rb') as file:
    LinearRegression_Model = pickle.load(file)
    
# Load scaler
with open('StandardScalerUber.pkl', 'rb') as file:
    scaler = pickle.load(file)

def main():
    st.sidebar.title("Uber Fare Prediction")
    menu = ["Home", "Predict Fare"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Uber Fare Prediction App!")
    elif choice == "Predict Fare":
        run_prediction_app()

def run_prediction_app():
    pickup_lat = st.number_input("Pickup Latitude", value=40.7614327, format="%.6f")
    pickup_long = st.number_input("Pickup Longitude", value=-73.9798156, format="%.6f")
    dropoff_lat = st.number_input("Dropoff Latitude", value=40.6513111, format="%.6f")
    dropoff_long = st.number_input("Dropoff Longitude", value=-73.8803331, format="%.6f")

    # Calculate distance using Haversine formula
    distance = haversine_array(pickup_long, pickup_lat, dropoff_long, dropoff_lat)
    st.write(f"Calculated Distance: {distance:.2f} km")

    # Predict fare using the LGBM model
    if st.button('Predict Fare'):
        distance = scaler.transform([[distance]])
        fare_pred = LinearRegression_Model.predict(distance)
        st.write(f"Predicted Fare: ${fare_pred[0]:.2f}")

def haversine_array(pickup_long, pickup_lat, dropoff_long, dropoff_lat):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    pickup_long, pickup_lat, dropoff_long, dropoff_lat = map(lambda x: x/360.*(2*np.pi), [pickup_long, pickup_lat, dropoff_long, dropoff_lat])
    # haversine formula
    dlon = dropoff_long - pickup_long
    dlat = dropoff_lat - pickup_lat
    a = np.sin(dlat/2)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

if __name__ == "__main__":
    main()
