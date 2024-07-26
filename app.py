import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('model.sav')

# Function to predict weather
def value_predictor(features):
    features = np.array(features).reshape(1, -1)  # Reshape for prediction
    result = model.predict(features)[0]
    weather_mapping = {
        0: 'Cloudy',
        1: 'Rainy',
        2: 'Snowy',
        3: 'Sunny'
    }
    return weather_mapping.get(result, 'Unknown')

# Streamlit application
def main():
    st.title('Weather Prediction by Widya S.A.')

    # Input fields
    temperature = st.number_input('Temperature', format="%.2f")
    humidity = st.number_input('Humidity', format="%.2f")
    wind_speed = st.number_input('Wind Speed', format="%.2f")
    precipitation = st.number_input('Precipitation (%)', format="%.2f")
    cloud_cover = st.selectbox('Cloud Cover', ['Clear', 'Cloudy', 'Overcast', 'Partly Cloudy'])
    atmospheric_pressure = st.number_input('Atmospheric Pressure', format="%.2f")
    uv_index = st.number_input('UV Index', format="%.2f")
    season = st.selectbox('Season', ['Autumn', 'Spring', 'Summer', 'Winter'])
    visibility = st.number_input('Visibility (km)', format="%.2f")
    location = st.selectbox('Location', ['Coastal', 'Inland', 'Mountain'])

    # Convert selections to numerical values
    cloud_cover = {'Clear': 0, 'Cloudy': 1, 'Overcast': 2, 'Partly Cloudy': 3}[cloud_cover]
    season = {'Autumn': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}[season]
    location = {'Coastal': 0, 'Inland': 1, 'Mountain': 2}[location]

    features = [temperature, humidity, wind_speed, precipitation, cloud_cover, atmospheric_pressure, uv_index, season, visibility, location]

    # Prediction button
    if st.button('Predict'):
        result = value_predictor(features)
        st.write(f'The predicted weather is: {result}')

if __name__ == '__main__':
    main()
