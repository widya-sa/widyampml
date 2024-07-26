import streamlit as st
import joblib
import numpy as np

# Fungsi untuk memuat model dan melakukan prediksi
def value_predictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, -1)  # Membentuk array 2D dengan 1 baris dan 10 kolom
    loaded_model = joblib.load('model.sav')
    result = loaded_model.predict(to_predict)[0]  # Ambil hasil prediksi
    weather_mapping = {
        0: 'Cloudy',
        1: 'Rainy',
        2: 'Snowy',
        3: 'Sunny'
    }
    return weather_mapping.get(result, 'Unknown')  # Mengembalikan kategori cuaca

# Antarmuka pengguna Streamlit
st.title("Weather Prediction")

# Input form
temperature = st.number_input('Temperature', format="%f")
humidity = st.number_input('Humidity', format="%f")
wind_speed = st.number_input('Wind Speed', format="%f")
precipitation = st.number_input('Precipitation (%)', format="%f")
cloud_cover = st.number_input('Cloud Cover', format="%f")
atmospheric_pressure = st.number_input('Atmospheric Pressure', format="%f")
uv_index = st.number_input('UV Index', format="%f")
season = st.number_input('Season', format="%f")
visibility = st.number_input('Visibility (km)', format="%f")
location = st.number_input('Location', format="%f")

# Tombol prediksi
if st.button('Predict'):
    to_predict_list = [
        temperature, humidity, wind_speed, precipitation,
        cloud_cover, atmospheric_pressure, uv_index, season,
        visibility, location
    ]
    
    try:
        result = value_predictor(to_predict_list)
        st.write(f"Predicted Weather Type: {result}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

