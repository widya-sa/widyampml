import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as imPipeline

# Function to train the model and save it
def train_and_save_model():
    # Dataset
    df_cleaned = pd.read_csv('df_cleaned.csv')

    # Identifikasi variabel kategorik
    Categorical_Cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype == 'object']

    # Identifikasi variabel numerik
    Numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
    Numerical_cols.remove('Weather Type')  # Remove the target column if present

    # Separation
    X = df_cleaned.drop('Weather Type', axis='columns')
    y = df_cleaned['Weather Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

    # Define preprocessing for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), Numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), Categorical_Cols)
        ])

    # Define SMOTE and ENN
    smote = SMOTE(random_state=47)
    enn = EditedNearestNeighbours()

    # Create an imbalanced-learn pipeline
    pipeline = imPipeline([
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('enn', enn),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=47))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Save model
    joblib.dump(pipeline, 'model_with_smote_enn.sav')

# Train and save the model
train_and_save_model()

# Load the model
model = joblib.load('model_with_smote_enn.sav')

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
