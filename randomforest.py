# Library
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as imPipeline

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
filename = 'model_with_smote_enn.sav'
joblib.dump(pipeline, filename)

# Load model
loaded_model = joblib.load(filename)

# Make predictions
predictions = loaded_model.predict(X_test)

# Evaluate model performance
print(classification_report(y_test, predictions))
