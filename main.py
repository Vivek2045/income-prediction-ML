import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Using Logistic Regression (less memory-intensive)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load Dataset
data = pd.read_csv("data.csv")
print("Dataset Loaded Successfully")
print(data.head())

# Check for null values or missing data
missing_data = data.isnull().sum()
print(f"Columns with missing values:\n{missing_data[missing_data > 0]}")

# Fill missing values (optional, adjust as needed)
data = data.fillna(data.mode().iloc[0])  # Replace with mode for categorical, mean/median for numerical

# Separate categorical and numerical columns
cat_columns = data.select_dtypes(include="object").columns
num_columns = data.select_dtypes(include="number").columns

print(f"Categorical columns: {list(cat_columns)}")
print(f"Numerical columns: {list(num_columns)}")

# Encode categorical columns
encoder = LabelEncoder()
for col in cat_columns:
    data[col] = encoder.fit_transform(data[col])

# Feature-Target Split
X = data.drop('Income', axis=1)  # Predicting 'Income'
y = data['Income']

# Use a subset of the data (optional, adjust as needed)
data_subset = data.sample(n=10000, random_state=42)  # Using a smaller subset for training

# Feature-Target Split again after selecting the subset
X = data_subset.drop('Income', axis=1)
y = data_subset['Income']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Scaling (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training: Using Logistic Regression (memory-efficient)
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Print the Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model and Scaler
os.makedirs('model', exist_ok=True)
model_path = 'model/salary_prediction_model.pkl'
scaler_path = 'model/scaler.pkl'

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
