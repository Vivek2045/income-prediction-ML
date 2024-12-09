import numpy as np
from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Initialize Flask App
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model/salary_prediction_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Feature names (adjust based on how your model was trained)
feature_names = ['Education', 'Work Experience', 'Occupation', 'Employment Status', 'Other_Feature1', 'Other_Feature2', 'Other_Feature3', 'Other_Feature4', 'Other_Feature5', 'Other_Feature6', 'Other_Feature7', 'Other_Feature8', 'Other_Feature9']

# Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from user input and convert them to appropriate types
    education = int(request.form['education'])  # Convert to integer
    work_experience = int(request.form['work_experience'])  # Convert to integer
    occupation = int(request.form['occupation'])  # Convert to integer
    employment_status = int(request.form['employment_status'])  # Convert to integer

    # Add other necessary features that were used during training (dummy example)
    # Assuming other features need to be zero-filled or provided from somewhere
    # Adjust this to match your real dataset
    other_features = np.zeros(len(feature_names) - 4)  # Assuming you have 9 other features
    
    # Combine the features
    features = np.array([education, work_experience, occupation, employment_status] + list(other_features)).reshape(1, -1)

    # Scale the features using the same scaler as in training
    scaled_features = scaler.transform(features)

    # Make prediction using the model
    prediction = model.predict(scaled_features)

    # Return the prediction result
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

