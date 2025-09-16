import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# Initialize the Flask application
app = Flask(__name__)

# --- Load the trained model and scaler ---
# These are loaded once when the application starts
try:
    # Corrected filenames to match your uploaded files
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: 'model (1).pkl' or 'scaler (1).pkl' not found. Make sure the files are in the same directory.")
    model = None
    scaler = None

# --- Define the feature order the model expects ---
# This must be the exact same order as the columns used during training.
# This list is now updated to include the missing 'id' and 'dataset_*' columns.
EXPECTED_FEATURES = [
    'id', 'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'sex_Female',
    'sex_Male', 'dataset_Cleveland', 'dataset_Hungary', 'dataset_Switzerland',
    'dataset_VA Long Beach', 'cp_asymptomatic', 'cp_atypical angina',
    'cp_non-anginal', 'cp_typical angina', 'fbs_False', 'fbs_True',
    'restecg_lv hypertrophy', 'restecg_normal', 'restecg_st-t abnormality',
    'exang_False', 'exang_True', 'slope_downsloping', 'slope_flat',
    'slope_upsloping', 'thal_fixed defect', 'thal_normal',
    'thal_reversable defect'
]

# --- Define Routes ---

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives patient data, makes a prediction, and returns it."""
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        # --- Data Preprocessing ---
        data = request.get_json()
        print(f"Received data: {data}")

        # Create a dictionary with all expected features initialized to 0
        feature_dict = {feature: 0 for feature in EXPECTED_FEATURES}
        
        # **FIX**: Add placeholder values for features the model expects but the user doesn't provide.
        feature_dict['id'] = 0 # 'id' is not a predictive feature, so a placeholder is fine.
        feature_dict['dataset_Cleveland'] = 1 # Default to the most common dataset.

        # Populate the dictionary with numerical data from the request
        numerical_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
        for feature in numerical_features:
            # Use .get() with a default value to avoid errors if a key is missing
            feature_dict[feature] = float(data.get(feature, 0))

        # Handle one-hot encoded features from user input
        # Note: The keys are now more robust to avoid errors if a value is missing.
        if data.get('sex') == '1': feature_dict['sex_Male'] = 1
        else: feature_dict['sex_Female'] = 1
        
        if data.get('fbs') == '1': feature_dict['fbs_True'] = 1
        else: feature_dict['fbs_False'] = 1
            
        if data.get('exang') == '1': feature_dict['exang_True'] = 1
        else: feature_dict['exang_False'] = 1

        # Use .get() to safely access keys that might not be present
        cp_key = f"cp_{data.get('cp')}"
        if cp_key in feature_dict: feature_dict[cp_key] = 1

        restecg_key = f"restecg_{data.get('restecg')}"
        if restecg_key in feature_dict: feature_dict[restecg_key] = 1
        
        slope_key = f"slope_{data.get('slope')}"
        if slope_key in feature_dict: feature_dict[slope_key] = 1
            
        thal_key = f"thal_{data.get('thal')}"
        if thal_key in feature_dict: feature_dict[thal_key] = 1

        # Convert the dictionary to a pandas DataFrame in the correct order
        input_df = pd.DataFrame([feature_dict])
        input_df = input_df[EXPECTED_FEATURES] # Ensure column order matches training

        print(f"Processed DataFrame:\n{input_df.to_string()}")

        # Scale the features
        scaled_features = scaler.transform(input_df)
        
        # --- Make Prediction ---
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)

        # The result is 0 (low risk) or 1 (high risk)
        risk_level = int(prediction[0])
        
        # Confidence is the probability of the predicted class
        confidence = prediction_proba[0][risk_level]
        
        print(f"Prediction: {risk_level}, Confidence: {confidence:.2f}")

        # --- Return the result as JSON ---
        return jsonify({
            'prediction': risk_level,
            'confidence': round(confidence * 100)
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

