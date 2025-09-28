import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Configure CORS to allow requests from our React frontend
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Path to the dataset
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')

# Global variables for the model and data
model = None
scaler = None
model_accuracy = 0.0
feature_names = None
raw_data = None
raw_data_for_training = None

def load_data():
    """Loads and preprocesses the entire dataset, handling common data issues."""
    global raw_data, raw_data_for_training

    if raw_data is not None:
        return

    try:
        data = pd.read_csv(DATA_PATH)
        data = data.drop('id', axis=1)

        # Get the diagnosis column before converting other columns
        diagnosis_column = data['diagnosis']
        data = data.drop('diagnosis', axis=1)

        # Convert feature columns to numeric, coercing errors to NaN
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Drop feature columns that are all NaN
        data.dropna(axis=1, how='all', inplace=True)
        
        # Add the diagnosis column back to the DataFrame
        data['diagnosis'] = diagnosis_column

        # Store a version of the data for JSON serialization (NaN -> None)
        raw_data = data.replace({np.nan: None})
        
        # Store a version for model training (with mean imputation)
        data_for_training = data.fillna(data.mean())
        data_for_training['diagnosis'] = diagnosis_column
        raw_data_for_training = data_for_training

    except Exception as e:
        print(f"Error loading data: {e}")
        raw_data = None
        raw_data_for_training = None

def train_model(data_for_training=None):
    """Trains or re-trains the model on a given dataset."""
    global model, scaler, model_accuracy, feature_names

    if data_for_training is None:
        if raw_data_for_training is not None:
            data_for_training = raw_data_for_training.copy()
        else:
            print("Error: No data available for training.")
            model = None
            scaler = None
            return

    try:
        # Map diagnosis to numerical values
        data_for_training['diagnosis'] = data_for_training['diagnosis'].map({'M': 1, 'B': 0})
        
        X = data_for_training.drop('diagnosis', axis=1)
        y = data_for_training['diagnosis']

        feature_names = X.columns
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        model_accuracy = accuracy_score(y_test, y_pred)
        
        print("Model training complete.")
        print(f"Model Accuracy: {model_accuracy:.4f}")

    except Exception as e:
        print(f"Error during model training: {e}")
        model = None
        scaler = None
        feature_names = None

@app.route('/')
def home():
    return 'Flask Backend is running!'

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions on new data."""
    if model is None or scaler is None or feature_names is None:
        return jsonify({'error': 'Model is not trained'}), 503

    try:
        data = request.get_json(force=True)
        input_features = data['features']
        
        if len(input_features) < len(feature_names):
            padded_features = input_features + [0] * (len(feature_names) - len(input_features))
        else:
            padded_features = input_features

        features_df = pd.DataFrame([padded_features], columns=feature_names)
        
        features_scaled = scaler.transform(features_df)
        
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        diagnosis = 'Malignant' if prediction[0] == 1 else 'Benign'
        confidence = float(np.max(probabilities))
        
        return jsonify({
            'diagnosis': diagnosis,
            'confidence': confidence,
            'accuracy': float(model_accuracy)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/data', methods=['GET'])
def get_data():
    """API endpoint to get dataset with optional filtering."""
    if raw_data is None:
        return jsonify({'error': 'Data not available'}), 503

    data = raw_data.copy()
    diagnosis_filter = request.args.get('diagnosis')
    if diagnosis_filter:
        data = data[data['diagnosis'] == diagnosis_filter]

    return jsonify(data.to_dict('records'))

@app.route('/status', methods=['GET'])
def get_status():
    """New API endpoint to provide model status and statistics to the dashboard."""
    if model is None or scaler is None:
        return jsonify({'trained': False, 'accuracy': 0, 'total_samples': 0})
    
    return jsonify({
        'trained': True,
        'accuracy': float(model_accuracy),
        'total_samples': len(raw_data) if raw_data is not None else 0
    })

@app.route('/train', methods=['POST'])
def retrain_model():
    """API endpoint to retrain the model on a selected subset of the data."""
    if raw_data_for_training is None:
        return jsonify({'error': 'Training data not available'}), 503

    try:
        data_to_train = raw_data_for_training.copy()
        selected_ids = request.get_json(force=True).get('ids', [])
        
        if selected_ids:
            # Filter the training data based on the provided IDs
            data_to_train = data_to_train[data_to_train['id'].isin(selected_ids)]
        
        # Call the train function with the selected data
        train_model(data_to_train)

        return jsonify({
            'status': 'Model retraining initiated',
            'new_accuracy': float(model_accuracy)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Train the model and load data when the application starts
if __name__ == '__main__':
    load_data()
    train_model()
    app.run(debug=True)