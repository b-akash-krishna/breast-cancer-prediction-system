import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS to allow requests from our React frontend
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:3000"]}})

# Global variables for the model and data
model = None
scaler = None
model_accuracy = 0.0
feature_names = None
raw_data = None
raw_data_for_training = None

def load_data():
    """Loads and preprocesses the dataset. Uses sklearn breast cancer dataset as fallback."""
    global raw_data, raw_data_for_training, feature_names

    if raw_data is not None:
        logger.info("Data already loaded, skipping...")
        return

    try:
        # First try to load from CSV file
        base_dir = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(base_dir, 'data', 'data.csv')
        
        if os.path.exists(data_path):
            logger.info(f"Loading data from CSV: {data_path}")
            data = pd.read_csv(data_path)
            
            # Handle potential 'id' column
            if 'id' in data.columns:
                data = data.drop('id', axis=1)
            
            # Ensure we have diagnosis column
            if 'diagnosis' not in data.columns:
                raise ValueError("CSV must contain 'diagnosis' column")
                
        else:
            # Fallback to sklearn breast cancer dataset
            logger.warning("CSV file not found, using sklearn breast cancer dataset as fallback")
            sklearn_data = load_breast_cancer()
            
            # Create DataFrame from sklearn data
            data = pd.DataFrame(sklearn_data.data, columns=sklearn_data.feature_names)
            data['diagnosis'] = ['M' if target == 0 else 'B' for target in sklearn_data.target]
            
            logger.info(f"Loaded sklearn dataset with {len(data)} samples and {len(data.columns)-1} features")

        # Get the diagnosis column before converting other columns
        diagnosis_column = data['diagnosis']
        feature_data = data.drop('diagnosis', axis=1)

        # Convert feature columns to numeric, coercing errors to NaN
        for col in feature_data.columns:
            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
        
        # Drop feature columns that are all NaN
        feature_data = feature_data.dropna(axis=1, how='all')
        
        # Store feature names
        feature_names = feature_data.columns.tolist()
        
        # Add the diagnosis column back to the DataFrame
        feature_data['diagnosis'] = diagnosis_column

        # Store a version of the data for JSON serialization (NaN -> None)
        raw_data = feature_data.replace({np.nan: None})
        
        # Store a version for model training (with mean imputation)
        data_for_training = feature_data.copy()
        
        # Fill NaN with column means for numeric columns only
        numeric_columns = data_for_training.select_dtypes(include=[np.number]).columns
        data_for_training[numeric_columns] = data_for_training[numeric_columns].fillna(
            data_for_training[numeric_columns].mean()
        )
        
        raw_data_for_training = data_for_training
        
        logger.info(f"Data loading complete: {len(raw_data)} samples, {len(feature_names)} features")

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raw_data = None
        raw_data_for_training = None
        feature_names = None

def train_model(data_for_training=None):
    """Trains or re-trains the model on a given dataset."""
    global model, scaler, model_accuracy, feature_names

    if data_for_training is None:
        if raw_data_for_training is not None:
            data_for_training = raw_data_for_training.copy()
        else:
            logger.error("No data available for training.")
            model = None
            scaler = None
            return False

    try:
        # Map diagnosis to numerical values
        data_for_training = data_for_training.copy()
        data_for_training['diagnosis'] = data_for_training['diagnosis'].map({'M': 1, 'B': 0})
        
        # Remove any rows with NaN in diagnosis (invalid mappings)
        data_for_training = data_for_training.dropna(subset=['diagnosis'])
        
        X = data_for_training.drop('diagnosis', axis=1)
        y = data_for_training['diagnosis']

        if len(X) == 0:
            raise ValueError("No valid training samples after preprocessing")

        # Update feature names if not set
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        model_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model training complete. Accuracy: {model_accuracy:.4f}")
        return True

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        model = None
        scaler = None
        feature_names = None
        return False

@app.route('/')
def home():
    return jsonify({
        'message': 'Flask Backend is running!',
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': raw_data is not None,
        'feature_count': len(feature_names) if feature_names else 0,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions on new data."""
    if model is None or scaler is None or feature_names is None:
        return jsonify({
            'error': 'Model is not trained or loaded properly',
            'details': {
                'model_loaded': model is not None,
                'scaler_loaded': scaler is not None,
                'features_loaded': feature_names is not None
            }
        }), 503

    try:
        data = request.get_json(force=True)
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features in request body'}), 400
            
        input_features = data['features']
        
        if not isinstance(input_features, list):
            return jsonify({'error': 'Features must be a list'}), 400
            
        if len(input_features) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {len(input_features)}',
                'expected_features': feature_names
            }), 400

        # Convert to DataFrame for consistency
        features_df = pd.DataFrame([input_features], columns=feature_names)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        diagnosis = 'Malignant' if prediction[0] == 1 else 'Benign'
        confidence = float(np.max(probabilities)) * 100  # Convert to percentage
        
        logger.info(f"Prediction made: {diagnosis} with {confidence:.1f}% confidence")
        
        return jsonify({
            'diagnosis': diagnosis,
            'confidence': confidence,
            'accuracy': float(model_accuracy * 100),  # Convert to percentage
            'prediction_timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/data', methods=['GET'])
def get_data():
    """API endpoint to get dataset with optional filtering."""
    if raw_data is None:
        return jsonify({'error': 'Data not available'}), 503

    try:
        data = raw_data.copy()
        diagnosis_filter = request.args.get('diagnosis')
        limit = request.args.get('limit', type=int)
        
        if diagnosis_filter and diagnosis_filter in ['M', 'B']:
            data = data[data['diagnosis'] == diagnosis_filter]
            
        if limit and limit > 0:
            data = data.head(limit)

        return jsonify({
            'data': data.to_dict('records'),
            'total_count': len(data),
            'feature_names': feature_names,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Data retrieval error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """API endpoint to provide model status and statistics."""
    try:
        if model is None or scaler is None:
            return jsonify({
                'trained': False,
                'accuracy': 0,
                'total_samples': 0,
                'feature_count': 0,
                'error': 'Model not trained'
            })
        
        total_samples = len(raw_data) if raw_data is not None else 0
        
        return jsonify({
            'trained': True,
            'accuracy': float(model_accuracy * 100),  # Convert to percentage
            'total_samples': total_samples,
            'feature_count': len(feature_names) if feature_names else 0,
            'feature_names': feature_names,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def retrain_model():
    """API endpoint to retrain the model on a selected subset of the data."""
    if raw_data_for_training is None:
        return jsonify({'error': 'Training data not available'}), 503

    try:
        request_data = request.get_json(force=True) if request.is_json else {}
        selected_ids = request_data.get('ids', [])
        
        data_to_train = raw_data_for_training.copy()
        
        if selected_ids and 'id' in data_to_train.columns:
            # Filter the training data based on the provided IDs
            data_to_train = data_to_train[data_to_train['id'].isin(selected_ids)]
        
        # Call the train function with the selected data
        success = train_model(data_to_train)
        
        if success:
            return jsonify({
                'status': 'Model retraining completed successfully',
                'new_accuracy': float(model_accuracy * 100),
                'samples_used': len(data_to_train),
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                'status': 'Model retraining failed',
                'error': 'Training process encountered an error'
            }), 500

    except Exception as e:
        logger.error(f"Retraining error: {e}")
        return jsonify({'error': str(e)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize the application
def init_app():
    """Initialize the application with data and model."""
    logger.info("Initializing application...")
    load_data()
    if raw_data_for_training is not None:
        train_model()
        logger.info("Application initialization complete")
    else:
        logger.error("Failed to initialize application - no data available")

if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5000)