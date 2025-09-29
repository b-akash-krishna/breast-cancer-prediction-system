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
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import json

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

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global variables for the model and data
model = None
scaler = None
selector = None
model_accuracy = 0.0
feature_names = None
raw_data = None
raw_data_for_training = None
current_dataset_info = {'source': 'sklearn', 'filename': 'default_dataset'}


def allowed_file(filename):
    """Checks if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_data(data_path=None):
    """Loads and preprocesses the dataset. Uses sklearn breast cancer dataset as fallback."""
    global raw_data, raw_data_for_training, feature_names, current_dataset_info

    if data_path:
        logger.info(f"Loading data from CSV: {data_path}")
        try:
            data = pd.read_csv(data_path)
            current_dataset_info = {'source': 'file', 'filename': os.path.basename(data_path)}
        except Exception as e:
            logger.error(f"Error loading CSV data from {data_path}: {e}")
            return False
    else:
        # Fallback to sklearn breast cancer dataset
        logger.warning("No CSV file provided, using sklearn breast cancer dataset as fallback")
        sklearn_data = load_breast_cancer()
        data = pd.DataFrame(sklearn_data.data, columns=sklearn_data.feature_names)
        data['diagnosis'] = ['M' if target == 0 else 'B' for target in sklearn_data.target]
        current_dataset_info = {'source': 'sklearn', 'filename': 'default_dataset'}
        logger.info(f"Loaded sklearn dataset with {len(data)} samples and {len(data.columns) - 1} features")

    try:
        # Check for diagnosis column
        if 'diagnosis' not in data.columns:
            raise ValueError("Dataset must contain a 'diagnosis' column")

        # Handle potential 'id' column
        if 'id' in data.columns:
            # Store the id column for later reference but drop from training features
            data = data.drop('id', axis=1)

        # Separate diagnosis and features
        diagnosis_column = data['diagnosis']
        feature_data = data.drop('diagnosis', axis=1)

        # Convert feature columns to numeric, coercing errors
        for col in feature_data.columns:
            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')

        # Drop feature columns that are all NaN or have too many missing values
        feature_data = feature_data.dropna(axis=1, how='all')
        feature_data = feature_data.dropna(axis=1, thresh=len(feature_data) * 0.5)

        # Fill remaining NaN with the mean of each column
        numeric_columns = feature_data.select_dtypes(include=[np.number]).columns
        feature_data[numeric_columns] = feature_data[numeric_columns].fillna(feature_data[numeric_columns].mean())

        # Combine features and diagnosis back for storage
        processed_data = feature_data.copy()
        processed_data['diagnosis'] = diagnosis_column
        raw_data = processed_data.replace({np.nan: None})
        raw_data_for_training = processed_data
        
        feature_names = raw_data_for_training.drop('diagnosis', axis=1).columns.tolist()

        logger.info(f"Data loading complete: {len(raw_data_for_training)} samples, {len(feature_names)} features")
        return True

    except Exception as e:
        logger.error(f"Error loading and processing data: {e}")
        raw_data = None
        raw_data_for_training = None
        feature_names = None
        return False


def select_features(X, y, k=15):
    """Selects the top k features using f_classif."""
    global selector

    if k >= X.shape[1]:
        logger.warning(f"Number of features to select ({k}) is greater than or equal to total features ({X.shape[1]}). Skipping feature selection.")
        selector = None
        return X, X.columns.tolist()

    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    mask = selector.get_support()
    selected_features = X.columns[mask].tolist()
    
    logger.info(f"Selected {len(selected_features)} features.")
    return pd.DataFrame(X_new, columns=selected_features), selected_features


def train_model(data_for_training=None):
    """Trains or re-trains the model on a given dataset."""
    global model, scaler, model_accuracy, feature_names, selector

    if data_for_training is None:
        if raw_data_for_training is not None:
            data_for_training = raw_data_for_training.copy()
        else:
            logger.error("No data available for training.")
            model = None
            scaler = None
            selector = None
            return False

    try:
        data_for_training = data_for_training.copy()
        data_for_training['diagnosis'] = data_for_training['diagnosis'].map({'M': 1, 'B': 0})
        data_for_training = data_for_training.dropna(subset=['diagnosis'])

        X = data_for_training.drop('diagnosis', axis=1)
        y = data_for_training['diagnosis']

        if len(X) == 0:
            raise ValueError("No valid training samples after preprocessing")

        # Feature Selection
        X_selected, selected_features = select_features(X, y, k=15)
        feature_names = selected_features

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
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
        selector = None
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

        # Create a DataFrame with the correct column names for prediction
        # Ensure the input features are in the same order as the trained features
        if len(input_features) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, but got {len(input_features)}. '
                         f'Please provide values for the following features: {", ".join(feature_names)}'
            }), 400

        features_df = pd.DataFrame([input_features], columns=feature_names)

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Make prediction
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)

        diagnosis = 'Malignant' if prediction[0] == 1 else 'Benign'
        confidence = float(np.max(probabilities)) * 100
        logger.info(f"Prediction made: {diagnosis} with {confidence:.1f}% confidence")

        return jsonify({
            'diagnosis': diagnosis,
            'confidence': confidence,
            'accuracy': float(model_accuracy * 100),
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
            'total_count': len(raw_data),
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
                'current_dataset': current_dataset_info,
                'error': 'Model not trained'
            })

        total_samples = len(raw_data) if raw_data is not None else 0

        return jsonify({
            'trained': True,
            'accuracy': float(model_accuracy * 100),
            'total_samples': total_samples,
            'feature_count': len(feature_names) if feature_names else 0,
            'feature_names': feature_names,
            'current_dataset': current_dataset_info,
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
            data_to_train = data_to_train[data_to_train['id'].isin(selected_ids)]

        success = train_model(data_to_train)

        if success:
            return jsonify({
                'status': 'Model retraining completed successfully',
                'new_accuracy': float(model_accuracy * 100),
                'samples_used': len(data_to_train),
                'current_dataset': current_dataset_info,
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


@app.route('/upload', methods=['POST'])
def upload_file():
    """API endpoint to upload a new dataset and retrain the model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            logger.info(f"File saved successfully at {filepath}")
            success = load_data(data_path=filepath)
            if success:
                train_model()
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and model retrained successfully',
                    'filename': filename,
                    'columns': feature_names
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to process the uploaded file'
                }), 500
        except Exception as e:
            logger.error(f"File upload or processing error: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500
    else:
        return jsonify({'error': 'Allowed file types are csv'}), 400


@app.route('/files', methods=['GET'])
def list_files():
    """API endpoint to list all available uploaded files."""
    try:
        files = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                stat = os.stat(filepath)
                files.append({
                    'filename': filename,
                    'upload_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'size': stat.st_size,
                })
        return jsonify({'files': files}), 200
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    """API endpoint to delete an uploaded file."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return jsonify({'success': True, 'message': f'File {filename} deleted successfully'}), 200
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    else:
        return jsonify({'success': False, 'message': 'File not found'}), 404


@app.route('/files/<filename>/analyze', methods=['GET'])
def analyze_file(filename):
    """API endpoint to analyze a specific uploaded file."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        data = pd.read_csv(filepath)
        
        # Data profiling
        column_types = {col: str(data[col].dtype) for col in data.columns}
        missing_values = {col: data[col].isnull().sum() for col in data.columns}
        sample_data = data.head(5).to_dict('records')
        total_rows = len(data)
        columns = data.columns.tolist()

        return jsonify({
            'filename': filename,
            'columns': columns,
            'total_rows': total_rows,
            'sample_data': sample_data,
            'column_types': column_types,
            'missing_values': missing_values,
        }), 200

    except Exception as e:
        logger.error(f"File analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413


# Initialize the application
def init_app():
    """Initialize the application with data and model."""
    logger.info("Initializing application...")
    success = load_data()
    if success and raw_data_for_training is not None:
        train_model()
        logger.info("Application initialization complete")
    else:
        logger.error("Failed to initialize application - no data available")


if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5000)