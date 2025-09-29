import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.datasets import load_breast_cancer
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

# Import our new training module
from model_trainer import BreastCancerModelTrainer

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
trainer = BreastCancerModelTrainer(n_features=10, test_size=0.2, random_state=42)
raw_data = None
raw_data_for_training = None
current_dataset_info = {'source': 'sklearn', 'filename': 'default_dataset'}


def clean_json_response(obj):
    """Convert NaN, Infinity to None for valid JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_json_response(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_response(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def allowed_file(filename):
    """Checks if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_data(data_path=None):
    """Loads the dataset. Uses sklearn breast cancer dataset as fallback."""
    global raw_data, raw_data_for_training, current_dataset_info

    if data_path:
        logger.info(f"Loading data from CSV: {data_path}")
        try:
            data = pd.read_csv(data_path)
            # Replace NaN with None immediately after loading
            data = data.replace([np.nan, np.inf, -np.inf], None)
            current_dataset_info = {'source': 'file', 'filename': os.path.basename(data_path)}
            logger.info(f"Loaded CSV with {len(data)} samples and {len(data.columns)} columns")
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
        # Store both raw and training versions
        raw_data = data.copy()
        raw_data_for_training = data.copy()
        
        logger.info(f"Data loading complete: {len(data)} samples")
        return True

    except Exception as e:
        logger.error(f"Error storing data: {e}")
        raw_data = None
        raw_data_for_training = None
        return False


def train_model(data_for_training=None):
    """Trains or re-trains the model using the new BreastCancerModelTrainer."""
    global trainer

    if data_for_training is None:
        if raw_data_for_training is not None:
            data_for_training = raw_data_for_training.copy()
        else:
            logger.error("No data available for training.")
            return False

    try:
        # Use the new trainer
        success = trainer.train(data_for_training)
        return success

    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
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
        'model_loaded': trainer.model is not None,
        'data_loaded': raw_data is not None,
        'feature_count': len(trainer.selected_feature_names),
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions on new data."""
    if trainer.model is None or trainer.scaler is None:
        return jsonify({
            'error': 'Model is not trained or loaded properly',
            'details': {
                'model_loaded': trainer.model is not None,
                'scaler_loaded': trainer.scaler is not None,
            }
        }), 503

    try:
        data = request.get_json(force=True)

        if 'features' not in data:
            return jsonify({'error': 'Missing features in request body'}), 400

        input_features = data['features']

        if not isinstance(input_features, list):
            return jsonify({'error': 'Features must be a list'}), 400

        # Use the trainer's predict method
        result = trainer.predict(input_features)
        
        # Add model accuracy to response
        result['accuracy'] = float(trainer.metrics.get('test_accuracy', 0) * 100)
        result['prediction_timestamp'] = datetime.utcnow().isoformat()
        
        logger.info(f"Prediction made: {result['diagnosis']} with {result['confidence']:.1f}% confidence")

        return jsonify(clean_json_response(result))

    except ValueError as e:
        logger.error(f"Prediction validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


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

        # Only apply limit if it's positive (0 or None = all data)
        if limit and limit > 0:
            data = data.head(limit)

        # Replace NaN values before converting to dict
        data = data.replace([np.nan, np.inf, -np.inf], None)

        return jsonify({
            'data': data.to_dict('records'),
            'total_count': len(raw_data),
            'feature_names': trainer.selected_feature_names,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Data retrieval error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/data/random-sample', methods=['GET'])
def get_random_sample():
    """API endpoint to get a random sample from the dataset."""
    if raw_data is None or len(raw_data) == 0:
        return jsonify({'error': 'No data available'}), 503
    
    try:
        # Get a random row
        random_row = raw_data.sample(n=1).iloc[0]
        
        # Extract features for the selected feature names
        features = []
        for feature_name in trainer.selected_feature_names:
            if feature_name in random_row:
                value = random_row[feature_name]
                # Convert to float, handle NaN
                if pd.isna(value):
                    features.append(0.0)
                else:
                    features.append(float(value))
            else:
                features.append(0.0)
        
        # Get diagnosis if available
        diagnosis = random_row.get('diagnosis', None)
        
        return jsonify({
            'features': features,
            'diagnosis': diagnosis,
            'feature_names': trainer.selected_feature_names
        })
    
    except Exception as e:
        logger.error(f"Random sample error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions from CSV upload."""
    if trainer.model is None or trainer.scaler is None:
        return jsonify({'error': 'Model not trained'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files allowed'}), 400
    
    try:
        # Read CSV
        data = pd.read_csv(file)
        logger.info(f"Batch prediction: loaded {len(data)} rows, {len(data.columns)} columns")
        
        # Extract only the features needed for prediction
        missing_features = []
        features_data = []
        
        for feature_name in trainer.selected_feature_names:
            if feature_name not in data.columns:
                missing_features.append(feature_name)
                # Use zeros for missing features
                features_data.append(np.zeros(len(data)))
            else:
                # Convert to numeric, fill NaN with 0
                col_data = pd.to_numeric(data[feature_name], errors='coerce').fillna(0).values
                features_data.append(col_data)
        
        if missing_features:
            logger.warning(f"Missing features in uploaded file: {missing_features}. Using zeros.")
        
        # Transpose to get (n_samples, n_features)
        features_array = np.array(features_data).T
        
        # Make predictions
        predictions = []
        for idx, features in enumerate(features_array):
            try:
                result = trainer.predict(features.tolist())
                predictions.append({
                    'row_index': idx,
                    'diagnosis': result['diagnosis'],
                    'confidence': result['confidence'],
                    'probability_benign': result['probability_benign'],
                    'probability_malignant': result['probability_malignant']
                })
            except Exception as e:
                logger.error(f"Prediction failed for row {idx}: {e}")
                predictions.append({
                    'row_index': idx,
                    'diagnosis': 'Error',
                    'confidence': 0,
                    'probability_benign': 0,
                    'probability_malignant': 0,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'total_predictions': len(predictions),
            'missing_features': missing_features,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """API endpoint to provide model status and comprehensive statistics."""
    try:
        if trainer.model is None:
            return jsonify({
                'trained': False,
                'accuracy': 0,
                'total_samples': 0,
                'feature_count': 0,
                'current_dataset': current_dataset_info,
                'error': 'Model not trained'
            })

        total_samples = len(raw_data) if raw_data is not None else 0

        # Get comprehensive model summary from trainer
        summary = trainer.get_model_summary()
        
        # Add additional context
        summary['accuracy'] = float(trainer.metrics.get('test_accuracy', 0) * 100)
        summary['total_samples'] = total_samples
        summary['current_dataset'] = current_dataset_info
        summary['timestamp'] = datetime.utcnow().isoformat()

        return jsonify(clean_json_response(summary))

    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def retrain_model():
    """API endpoint to retrain the model on a selected subset of the data."""
    if raw_data_for_training is None:
        return jsonify({'error': 'Training data not available'}), 503

    try:
        request_data = request.get_json(force=True) if request.is_json else {}
        selected_indices = request_data.get('indices', [])

        data_to_train = raw_data_for_training.copy()

        # Filter by row indices if provided
        if selected_indices and len(selected_indices) > 0:
            try:
                # Convert string indices to integers
                indices = [int(idx) for idx in selected_indices]
                # Filter by iloc position
                data_to_train = data_to_train.iloc[indices]
                logger.info(f"Training on {len(data_to_train)} selected samples (indices: {len(indices)}) out of {len(raw_data_for_training)}")
            except (ValueError, IndexError) as e:
                logger.error(f"Invalid indices provided: {e}")
                return jsonify({
                    'status': 'Model retraining failed',
                    'error': f'Invalid sample indices provided'
                }), 400
        else:
            # Train on ALL data if no selection
            logger.info(f"Training on all {len(data_to_train)} samples")

        if len(data_to_train) < 10:
            return jsonify({
                'status': 'Model retraining failed',
                'error': f'Insufficient samples for training. Need at least 10, got {len(data_to_train)}. Available in dataset: {len(raw_data_for_training)}'
            }), 400

        success = train_model(data_to_train)

        if success:
            return jsonify(clean_json_response({
                'status': 'Model retraining completed successfully',
                'new_accuracy': float(trainer.metrics.get('test_accuracy', 0) * 100),
                'samples_used': len(data_to_train),
                'metrics': trainer.metrics,
                'feature_importance': trainer.get_feature_importance_data(),
                'confusion_matrix': trainer.get_confusion_matrix_data(),
                'current_dataset': current_dataset_info,
                'timestamp': datetime.utcnow().isoformat()
            }))
        else:
            return jsonify({
                'status': 'Model retraining failed',
                'error': 'Training process encountered an error'
            }), 500

    except Exception as e:
        logger.error(f"Retraining error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


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
                train_success = train_model()
                
                if train_success:
                    return jsonify(clean_json_response({
                        'success': True,
                        'message': 'File uploaded and model retrained successfully',
                        'filename': filename,
                        'columns': trainer.selected_feature_names,
                        'metrics': trainer.metrics,
                        'feature_importance': trainer.get_feature_importance_data(),
                    })), 200
                else:
                    return jsonify({
                        'success': False,
                        'message': 'File uploaded but model training failed',
                        'error': 'Check logs for details'
                    }), 500
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to process the uploaded file'
                }), 500
                
        except Exception as e:
            logger.error(f"File upload or processing error: {e}", exc_info=True)
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
        
        # Replace NaN before analysis
        data_clean = data.replace([np.nan, np.inf, -np.inf], None)
        
        # Data profiling
        column_types = {col: str(data[col].dtype) for col in data.columns}
        missing_values = {col: int(data[col].isnull().sum()) for col in data.columns}
        sample_data = data_clean.head(5).to_dict('records')
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
        logger.error(f"File analysis error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
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