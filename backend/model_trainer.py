"""
Advanced Model Training Module for Breast Cancer Prediction
Handles preprocessing, feature selection, training, and metrics calculation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class BreastCancerModelTrainer:
    """
    Comprehensive model trainer for breast cancer prediction.
    Handles data preprocessing, feature selection, training, and metrics calculation.
    """
    
    def __init__(self, n_features: int = 10, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            n_features: Number of top features to select (default: 10)
            test_size: Proportion of dataset for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.n_features = n_features
        self.test_size = test_size
        self.random_state = random_state
        
        # Model components
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_selector: Optional[SelectKBest] = None
        
        # Training artifacts
        self.selected_feature_names: List[str] = []
        self.feature_importance_scores: Dict[str, float] = {}
        self.all_feature_scores: Dict[str, float] = {}
        
        # Metrics
        self.metrics: Dict[str, Any] = {}
        self.confusion_matrix_data: Optional[np.ndarray] = None
        
        # Training data info
        self.X_train_original: Optional[pd.DataFrame] = None
        self.X_test_original: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the dataset with robust handling of missing values and data quality issues.
        
        Args:
            data: Raw dataset with 'diagnosis' column
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info(f"Starting preprocessing. Initial shape: {data.shape}")
        
        # Create a copy to avoid modifying original
        df = data.copy()
        
        # Check for diagnosis column
        if 'diagnosis' not in df.columns:
            raise ValueError("Dataset must contain a 'diagnosis' column")
        
        # Remove ID column if present (not useful for prediction)
        if 'id' in df.columns:
            logger.info("Removing 'id' column from features")
            df = df.drop('id', axis=1)
        
        # Separate features and labels
        y = df['diagnosis'].copy()
        X = df.drop('diagnosis', axis=1)
        
        # Convert all feature columns to numeric
        logger.info("Converting features to numeric...")
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Log missing value statistics before handling
        missing_before = X.isnull().sum()
        if missing_before.sum() > 0:
            logger.warning(f"Found {missing_before.sum()} missing values across {(missing_before > 0).sum()} columns")
        
        # Drop columns with >50% missing values
        threshold = len(X) * 0.5
        cols_to_drop = X.columns[X.isnull().sum() > threshold].tolist()
        if cols_to_drop:
            logger.warning(f"Dropping {len(cols_to_drop)} columns with >50% missing values: {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)
        
        # Drop columns that are all NaN
        X = X.dropna(axis=1, how='all')
        
        # Fill remaining missing values with column mean
        for col in X.columns:
            if X[col].isnull().any():
                mean_val = X[col].mean()
                X[col].fillna(mean_val, inplace=True)
                logger.info(f"Filled {missing_before[col]} missing values in '{col}' with mean: {mean_val:.4f}")
        
        # Drop rows with invalid diagnosis labels
        valid_labels = ['M', 'B', 1, 0, '1', '0']
        invalid_mask = ~y.isin(valid_labels)
        if invalid_mask.any():
            logger.warning(f"Dropping {invalid_mask.sum()} rows with invalid diagnosis labels")
            X = X[~invalid_mask]
            y = y[~invalid_mask]
        
        # Standardize diagnosis labels to binary (1=Malignant, 0=Benign)
        y = y.map({'M': 1, 'B': 0, 1: 1, 0: 0, '1': 1, '0': 0})
        
        # Final validation
        if len(X) == 0:
            raise ValueError("No valid samples remaining after preprocessing")
        
        if X.isnull().any().any():
            raise ValueError("Preprocessing failed: NaN values still present")
        
        logger.info(f"Preprocessing complete. Final shape: {X.shape}, Labels: {len(y)}")
        logger.info(f"Class distribution - Malignant: {(y == 1).sum()}, Benign: {(y == 0).sum()}")
        
        return X, y
    
    def select_top_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Select top N most important features using ANOVA F-statistic.
        
        Args:
            X: Feature dataframe
            y: Target labels
            
        Returns:
            DataFrame with only selected features
        """
        n_available_features = X.shape[1]
        n_to_select = min(self.n_features, n_available_features)
        
        if n_to_select >= n_available_features:
            logger.warning(f"Requested {self.n_features} features but only {n_available_features} available. Using all.")
            self.selected_feature_names = X.columns.tolist()
            self.feature_selector = None
            
            # Calculate scores for all features anyway
            selector_temp = SelectKBest(f_classif, k='all')
            selector_temp.fit(X, y)
            self.all_feature_scores = dict(zip(X.columns, selector_temp.scores_))
            self.feature_importance_scores = self.all_feature_scores.copy()
            
            return X
        
        # Perform feature selection
        logger.info(f"Selecting top {n_to_select} features from {n_available_features}...")
        self.feature_selector = SelectKBest(f_classif, k=n_to_select)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names and their scores
        mask = self.feature_selector.get_support()
        self.selected_feature_names = X.columns[mask].tolist()
        
        # Store all feature scores (for potential analysis)
        self.all_feature_scores = dict(zip(X.columns, self.feature_selector.scores_))
        
        # Store selected feature scores
        self.feature_importance_scores = {
            name: score for name, score in self.all_feature_scores.items() 
            if name in self.selected_feature_names
        }
        
        logger.info(f"Selected features: {self.selected_feature_names}")
        logger.info(f"Feature importance scores: {self.feature_importance_scores}")
        
        return pd.DataFrame(X_selected, columns=self.selected_feature_names)
    
    def train(self, data: pd.DataFrame) -> bool:
        """
        Complete training pipeline: preprocess, select features, train model, calculate metrics.
        
        Args:
            data: Raw dataset with 'diagnosis' column
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING MODEL TRAINING PIPELINE")
            logger.info("=" * 80)
            
            # Step 1: Preprocess data
            X, y = self.preprocess_data(data)
            
            # Step 2: Select top features
            X_selected = self.select_top_features(X, y)
            
            # Step 3: Split data
            logger.info(f"Splitting data: {100 * (1 - self.test_size):.0f}% train, {100 * self.test_size:.0f}% test")
            self.X_train_original, self.X_test_original, self.y_train, self.y_test = train_test_split(
                X_selected, y, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y  # Maintain class distribution
            )
            
            logger.info(f"Train set: {len(self.X_train_original)} samples")
            logger.info(f"Test set: {len(self.X_test_original)} samples")
            
            # Step 4: Scale features
            logger.info("Scaling features with StandardScaler...")
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train_original)
            X_test_scaled = self.scaler.transform(self.X_test_original)
            
            # Step 5: Train model
            logger.info("Training Logistic Regression model...")
            self.model = LogisticRegression(
                solver='liblinear',
                random_state=self.random_state,
                max_iter=1000,
                C=1.0  # Regularization parameter
            )
            self.model.fit(X_train_scaled, self.y_train)
            
            # Step 6: Calculate metrics
            logger.info("Calculating performance metrics...")
            self._calculate_metrics(X_train_scaled, X_test_scaled)
            
            logger.info("=" * 80)
            logger.info(f"TRAINING COMPLETE - Accuracy: {self.metrics['test_accuracy']:.4f}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self._reset_state()
            return False
    
    def _calculate_metrics(self, X_train_scaled: np.ndarray, X_test_scaled: np.ndarray):
        """Calculate comprehensive model performance metrics."""
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Probabilities for ROC-AUC
        y_train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Confusion Matrix
        self.confusion_matrix_data = confusion_matrix(self.y_test, y_test_pred)
        
        # Comprehensive metrics
        self.metrics = {
            # Accuracy
            'train_accuracy': float(accuracy_score(self.y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(self.y_test, y_test_pred)),
            
            # Precision, Recall, F1 (for both classes)
            'precision': float(precision_score(self.y_test, y_test_pred, average='binary')),
            'recall': float(recall_score(self.y_test, y_test_pred, average='binary')),
            'f1_score': float(f1_score(self.y_test, y_test_pred, average='binary')),
            
            # ROC-AUC
            'train_roc_auc': float(roc_auc_score(self.y_train, y_train_proba)),
            'test_roc_auc': float(roc_auc_score(self.y_test, y_test_proba)),
            
            # Confusion Matrix breakdown
            'true_negatives': int(self.confusion_matrix_data[0, 0]),
            'false_positives': int(self.confusion_matrix_data[0, 1]),
            'false_negatives': int(self.confusion_matrix_data[1, 0]),
            'true_positives': int(self.confusion_matrix_data[1, 1]),
            
            # Sample counts
            'train_samples': len(self.y_train),
            'test_samples': len(self.y_test),
            'total_samples': len(self.y_train) + len(self.y_test),
        }
        
        logger.info(f"Train Accuracy: {self.metrics['train_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {self.metrics['test_accuracy']:.4f}")
        logger.info(f"Precision: {self.metrics['precision']:.4f}")
        logger.info(f"Recall: {self.metrics['recall']:.4f}")
        logger.info(f"F1 Score: {self.metrics['f1_score']:.4f}")
        logger.info(f"Test ROC-AUC: {self.metrics['test_roc_auc']:.4f}")
        logger.info(f"Confusion Matrix:\n{self.confusion_matrix_data}")
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Make a prediction on new data.
        
        Args:
            features: List of feature values in the same order as selected_feature_names
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if len(features) != len(self.selected_feature_names):
            raise ValueError(
                f"Expected {len(self.selected_feature_names)} features, got {len(features)}. "
                f"Required features: {self.selected_feature_names}"
            )
        
        # Create DataFrame with proper feature names
        features_df = pd.DataFrame([features], columns=self.selected_feature_names)
        
        # Scale
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        diagnosis = 'Malignant' if prediction == 1 else 'Benign'
        confidence = float(np.max(probabilities)) * 100
        
        return {
            'diagnosis': diagnosis,
            'confidence': confidence,
            'probability_benign': float(probabilities[0]),
            'probability_malignant': float(probabilities[1]),
        }
    
    def get_feature_importance_data(self) -> List[Dict[str, Any]]:
        """
        Get feature importance data formatted for frontend visualization.
        
        Returns:
            List of dictionaries with feature names and importance scores
        """
        if not self.feature_importance_scores:
            return []
        
        # Sort by importance (highest first)
        sorted_features = sorted(
            self.feature_importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                'feature': name,
                'importance': float(score),
                'rank': idx + 1
            }
            for idx, (name, score) in enumerate(sorted_features)
        ]
    
    def get_confusion_matrix_data(self) -> Dict[str, Any]:
        """
        Get confusion matrix data formatted for frontend visualization.
        
        Returns:
            Dictionary with confusion matrix data and labels
        """
        if self.confusion_matrix_data is None:
            return {}
        
        return {
            'matrix': self.confusion_matrix_data.tolist(),
            'labels': ['Benign', 'Malignant'],
            'values': {
                'true_negatives': int(self.confusion_matrix_data[0, 0]),
                'false_positives': int(self.confusion_matrix_data[0, 1]),
                'false_negatives': int(self.confusion_matrix_data[1, 0]),
                'true_positives': int(self.confusion_matrix_data[1, 1]),
            }
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary for API responses.
        
        Returns:
            Dictionary with all model information
        """
        return {
            'trained': self.model is not None,
            'n_features': len(self.selected_feature_names),
            'feature_names': self.selected_feature_names,
            'metrics': self.metrics,
            'feature_importance': self.get_feature_importance_data(),
            'confusion_matrix': self.get_confusion_matrix_data(),
        }
    
    def _reset_state(self):
        """Reset all model state (used after training failure)."""
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_feature_names = []
        self.feature_importance_scores = {}
        self.all_feature_scores = {}
        self.metrics = {}
        self.confusion_matrix_data = None