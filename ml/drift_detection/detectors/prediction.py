"""
Prediction drift detection for monitoring changes in model predictions.
"""

from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
import warnings

from ml.drift_detection.detectors.base import DriftDetector, DriftResult, DriftStatus


class PredictionDriftDetector(DriftDetector):
    """
    Detector for changes in model predictions.
    
    This detector monitors shifts in:
    1. Prediction distribution (e.g., class probabilities)
    2. Confidence scores
    3. Error rates or metrics
    
    This can help identify concept drift even when input features 
    appear stable, but the relationship between features and targets
    has changed.
    """
    
    def __init__(
        self,
        reference_predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        reference_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        method: str = "distribution",
        warning_threshold: float = 0.1,
        drift_threshold: float = 0.2,
        model: Optional[Any] = None,
        predict_fn: Optional[Callable] = None,
        task_type: str = "classification"
    ):
        """
        Initialize prediction drift detector.
        
        Args:
            reference_predictions: Model predictions on reference data
            reference_labels: True labels for reference data
            method: Detection method ('distribution', 'confidence', 'error')
            warning_threshold: Threshold for warnings
            drift_threshold: Threshold for drift detection
            model: Model object (optional)
            predict_fn: Function to get predictions from model (optional)
            task_type: 'classification' or 'regression'
        """
        self.method = method.lower()
        self.task_type = task_type.lower()
        self.model = model
        self.predict_fn = predict_fn
        self._reference_predictions = None
        self._reference_labels = None
        
        # Check if method is supported
        supported_methods = ["distribution", "confidence", "error"]
        if self.method not in supported_methods:
            raise ValueError(f"Unsupported method: {method}. "
                           f"Choose from: {', '.join(supported_methods)}")
            
        # Store reference predictions if provided
        if reference_predictions is not None:
            if isinstance(reference_predictions, torch.Tensor):
                reference_predictions = reference_predictions.detach().cpu().numpy()
            self._reference_predictions = reference_predictions
            
        # Store reference labels if provided
        if reference_labels is not None:
            if isinstance(reference_labels, torch.Tensor):
                reference_labels = reference_labels.detach().cpu().numpy()
            self._reference_labels = reference_labels
            
        # For error method, we need both predictions and labels
        if self.method == "error" and (self._reference_predictions is None or self._reference_labels is None):
            raise ValueError("Both reference predictions and labels must be provided for error method")
            
        # Initialize with no reference data (we don't use input data for this detector)
        super().__init__(None, warning_threshold, drift_threshold)
        
        # Compute reference statistics if we have predictions
        if self._reference_predictions is not None:
            self._compute_reference_statistics()
    
    def set_reference_predictions(
        self, 
        predictions: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> None:
        """
        Set reference predictions.
        
        Args:
            predictions: Model predictions on reference data
            labels: True labels for reference data
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
            
        self._reference_predictions = predictions
        
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            self._reference_labels = labels
            
        self._compute_reference_statistics()
    
    def _compute_reference_statistics(self) -> None:
        """
        Compute statistics on reference predictions.
        """
        if self._reference_predictions is None:
            return
        
        self._reference_statistics = {}
        
        if self.method == "distribution":
            # For classification, compute class distribution
            if self.task_type == "classification":
                if len(self._reference_predictions.shape) > 1:
                    # We have class probabilities
                    class_probs = self._reference_predictions
                    self._reference_statistics["class_distribution"] = np.mean(class_probs, axis=0)
                else:
                    # We have class predictions
                    class_preds = self._reference_predictions.astype(int)
                    n_classes = max(np.max(class_preds) + 1, 2)  # At least binary
                    class_counts = np.bincount(class_preds, minlength=n_classes)
                    self._reference_statistics["class_distribution"] = class_counts / len(class_preds)
            
            # For regression, compute value distribution
            else:
                # Compute histogram
                hist, edges = np.histogram(self._reference_predictions, bins=20, density=True)
                self._reference_statistics["value_distribution"] = hist
                self._reference_statistics["bin_edges"] = edges
                
                # Also compute basic statistics
                self._reference_statistics["mean"] = np.mean(self._reference_predictions)
                self._reference_statistics["std"] = np.std(self._reference_predictions)
                self._reference_statistics["min"] = np.min(self._reference_predictions)
                self._reference_statistics["max"] = np.max(self._reference_predictions)
                
        elif self.method == "confidence":
            # Compute confidence statistics
            if self.task_type == "classification" and len(self._reference_predictions.shape) > 1:
                # Get max probability for each prediction
                confidence = np.max(self._reference_predictions, axis=1)
                self._reference_statistics["mean_confidence"] = np.mean(confidence)
                self._reference_statistics["confidence_hist"], self._reference_statistics["confidence_bins"] = \
                    np.histogram(confidence, bins=10, range=(0, 1), density=True)
            else:
                # For regression or class predictions, confidence doesn't apply
                warnings.warn("Confidence method only applicable to classification with probabilities")
                
        elif self.method == "error":
            if self._reference_labels is None:
                return
                
            # Compute error metrics
            if self.task_type == "classification":
                if len(self._reference_predictions.shape) > 1:
                    # Convert probabilities to class predictions
                    preds = np.argmax(self._reference_predictions, axis=1)
                else:
                    preds = self._reference_predictions.astype(int)
                    
                # Compute accuracy
                accuracy = np.mean(preds == self._reference_labels)
                self._reference_statistics["accuracy"] = accuracy
                
                # Compute per-class metrics if multi-class
                n_classes = max(np.max(preds), np.max(self._reference_labels)) + 1
                if n_classes > 2:
                    per_class_acc = {}
                    for c in range(n_classes):
                        mask = self._reference_labels == c
                        if np.sum(mask) > 0:
                            per_class_acc[c] = np.mean(preds[mask] == c)
                    self._reference_statistics["per_class_accuracy"] = per_class_acc
                    
            else:  # regression
                # Compute MSE
                mse = np.mean((self._reference_predictions - self._reference_labels) ** 2)
                self._reference_statistics["mse"] = mse
                
                # Compute MAE
                mae = np.mean(np.abs(self._reference_predictions - self._reference_labels))
                self._reference_statistics["mae"] = mae
    
    def detect(
        self, 
        predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> DriftResult:
        """
        Detect prediction drift.
        
        Args:
            predictions: Current model predictions
            data: Input data (used with model to get predictions if provided)
            labels: True labels (for error method)
            
        Returns:
            DriftResult containing drift metrics
        """
        if self._reference_predictions is None:
            raise ValueError("Reference predictions not set. Call set_reference_predictions() first.")
        
        # Get predictions
        if predictions is None and data is not None and (self.model is not None or self.predict_fn is not None):
            # Generate predictions using model or predict function
            if self.predict_fn is not None:
                predictions = self.predict_fn(data)
            else:
                if isinstance(data, np.ndarray):
                    data_tensor = torch.tensor(data)
                else:
                    data_tensor = data
                    
                with torch.no_grad():
                    predictions = self.model(data_tensor)
                    
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.detach().cpu().numpy()
        
        if predictions is None:
            raise ValueError("Either predictions or both data and model/predict_fn must be provided")
            
        # Convert torch tensors to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
            
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            
        # For error method, we need labels
        if self.method == "error" and labels is None:
            raise ValueError("Labels must be provided for error method")
            
        if self.method == "distribution":
            return self._detect_distribution_drift(predictions)
        elif self.method == "confidence":
            return self._detect_confidence_drift(predictions)
        elif self.method == "error":
            return self._detect_error_drift(predictions, labels)
    
    def _detect_distribution_drift(self, predictions: np.ndarray) -> DriftResult:
        """
        Detect drift in prediction distribution.
        
        Args:
            predictions: Current predictions
            
        Returns:
            DriftResult for distribution-based drift
        """
        if self.task_type == "classification":
            # Get class distributions
            if len(predictions.shape) > 1:
                # We have class probabilities
                current_dist = np.mean(predictions, axis=0)
            else:
                # We have class predictions
                class_preds = predictions.astype(int)
                n_classes = max(np.max(class_preds) + 1, 2)  # At least binary
                class_counts = np.bincount(class_preds, minlength=n_classes)
                current_dist = class_counts / len(class_preds)
                
            ref_dist = self._reference_statistics.get("class_distribution")
            
            # Ensure distributions are the same length
            if len(current_dist) != len(ref_dist):
                # Pad the shorter one
                max_len = max(len(current_dist), len(ref_dist))
                if len(current_dist) < max_len:
                    current_dist = np.pad(current_dist, (0, max_len - len(current_dist)))
                if len(ref_dist) < max_len:
                    ref_dist = np.pad(ref_dist, (0, max_len - len(ref_dist)))
            
            # Compute Jensen-Shannon distance
            drift_score = jensenshannon(current_dist, ref_dist)
            
            # Create per-class drift scores
            feature_drifts = {}
            for i in range(len(ref_dist)):
                feature_drifts[f"class_{i}"] = abs(current_dist[i] - ref_dist[i])
                
            details = {
                "reference_distribution": ref_dist.tolist(),
                "current_distribution": current_dist.tolist(),
                "class_differences": [abs(c - r) for c, r in zip(current_dist, ref_dist)]
            }
            
        else:  # regression
            # Get current value distribution
            ref_edges = self._reference_statistics.get("bin_edges")
            current_hist, _ = np.histogram(predictions, bins=ref_edges, density=True)
            ref_hist = self._reference_statistics.get("value_distribution")
            
            # Compute Jensen-Shannon distance
            drift_score = jensenshannon(current_hist, ref_hist)
            
            # No per-feature drifts for regression
            feature_drifts = {"overall": drift_score}
            
            # Compute basic statistics
            current_mean = np.mean(predictions)
            current_std = np.std(predictions)
            
            details = {
                "reference_stats": {
                    "mean": self._reference_statistics.get("mean"),
                    "std": self._reference_statistics.get("std"),
                    "min": self._reference_statistics.get("min"),
                    "max": self._reference_statistics.get("max")
                },
                "current_stats": {
                    "mean": current_mean,
                    "std": current_std,
                    "min": np.min(predictions),
                    "max": np.max(predictions)
                },
                "mean_difference": abs(current_mean - self._reference_statistics.get("mean")),
                "std_difference": abs(current_std - self._reference_statistics.get("std"))
            }
            
        # Determine drift status
        status = DriftStatus.NO_DRIFT
        if drift_score > self.drift_threshold:
            status = DriftStatus.DRIFT
        elif drift_score > self.warning_threshold:
            status = DriftStatus.WARNING
            
        return DriftResult(
            status=status,
            p_value=None,
            drift_score=drift_score,
            metric_name="js_distribution_distance",
            threshold=self.drift_threshold,
            feature_drifts=feature_drifts,
            details=details
        )
    
    def _detect_confidence_drift(self, predictions: np.ndarray) -> DriftResult:
        """
        Detect drift in prediction confidence.
        
        Args:
            predictions: Current predictions
            
        Returns:
            DriftResult for confidence-based drift
        """
        if self.task_type != "classification" or len(predictions.shape) <= 1:
            raise ValueError("Confidence method only applicable to classification with probabilities")
            
        # Get confidence values
        confidence = np.max(predictions, axis=1)
        current_mean_conf = np.mean(confidence)
        ref_mean_conf = self._reference_statistics.get("mean_confidence")
        
        # Compute confidence distribution
        ref_bins = self._reference_statistics.get("confidence_bins")
        current_hist, _ = np.histogram(confidence, bins=ref_bins, density=True)
        ref_hist = self._reference_statistics.get("confidence_hist")
        
        # Compute histogram distance
        drift_score = jensenshannon(current_hist, ref_hist)
        
        # Determine drift status
        status = DriftStatus.NO_DRIFT
        if drift_score > self.drift_threshold:
            status = DriftStatus.DRIFT
        elif drift_score > self.warning_threshold:
            status = DriftStatus.WARNING
            
        # Create detailed results
        details = {
            "reference_mean_confidence": ref_mean_conf,
            "current_mean_confidence": current_mean_conf,
            "mean_confidence_change": abs(current_mean_conf - ref_mean_conf),
            "confidence_decreasing": current_mean_conf < ref_mean_conf
        }
        
        return DriftResult(
            status=status,
            p_value=None,
            drift_score=drift_score,
            metric_name="confidence_distribution_change",
            threshold=self.drift_threshold,
            feature_drifts={"mean_confidence": abs(current_mean_conf - ref_mean_conf)},
            details=details
        )
    
    def _detect_error_drift(self, predictions: np.ndarray, labels: np.ndarray) -> DriftResult:
        """
        Detect drift in error rates.
        
        Args:
            predictions: Current predictions
            labels: True labels
            
        Returns:
            DriftResult for error-based drift
        """
        if self.task_type == "classification":
            if len(predictions.shape) > 1:
                # Convert probabilities to class predictions
                preds = np.argmax(predictions, axis=1)
            else:
                preds = predictions.astype(int)
                
            # Compute accuracy
            current_accuracy = np.mean(preds == labels)
            ref_accuracy = self._reference_statistics.get("accuracy")
            
            # Compute accuracy change
            drift_score = abs(ref_accuracy - current_accuracy)
            
            # Compute per-class metrics if multi-class
            feature_drifts = {}
            per_class_details = {}
            
            ref_per_class = self._reference_statistics.get("per_class_accuracy", {})
            
            n_classes = max(np.max(preds), np.max(labels)) + 1
            if n_classes > 2:
                for c in range(n_classes):
                    mask = labels == c
                    if np.sum(mask) > 0:
                        current_class_acc = np.mean(preds[mask] == c)
                        ref_class_acc = ref_per_class.get(c, 0)
                        class_drift = abs(current_class_acc - ref_class_acc)
                        feature_drifts[f"class_{c}"] = class_drift
                        per_class_details[f"class_{c}"] = {
                            "reference_accuracy": ref_class_acc,
                            "current_accuracy": current_class_acc,
                            "accuracy_change": class_drift
                        }
            
            details = {
                "reference_accuracy": ref_accuracy,
                "current_accuracy": current_accuracy,
                "accuracy_change": drift_score,
                "accuracy_improving": current_accuracy > ref_accuracy,
                "per_class": per_class_details
            }
            
        else:  # regression
            # Compute MSE
            current_mse = np.mean((predictions - labels) ** 2)
            ref_mse = self._reference_statistics.get("mse")
            
            # Compute MAE
            current_mae = np.mean(np.abs(predictions - labels))
            ref_mae = self._reference_statistics.get("mae")
            
            # Use relative change in MSE as drift score
            if ref_mse > 0:
                mse_change = abs(current_mse - ref_mse) / ref_mse
            else:
                mse_change = abs(current_mse - ref_mse)
                
            drift_score = mse_change
            
            feature_drifts = {
                "mse": mse_change,
                "mae": abs(current_mae - ref_mae) / max(ref_mae, 1e-10)
            }
            
            details = {
                "reference_metrics": {
                    "mse": ref_mse,
                    "mae": ref_mae
                },
                "current_metrics": {
                    "mse": current_mse,
                    "mae": current_mae
                },
                "metric_changes": {
                    "mse_change": mse_change,
                    "mae_change": abs(current_mae - ref_mae) / max(ref_mae, 1e-10)
                },
                "metrics_improving": current_mse < ref_mse
            }
            
        # Determine drift status
        status = DriftStatus.NO_DRIFT
        if drift_score > self.drift_threshold:
            status = DriftStatus.DRIFT
        elif drift_score > self.warning_threshold:
            status = DriftStatus.WARNING
            
        return DriftResult(
            status=status,
            p_value=None,
            drift_score=drift_score,
            metric_name="error_rate_change",
            threshold=self.drift_threshold,
            feature_drifts=feature_drifts,
            details=details
        ) 