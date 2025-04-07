"""
Feature drift detection for monitoring changes in feature importance and relationships.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

from ml.drift_detection.detectors.base import DriftDetector, DriftResult, DriftStatus


class FeatureDriftDetector(DriftDetector):
    """
    Detector for changes in feature importance and relationships.
    
    This detector tracks changes in:
    1. Feature correlations
    2. Feature importance (using model-based approaches)
    3. Feature distributional shifts
    """
    
    def __init__(
        self,
        reference_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        reference_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        method: str = "correlation",
        warning_threshold: float = 0.15,
        drift_threshold: float = 0.25,
        feature_names: Optional[List[str]] = None,
        task_type: str = "classification"
    ):
        """
        Initialize feature drift detector.
        
        Args:
            reference_data: Initial reference data
            reference_labels: Labels for reference data (if using importance method)
            method: Detection method ('correlation', 'importance')
            warning_threshold: Threshold for warnings
            drift_threshold: Threshold for drift detection
            feature_names: Names of features
            task_type: 'classification' or 'regression' (for importance method)
        """
        self.method = method.lower()
        self.task_type = task_type.lower()
        self._reference_labels = None
        
        # Check if method is supported
        supported_methods = ["correlation", "importance"]
        if self.method not in supported_methods:
            raise ValueError(f"Unsupported method: {method}. "
                           f"Choose from: {', '.join(supported_methods)}")
            
        # Store reference labels if provided
        if reference_labels is not None:
            if isinstance(reference_labels, torch.Tensor):
                reference_labels = reference_labels.detach().cpu().numpy()
            self._reference_labels = reference_labels
            
        # For importance method, we need labels
        if self.method == "importance" and self._reference_labels is None:
            raise ValueError("Reference labels must be provided for importance method")
            
        super().__init__(reference_data, warning_threshold, drift_threshold, feature_names)
    
    def _compute_reference_statistics(self) -> None:
        """
        Compute feature statistics on reference data.
        """
        if self._reference_data is None:
            return
        
        if self.method == "correlation":
            # Compute correlation matrix
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._reference_statistics["correlation_matrix"] = np.corrcoef(self._reference_data, rowvar=False)
                
        elif self.method == "importance":
            if self._reference_labels is None:
                return
                
            # Train a model to get feature importances
            if self.task_type == "classification":
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:  # regression
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                
            # Fit the model
            model.fit(self._reference_data, self._reference_labels)
            
            # Get feature importances
            self._reference_statistics["feature_importances"] = model.feature_importances_
            self._reference_statistics["model"] = model
    
    def detect(self, data: Union[np.ndarray, torch.Tensor],
              labels: Optional[Union[np.ndarray, torch.Tensor]] = None) -> DriftResult:
        """
        Detect feature drift between current data and reference data.
        
        Args:
            data: Current data to check for drift
            labels: Labels for current data (if using importance method)
            
        Returns:
            DriftResult containing drift metrics
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        # Convert torch tensors to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # For importance method, we need labels
        if self.method == "importance" and labels is None:
            raise ValueError("Labels must be provided for importance method")
            
        # Ensure data has the same shape as reference
        if data.shape[1:] != self._reference_data.shape[1:]:
            raise ValueError(f"Data shape {data.shape} does not match "
                           f"reference shape {self._reference_data.shape}")
        
        # Get feature names
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        if self.method == "correlation":
            return self._detect_correlation_drift(data, feature_names)
        elif self.method == "importance":
            return self._detect_importance_drift(data, labels, feature_names)
    
    def _detect_correlation_drift(self, data: np.ndarray, feature_names: List[str]) -> DriftResult:
        """
        Detect drift in feature correlations.
        
        Args:
            data: Current data
            feature_names: Feature names
            
        Returns:
            DriftResult for correlation-based drift
        """
        # Compute current correlation matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            current_corr_matrix = np.corrcoef(data, rowvar=False)
        
        # Get reference correlation matrix
        ref_corr_matrix = self._reference_statistics.get("correlation_matrix")
        
        if ref_corr_matrix is None:
            raise ValueError("Reference correlation matrix not computed")
            
        # Compute drift score as mean absolute difference in correlations
        corr_diff = np.abs(current_corr_matrix - ref_corr_matrix)
        drift_score = np.mean(corr_diff)
        
        # Get per-feature drift (mean absolute diff for each feature's correlations)
        feature_drifts = {}
        for i, name in enumerate(feature_names):
            feature_drifts[name] = np.mean(corr_diff[i, :])
        
        # Determine drift status
        status = DriftStatus.NO_DRIFT
        if drift_score > self.drift_threshold:
            status = DriftStatus.DRIFT
        elif drift_score > self.warning_threshold:
            status = DriftStatus.WARNING
            
        # Create results
        details = {
            "reference_correlation_matrix": ref_corr_matrix.tolist(),
            "current_correlation_matrix": current_corr_matrix.tolist(),
            "correlation_difference_matrix": corr_diff.tolist(),
            "max_correlation_change": np.max(corr_diff)
        }
        
        return DriftResult(
            status=status,
            p_value=None,
            drift_score=drift_score,
            metric_name="correlation_difference",
            threshold=self.drift_threshold,
            feature_drifts=feature_drifts,
            details=details
        )
    
    def _detect_importance_drift(self, data: np.ndarray, labels: np.ndarray, 
                               feature_names: List[str]) -> DriftResult:
        """
        Detect drift in feature importance.
        
        Args:
            data: Current data
            labels: Current labels
            feature_names: Feature names
            
        Returns:
            DriftResult for importance-based drift
        """
        # Get reference model and importances
        ref_model = self._reference_statistics.get("model")
        ref_importances = self._reference_statistics.get("feature_importances")
        
        if ref_model is None or ref_importances is None:
            raise ValueError("Reference model or importances not computed")
            
        # Train new model on current data
        if self.task_type == "classification":
            current_model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:  # regression
            current_model = RandomForestRegressor(n_estimators=50, random_state=42)
            
        # Fit the model
        current_model.fit(data, labels)
        
        # Get current feature importances
        current_importances = current_model.feature_importances_
        
        # Compute importance differences
        importance_diff = np.abs(current_importances - ref_importances)
        drift_score = np.mean(importance_diff)
        
        # Get per-feature drift
        feature_drifts = {}
        for i, name in enumerate(feature_names):
            feature_drifts[name] = importance_diff[i]
        
        # Determine drift status
        status = DriftStatus.NO_DRIFT
        if drift_score > self.drift_threshold:
            status = DriftStatus.DRIFT
        elif drift_score > self.warning_threshold:
            status = DriftStatus.WARNING
            
        # Create results
        details = {
            "reference_importances": ref_importances.tolist(),
            "current_importances": current_importances.tolist(),
            "importance_differences": importance_diff.tolist(),
            "max_importance_change": np.max(importance_diff),
            "reference_top_features": [
                feature_names[i] for i in np.argsort(ref_importances)[::-1][:5]
            ],
            "current_top_features": [
                feature_names[i] for i in np.argsort(current_importances)[::-1][:5]
            ]
        }
        
        return DriftResult(
            status=status,
            p_value=None,
            drift_score=drift_score,
            metric_name="importance_difference",
            threshold=self.drift_threshold,
            feature_drifts=feature_drifts,
            details=details
        ) 