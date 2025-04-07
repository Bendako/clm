"""
Distribution-based drift detection.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon, wasserstein_distance

from ml.drift_detection.detectors.base import DriftDetector, DriftResult, DriftStatus


class DistributionDriftDetector(DriftDetector):
    """
    Drift detector using distribution distance metrics.
    
    This detector measures the distance between distributions using
    metrics like KL-divergence, Jensen-Shannon distance, or
    Wasserstein distance.
    """
    
    def __init__(
        self,
        reference_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        distance_metric: str = "js",
        warning_threshold: float = 0.1,
        drift_threshold: float = 0.2,
        feature_names: Optional[List[str]] = None,
        n_bins: int = 30
    ):
        """
        Initialize distribution drift detector.
        
        Args:
            reference_data: Initial reference data
            distance_metric: Distribution distance measure ('js', 'wasserstein')
            warning_threshold: Distance threshold for warnings
            drift_threshold: Distance threshold for drift detection
            feature_names: Names of features for more informative results
            n_bins: Number of bins for histogram approximation
        """
        self.distance_metric = distance_metric.lower()
        self.n_bins = n_bins
        
        # Reverse thresholds logic for distance metrics (larger = more drift)
        self._warning_threshold = warning_threshold
        self._drift_threshold = drift_threshold
        
        # Check if distance metric is supported
        supported_metrics = ["js", "wasserstein"]
        if self.distance_metric not in supported_metrics:
            raise ValueError(f"Unsupported distance metric: {distance_metric}. "
                           f"Choose from: {', '.join(supported_metrics)}")
            
        super().__init__(reference_data, warning_threshold, drift_threshold, feature_names)
    
    def _compute_reference_statistics(self) -> None:
        """
        Compute histograms on reference data.
        """
        if self._reference_data is None:
            return
        
        n_features = self._reference_data.shape[1]
        histograms = []
        bin_edges = []
        
        # Compute histogram for each feature
        for i in range(n_features):
            feature_data = self._reference_data[:, i]
            hist, edges = np.histogram(
                feature_data, bins=self.n_bins, density=True
            )
            histograms.append(hist)
            bin_edges.append(edges)
        
        self._reference_statistics = {
            "histograms": histograms,
            "bin_edges": bin_edges,
            "min": np.min(self._reference_data, axis=0),
            "max": np.max(self._reference_data, axis=0)
        }
    
    def detect(self, data: Union[np.ndarray, torch.Tensor]) -> DriftResult:
        """
        Detect distribution drift between current data and reference data.
        
        Args:
            data: Current data to check for drift
            
        Returns:
            DriftResult containing distance metrics and drift status
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        # Convert torch tensors to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Ensure data has the same shape as reference
        if data.shape[1:] != self._reference_data.shape[1:]:
            raise ValueError(f"Data shape {data.shape} does not match "
                           f"reference shape {self._reference_data.shape}")
        
        # Get feature names
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        # Initialize results
        all_distances = []
        feature_drifts = {}
        
        # Compute distances for each feature
        for i in range(data.shape[1]):
            # Get reference histogram and bin edges
            ref_hist = self._reference_statistics["histograms"][i]
            bin_edges = self._reference_statistics["bin_edges"][i]
            
            # Compute histogram for current data
            current_hist, _ = np.histogram(
                data[:, i], bins=bin_edges, density=True
            )
            
            # Compute distance
            distance = self._compute_distance(current_hist, ref_hist)
            all_distances.append(distance)
            feature_drifts[feature_names[i]] = distance
        
        # Get overall drift score (mean distance across features)
        mean_distance = np.mean(all_distances)
        
        # Determine drift status (note: higher distance = more drift)
        status = self._get_drift_status_for_distance(mean_distance)
        
        # Create detailed results
        details = {
            "per_feature_distances": dict(zip(feature_names, all_distances)),
            "reference_min_max": {
                "min": self._reference_statistics["min"].tolist(),
                "max": self._reference_statistics["max"].tolist()
            },
            "current_min_max": {
                "min": np.min(data, axis=0).tolist(),
                "max": np.max(data, axis=0).tolist()
            },
            "distance_metric": self.distance_metric
        }
        
        return DriftResult(
            status=status,
            p_value=None,  # Not applicable for distance metrics
            drift_score=mean_distance,
            metric_name=self.distance_metric,
            threshold=self._drift_threshold,
            feature_drifts=feature_drifts,
            details=details
        )
    
    def _compute_distance(self, current_hist: np.ndarray, reference_hist: np.ndarray) -> float:
        """
        Compute distance between two histograms using the selected metric.
        
        Args:
            current_hist: Histogram of current data
            reference_hist: Histogram of reference data
            
        Returns:
            Distance value
        """
        # Add small constant to avoid zeros
        current_hist = current_hist + 1e-10
        reference_hist = reference_hist + 1e-10
        
        # Normalize if needed
        current_hist = current_hist / np.sum(current_hist)
        reference_hist = reference_hist / np.sum(reference_hist)
        
        if self.distance_metric == "js":
            # Jensen-Shannon distance
            return jensenshannon(current_hist, reference_hist)
            
        elif self.distance_metric == "wasserstein":
            # Wasserstein distance (Earth Mover's Distance)
            return wasserstein_distance(
                np.arange(len(current_hist)), 
                np.arange(len(reference_hist)),
                current_hist, 
                reference_hist
            )
    
    def _get_drift_status_for_distance(self, distance: float) -> DriftStatus:
        """
        Determine drift status based on distance and thresholds.
        
        For distance metrics, higher values indicate more drift,
        opposite of p-values.
        
        Args:
            distance: Calculated distance metric
            
        Returns:
            DriftStatus enum value
        """
        if distance > self._drift_threshold:
            return DriftStatus.DRIFT
        elif distance > self._warning_threshold:
            return DriftStatus.WARNING
        else:
            return DriftStatus.NO_DRIFT 