"""
Statistical drift detection using hypothesis tests.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from scipy import stats

from ml.drift_detection.detectors.base import DriftDetector, DriftResult, DriftStatus


class StatisticalDriftDetector(DriftDetector):
    """
    Drift detector using statistical hypothesis tests.
    
    This detector applies statistical tests (KS-test, Chi-squared, etc.)
    to determine if the distribution of new data has significantly 
    changed from the reference distribution.
    """
    
    def __init__(
        self,
        reference_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        test_method: str = "ks",
        warning_threshold: float = 0.05,
        drift_threshold: float = 0.01,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize statistical drift detector.
        
        Args:
            reference_data: Initial reference data
            test_method: Statistical test to use ('ks', 'chi2', 't-test')
            warning_threshold: P-value threshold for warnings
            drift_threshold: P-value threshold for drift detection
            feature_names: Names of features for more informative results
        """
        self.test_method = test_method.lower()
        
        # Check if test method is supported
        supported_methods = ["ks", "chi2", "t-test", "mann-whitney"]
        if self.test_method not in supported_methods:
            raise ValueError(f"Unsupported test method: {test_method}. "
                           f"Choose from: {', '.join(supported_methods)}")
            
        super().__init__(reference_data, warning_threshold, drift_threshold, feature_names)
    
    def _compute_reference_statistics(self) -> None:
        """
        Compute summary statistics on reference data.
        """
        if self._reference_data is None:
            return
        
        # For quick reference access during detection
        self._reference_statistics = {
            "mean": np.mean(self._reference_data, axis=0),
            "std": np.std(self._reference_data, axis=0),
            "min": np.min(self._reference_data, axis=0),
            "max": np.max(self._reference_data, axis=0),
            "q25": np.percentile(self._reference_data, 25, axis=0),
            "median": np.median(self._reference_data, axis=0),
            "q75": np.percentile(self._reference_data, 75, axis=0)
        }
    
    def detect(self, data: Union[np.ndarray, torch.Tensor]) -> DriftResult:
        """
        Detect statistical drift between current data and reference data.
        
        Args:
            data: Current data to check for drift
            
        Returns:
            DriftResult containing test results and drift status
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
        all_p_values = []
        feature_drifts = {}
        
        # Apply test for each feature
        for i in range(data.shape[1]):
            p_value = self._apply_test(data[:, i], self._reference_data[:, i])
            all_p_values.append(p_value)
            feature_drifts[feature_names[i]] = p_value
        
        # Get overall p-value (minimum across features, Bonferroni correction)
        min_p_value = min(all_p_values) * len(all_p_values)  # Bonferroni correction
        min_p_value = min(1.0, min_p_value)  # Cap at 1.0
        
        # Determine drift status
        status = self._get_drift_status(min_p_value)
        
        # Create detailed results
        details = {
            "per_feature_p_values": dict(zip(feature_names, all_p_values)),
            "reference_stats": {k: v.tolist() for k, v in self._reference_statistics.items()},
            "current_stats": {
                "mean": np.mean(data, axis=0).tolist(),
                "std": np.std(data, axis=0).tolist(),
                "min": np.min(data, axis=0).tolist(),
                "max": np.max(data, axis=0).tolist()
            },
            "test_method": self.test_method
        }
        
        return DriftResult(
            status=status,
            p_value=min_p_value,
            drift_score=1.0 - min_p_value,
            metric_name=self.test_method,
            threshold=self.drift_threshold,
            feature_drifts=feature_drifts,
            details=details
        )
    
    def _apply_test(self, current_data: np.ndarray, reference_data: np.ndarray) -> float:
        """
        Apply the selected statistical test to a single feature.
        
        Args:
            current_data: Current data for a single feature
            reference_data: Reference data for a single feature
            
        Returns:
            p-value from the statistical test
        """
        if self.test_method == "ks":
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(current_data, reference_data)
        
        elif self.test_method == "chi2":
            # Chi-squared test (requires binning continuous data)
            bins = min(20, int(np.sqrt(len(reference_data))))
            
            # Use range from combined data
            all_data = np.concatenate([current_data, reference_data])
            range_min, range_max = np.min(all_data), np.max(all_data)
            
            # Create histograms
            hist1, edges = np.histogram(current_data, bins=bins, range=(range_min, range_max))
            hist2, _ = np.histogram(reference_data, bins=bins, range=(range_min, range_max))
            
            # Ensure no zeros (add small constant if needed)
            hist1 = hist1 + 0.001
            hist2 = hist2 + 0.001
            
            # Scale hist2 to match hist1 size
            scale_factor = np.sum(hist1) / np.sum(hist2)
            hist2_scaled = hist2 * scale_factor
            
            # Chi-squared test
            statistic, p_value = stats.chisquare(hist1, hist2_scaled)
        
        elif self.test_method == "t-test":
            # Student's t-test
            statistic, p_value = stats.ttest_ind(current_data, reference_data, equal_var=False)
            
            # Convert NaN to 1.0 (no drift)
            if np.isnan(p_value):
                p_value = 1.0
                
        elif self.test_method == "mann-whitney":
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(current_data, reference_data, alternative='two-sided')
        
        return p_value 