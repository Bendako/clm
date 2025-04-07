"""
Base class for drift detectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from enum import Enum


class DriftStatus(Enum):
    """
    Enum representing the drift status.
    """
    NO_DRIFT = "no_drift"
    WARNING = "warning"
    DRIFT = "drift"


class DriftResult:
    """
    Class to hold drift detection result.
    """
    def __init__(
        self,
        status: DriftStatus,
        p_value: Optional[float] = None,
        drift_score: Optional[float] = None,
        metric_name: Optional[str] = None,
        threshold: Optional[float] = None,
        feature_drifts: Optional[Dict[str, float]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.status = status
        self.p_value = p_value
        self.drift_score = drift_score
        self.metric_name = metric_name
        self.threshold = threshold
        self.feature_drifts = feature_drifts or {}
        self.details = details or {}
    
    def __str__(self) -> str:
        result = f"DriftResult(status={self.status.value}"
        if self.p_value is not None:
            result += f", p_value={self.p_value:.4f}"
        if self.drift_score is not None:
            result += f", drift_score={self.drift_score:.4f}"
        if self.metric_name is not None:
            result += f", metric={self.metric_name}"
        if self.threshold is not None:
            result += f", threshold={self.threshold:.4f}"
        if self.feature_drifts:
            result += f", features_with_drift={len([f for f, v in self.feature_drifts.items() if v > (self.threshold or 0)])}"
        result += ")"
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the drift result to a dictionary for serialization."""
        return {
            "status": self.status.value,
            "p_value": self.p_value,
            "drift_score": self.drift_score,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "feature_drifts": self.feature_drifts,
            "details": self.details
        }


class DriftDetector(ABC):
    """
    Abstract base class for drift detectors.
    
    Drift detectors identify changes in data distributions over time, which
    is crucial for continual learning systems to adapt to changing environments.
    """
    
    def __init__(
        self,
        reference_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        warning_threshold: float = 0.05,
        drift_threshold: float = 0.01,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: Initial reference data representing the baseline distribution
            warning_threshold: P-value threshold for issuing a warning
            drift_threshold: P-value threshold for detecting drift
            feature_names: Names of features for more informative results
        """
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.feature_names = feature_names
        
        # Reference statistics
        self._reference_data = None
        self._reference_statistics = {}
        
        # Set reference data if provided
        if reference_data is not None:
            self.set_reference(reference_data)
    
    def set_reference(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Set reference data for baseline distribution.
        
        Args:
            data: Data to use as reference
        """
        # Convert torch tensors to numpy arrays if needed
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        self._reference_data = data
        self._compute_reference_statistics()
    
    @abstractmethod
    def _compute_reference_statistics(self) -> None:
        """
        Compute statistics on reference data.
        
        This method should be implemented by subclasses to compute
        relevant statistics on the reference data.
        """
        pass
    
    @abstractmethod
    def detect(self, data: Union[np.ndarray, torch.Tensor]) -> DriftResult:
        """
        Detect drift in current data compared to reference data.
        
        Args:
            data: Current data to check for drift
            
        Returns:
            DriftResult with drift status and metrics
        """
        pass
    
    def _get_drift_status(self, p_value: float) -> DriftStatus:
        """
        Determine drift status based on p-value and thresholds.
        
        Args:
            p_value: Calculated p-value from statistical test
            
        Returns:
            DriftStatus enum value
        """
        if p_value < self.drift_threshold:
            return DriftStatus.DRIFT
        elif p_value < self.warning_threshold:
            return DriftStatus.WARNING
        else:
            return DriftStatus.NO_DRIFT
    
    def update_reference(self, data: Union[np.ndarray, torch.Tensor], 
                         alpha: float = 0.05) -> None:
        """
        Update reference data with new samples using exponential weighting.
        
        Args:
            data: New data to incorporate
            alpha: Weight for new data (between 0 and 1)
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        if self._reference_data is None:
            self.set_reference(data)
        else:
            # Exponential weighting update
            self._reference_data = (1 - alpha) * self._reference_data + alpha * data
            self._compute_reference_statistics() 