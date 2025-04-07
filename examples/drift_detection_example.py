#!/usr/bin/env python
"""
Example of using the CLM drift detection module to detect and monitor
distribution shifts in continual learning scenarios.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any

# Import drift detectors
from ml.drift_detection import (
    DriftDetector,
    StatisticalDriftDetector,
    DistributionDriftDetector,
    FeatureDriftDetector,
    PredictionDriftDetector
)

# Import from CLM framework for integration example
from ml.training.continual import ContinualTrainer


# Simple classifier model for demonstration
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)


def generate_synthetic_data_with_drift(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    drift_type: str = "feature",
    drift_severity: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data with controlled drift for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes (for classification)
        drift_type: Type of drift to simulate ('feature', 'label', 'concept')
        drift_severity: How severe the drift should be (0-1)
        
    Returns:
        Tuple of (reference_X, reference_y, drifted_X, drifted_y)
    """
    # Generate reference data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=n_classes,
        random_state=42
    )
    
    # Split into reference and data to be drifted
    X_ref, X_drift, y_ref, y_drift = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Apply different types of drift
    if drift_type == "feature":
        # Feature drift: change the distribution of features
        feature_shift = np.random.normal(0, drift_severity, X_drift.shape[1])
        X_drift = X_drift + feature_shift
        
    elif drift_type == "label":
        # Label drift: change the class balance
        if n_classes > 1:
            # Oversample one class to change distribution
            target_class = np.random.randint(0, n_classes)
            class_indices = np.where(y_drift == target_class)[0]
            
            # Select samples to duplicate
            n_duplicates = int(len(class_indices) * drift_severity)
            duplicate_indices = np.random.choice(class_indices, n_duplicates, replace=False)
            
            # Duplicate samples
            X_drift = np.vstack([X_drift, X_drift[duplicate_indices]])
            y_drift = np.hstack([y_drift, y_drift[duplicate_indices]])
            
    elif drift_type == "concept":
        # Concept drift: change the relationship between features and target
        # We'll simulate this by swapping the importance of features
        if n_features >= 4:
            # Swap some features to change their relationship with the target
            n_swaps = int(n_features * drift_severity / 2) * 2  # Ensure even number
            if n_swaps >= 2:
                swap_indices = np.random.choice(n_features, n_swaps, replace=False)
                for i in range(0, n_swaps, 2):
                    if i+1 < n_swaps:
                        X_drift[:, swap_indices[i]], X_drift[:, swap_indices[i+1]] = \
                            X_drift[:, swap_indices[i+1]].copy(), X_drift[:, swap_indices[i]].copy()
                
                # Optionally, flip some labels to simulate concept change
                if np.random.random() < drift_severity and n_classes == 2:
                    flip_indices = np.random.choice(
                        len(y_drift), 
                        int(len(y_drift) * drift_severity * 0.3), 
                        replace=False
                    )
                    y_drift[flip_indices] = 1 - y_drift[flip_indices]
    
    return X_ref, y_ref, X_drift, y_drift


def train_model(X: np.ndarray, y: np.ndarray, input_dim: int, hidden_dim: int = 64, 
                output_dim: int = 2, epochs: int = 10) -> Tuple[SimpleClassifier, List[float]]:
    """
    Train a simple classifier on the data.
    
    Args:
        X: Input features
        y: Target labels
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (number of classes)
        epochs: Number of training epochs
        
    Returns:
        Trained model and loss history
    """
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create model
    model = SimpleClassifier(input_dim, hidden_dim, output_dim)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
    return model, loss_history


def get_model_predictions(model: nn.Module, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get model predictions.
    
    Args:
        model: Trained model
        X: Input features
        
    Returns:
        Tuple of (class_predictions, class_probabilities)
    """
    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        
    # Convert to numpy
    probs_np = probs.numpy()
    preds_np = np.argmax(probs_np, axis=1)
    
    return preds_np, probs_np


def plot_drift_results(
    detector_name: str,
    drift_results: Dict[str, Any],
    feature_names: List[str] = None,
    savefig: bool = False
):
    """
    Visualize drift detection results.
    
    Args:
        detector_name: Name of the detector
        drift_results: Drift detection results
        feature_names: Names of features
        savefig: Whether to save the figure
    """
    result = drift_results['result']
    data = drift_results.get('data', {})
    
    plt.figure(figsize=(14, 8))
    
    # Create subplot grid
    if detector_name == "StatisticalDriftDetector":
        n_plots = 3
    elif detector_name == "DistributionDriftDetector":
        n_plots = 2
    elif detector_name == "FeatureDriftDetector":
        n_plots = 3
    elif detector_name == "PredictionDriftDetector":
        n_plots = 2
    else:
        n_plots = 1
    
    # Plot 1: Drift status and overall score
    plt.subplot(1, n_plots, 1)
    status_color = {
        "no_drift": "green",
        "warning": "orange",
        "drift": "red"
    }
    
    plt.bar(["Drift Score"], [result.drift_score], 
            color=status_color.get(result.status.value, "blue"))
    plt.axhline(y=result.threshold, color='r', linestyle='--', label="Drift Threshold")
    
    if hasattr(result, 'warning_threshold') and result.warning_threshold:
        plt.axhline(y=result.warning_threshold, color='orange', 
                   linestyle='--', label="Warning Threshold")
    
    plt.title(f"{detector_name}: {result.status.value.upper()}")
    plt.ylim(0, max(1.0, result.drift_score * 1.2))
    plt.legend()
    
    # Plot 2: Feature-specific drift
    if feature_names is None:
        feature_names = list(result.feature_drifts.keys())
    
    # Only plot if we have feature drifts
    if result.feature_drifts and len(feature_names) > 0:
        plt.subplot(1, n_plots, 2)
        
        # Sort feature drifts for better visualization
        feature_drifts = {k: result.feature_drifts[k] for k in feature_names 
                         if k in result.feature_drifts}
        sorted_features = sorted(feature_drifts.items(), key=lambda x: x[1], reverse=True)
        
        # Plot top N features
        top_n = min(10, len(sorted_features))
        feat_names = [x[0] for x in sorted_features[:top_n]]
        feat_drifts = [x[1] for x in sorted_features[:top_n]]
        
        colors = [status_color["drift"] if x > result.threshold else 
                 (status_color["warning"] if x > (result.warning_threshold if hasattr(result, 'warning_threshold') else 0) else 
                 status_color["no_drift"]) for x in feat_drifts]
        
        plt.barh(feat_names, feat_drifts, color=colors)
        plt.axvline(x=result.threshold, color='r', linestyle='--', label="Drift Threshold")
        plt.title("Feature-specific Drift")
        plt.xlabel("Drift Score")
        plt.tight_layout()
    
    # Plot 3: Detector-specific visualization
    if n_plots >= 3:
        plt.subplot(1, n_plots, 3)
        
        if detector_name == "StatisticalDriftDetector":
            # Plot distributions for most drifted feature
            if feature_names and data.get('X_ref') is not None and data.get('X_drift') is not None:
                top_feature_idx = feature_names.index(sorted_features[0][0]) if sorted_features else 0
                
                sns.kdeplot(data['X_ref'][:, top_feature_idx], label="Reference")
                sns.kdeplot(data['X_drift'][:, top_feature_idx], label="Current")
                plt.title(f"Distribution of {feat_names[0]}")
                plt.legend()
                
        elif detector_name == "FeatureDriftDetector" and result.method == "correlation":
            # Plot correlation matrices
            if 'reference_correlation_matrix' in result.details and 'current_correlation_matrix' in result.details:
                ref_corr = np.array(result.details['reference_correlation_matrix'])
                curr_corr = np.array(result.details['current_correlation_matrix'])
                
                # Plot correlation difference
                plt.imshow(np.abs(curr_corr - ref_corr), cmap='YlOrRd', vmin=0, vmax=1)
                plt.colorbar(label="Correlation Change")
                plt.title("Correlation Matrix Differences")
                if len(feature_names) <= 10:
                    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
                    plt.yticks(range(len(feature_names)), feature_names)
    
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f"{detector_name}_drift_results.png", dpi=150, bbox_inches='tight')
    
    plt.show()


def main():
    # Generate synthetic data with various types of drift
    n_features = 10
    n_classes = 3
    
    print("Generating synthetic data with drift...")
    # Generate data with feature distribution drift
    X_ref, y_ref, X_feature_drift, y_feature_drift = generate_synthetic_data_with_drift(
        n_features=n_features, n_classes=n_classes, drift_type="feature", drift_severity=0.8
    )
    
    # Generate data with label distribution drift
    _, _, X_label_drift, y_label_drift = generate_synthetic_data_with_drift(
        n_features=n_features, n_classes=n_classes, drift_type="label", drift_severity=0.7
    )
    
    # Generate data with concept drift
    _, _, X_concept_drift, y_concept_drift = generate_synthetic_data_with_drift(
        n_features=n_features, n_classes=n_classes, drift_type="concept", drift_severity=0.6
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    print("\n1. Statistical Drift Detection (Feature Distribution Shift)")
    # Initialize statistical drift detector
    stat_detector = StatisticalDriftDetector(
        reference_data=X_ref,
        test_method="ks",
        warning_threshold=0.05,
        drift_threshold=0.01,
        feature_names=feature_names
    )
    
    # Detect drift
    stat_result = stat_detector.detect(X_feature_drift)
    print(f"Statistical Drift Result: {stat_result}")
    print(f"Status: {stat_result.status.value}, Drift Score: {stat_result.drift_score:.4f}")
    
    # Plot the top 3 most drifted features
    top_features = sorted(
        [(k, v) for k, v in stat_result.feature_drifts.items()], 
        key=lambda x: x[1]
    )[:3]
    print("Top drifted features:")
    for feature, p_value in top_features:
        print(f"  {feature}: p-value = {p_value:.4f}")
    
    # Visualize results
    plot_drift_results(
        "StatisticalDriftDetector",
        {'result': stat_result, 'data': {'X_ref': X_ref, 'X_drift': X_feature_drift}},
        feature_names
    )
    
    print("\n2. Distribution Drift Detection")
    # Initialize distribution drift detector
    dist_detector = DistributionDriftDetector(
        reference_data=X_ref,
        distance_metric="js",
        warning_threshold=0.1,
        drift_threshold=0.2,
        feature_names=feature_names
    )
    
    # Detect drift
    dist_result = dist_detector.detect(X_feature_drift)
    print(f"Distribution Drift Result: {dist_result}")
    print(f"Status: {dist_result.status.value}, Drift Score: {dist_result.drift_score:.4f}")
    
    # Visualize results
    plot_drift_results(
        "DistributionDriftDetector",
        {'result': dist_result},
        feature_names
    )
    
    print("\n3. Feature Correlation Drift Detection")
    # Initialize feature drift detector
    feat_detector = FeatureDriftDetector(
        reference_data=X_ref,
        method="correlation",
        warning_threshold=0.15,
        drift_threshold=0.25,
        feature_names=feature_names
    )
    
    # Detect drift
    feat_result = feat_detector.detect(X_concept_drift)
    print(f"Feature Drift Result: {feat_result}")
    print(f"Status: {feat_result.status.value}, Drift Score: {feat_result.drift_score:.4f}")
    
    # Visualize results
    plot_drift_results(
        "FeatureDriftDetector",
        {'result': feat_result},
        feature_names
    )
    
    print("\n4. Model Training for Prediction Drift")
    # Train model on reference data
    model, _ = train_model(
        X_ref, y_ref, input_dim=n_features, output_dim=n_classes, epochs=20
    )
    
    # Get predictions on reference data
    ref_preds, ref_probs = get_model_predictions(model, X_ref)
    
    # Get predictions on concept drift data
    concept_preds, concept_probs = get_model_predictions(model, X_concept_drift)
    
    print("\n5. Prediction Distribution Drift Detection")
    # Initialize prediction drift detector
    pred_detector = PredictionDriftDetector(
        reference_predictions=ref_probs,
        method="distribution",
        warning_threshold=0.1,
        drift_threshold=0.2
    )
    
    # Detect drift
    pred_result = pred_detector.detect(predictions=concept_probs)
    print(f"Prediction Drift Result: {pred_result}")
    print(f"Status: {pred_result.status.value}, Drift Score: {pred_result.drift_score:.4f}")
    
    # Visualize results
    plot_drift_results(
        "PredictionDriftDetector",
        {'result': pred_result},
        [f"class_{i}" for i in range(n_classes)]
    )
    
    print("\n6. Error Rate Drift Detection")
    # Initialize error drift detector
    error_detector = PredictionDriftDetector(
        reference_predictions=ref_preds,
        reference_labels=y_ref,
        method="error",
        warning_threshold=0.1,
        drift_threshold=0.2,
        task_type="classification"
    )
    
    # Detect drift
    error_result = error_detector.detect(predictions=concept_preds, labels=y_concept_drift)
    print(f"Error Drift Result: {error_result}")
    print(f"Status: {error_result.status.value}, Drift Score: {error_result.drift_score:.4f}")
    
    # Print detailed metrics
    print("\nReference accuracy:", error_result.details.get("reference_accuracy"))
    print("Current accuracy:", error_result.details.get("current_accuracy"))
    print("Accuracy change:", error_result.details.get("accuracy_change"))
    
    # Visualize results
    plot_drift_results(
        "PredictionDriftDetector (Error)",
        {'result': error_result}
    )
    
    print("\n7. Integration with Continual Trainer")
    # This section demonstrates how drift detection would integrate with the ContinualTrainer
    
    # Create a simple detection-based trainer (pseudocode)
    print("\nPseudocode for integrating drift detection with ContinualTrainer:")
    print("""
    # In ContinualTrainer class:
    
    def setup_drift_detection(self, drift_config):
        # Initialize detectors based on config
        self.distribution_detector = DistributionDriftDetector(...)
        self.prediction_detector = PredictionDriftDetector(...)
    
    def train_task(self, train_loader, val_loader, task_name, task_id, num_epochs):
        # Check for drift before training
        X_sample, _ = self._get_batch_sample(train_loader)
        drift_result = self.distribution_detector.detect(X_sample)
        
        if drift_result.status == DriftStatus.DRIFT:
            # Adjust training based on drift detection
            # 1. Use more aggressive regularization
            # 2. Allocate more replay buffer samples
            # 3. Lower learning rate
            # 4. Log drift event
        
        # Continue with normal training...
        
        # After training, update reference statistics
        self.distribution_detector.update_reference(X_sample)
    """)


if __name__ == "__main__":
    main() 