"""Metrics tracking and evaluation utilities for TransJect."""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track and compute metrics during training and evaluation.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = defaultdict(list)
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)
    
    def update(self, metrics_dict: Dict[str, float]):
        """
        Update metrics with new values.
        
        Args:
            metrics_dict: Dictionary of metric names and values
        """
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
    
    def get_average(self, key: str) -> float:
        """
        Get average value for a metric.
        
        Args:
            key: Metric name
            
        Returns:
            Average value
        """
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return np.mean(self.metrics[key])
    
    def get_all_averages(self) -> Dict[str, float]:
        """
        Get average values for all metrics.
        
        Returns:
            Dictionary of metric averages
        """
        return {key: self.get_average(key) for key in self.metrics.keys()}
    
    def get_last(self, key: str) -> float:
        """
        Get last value for a metric.
        
        Args:
            key: Metric name
            
        Returns:
            Last value
        """
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return self.metrics[key][-1]


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_labels: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        num_labels: Number of unique labels
        
    Returns:
        Dictionary of metrics
    """
    accuracy = np.mean(predictions == labels)
    
    metrics = {
        'accuracy': accuracy,
        'num_samples': len(labels)
    }
    
    # Per-class metrics if num_labels is known
    if num_labels is not None:
        for i in range(num_labels):
            mask = labels == i
            if mask.sum() > 0:
                class_acc = np.mean(predictions[mask] == labels[mask])
                metrics[f'accuracy_class_{i}'] = class_acc
    
    return metrics


def compute_language_modeling_metrics(
    loss: float,
    perplexity: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute language modeling metrics.
    
    Args:
        loss: Cross-entropy loss
        perplexity: Perplexity (computed if None)
        
    Returns:
        Dictionary of metrics
    """
    if perplexity is None:
        perplexity = np.exp(loss)
    
    return {
        'loss': loss,
        'perplexity': perplexity
    }


def compute_distillation_metrics(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray
) -> Dict[str, float]:
    """
    Compute knowledge distillation metrics.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        
    Returns:
        Dictionary of metrics
    """
    # Compute agreement
    student_preds = np.argmax(student_logits, axis=-1)
    teacher_preds = np.argmax(teacher_logits, axis=-1)
    agreement = np.mean(student_preds == teacher_preds)
    
    # Compute KL divergence
    from scipy.special import softmax
    from scipy.stats import entropy
    
    student_probs = softmax(student_logits, axis=-1)
    teacher_probs = softmax(teacher_logits, axis=-1)
    
    kl_div = np.mean([entropy(teacher_probs[i], student_probs[i]) 
                      for i in range(len(student_probs))])
    
    return {
        'teacher_student_agreement': agreement,
        'kl_divergence': kl_div
    }
