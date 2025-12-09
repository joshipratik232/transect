"""Configuration management for TransJect framework."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json


@dataclass
class TransJectConfig:
    """
    Configuration class for TransJect training.
    
    This class handles all hyperparameters and settings for the knowledge
    transfer process.
    
    Args:
        student_layers: Number of layers to use from student model. 
                       -1 means use all layers (no slicing).
        temperature: Temperature for knowledge distillation softmax.
        alpha: Weight for distillation loss vs task loss.
        learning_rate: Learning rate for optimization.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        max_grad_norm: Maximum gradient norm for clipping.
        accumulation_steps: Gradient accumulation steps.
        use_meta_learning: Whether to use meta-learning with meta dataloaders.
        meta_learning_rate: Learning rate for meta-learning updates.
        log_interval: Steps between logging.
        eval_interval: Steps between evaluations.
        save_interval: Steps between model checkpoints.
        seed: Random seed for reproducibility.
        fp16: Whether to use mixed precision training.
        gradient_checkpointing: Whether to use gradient checkpointing.
    """
    
    # Model configuration
    student_layers: int = -1  # -1 means no slicing, use full model
    temperature: float = 2.0
    alpha: float = 0.5
    
    # Training configuration
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1
    
    # Meta-learning configuration
    use_meta_learning: bool = True
    meta_learning_rate: float = 1e-4
    meta_steps_per_batch: int = 1
    
    # Logging and evaluation
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Reproducibility
    seed: int = 42
    
    # Performance optimization
    fp16: bool = False
    gradient_checkpointing: bool = False
    
    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        return config_dict
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransJectConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'TransJectConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra_params[key] = value
