"""
TransJect: A novel knowledge transfer framework for neural networks.

This package provides a clean, HuggingFace-like API for knowledge transfer
between teacher and student models, supporting both classification and 
language modeling tasks.
"""

from .config import TransJectConfig
from .models import SequenceClassification, AutoModel
from .__version__ import __version__

__all__ = [
    "TransJectConfig",
    "SequenceClassification",
    "AutoModel",
    "__version__",
]
