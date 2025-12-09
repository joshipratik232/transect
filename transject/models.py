"""Core model architectures for TransJect framework."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, Tuple
import logging

from .config import TransJectConfig
from .trainer import TransJectTrainer

logger = logging.getLogger(__name__)


class BaseTransJectModel(nn.Module):
    """
    Base class for TransJect models.
    
    Provides common functionality for knowledge transfer between teacher and student models.
    """
    
    def __init__(
        self,
        student_model: str,
        teacher_model: str,
        config: Optional[TransJectConfig] = None,
        num_labels: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize base TransJect model.
        
        Args:
            student_model: HuggingFace model identifier for student
            teacher_model: HuggingFace model identifier for teacher
            config: TransJect configuration
            num_labels: Number of labels for classification tasks
            **kwargs: Additional arguments passed to config
        """
        super().__init__()
        
        self.config = config if config is not None else TransJectConfig()
        self.config.update(**kwargs)
        
        self.student_model_name = student_model
        self.teacher_model_name = teacher_model
        self.num_labels = num_labels
        
        # Will be initialized by subclasses
        self.student_model = None
        self.teacher_model = None
        self.tokenizer = None
        
    def _slice_student_layers(self, model, num_layers: int):
        """
        Slice student model to specified number of layers.
        
        Args:
            model: The model to slice
            num_layers: Number of layers to keep (-1 for all layers)
            
        Returns:
            The potentially sliced model
        """
        if num_layers == -1:
            # Use full model, no slicing
            logger.info("Using full student model (student_layers=-1)")
            return model
        
        # Try to slice the model based on its architecture
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style architecture
            original_layers = len(model.transformer.h)
            if num_layers < original_layers:
                model.transformer.h = model.transformer.h[:num_layers]
                logger.info(f"Sliced student model from {original_layers} to {num_layers} layers")
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama style architecture
            original_layers = len(model.model.layers)
            if num_layers < original_layers:
                model.model.layers = model.model.layers[:num_layers]
                logger.info(f"Sliced student model from {original_layers} to {num_layers} layers")
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT style architecture
            original_layers = len(model.encoder.layer)
            if num_layers < original_layers:
                model.encoder.layer = model.encoder.layer[:num_layers]
                logger.info(f"Sliced student model from {original_layers} to {num_layers} layers")
        else:
            logger.warning(f"Could not slice model - architecture not recognized")
        
        return model
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            temperature: Temperature for softening distributions
            
        Returns:
            Distillation loss
        """
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        
        distillation_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return distillation_loss
    
    def fit(
        self,
        train_dataloader,
        meta_dataloader: Optional[Dict[str, Any]] = None,
        val_dataloader: Optional[Any] = None,
        epochs: int = 3,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        report_to: Optional[str] = None,
        output_dir: str = "./output",
        **kwargs
    ):
        """
        Train the model using TransJect methodology.
        
        Args:
            train_dataloader: Primary training data loader
            meta_dataloader: Dictionary of meta data loaders for meta-learning
            val_dataloader: Validation data loader
            epochs: Number of training epochs
            optimizer: PyTorch optimizer (created if None)
            scheduler: Learning rate scheduler
            report_to: Logging backend ("wandb", "tensorboard", or None)
            output_dir: Directory for saving checkpoints
            **kwargs: Additional training arguments
        """
        trainer = TransJectTrainer(
            model=self,
            config=self.config,
            train_dataloader=train_dataloader,
            meta_dataloader=meta_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            report_to=report_to,
            output_dir=output_dir,
            **kwargs
        )
        
        trainer.train(epochs=epochs)


class SequenceClassification(BaseTransJectModel):
    """
    TransJect model for sequence classification tasks.
    
    Supports knowledge transfer for classification tasks like sentiment analysis,
    natural language inference, etc.
    
    Example:
        >>> from transject import SequenceClassification
        >>> model = SequenceClassification(
        ...     student_model="distilbert-base-uncased",
        ...     teacher_model="bert-base-uncased",
        ...     num_labels=3,
        ...     student_layers=-1
        ... )
    """
    
    def __init__(
        self,
        student_model: str,
        teacher_model: str,
        num_labels: int = 2,
        config: Optional[TransJectConfig] = None,
        token: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize sequence classification model.
        
        Args:
            student_model: HuggingFace model identifier for student
            teacher_model: HuggingFace model identifier for teacher
            num_labels: Number of classification labels
            config: TransJect configuration
            token: HuggingFace token for accessing gated models
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            student_model=student_model,
            teacher_model=teacher_model,
            config=config,
            num_labels=num_labels,
            **kwargs
        )
        
        logger.info(f"Initializing SequenceClassification with student={student_model}, teacher={teacher_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            student_model,
            token=token,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load models
        self.student_model = AutoModelForSequenceClassification.from_pretrained(
            student_model,
            num_labels=num_labels,
            token=token,
            trust_remote_code=True
        )
        
        # Apply layer slicing if needed
        if self.config.student_layers != -1:
            self.student_model = self._slice_student_layers(
                self.student_model, 
                self.config.student_layers
            )
        
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            teacher_model,
            num_labels=num_labels,
            token=token,
            trust_remote_code=True
        )
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        logger.info("SequenceClassification model initialized successfully")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with knowledge distillation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing loss, logits, and other outputs
        """
        # Student forward pass
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # Compute losses
        task_loss = student_outputs.loss if labels is not None else torch.tensor(0.0)
        
        distillation_loss = self.compute_distillation_loss(
            student_logits,
            teacher_logits,
            self.config.temperature
        )
        
        # Combined loss
        total_loss = (
            self.config.alpha * distillation_loss +
            (1 - self.config.alpha) * task_loss
        )
        
        if return_dict:
            return {
                'loss': total_loss,
                'task_loss': task_loss,
                'distillation_loss': distillation_loss,
                'logits': student_logits,
                'teacher_logits': teacher_logits
            }
        
        return total_loss


class AutoModel(BaseTransJectModel):
    """
    TransJect model for causal language modeling tasks.
    
    Supports knowledge transfer for language modeling tasks like text generation,
    instruction following, etc. Compatible with datasets like Alpaca.
    
    Example:
        >>> from transject import AutoModel
        >>> model = AutoModel(
        ...     student_model="gpt2",
        ...     teacher_model="gpt2-large",
        ...     student_layers=-1
        ... )
    """
    
    def __init__(
        self,
        student_model: str,
        teacher_model: str,
        config: Optional[TransJectConfig] = None,
        token: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize causal language model.
        
        Args:
            student_model: HuggingFace model identifier for student
            teacher_model: HuggingFace model identifier for teacher
            config: TransJect configuration
            token: HuggingFace token for accessing gated models
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            student_model=student_model,
            teacher_model=teacher_model,
            config=config,
            **kwargs
        )
        
        logger.info(f"Initializing AutoModel with student={student_model}, teacher={teacher_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            student_model,
            token=token,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load models
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model,
            token=token,
            trust_remote_code=True
        )
        
        # Apply layer slicing if needed
        if self.config.student_layers != -1:
            self.student_model = self._slice_student_layers(
                self.student_model,
                self.config.student_layers
            )
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model,
            token=token,
            trust_remote_code=True
        )
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        logger.info("AutoModel initialized successfully")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with knowledge distillation for language modeling.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target token IDs for language modeling
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing loss, logits, and other outputs
        """
        # Student forward pass
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # Compute losses
        task_loss = student_outputs.loss if labels is not None else torch.tensor(0.0)
        
        # For language modeling, compute distillation loss only on valid positions
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_student_logits = student_logits[..., :-1, :].contiguous()
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            
            # Flatten the logits
            shift_student_logits = shift_student_logits.view(-1, shift_student_logits.size(-1))
            shift_teacher_logits = shift_teacher_logits.view(-1, shift_teacher_logits.size(-1))
            
            distillation_loss = self.compute_distillation_loss(
                shift_student_logits,
                shift_teacher_logits,
                self.config.temperature
            )
        else:
            distillation_loss = self.compute_distillation_loss(
                student_logits.view(-1, student_logits.size(-1)),
                teacher_logits.view(-1, teacher_logits.size(-1)),
                self.config.temperature
            )
        
        # Combined loss
        total_loss = (
            self.config.alpha * distillation_loss +
            (1 - self.config.alpha) * task_loss
        )
        
        if return_dict:
            return {
                'loss': total_loss,
                'task_loss': task_loss,
                'distillation_loss': distillation_loss,
                'logits': student_logits,
                'teacher_logits': teacher_logits
            }
        
        return total_loss
    
    def generate(self, *args, **kwargs):
        """Generate text using the student model."""
        return self.student_model.generate(*args, **kwargs)
