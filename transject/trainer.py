"""Training utilities for TransJect framework."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os
import logging
from tqdm.auto import tqdm
import numpy as np

from .config import TransJectConfig
from .metrics import MetricsTracker

logger = logging.getLogger(__name__)


class TransJectTrainer:
    """
    Trainer class for TransJect models.
    
    Handles the training loop, meta-learning, validation, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TransJectConfig,
        train_dataloader: DataLoader,
        meta_dataloader: Optional[Dict[str, DataLoader]] = None,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        report_to: Optional[str] = None,
        output_dir: str = "./output",
        **kwargs
    ):
        """
        Initialize the trainer.
        
        Args:
            model: TransJect model to train
            config: Training configuration
            train_dataloader: Primary training data loader
            meta_dataloader: Dictionary of meta data loaders
            val_dataloader: Validation data loader
            optimizer: Optimizer (created if None)
            scheduler: Learning rate scheduler
            report_to: Logging backend ("wandb", "tensorboard", or None)
            output_dir: Output directory for checkpoints
            **kwargs: Additional arguments
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.meta_dataloader = meta_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        self.report_to = report_to
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Move models to device
        self.model = self.model.to(self.device)
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.student_model.parameters(),
                lr=config.learning_rate
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker()
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # Meta-learning setup
        if meta_dataloader is not None:
            self.meta_optimizers = {}
            for key in meta_dataloader.keys():
                self.meta_optimizers[key] = torch.optim.AdamW(
                    self.model.student_model.parameters(),
                    lr=config.meta_learning_rate
                )
            logger.info(f"Meta-learning enabled with {len(meta_dataloader)} meta loaders")
    
    def _setup_logging(self):
        """Setup logging backend."""
        if self.report_to == "wandb":
            try:
                import wandb
                self.wandb = wandb
                logger.info("Weights & Biases logging enabled")
            except ImportError:
                logger.warning("wandb not installed, disabling W&B logging")
                self.report_to = None
        elif self.report_to == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "logs"))
                logger.info("TensorBoard logging enabled")
            except ImportError:
                logger.warning("tensorboard not installed, disabling TB logging")
                self.report_to = None
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """Log metrics to configured backend."""
        # Console logging
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"[{prefix}] Step {step} | {metrics_str}")
        
        # WandB logging
        if self.report_to == "wandb" and hasattr(self, 'wandb'):
            log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
            log_dict["step"] = step
            self.wandb.log(log_dict)
        
        # TensorBoard logging
        if self.report_to == "tensorboard" and hasattr(self, 'tb_writer'):
            for k, v in metrics.items():
                self.tb_writer.add_scalar(f"{prefix}/{k}", v, step)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
        else:
            outputs = self.model(**batch)
        
        loss = outputs['loss'] / self.config.accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Metrics
        metrics = {
            'loss': outputs['loss'].item(),
            'task_loss': outputs['task_loss'].item(),
            'distillation_loss': outputs['distillation_loss'].item(),
        }
        
        return metrics
    
    def meta_step(self, meta_key: str, meta_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a meta-learning step.
        
        Args:
            meta_key: Key identifying the meta dataloader
            meta_batch: Batch of meta data
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Move batch to device
        meta_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in meta_batch.items()}
        
        # Forward pass
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(**meta_batch)
        else:
            outputs = self.model(**meta_batch)
        
        loss = outputs['loss']
        
        # Meta backward pass
        meta_optimizer = self.meta_optimizers[meta_key]
        meta_optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(meta_optimizer)
        else:
            loss.backward()
            meta_optimizer.step()
        
        # Metrics
        metrics = {
            f'meta_{meta_key}_loss': loss.item(),
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation loop.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_task_loss = 0.0
        total_distillation_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                
                total_loss += outputs['loss'].item()
                total_task_loss += outputs['task_loss'].item()
                total_distillation_loss += outputs['distillation_loss'].item()
                num_batches += 1
                
                # Collect predictions for metrics
                if 'labels' in batch:
                    preds = torch.argmax(outputs['logits'], dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
        
        metrics = {
            'loss': total_loss / num_batches,
            'task_loss': total_task_loss / num_batches,
            'distillation_loss': total_distillation_loss / num_batches,
        }
        
        # Compute accuracy if we have labels
        if all_labels:
            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            metrics['accuracy'] = accuracy
        
        return metrics
    
    def train(self, epochs: int = 3):
        """
        Main training loop.
        
        Args:
            epochs: Number of training epochs
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Total training steps: {len(self.train_dataloader) * epochs}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"{'='*50}")
            
            # Training loop
            epoch_metrics = []
            progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Regular training step
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                # Meta-learning steps
                if self.meta_dataloader is not None and self.config.use_meta_learning:
                    for meta_key, meta_loader in self.meta_dataloader.items():
                        try:
                            meta_batch = next(iter(meta_loader))
                            meta_metrics = self.meta_step(meta_key, meta_batch)
                            metrics.update(meta_metrics)
                        except StopIteration:
                            pass
                
                # Gradient accumulation and optimizer step
                if (step + 1) % self.config.accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.student_model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': metrics['loss'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    self._log_metrics(metrics, self.global_step, prefix="train")
                
                # Validation
                if self.global_step % self.config.eval_interval == 0:
                    val_metrics = self.validate()
                    if val_metrics:
                        self._log_metrics(val_metrics, self.global_step, prefix="val")
                
                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint()
            
            # End of epoch validation
            logger.info("\nRunning end-of-epoch validation...")
            val_metrics = self.validate()
            if val_metrics:
                self._log_metrics(val_metrics, self.global_step, prefix="val")
            
            # Save epoch checkpoint
            self.save_checkpoint(epoch=epoch)
        
        logger.info("\nTraining completed!")
        
        # Final validation
        final_metrics = self.validate()
        if final_metrics:
            logger.info("\nFinal validation metrics:")
            for k, v in final_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
    
    def save_checkpoint(self, epoch: Optional[int] = None):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch (if None, uses global step)
        """
        if epoch is not None:
            save_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
        else:
            save_dir = os.path.join(self.output_dir, f"checkpoint-step-{self.global_step}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save student model
        self.model.student_model.save_pretrained(os.path.join(save_dir, "student_model"))
        
        # Save tokenizer
        if hasattr(self.model, 'tokenizer'):
            self.model.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        
        # Save config
        self.config.to_json(os.path.join(save_dir, "config.json"))
        
        # Save optimizer state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'global_step': self.global_step,
            'epoch': self.current_epoch,
        }, os.path.join(save_dir, "trainer_state.pt"))
        
        logger.info(f"Checkpoint saved to {save_dir}")
