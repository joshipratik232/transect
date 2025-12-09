"""Data loading utilities for TransJect."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class SuperGLUEDataset(Dataset):
    """
    Dataset wrapper for SuperGLUE tasks.
    
    Supports tasks like CB (CommitmentBank), RTE, etc.
    """
    
    def __init__(
        self,
        task_name: str,
        split: str,
        tokenizer,
        max_length: int = 128,
        num_samples: Optional[int] = None
    ):
        """
        Initialize SuperGLUE dataset.
        
        Args:
            task_name: Name of the task (e.g., 'cb', 'rte')
            split: Dataset split ('train', 'validation', 'test')
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            num_samples: Number of samples to use (None for all)
        """
        self.task_name = task_name.lower()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading SuperGLUE task: {task_name}, split: {split}")
        
        # Load dataset
        self.dataset = load_dataset("super_glue", self.task_name, split=split)
        
        if num_samples is not None:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        logger.info(f"Loaded {len(self.dataset)} samples")
        
        # Get label information
        if hasattr(self.dataset.features['label'], 'num_classes'):
            self.num_labels = self.dataset.features['label'].num_classes
        else:
            self.num_labels = len(set(self.dataset['label']))
        
        logger.info(f"Number of labels: {self.num_labels}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.dataset[idx]
        
        # Handle different task formats
        if self.task_name == 'cb':
            text_a = item['premise']
            text_b = item['hypothesis']
        elif self.task_name == 'rte':
            text_a = item['premise']
            text_b = item['hypothesis']
        elif self.task_name == 'wic':
            text_a = item['sentence1']
            text_b = item['sentence2']
        else:
            # Default handling
            text_a = item.get('sentence1', item.get('premise', ''))
            text_b = item.get('sentence2', item.get('hypothesis', ''))
        
        # Tokenize
        encoding = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare output
        output = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }
        
        return output


class AlpacaDataset(Dataset):
    """
    Dataset wrapper for Alpaca instruction-following dataset.
    """
    
    def __init__(
        self,
        split: str,
        tokenizer,
        max_length: int = 512,
        num_samples: Optional[int] = None,
        dataset_name: str = "tatsu-lab/alpaca"
    ):
        """
        Initialize Alpaca dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            num_samples: Number of samples to use (None for all)
            dataset_name: HuggingFace dataset name
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading Alpaca dataset: {dataset_name}, split: {split}")
        
        try:
            # Try to load from HuggingFace
            self.dataset = load_dataset(dataset_name, split=split)
        except:
            # Fallback to train split and manually split if needed
            logger.warning(f"Split {split} not found, using train split")
            full_dataset = load_dataset(dataset_name, split='train')
            
            # Manual split
            if split == 'train':
                self.dataset = full_dataset.select(range(int(0.9 * len(full_dataset))))
            elif split == 'validation':
                self.dataset = full_dataset.select(range(int(0.9 * len(full_dataset)), len(full_dataset)))
            else:
                self.dataset = full_dataset
        
        if num_samples is not None:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        logger.info(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.dataset[idx]
        
        # Format Alpaca instruction
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        
        # Create prompt
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create labels (mask padding tokens)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        return output


def create_superglue_dataloaders(
    task_name: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 128,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train and validation dataloaders for SuperGLUE tasks.
    
    Args:
        task_name: SuperGLUE task name
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_train_samples: Number of training samples (None for all)
        num_val_samples: Number of validation samples (None for all)
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, num_labels)
    """
    train_dataset = SuperGLUEDataset(
        task_name=task_name,
        split='train',
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_train_samples
    )
    
    val_dataset = SuperGLUEDataset(
        task_name=task_name,
        split='validation',
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_val_samples
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_dataloader, val_dataloader, train_dataset.num_labels


def create_alpaca_dataloaders(
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
    dataset_name: str = "tatsu-lab/alpaca"
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for Alpaca dataset.
    
    Args:
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_train_samples: Number of training samples (None for all)
        num_val_samples: Number of validation samples (None for all)
        dataset_name: HuggingFace dataset name
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_dataset = AlpacaDataset(
        split='train',
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_train_samples,
        dataset_name=dataset_name
    )
    
    val_dataset = AlpacaDataset(
        split='validation',
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_val_samples,
        dataset_name=dataset_name
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_dataloader, val_dataloader


def create_meta_dataloaders(
    tokenizer,
    tasks: List[str],
    batch_size: int = 8,
    max_length: int = 128,
    num_samples_per_task: Optional[int] = 100
) -> Dict[str, DataLoader]:
    """
    Create meta-learning dataloaders for multiple tasks.
    
    Args:
        tokenizer: Tokenizer
        tasks: List of task names
        batch_size: Batch size
        max_length: Maximum sequence length
        num_samples_per_task: Number of samples per task
        
    Returns:
        Dictionary mapping task names to dataloaders
    """
    meta_dataloaders = {}
    
    for task in tasks:
        try:
            dataset = SuperGLUEDataset(
                task_name=task,
                split='train',
                tokenizer=tokenizer,
                max_length=max_length,
                num_samples=num_samples_per_task
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            
            meta_dataloaders[task] = dataloader
            logger.info(f"Created meta-dataloader for task: {task}")
        except Exception as e:
            logger.warning(f"Failed to create meta-dataloader for {task}: {e}")
    
    return meta_dataloaders
