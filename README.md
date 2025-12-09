# TransJect

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshipratik232/transject/blob/main/TransJect_Demo.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel knowledge transfer framework for neural networks with a clean, HuggingFace-like API.

> **🚀 Quick Start**: Click the "Open in Colab" badge above to try TransJect immediately!

## Features

- 🚀 **Simple API**: HuggingFace-style interface for easy integration
- 🎯 **Task Flexibility**: Support for both classification and language modeling
- 🔄 **Meta-Learning**: Built-in meta-learning support with multiple data loaders
- 📊 **Logging**: Integration with Weights & Biases and TensorBoard
- 🎓 **Knowledge Transfer**: Efficient teacher-student distillation
- 🔧 **Configurable**: Extensive configuration options via `TransJectConfig`
- 📦 **Production Ready**: Pip-installable package with proper structure

## Installation

### From GitHub (Recommended)

```bash
git clone https://github.com/joshipratik232/transject.git
cd transject
pip install -r requirements.txt
```

### For Google Colab

```python
!git clone https://github.com/joshipratik232/transject.git
%cd transject
!pip install -r requirements.txt -q
```

### From PyPI (when available)

```bash
pip install transject
```

### From Source

```bash
git clone https://github.com/transject/transject.git
cd transject
pip install -e .
```

### With Optional Dependencies

```bash
# For W&B logging
pip install transject[wandb]

# For TensorBoard logging
pip install transject[tensorboard]

# Install everything
pip install transject[all]
```

## Quick Start

### Classification Task (e.g., CB from SuperGLUE)

```python
from transject import SequenceClassification, TransJectConfig
from transject.data_utils import create_superglue_dataloaders

# Create model
model = SequenceClassification(
    student_model="distilbert-base-uncased",
    teacher_model="bert-base-uncased",
    num_labels=3,
    student_layers=-1,  # -1 means use full model
    temperature=2.0,
    alpha=0.5
)

# Create dataloaders
train_loader, val_loader, num_labels = create_superglue_dataloaders(
    task_name="cb",
    tokenizer=model.tokenizer,
    batch_size=8
)

# Train the model
model.fit(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=3,
    report_to="wandb",  # or "tensorboard", or None
    output_dir="./output/cb"
)

# Save the student model
model.student_model.save_pretrained("./output/cb/final_model")
```

### Language Modeling Task (e.g., Alpaca)

```python
from transject import AutoModel
from transject.data_utils import create_alpaca_dataloaders

# Create model
model = AutoModel(
    student_model="gpt2",
    teacher_model="gpt2-large",
    student_layers=-1,
    temperature=2.0,
    alpha=0.5
)

# Create dataloaders
train_loader, val_loader = create_alpaca_dataloaders(
    tokenizer=model.tokenizer,
    batch_size=4,
    num_train_samples=1000  # Use subset for quick testing
)

# Train the model
model.fit(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=3,
    report_to="wandb",
    output_dir="./output/alpaca"
)

# Save the student model
model.student_model.save_pretrained("./output/alpaca/final_model")
```

### With Meta-Learning

```python
from transject import SequenceClassification
from transject.data_utils import create_superglue_dataloaders, create_meta_dataloaders

# Create model
model = SequenceClassification(
    student_model="distilbert-base-uncased",
    teacher_model="bert-base-uncased",
    num_labels=3,
    use_meta_learning=True
)

# Create main dataloader
train_loader, val_loader, _ = create_superglue_dataloaders(
    task_name="cb",
    tokenizer=model.tokenizer,
    batch_size=8
)

# Create meta-learning dataloaders
meta_loaders = create_meta_dataloaders(
    tokenizer=model.tokenizer,
    tasks=["rte", "wic"],  # Additional tasks for meta-learning
    batch_size=8,
    num_samples_per_task=100
)

# Train with meta-learning
model.fit(
    train_dataloader=train_loader,
    meta_dataloader=meta_loaders,  # Dictionary of dataloaders
    val_dataloader=val_loader,
    epochs=3,
    report_to="wandb"
)
```

## Advanced Configuration

```python
from transject import TransJectConfig, SequenceClassification

# Custom configuration
config = TransJectConfig(
    student_layers=-1,  # -1 for full model, or specify number of layers
    temperature=2.0,
    alpha=0.5,
    learning_rate=5e-5,
    warmup_steps=100,
    max_grad_norm=1.0,
    accumulation_steps=4,
    use_meta_learning=True,
    meta_learning_rate=1e-4,
    log_interval=10,
    eval_interval=100,
    save_interval=500,
    fp16=True,  # Mixed precision training
    gradient_checkpointing=False
)

# Create model with config
model = SequenceClassification(
    student_model="distilbert-base-uncased",
    teacher_model="bert-base-uncased",
    num_labels=3,
    config=config
)
```

## Supported Models

TransJect works with any HuggingFace model. Some examples:

### Classification Models
- BERT, RoBERTa, DistilBERT
- ELECTRA, ALBERT
- DeBERTa, DeBERTa-v3

### Language Models
- GPT-2 (all sizes)
- Llama-3 (8B and larger)
- OPT, BLOOM
- GPT-Neo, GPT-J

## Supported Datasets

### Classification
- SuperGLUE tasks (CB, RTE, WiC, etc.)
- GLUE tasks
- Custom classification datasets

### Language Modeling
- Alpaca
- Dolly
- Custom instruction datasets

## API Reference

### Models

- `SequenceClassification`: For classification tasks
- `AutoModel`: For causal language modeling

### Configuration

- `TransJectConfig`: Configuration class for all hyperparameters

### Data Utilities

- `create_superglue_dataloaders()`: Create dataloaders for SuperGLUE tasks
- `create_alpaca_dataloaders()`: Create dataloaders for Alpaca dataset
- `create_meta_dataloaders()`: Create meta-learning dataloaders

## Examples

See the `examples/` directory for complete examples:

- `TransJect_Demo.ipynb`: Complete demo with CB and Alpaca
- Classification examples
- Language modeling examples
- Meta-learning examples

## Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## Citation

If you use TransJect in your research, please cite:

```bibtex
@software{transject2024,
  title={TransJect: A Novel Knowledge Transfer Framework},
  author={TransJect Team},
  year={2024},
  url={https://github.com/transject/transject}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built with PyTorch and HuggingFace Transformers.
