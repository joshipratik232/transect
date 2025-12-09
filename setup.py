"""Setup configuration for TransJect package."""

from setuptools import setup, find_packages
import os

# Read version
version = {}
with open(os.path.join("transject", "__version__.py")) as f:
    exec(f.read(), version)

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="transject",
    version=version["__version__"],
    author="TransJect Team",
    author_email="transject@example.com",
    description="A novel knowledge transfer framework for neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/transject/transject",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
        "tensorboard": [
            "tensorboard>=2.13.0",
        ],
        "all": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "transject=transject.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
