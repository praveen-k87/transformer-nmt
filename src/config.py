# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Configuration module for the Neural Machine Translation project.

This module centralizes all project paths, dataset configurations, special tokens,
device selections, and hyperparameters (for both the model and the training loop).
Keeping these constants in a single file ensures consistency across the entire codebase.
"""

from pathlib import Path

import torch

# =============================================================================
# Project Paths
# =============================================================================
# Root directory of the project, resolved absolutely to avoid relative path issues.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Standardized directories for storing various artifacts.
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
SRC_DIR: Path = PROJECT_ROOT / "src"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
REPORT_DIR: Path = PROJECT_ROOT / "report"

# =============================================================================
# Dataset Configuration
# =============================================================================
# Hugging Face dataset identifier.
DATASET_NAME: str = "bentrevett/multi30k"

# Language codes used to index into the dataset dictionaries.
SRC_LANG: str = "en"  # Source language: English
TGT_LANG: str = "de"  # Target language: German

# =============================================================================
# Special Tokens
# =============================================================================
# String representations for special tokens used by the BPE tokenizers.
PAD_TOKEN: str = "<pad>"  # Used to pad sequences to the same length in a batch.
BOS_TOKEN: str = "<bos>"  # Beginning-Of-Sequence token.
EOS_TOKEN: str = "<eos>"  # End-Of-Sequence token.
UNK_TOKEN: str = "<unk>"  # Unknown token for out-of-vocabulary words.

# Fixed numerical indices corresponding to the special tokens.
# These must match the order in which they are added to the tokenizer.
PAD_IDX: int = 0
BOS_IDX: int = 1
EOS_IDX: int = 2
UNK_IDX: int = 3

# =============================================================================
# Device Selection
# =============================================================================
# Automatically detect and assign the best available hardware accelerator.
# MPS (Metal Performance Shaders) is prioritized for Apple Silicon, followed by CUDA for NVIDIA GPUs.
if torch.backends.mps.is_available():
    DEVICE: torch.device = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE: torch.device = torch.device("cuda")
else:
    DEVICE: torch.device = torch.device("cpu")

# =============================================================================
# Model Hyperparameters
# =============================================================================
# D_MODEL: The embedding dimension and hidden state size across all layers.
# Must be perfectly divisible by N_HEADS.
D_MODEL: int = 256
N_HEADS: int = 8  # Number of attention heads.
NUM_ENCODER_LAYERS: int = 3  # Number of stacked layers in the Encoder.
NUM_DECODER_LAYERS: int = 3  # Number of stacked layers in the Decoder.
D_FF: int = 512  # Dimensionality of the inner Position-wise Feed-Forward network.
DROPOUT: float = 0.1  # Dropout probability applied throughout the network.

# Maximum allowed subword vocabulary size for the BPE tokenizers.
VOCAB_SIZE: int = 8000
# Maximum absolute length (in subwords) of any sequence. Used to instantiate Positional Encoding matrices.
MAX_LEN: int = 80

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE: int = 32  # Number of sentence pairs per training step.
EPOCHS: int = 8  # Number of complete passes over the dataset.
WARMUP_STEPS: int = 4000  # Number of steps during which the learning rate increases linearly.

# Initial base learning rate. The actual learning rate is dynamically controlled by NoamScheduler.
LEARNING_RATE: float = 1e-4

# Adam optimizer configuration, closely following the "Attention is All You Need" paper.
ADAM_BETAS: tuple[float, float] = (0.9, 0.98)
ADAM_EPS: float = 1e-9

# Maximum norm for gradient clipping to prevent exploding gradients.
GRADIENT_CLIPPING: float = 1.0

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_SEED: int = 42  # Seed to ensure deterministic behavior across runs.
