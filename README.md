# Full Sequence-to-Sequence Encoder-Decoder Transformer for Neural Machine Translation

**Implemented from scratch in PyTorch without pretrained translation models or torch.nn.Transformer.**

## Project Overview

This repository contains a mathematically precise implementation of the original "Attention Is All You Need" Transformer architecture. Built entirely from scratch, the model maps English source sequences to German target sequences using a full Encoder-Decoder paradigm with Cross-Attention.

The model is trained on the `bentrevett/multi30k` dataset, a high-quality corpus of bilingual image captions.

## Key Features

- **Zero Pretrained Libraries**: No `torch.nn.Transformer`, Hugging Face pipelines, or pre-trained translation weights are used.
- **Dual BPE Tokenizers**: Two distinct Word-Level Byte-Pair Encoding (BPE) tokenizers trained strictly on the training set to respect the independent morphology of English and German.
- **Cross-Attention Mapping**: Explicit Query (from Decoder), Key (from Encoder), and Value (from Encoder) passing for robust autoregressive alignment.
- **Noam Scheduler**: Custom learning rate scaling with warmup steps to prevent early gradient explosion during deep attention initialization.
- **Robust Evaluation**: Comprehensive SacreBLEU evaluation for both Greedy Decoding and Beam Search Decoding.

## Architecture Summary

| Component           | Implementation Details                                                       |
|---------------------|------------------------------------------------------------------------------|
| **Embedding**       | Vocabulary Lookup + Sinusoidal Positional Encoding                           |
| **Encoder**         | Multi-Head Self-Attention + Position-Wise Feed-Forward Network               |
| **Decoder**         | Masked Self-Attention + Cross-Attention + Position-Wise Feed-Forward Network |
| **Hyperparameters** | `d_model=256`, `heads=8`, `layers=3`, `dropout=0.1`                          |

## Repository Structure

```text
transformer-nmt/
│
├── notebooks/
│   └── transformer_nmt.ipynb    # Executable academic notebook with full tasks mapped
├── report/
│   ├── Project_Report.md        # Detailed academic report
│   └── Project_Report.pdf       # Exported academic submission report
├── src/
│   ├── config.py                # Hyperparameters and path configurations
│   ├── data.py                  # Dual-language preprocessing (NFC normalization)
│   ├── dataset.py               # Dataset class with dynamic padding and MAX_LEN safeties
│   ├── inference.py             # Greedy and Beam Search decoding implementations
│   ├── masks.py                 # Source padding, Target causal, and Cross-Attention masking
│   ├── scheduler.py             # Custom Noam Learning Rate scheduler
│   ├── tokenizer_train.py       # BPE tokenizer training logic
│   ├── transformer.py           # Core Transformer architecture (Encoder/Decoder/MHA)
│   └── utils.py                 # Seeds, parameter counting, and plotting
│
├── outputs/                     # Generated artifacts (ignored by git unless requested)
│   ├── src_tokenizer.json       # Trained English BPE vocabulary
│   ├── tgt_tokenizer.json       # Trained German BPE vocabulary
│   ├── model.pt                 # Best epoch PyTorch weights
│   ├── training_curve.png       # Loss visualization
│   ├── bleu_score.txt           # Final greedy and beam search SacreBLEU metrics
│   └── sample_translations.csv  # Output samples with qualitative text analysis
│
├── train.py                     # Execution script for the training loop with Teacher Forcing
├── evaluate.py                  # Execution script for test set SacreBLEU evaluation
├── translate.py                 # Interactive terminal for live translations
└── requirements.txt             # Environment dependencies
```

## Assignment Task Mapping

This repository rigorously adheres to all academic assignment objectives:

- **Task 1: Dual-Language Preprocessing**: Located in `src/data.py` with NFC preservation for German characters.
- **Task 2: Shared vs Separate Tokenizer Training**: Located in `src/tokenizer_train.py` (Separate BPEs used for morphological independence).
- **Task 3: Cross-Attention Mechanism**: Located in `src/transformer.py` (`DecoderLayer`).
- **Task 4: Positional and Padding Masking**: Located in `src/masks.py` and `src/inference.py`.
- **Task 5: Training Loop with Teacher Forcing**: Located in `train.py`.
- **Task 6: Qualitative and Quantitative Translation Analysis**: Located in `evaluate.py`.

## Setup Instructions

1. **Clone the repository**:

```bash
git clone <repository-url>
cd transformer-nmt
```

2. **Create and activate a virtual environment**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

*(Note: The codebase automatically detects hardware acceleration, prioritizing `MPS` for Apple Silicon, `CUDA` for NVIDIA GPUs, and falling back to `CPU`).*

## Commands

### 1. Training

To train the Seq2Seq NMT model from scratch, run:

```bash
python train.py
```

*Outputs: `model.pt`, `training_curve.png`, `training_summary.txt`, `training_history.csv`*

### 2. Evaluation

To evaluate the model on the hold-out test set and generate BLEU scores and samples, run:

```bash
python evaluate.py
```

*Outputs: `bleu_score.txt`, `sample_translations.csv`*

### 3. Interactive Translation

To test the model live with custom English input, run:

```bash
python translate.py
```

## Limitations

- **Domain Overfitting**: The model is highly specific to the `multi30k` dataset, which consists exclusively of physical image descriptions. It will degrade or hallucinate when presented with complex, conversational English.
- **Fixed Max Sequence Length**: The hard limit of `MAX_LEN=80` prevents processing very long documents, dropping the trailing syntax.

## Future Improvements

- **Key-Value Caching**: The current autoregressive decoder recomputes keys and values for all past tokens at every step. Implementing a KV cache mechanism would significantly accelerate inference.
- **WMT14 Expansion**: Upgrading the training corpus to the larger WMT14 English-German dataset to grant the model general conversational understanding.

## License

MIT License.
