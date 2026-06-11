# Full Sequence-to-Sequence Encoder-Decoder Transformer for Neural Machine Translation

**Implemented from scratch in PyTorch without pretrained translation models or `torch.nn.Transformer`.**

## Project Overview
This project provides a complete, from-scratch implementation of the Transformer architecture for Neural Machine Translation (NMT). The model is trained to translate English to German using the Multi30k dataset. It serves as a deep dive into the core mechanisms of the Transformer, including self-attention, cross-attention, and positional encodings, all built manually.

## Key Features
- **Full Encoder-Decoder Architecture**: Implements the complete sequence-to-sequence pipeline.
- **Manual Implementation**: Built from scratch to demonstrate a fundamental understanding of the Transformer architecture.
- **Separate Tokenizers**: Justified use of separate BPE tokenizers for source (English) and target (German) languages.
- **Noam Scheduler**: Implements the learning rate scheduler from the original "Attention is All You Need" paper.
- **Inference Strategies**: Supports both Greedy Decoding and Beam Search for translation.
- **Device Support**: Runs on MPS (Apple Silicon), CUDA, or CPU.
- **Reproducible**: Fully self-contained with clear setup and execution steps.

## Architecture Summary
The model is a faithful implementation of the original Transformer, consisting of:
- **Encoder**: A stack of `N` identical layers, each with a multi-head self-attention mechanism and a position-wise feed-forward network.
- **Decoder**: A stack of `N` identical layers, each with a masked multi-head self-attention, a cross-attention mechanism to attend to the encoder's output, and a feed-forward network.
- **Cross-Attention**: The key to translation, where the decoder queries the encoder's output to align source and target sequences.

## Repository Structure
```
encoder-decoder-transformer-nmt/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── check_device.py
├── train.py
├── evaluate.py
├── translate.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── tokenizer_train.py
│   ├── masks.py
│   ├── transformer.py
│   ├── scheduler.py
│   ├── inference.py
│   └── utils.py
├── notebooks/
│   └── transformer_nmt_walkthrough.ipynb
├── report/
│   └── technical_report.md
└── outputs/
    └── .gitkeep
```

## Setup Instructions
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/encoder-decoder-transformer-nmt.git
    cd encoder-decoder-transformer-nmt
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## How to Run
### 1. Check Device
Verify your PyTorch installation and see which device (MPS, CUDA, or CPU) is selected.
```bash
python check_device.py
```

### 2. Train the Model
This script will train the tokenizers (if they don't exist), train the Transformer model, and save the best checkpoint and training artifacts to the `outputs/` directory.
```bash
python train.py
```

### 3. Evaluate the Model
This script loads the trained model and evaluates it on the test set, calculating the BLEU score and generating sample translations.
```bash
python evaluate.py
```

### 4. Interactive Translation
Use this script to translate your own English sentences to German.
```bash
python translate.py
```

## Output Files
After running the scripts, the `outputs/` directory will contain:
- `src_tokenizer.json`: The trained source tokenizer.
- `tgt_tokenizer.json`: The trained target tokenizer.
- `model.pt`: The best model checkpoint.
- `training_curve.png`: A plot of training and validation loss.
- `training_summary.txt`: A summary of the training process.
- `sample_translations.csv`: A CSV file with source, reference, and translated sentences.
- `bleu_score.txt`: The final BLEU score.

## Results
*The following results are generated after a full training run.*
- **Trainable Parameters**: [To be filled]
- **BLEU Score**: [To be filled]

## Limitations
- The model is trained on a small dataset (Multi30k), which limits its ability to generalize.
- The training hyperparameters are not exhaustively tuned.

## Future Improvements
- Train on a larger dataset like WMT.
- Implement more advanced techniques like label smoothing.
- Experiment with different model sizes and hyperparameters.

## References
- The Annotated Transformer by Harvard NLP.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
