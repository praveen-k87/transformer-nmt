# Full Sequence-to-Sequence Encoder-Decoder Transformer for Neural Machine Translation

**Implemented from scratch in PyTorch without pretrained translation models or `torch.nn.Transformer`.**

## Project Overview
This project provides a complete, from-scratch implementation of the Transformer architecture for Neural Machine Translation (NMT). The model is trained to translate English to German using the `multi30k` dataset. It serves as a deep dive into the core mechanisms of the Transformer, including self-attention, cross-attention, causal masking, and positional encodings, all built manually.

## Key Features
- **Full Encoder-Decoder Architecture**: Implements the complete sequence-to-sequence pipeline.
- **Manual Implementation**: Built from scratch to demonstrate a fundamental understanding of the Transformer architecture.
- **Separate Tokenizers**: Justified use of separate BPE tokenizers for source (English) and target (German) languages.
- **Noam Scheduler**: Implements the learning rate scheduler and Teacher Forcing from the original "Attention is All You Need" paper.
- **Inference Strategies**: Supports both Greedy Decoding and Beam Search for translation generation.
- **Device Support**: Runs seamlessly on MPS (Apple Silicon), CUDA, or CPU.
- **SonarQube Compliant**: The entire codebase achieves a 10.00/10 Pylint score and enforces strict static typing.

## Repository Structure
```
encoder-decoder-transformer-nmt/
├── README.md
├── requirements.txt
├── .env                  <-- (Create this for HuggingFace Token, ignored by git)
├── .gitignore
├── .pylintrc             <-- Strict linting configuration
├── train.py
├── evaluate.py
├── translate.py
├── src/
│   ├── config.py         <-- Centralized hyperparameters
│   ├── data.py           <-- Dataset fetching and text preprocessing
│   ├── dataset.py        <-- PyTorch Dataset and collate functions
│   ├── tokenizer_train.py
│   ├── masks.py
│   ├── transformer.py    <-- Core Architecture
│   ├── scheduler.py
│   ├── inference.py      <-- Greedy and Beam search logic
│   └── utils.py
├── notebooks/
│   └── transformer_nmt.ipynb  <-- Gold standard submission notebook
└── outputs/              <-- Generated artifacts go here
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

3.  **HuggingFace API Token Setup:**
    The dataset (`bentrevett/multi30k`) is downloaded via the HuggingFace datasets library. To prevent anonymous rate limits or connection warnings, provide your HF Token.
    - Create a `.env` file in the root directory (this is automatically ignored by Git).
    - Add the following line:
      ```env
      HF_TOKEN=hf_your_actual_token_here
      ```
    Alternatively, export it in your terminal: `export HF_TOKEN="hf_..."`

## How to Run

### 1. Train the Model
This script will fetch the data, train the tokenizers (if they don't exist), initialize the Transformer, execute the training loop, and save the best checkpoint and training artifacts to the `outputs/` directory.
```bash
python train.py
```

### 2. Evaluate the Model
This script strictly loads the existing tokenizers (preventing data leakage) and evaluates the trained model on the hold-out test set. It calculates the SacreBLEU score and generates qualitative CSV analyses using both Greedy and Beam Search decoding.
```bash
python evaluate.py
```

### 3. Interactive Translation
Use this script to translate your own custom English sentences to German using your trained model.
```bash
python translate.py
```

### 4. Jupyter Notebook (Assignment Submission)
For the final assignment submission, open the meticulously documented Jupyter Notebook. It contains the written theoretical justifications required by the rubric alongside the executable code.
```bash
jupyter notebook notebooks/transformer_nmt.ipynb
```

## Output Artifacts
After running the scripts, the `outputs/` directory will contain:
- `src_tokenizer.json` / `tgt_tokenizer.json`: The trained BPE tokenizers.
- `model.pt`: The optimal model checkpoint.
- `training_curve.png`: A visual plot of training and validation loss.
- `sample_translations.csv`: Translation outputs with qualitative heuristic feedback.
- `bleu_score.txt`: The numeric SacreBLEU evaluation metric.

## License
This project is licensed under the MIT License.
