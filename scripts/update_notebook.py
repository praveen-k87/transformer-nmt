import json
from pathlib import Path

notebook_path = Path("/Users/praveen/Workspace/GitHub/python/transformer-nmt/notebooks/transformer_nmt.ipynb")

with open(notebook_path, 'r') as f:
    notebook = json.load(f)

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))]
    }

cells_to_add = [
    """import torch
import pandas as pd
from pathlib import Path
import sys
import os

# Ensure src is in the python path
sys.path.append(os.path.abspath('..'))

from src.config import *
from src.data import load_and_preprocess_data, show_preprocessing_examples
from src.tokenizer_train import get_or_train_tokenizers
from src.masks import create_masks
from src.transformer import Seq2SeqTransformer
from src.utils import count_trainable_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")""",

    """print("Loading dataset...")
train_pairs, val_pairs, test_pairs = load_and_preprocess_data()
print(f"Loaded {len(train_pairs)} training, {len(val_pairs)} validation, and {len(test_pairs)} test pairs.")

print("\\nShowing preprocessing examples...")
from datasets import load_dataset
raw_dataset = load_dataset(DATASET_NAME)
raw_train_pairs = [(ex[SRC_LANG], ex[TGT_LANG]) for ex in raw_dataset["train"]]
show_preprocessing_examples(raw_train_pairs, train_pairs, num_examples=3)""",

    """print("Loading or training tokenizers...")
src_tokenizer, tgt_tokenizer = get_or_train_tokenizers(train_pairs)
print(f"Source vocabulary size: {src_tokenizer.get_vocab_size()}")
print(f"Target vocabulary size: {tgt_tokenizer.get_vocab_size()}")""",

    """print("Checking masks...")
# Create dummy data
dummy_src = torch.tensor([[1, 2, 3, PAD_IDX, PAD_IDX]]).to(device)
dummy_tgt = torch.tensor([[1, 4, 5, 6, PAD_IDX]]).to(device)

src_mask, tgt_mask = create_masks(dummy_src, dummy_tgt, device)
print(f"Source Mask Shape: {src_mask.shape}")
print(f"Target Mask Shape: {tgt_mask.shape}")
print(f"Target Mask (Causal + Padding):\\n{tgt_mask[0, 0]}")""",

    """print("Initializing model...")
model = Seq2SeqTransformer(
    src_vocab_size=src_tokenizer.get_vocab_size(),
    tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN
).to(device)

param_count = count_trainable_parameters(model)
print(f"Model initialized with {param_count:,} trainable parameters.")""",

    """# Check if outputs exist
sample_csv_path = OUTPUT_DIR / "sample_translations.csv"
bleu_score_path = OUTPUT_DIR / "bleu_score.txt"

if not sample_csv_path.exists() or not bleu_score_path.exists():
    print("Output files are missing!")
    print("Please run `python train.py` and then `python evaluate.py` first to generate the models and evaluations.")
else:
    with open(bleu_score_path, "r") as f:
        bleu_score = f.read().strip()
    print(f"Results found!\\n{bleu_score}")
    
    print("\\nSample Translations:")
    df = pd.read_csv(sample_csv_path)
    pd.set_option('display.max_colwidth', None)
    display(df.head())"""
]

for cell_source in cells_to_add:
    notebook['cells'].append(create_code_cell(cell_source))

with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully.")
