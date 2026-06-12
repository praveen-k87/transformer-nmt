# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Interactive Translation script."""


import torch
from src.config import (
    D_FF,
    D_MODEL,
    DEVICE,
    DROPOUT,
    MAX_LEN,
    NUM_DECODER_LAYERS,
    NUM_ENCODER_LAYERS,
    N_HEADS,
    OUTPUT_DIR,
)
from src.data import preprocess_text
from src.inference import greedy_decode
from src.masks import create_source_padding_mask
from src.tokenizer_train import decode_target, load_existing_tokenizers
from src.transformer import Seq2SeqTransformer
from src.utils import load_checkpoint

# pylint: disable=duplicate-code


def main():
    """Interactive translation interface."""
    print("Loading tokenizers...")
    src_tokenizer, tgt_tokenizer = load_existing_tokenizers()

    model = Seq2SeqTransformer(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(DEVICE)

    print(f"Loading model from {OUTPUT_DIR / 'model.pt'}...")
    try:
        load_checkpoint(model, OUTPUT_DIR / "model.pt")
    except FileNotFoundError:
        print("Error: Model checkpoint not found. Please run train.py first.")
        return

    model.eval()

    print(f"Using device: {DEVICE}")
    print("Welcome to the NMT Interactive Translator!")
    print("Type an English sentence and press Enter to translate it to German.")
    print("Type 'exit' or 'quit' to close the program.")

    while True:
        english_sentence = input("\nEnglish: ")
        if english_sentence.lower() == "exit" or english_sentence.lower() == "quit":
            break

        # Preprocess the input sentence
        preprocessed_sentence = preprocess_text(english_sentence)

        # Convert to tensor
        src_ids = src_tokenizer.encode(preprocessed_sentence).ids
        src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)
        src_mask = create_source_padding_mask(src, DEVICE)

        # Perform translation
        with torch.no_grad():
            output_tokens = greedy_decode(model, src, src_mask, MAX_LEN, DEVICE)
            german_translation = decode_target(tgt_tokenizer, output_tokens.tolist())

        print(f"German: {german_translation}")


if __name__ == "__main__":
    main()
