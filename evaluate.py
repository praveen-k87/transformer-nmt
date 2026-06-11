# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Evaluation and inference pipeline for the NMT model.

This module handles loading a fully trained model checkpoint, running both Greedy
and Beam Search decoding strategies on a test dataset, calculating the SacreBLEU
metric, and saving sample qualitative translations to CSV format.
"""

import pandas as pd
import sacrebleu
# pylint: disable=duplicate-code
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from src.data import load_and_preprocess_data
from src.dataset import TranslationDataset, collate_fn
from src.inference import beam_search_decode, greedy_decode
from src.masks import create_source_padding_mask
from src.tokenizer_train import decode_target, get_or_train_tokenizers
from src.transformer import Seq2SeqTransformer
from src.utils import ensure_output_dirs, load_checkpoint, qualitative_comment


def main() -> None:
    """Main entrypoint for model evaluation and inference."""
    ensure_output_dirs()

    print("Loading and preprocessing data...")
    # Discard train/val, we only care about the hold-out test set here.
    _, _, test_pairs = load_and_preprocess_data()

    print("Loading tokenizers...")
    # Get existing tokenizers (they must have been created during training).
    src_tokenizer, tgt_tokenizer = get_or_train_tokenizers(test_pairs)

    test_dataset = TranslationDataset(test_pairs, src_tokenizer, tgt_tokenizer)
    # Batch size is strictly 1 for autoregressive decoding implementation simplicity.
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Re-initialize the model architecture exactly as configured in src.config.
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

    # Load weights from the best checkpoint.
    checkpoint_path = OUTPUT_DIR / "model.pt"
    print(f"Loading model from {checkpoint_path}...")
    try:
        load_checkpoint(model, checkpoint_path)
    except FileNotFoundError:
        print("Error: Model checkpoint not found. Please run train.py first to generate a model.")
        return

    # Lock Dropout and BatchNorm layers for deterministic inference.
    model.eval()

    greedy_translations = []
    beam_translations = []
    source_sentences = []
    reference_sentences = []
    qualitative_comments = []

    print("Generating translations on the test set...")
    for src, _tgt, src_text, ref_text in tqdm(test_dataloader, desc="Translating"):
        src = src.to(DEVICE)

        # Only the source padding mask is needed for inference encoding.
        src_mask = create_source_padding_mask(src, DEVICE)

        # Execute Greedy Decoding
        greedy_output_tokens = greedy_decode(model, src, src_mask, MAX_LEN, DEVICE)
        greedy_translation = decode_target(tgt_tokenizer, greedy_output_tokens.tolist())

        # Execute Beam Search Decoding
        beam_output_tokens = beam_search_decode(model, src, src_mask, MAX_LEN, DEVICE, beam_width=3)
        beam_translation = decode_target(tgt_tokenizer, beam_output_tokens.tolist())

        # Collect results for qualitative review
        source_sentences.append(src_text[0])
        reference_sentences.append(ref_text[0])
        greedy_translations.append(greedy_translation)
        beam_translations.append(beam_translation)

        # Generate a placeholder review comment based on greedy results.
        qualitative_comments.append(
            qualitative_comment(src_text[0], ref_text[0], greedy_translation)
        )

    print("Calculating BLEU score...")
    # SacreBLEU is the industry-standard BLEU implementation.
    # It expects references as a list-of-lists: [[ref_1_sys_1, ref_1_sys_2...], [ref_2_sys_1...]]
    bleu = sacrebleu.corpus_bleu(greedy_translations, [reference_sentences])
    print(f"\\nFinal BLEU Score (Greedy Decoding): {bleu.score:.2f}")

    # Export translation samples to CSV for manual inspection.
    results_df = pd.DataFrame(
        {
            "source_english": source_sentences,
            "reference_german": reference_sentences,
            "greedy_translation": greedy_translations,
            "beam_translation": beam_translations,
            "qualitative_comment": qualitative_comments,
        }
    )

    csv_path = OUTPUT_DIR / "sample_translations.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Sample translations saved to {csv_path}")

    # Export the numeric BLEU score.
    bleu_path = OUTPUT_DIR / "bleu_score.txt"
    with open(bleu_path, "w", encoding="utf-8") as f:
        f.write(f"BLEU Score (Greedy Decoding): {bleu.score:.2f}\\n")
    print(f"BLEU score saved to {bleu_path}")


if __name__ == "__main__":
    main()
