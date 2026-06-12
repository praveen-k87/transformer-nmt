# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Data loading and preprocessing module for Neural Machine Translation.

This module provides utilities to download the bilingual dataset from Hugging Face,
clean and normalize the text (Unicode NFC, lowercasing, and spacing out punctuation),
and yield parallel sentence pairs ready for tokenization.
"""

import os
import re
import unicodedata
from typing import List, Tuple

from datasets import load_dataset
from dotenv import load_dotenv

from src.config import DATASET_NAME, SRC_LANG, TGT_LANG

# Load environment variables from .env file if present
load_dotenv()


def load_and_preprocess_data() -> (
    Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]
):
    """Loads the NMT dataset from Hugging Face and applies textual preprocessing.

    This function iterates through the 'train', 'validation', and 'test' splits of
    the configured dataset. For each example, it extracts the source and target language
    sentences, applies Unicode normalization and cleaning, and pairs them up.

    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
            A tuple containing three lists of sentence pairs (source, target) for the
            training, validation, and test sets respectively. Empty sentences are filtered out.
    """
    # Load the specified bilingual dataset (e.g., 'bentrevett/multi30k')
    # Use HF_TOKEN from environment if available to prevent rate limits or warnings.
    hf_token = os.environ.get("HF_TOKEN")
    dataset = load_dataset(DATASET_NAME, token=hf_token)

    train_pairs: List[Tuple[str, str]] = []
    val_pairs: List[Tuple[str, str]] = []
    test_pairs: List[Tuple[str, str]] = []

    for split in ["train", "validation", "test"]:
        for example in dataset[split]:
            src_sentence = preprocess_text(example[SRC_LANG])
            tgt_sentence = preprocess_text(example[TGT_LANG])

            # Ensure neither sentence became empty after preprocessing
            if src_sentence and tgt_sentence:
                if split == "train":
                    train_pairs.append((src_sentence, tgt_sentence))
                elif split == "validation":
                    val_pairs.append((src_sentence, tgt_sentence))
                else:
                    test_pairs.append((src_sentence, tgt_sentence))

    return train_pairs, val_pairs, test_pairs


def preprocess_text(text: str) -> str:
    """Cleans a raw text string by normalizing Unicode characters and handling punctuation.

    Steps applied:
        1. NFC Unicode normalization (crucial for languages like German with umlauts).
        2. Lowercasing to reduce vocabulary size.
        3. Isolating standard punctuation marks with spaces so tokenizers treat them
           as distinct subword tokens.
        4. Collapsing multiple spaces into a single space and stripping whitespace edges.

    Args:
        text (str): The raw input sentence to be processed.

    Returns:
        str: The fully preprocessed and normalized sentence.
    """
    # Step 1: Unicode normalization (NFC composes characters, useful for German umlauts)
    text = unicodedata.normalize("NFC", text)

    # Step 2: Lowercasing
    text = text.lower()

    # Step 3: Punctuation spacing (insert spaces around specific punctuation marks)
    text = re.sub(r"([?.!,¿])", r" \1 ", text)

    # Step 4: Collapse multiple consecutive whitespace characters into a single space
    text = re.sub(r" +", " ", text)

    return text.strip()


def show_preprocessing_examples(
    raw_pairs: List[Tuple[str, str]], processed_pairs: List[Tuple[str, str]], num_examples: int = 3
) -> None:
    """Utility function to print side-by-side comparisons of raw and preprocessed data.

    Args:
        raw_pairs (List[Tuple[str, str]]): List containing the original (source, target) tuples.
        processed_pairs (List[Tuple[str, str]]): List containing the processed (source, target) tuples.
        num_examples (int, optional): The number of examples to display. Defaults to 3.
    """
    print("--- Preprocessing Examples ---")
    for i in range(num_examples):
        raw_src, raw_tgt = raw_pairs[i]
        proc_src, proc_tgt = processed_pairs[i]
        print(f"\\nExample {i+1}:")
        print(f"  Raw SRC:       {raw_src}")
        print(f"  Processed SRC: {proc_src}")
        print(f"  Raw TGT:       {raw_tgt}")
        print(f"  Processed TGT: {proc_tgt}")
    print("----------------------------")


def main() -> None:
    """Local test execution."""
    # Local execution guard for testing the module directly
    hf_token = os.environ.get("HF_TOKEN")
    raw_dataset = load_dataset(DATASET_NAME, token=hf_token)
    raw_train_pairs = [(ex[SRC_LANG], ex[TGT_LANG]) for ex in raw_dataset["train"]]

    train_pairs, val_pairs, test_pairs = load_and_preprocess_data()
    print(f"Total training pairs: {len(train_pairs)}")
    print(f"Total validation pairs: {len(val_pairs)}")
    print(f"Total test pairs: {len(test_pairs)}")

    show_preprocessing_examples(raw_train_pairs, train_pairs)


if __name__ == "__main__":
    main()
