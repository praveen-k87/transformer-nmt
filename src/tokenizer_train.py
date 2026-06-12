# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Tokenizer training and serialization module.

This module provides functionality to train, save, load, and utilize Byte-Pair Encoding (BPE)
tokenizers using the Hugging Face `tokenizers` library. Separate tokenizers are
trained for the source and target languages to optimally model their distinct vocabularies.
"""

from pathlib import Path
from typing import List, Tuple

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from src.config import (
    BOS_TOKEN,
    EOS_TOKEN,
    OUTPUT_DIR,
    PAD_TOKEN,
    UNK_TOKEN,
    VOCAB_SIZE,
)


def train_tokenizer(texts: List[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Trains a BPE subword tokenizer on a provided list of sentences.

    The tokenizer splits words on whitespace first, then applies Byte-Pair Encoding.
    It automatically appends Beginning-Of-Sequence `<bos>` and End-Of-Sequence `<eos>`
    tokens to every encoded sequence via a post-processing template.

    Args:
        texts (List[str]): A list of string sentences to train the tokenizer on.
        vocab_size (int, optional): The maximum number of tokens in the vocabulary.
            Defaults to the `VOCAB_SIZE` defined in config.

    Returns:
        Tokenizer: A fully trained Hugging Face `Tokenizer` object.
    """
    # Initialize a BPE model with a designated unknown token.
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # Pre-tokenize strings by splitting on whitespace before BPE processing.
    tokenizer.pre_tokenizer = Whitespace()

    # Configure the BPE trainer with the target vocabulary size and required special tokens.
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
    )

    # Train the tokenizer strictly from the provided text iterator.
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Attach a post-processor to automatically wrap encoded sequences:
    # Example: "hello world" -> "<bos> hello world <eos>"
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    """Serializes a Tokenizer object to a JSON file.

    Args:
        tokenizer (Tokenizer): The trained tokenizer to save.
        path (Path): The destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def load_existing_tokenizers() -> Tuple[Tokenizer, Tokenizer]:
    """Strictly loads tokenizers from disk without training them.

    Used during evaluation to prevent data leakage.
    """
    src_path = OUTPUT_DIR / "src_tokenizer.json"
    tgt_path = OUTPUT_DIR / "tgt_tokenizer.json"

    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError("Tokenizer files not found. Run train.py first.")

    return load_tokenizer(src_path), load_tokenizer(tgt_path)


def load_tokenizer(path: Path) -> Tokenizer:
    """Loads a serialized Tokenizer object from a JSON file.

    Args:
        path (Path): The source file path.

    Returns:
        Tokenizer: The loaded tokenizer ready for encoding/decoding.
    """
    return Tokenizer.from_file(str(path))


def get_or_train_tokenizers(
    train_pairs: List[Tuple[str, str]],
) -> Tuple[Tokenizer, Tokenizer]:
    """Retrieves existing tokenizers from disk, or trains new ones if they don't exist.

    This ensures tokenizers are only trained once and reused for inference/evaluation.

    Args:
        train_pairs (List[Tuple[str, str]]): The dataset containing parallel source
            and target sentence tuples. Used for training if tokenizers are missing.

    Returns:
        Tuple[Tokenizer, Tokenizer]: A tuple containing the `(src_tokenizer, tgt_tokenizer)`.
    """
    src_tokenizer_path = OUTPUT_DIR / "src_tokenizer.json"
    tgt_tokenizer_path = OUTPUT_DIR / "tgt_tokenizer.json"

    if src_tokenizer_path.exists() and tgt_tokenizer_path.exists():
        print("Loading existing tokenizers...")
        src_tokenizer = load_tokenizer(src_tokenizer_path)
        tgt_tokenizer = load_tokenizer(tgt_tokenizer_path)
    else:
        print("Training new tokenizers...")
        # Unzip the parallel corpus into separate lists for training
        src_texts = [pair[0] for pair in train_pairs]
        tgt_texts = [pair[1] for pair in train_pairs]

        src_tokenizer = train_tokenizer(src_texts)
        tgt_tokenizer = train_tokenizer(tgt_texts)

        save_tokenizer(src_tokenizer, src_tokenizer_path)
        save_tokenizer(tgt_tokenizer, tgt_tokenizer_path)

    return src_tokenizer, tgt_tokenizer


def encode_source(tokenizer: Tokenizer, text: str) -> List[int]:
    """Encodes a single source sentence into a list of token IDs.

    Args:
        tokenizer (Tokenizer): The source tokenizer.
        text (str): The input string.

    Returns:
        List[int]: The encoded token IDs including `<bos>` and `<eos>`.
    """
    return tokenizer.encode(text).ids


def encode_target(tokenizer: Tokenizer, text: str) -> List[int]:
    """Encodes a single target sentence into a list of token IDs.

    Args:
        tokenizer (Tokenizer): The target tokenizer.
        text (str): The input string.

    Returns:
        List[int]: The encoded token IDs including `<bos>` and `<eos>`.
    """
    return tokenizer.encode(text).ids


def decode_source(tokenizer: Tokenizer, ids: List[int]) -> str:
    """Decodes a list of token IDs back into a source string.

    Args:
        tokenizer (Tokenizer): The source tokenizer.
        ids (List[int]): The sequence of token IDs.

    Returns:
        str: The decoded string, with special tokens stripped out.
    """
    return tokenizer.decode(ids, skip_special_tokens=True)


def decode_target(tokenizer: Tokenizer, ids: List[int]) -> str:
    """Decodes a list of token IDs back into a target string.

    Args:
        tokenizer (Tokenizer): The target tokenizer.
        ids (List[int]): The sequence of token IDs.

    Returns:
        str: The decoded string, with special tokens stripped out.
    """
    return tokenizer.decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    # Local execution guard for testing tokenizer training.
    from src.data import load_and_preprocess_data

    sample_pairs, _, _ = load_and_preprocess_data()
    src_tok, tgt_tok = get_or_train_tokenizers(sample_pairs)
    print(f"Source vocab size: {src_tok.get_vocab_size()}")
    print(f"Target vocab size: {tgt_tok.get_vocab_size()}")
