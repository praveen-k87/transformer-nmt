"""Dataset classes and collate functions for the NMT project.

Provides a standard PyTorch Dataset for loading translation sentence pairs
and a collate function to handle dynamic batch padding.
"""

from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from src.config import MAX_LEN, PAD_IDX


class TranslationDataset(Dataset):
    """Dataset class for parallel translation sentence pairs.

    Returns the original text strings along with the encoded tensors.

    Attributes:
        pairs (List[Tuple[str, str]]): List of parallel (source, target) string pairs.
        src_tokenizer: The trained source BPE tokenizer.
        tgt_tokenizer: The trained target BPE tokenizer.
        max_len (int): Maximum allowed sequence length.
    """

    def __init__(
        self, pairs: List[Tuple[str, str]], src_tokenizer, tgt_tokenizer, max_len: int = MAX_LEN
    ) -> None:
        """Initializes the TranslationDataset.

        Args:
            pairs (List[Tuple[str, str]]): List of (English, German) sentence tuples.
            src_tokenizer: Hugging Face Tokenizer for the source language.
            tgt_tokenizer: Hugging Face Tokenizer for the target language.
            max_len (int, optional): The maximum length for the sequences. Defaults to MAX_LEN.
        """
        self.pairs = pairs
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Returns the total number of sentence pairs in the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """Fetches and tokenizes the sentence pair at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str, str]: The encoded source and target token ID tensors, and the original strings.
        """
        src_text, tgt_text = self.pairs[idx]
        src_ids = self.src_tokenizer.encode(src_text).ids
        tgt_ids = self.tgt_tokenizer.encode(tgt_text).ids
        return torch.tensor(src_ids), torch.tensor(tgt_ids), src_text, tgt_text


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str, str]],
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Pads a batch of variable-length sequences to the length of the longest sequence in the batch.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, str, str]]): A batch of items.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]: Padded source and target tensors, and lists of strings.
    """
    src_batch, tgt_batch, src_texts, tgt_texts = [], [], [], []
    for src_item, tgt_item, src_text, tgt_text in batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)
        src_texts.append(src_text)
        tgt_texts.append(tgt_text)

    # Pad sequences dynamically using the PAD_IDX constant
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)

    return src_batch, tgt_batch, src_texts, tgt_texts
