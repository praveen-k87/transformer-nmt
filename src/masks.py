# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Tensor masking module for the Transformer architecture.

This module provides functions to create boolean masks for both the source
and target sequences. These masks ensure that the self-attention and cross-attention
mechanisms ignore padding tokens and (in the decoder) prevent attending to future tokens.
"""

from typing import Tuple

import torch

from src.config import PAD_IDX


def create_source_padding_mask(src: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Creates a boolean padding mask for the source sequence.

    This mask is applied in the Encoder's self-attention and the Decoder's
    cross-attention layers to prevent the model from attending to `<pad>` tokens.

    Args:
        src (torch.Tensor): The source sequence tensor of shape `(batch_size, src_len)`.
        device (torch.device): The computation device.

    Returns:
        torch.Tensor: A boolean tensor of shape `(batch_size, 1, 1, src_len)`.
            `True` indicates a valid token, while `False` indicates a padding token.
            The extra dimensions `(1, 1)` are added for broadcasting across the
            attention heads and the query sequence length.
    """
    # Create a boolean tensor where True represents non-padding tokens.
    # Unsqueeze twice to align dimensions for broadcasted element-wise operations
    # against the attention scores tensor of shape (batch, n_heads, seq_len, seq_len).
    return (src != PAD_IDX).unsqueeze(1).unsqueeze(2).to(device)


def create_target_padding_mask(tgt: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Creates a boolean padding mask for the target sequence.

    This mask is used to identify `<pad>` tokens in the target sequence so the
    Decoder's self-attention does not incorporate them.

    Args:
        tgt (torch.Tensor): The target sequence tensor of shape `(batch_size, tgt_len)`.
        device (torch.device): The computation device.

    Returns:
        torch.Tensor: A boolean tensor of shape `(batch_size, 1, 1, tgt_len)`.
    """
    return (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2).to(device)


def create_causal_mask(tgt_len: int, device: torch.device) -> torch.Tensor:
    """Creates a causal (look-ahead) mask for the target sequence.

    Because the Decoder generates sequences autoregressively, it must not "look ahead"
    at future tokens when predicting the next token. This function generates a lower
    triangular matrix to enforce causality.

    Args:
        tgt_len (int): The current length of the target sequence.
        device (torch.device): The computation device.

    Returns:
        torch.Tensor: A boolean tensor of shape `(tgt_len, tgt_len)`.
            `True` allows attention, `False` blocks it.
    """
    # torch.ones creates a matrix of 1s.
    # torch.tril zeroes out everything strictly above the main diagonal.
    # .bool() converts it to a boolean tensor where True permits attention.
    return torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()


def create_masks(
    src: torch.Tensor, tgt: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates all necessary masks (source and target) for a forward pass.

    This function coordinates the creation of the source padding mask and the
    target mask (which is a logical AND combination of the target padding mask
    and the causal look-ahead mask).

    Args:
        src (torch.Tensor): Source sequence tensor of shape `(batch_size, src_len)`.
        tgt (torch.Tensor): Target sequence tensor of shape `(batch_size, tgt_len)`.
        device (torch.device): The computation device.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - src_mask: The boolean source padding mask.
            - tgt_mask: The combined boolean target mask.
    """
    # Generate the source mask.
    src_mask = create_source_padding_mask(src, device)

    # Generate the components for the target mask.
    tgt_pad_mask = create_target_padding_mask(tgt, device)
    tgt_len = tgt.shape[1]
    tgt_causal_mask = create_causal_mask(tgt_len, device)

    # The final target mask requires the position to be valid in both the
    # causal mask (not a future token) AND the padding mask (not a pad token).
    tgt_mask = tgt_pad_mask & tgt_causal_mask

    return src_mask, tgt_mask


def main() -> None:
    """Run local sanity checks for masks."""
    # Sanity checks for local testing
    device_tensor = torch.device("cpu")
    src_tensor = torch.tensor([[1, 2, 3, 0, 0]])
    tgt_tensor = torch.tensor([[1, 4, 5, 6, 0]])

    src_mask_tensor, tgt_mask_tensor = create_masks(src_tensor, tgt_tensor, device_tensor)
    print("Source Mask Shape:", src_mask_tensor.shape)
    print("Source Mask:", src_mask_tensor)
    print("Target Mask Shape:", tgt_mask_tensor.shape)
    print("Target Mask:", tgt_mask_tensor)

if __name__ == "__main__":
    main()
