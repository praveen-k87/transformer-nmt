# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Inference decoding module for Neural Machine Translation.

This module provides autoregressive decoding strategies (Greedy Search and Beam Search)
used to generate translated target sequences from the encoded source sequences.
"""

import torch
from torch import nn
import torch.nn.functional

from src.config import BOS_IDX, EOS_IDX


def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Generates a causal look-ahead mask dynamically."""
    return torch.tril(torch.ones((size, size), device=device)).bool()


def greedy_decode(
    model: nn.Module, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, device: torch.device
) -> torch.Tensor:
    """Performs greedy decoding to generate a translation autoregressively.

    At each time step, this algorithm picks the token with the absolute highest probability.
    It is computationally cheap but might yield suboptimal global sequences because it
    never backtracks to explore alternative word choices.

    Args:
        model (nn.Module): The fully trained Seq2SeqTransformer model.
        src (torch.Tensor): The encoded source sentence tensor of shape `(1, src_len)`.
        src_mask (torch.Tensor): The boolean padding mask for the source sentence.
        max_len (int): The absolute maximum length the generated sequence can reach.
        device (torch.device): The computation device (CPU, CUDA, or MPS).

    Returns:
        torch.Tensor: A 1D tensor representing the generated target token sequence.
    """
    model.eval()
    with torch.no_grad():
        # Step 1: Encode the source sequence once.
        enc_output = model.encode(src, src_mask)

        # Step 2: Initialize the target sequence with just the Beginning-Of-Sequence token.
        # Shape: (batch_size=1, current_len=1)
        tgt = torch.ones(1, 1).fill_(BOS_IDX).long().to(device)

        for _ in range(max_len - 1):
            # Generate the causal (look-ahead) mask dynamically for the current sequence length.
            # Uses torch.triu to create an upper triangular matrix and transposes it.
            tgt_mask = _generate_causal_mask(tgt.size(1), device)

            # Step 3: Pass the current target sequence and encoder outputs through the Decoder.
            output = model.decode(tgt, enc_output, src_mask, tgt_mask)

            # Step 4: Greedily pick the token with the highest probability at the LAST time step.
            pred_token = output.argmax(2)[:, -1].item()

            # Append the predicted token to the target sequence.
            tgt = torch.cat([tgt, torch.ones(1, 1).fill_(pred_token).long().to(device)], 1)

            # Terminate early if the End-Of-Sequence token is generated.
            if pred_token == EOS_IDX:
                break

    # Squeeze out the batch dimension to return a 1D sequence tensor.
    return tgt.squeeze(0)


def beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    device: torch.device,
    beam_width: int = 3,
) -> torch.Tensor:
    """Performs beam search decoding to generate a translation autoregressively.

    Unlike greedy decoding, beam search maintains multiple active hypotheses (beams) at each
    time step. It expands all active beams, calculates their cumulative log probabilities,
    and keeps only the top `beam_width` sequences. This often yields globally better translations.

    Args:
        model (nn.Module): The fully trained Seq2SeqTransformer model.
        src (torch.Tensor): The encoded source sentence tensor of shape `(1, src_len)`.
        src_mask (torch.Tensor): The boolean padding mask for the source sentence.
        max_len (int): The absolute maximum length the generated sequence can reach.
        device (torch.device): The computation device (CPU, CUDA, or MPS).
        beam_width (int, optional): The number of concurrent hypotheses to track. Defaults to 3.

    Returns:
        torch.Tensor: A 1D tensor representing the best generated target token sequence.
    """
    model.eval()
    with torch.no_grad():
        # Encode the source sequence once.
        enc_output = model.encode(src, src_mask)

        # Initialize a list of beams. Each beam is a tuple: (sequence_tensor, log_probability_score)
        beams = [(torch.ones(1, 1).fill_(BOS_IDX).long().to(device), 0.0)]

        for _ in range(max_len - 1):
            new_beams = []

            # Iterate through all current active beams.
            for seq, score in beams:
                # If a beam has already predicted EOS, do not expand it further.
                if seq[0, -1].item() == EOS_IDX:
                    new_beams.append((seq, score))
                    continue

                # Generate the causal (look-ahead) mask dynamically.
                tgt_mask = _generate_causal_mask(seq.size(1), device)

                # Decode the current sequence.
                output = model.decode(seq, enc_output, src_mask, tgt_mask)

                # Extract log probabilities for the final predicted token step.
                log_probs = torch.nn.functional.log_softmax(output[:, -1], dim=-1)

                # Find the top `beam_width` next-word candidates for this specific beam.
                top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width, dim=-1)

                # Create new branches for each of the top-k candidates.
                for i in range(beam_width):
                    next_token = torch.tensor([[top_k_indices[0, i].item()]], device=device)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + top_k_log_probs[0, i].item()
                    new_beams.append((new_seq, new_score))

            # Sort all newly expanded beams by their cumulative log probability scores (descending).
            # Prune the list down to retain only the top `beam_width` beams overall.
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    # Return the sequence belonging to the best overall beam.
    best_seq, _ = beams[0]
    return best_seq.squeeze(0)
