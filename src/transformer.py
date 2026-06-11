# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Transformer architecture module.

This module provides the complete from-scratch implementation of the Sequence-to-Sequence
Encoder-Decoder Transformer model as described in the "Attention Is All You Need" paper.
It includes Token Embeddings, Positional Encoding, Multi-Head Attention (Self and Cross),
Position-wise Feed-Forward Networks, and the final structural Encoder and Decoder layers.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Converts vocabulary token indices into continuous dense vectors.

    Attributes:
        embedding (nn.Embedding): The PyTorch embedding layer.
        d_model (int): The dimensionality of the embedding vector.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """Initializes the TokenEmbedding layer.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimensionality of the embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds the token sequence and scales by sqrt(d_model).

        Scaling is necessary in the Transformer architecture to ensure the
        embeddings have sufficient variance before adding the positional encodings.

        Args:
            x (torch.Tensor): Token indices of shape `(batch_size, seq_len)`.

        Returns:
            torch.Tensor: Scaled embeddings of shape `(batch_size, seq_len, d_model)`.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    pe: torch.Tensor
    """Injects sequence order information into the token embeddings.

    Since self-attention operations are permutation-invariant, positional
    encodings give the model a sense of relative and absolute token positions.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied after adding positional encodings.
        pe (torch.Tensor): The pre-computed positional encoding matrix.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        """Initializes the PositionalEncoding matrix.

        Args:
            d_model (int): The dimensionality of the embeddings.
            max_len (int): The maximum expected sequence length.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize the encoding matrix
        pe = torch.zeros(max_len, d_model)

        # Compute the positional encodings using sine and cosine functions of different frequencies
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as a buffer so it's not considered a trainable parameter,
        # but it is automatically pushed to the correct device alongside the module.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape `(batch_size, seq_len, d_model)`.

        Returns:
            torch.Tensor: Position-encoded representations.
        """
        # Slices the pre-computed pe matrix to match the current sequence length of x
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """Computes scaled dot-product attention scores.

    This acts as the core attention mechanism inside the Multi-Head Attention block.

    Attributes:
        d_k (int): The dimension of the keys (and queries).
    """

    def __init__(self, d_k: int) -> None:
        """Initializes the ScaledDotProductAttention layer.

        Args:
            d_k (int): Dimension of keys/queries to scale the dot product.
        """
        super().__init__()
        self.d_k = d_k

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the attention outputs and weights.

        Args:
            q (torch.Tensor): Queries of shape `(batch_size, n_heads, query_len, d_k)`.
            k (torch.Tensor): Keys of shape `(batch_size, n_heads, key_len, d_k)`.
            v (torch.Tensor): Values of shape `(batch_size, n_heads, key_len, d_v)`.
            mask (Optional[torch.Tensor], optional): Mask tensor to ignore specific positions. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The attention outputs and the normalized attention weights.
        """
        # Computes Q * K^T / sqrt(d_k).
        # We transpose the last two dimensions of k to align the dot product.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply the mask (if provided) by filling masked positions with a very large negative value
        # so that they become zero after the softmax operation.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax across the last dimension (key sequence length) to get normalized probabilities.
        attn = torch.softmax(scores, dim=-1)

        # Multiply attention probabilities by the values (V).
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Executes Multi-Head Attention by projecting inputs into multiple sub-spaces.

    Allows the model to jointly attend to information from different representation
    subspaces at different positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        """Initializes the MultiHeadAttention block.

        Args:
            d_model (int): Total dimension of the model.
            n_heads (int): Number of parallel attention heads.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be perfectly divisible by n_heads"

        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # Linear projections for Queries, Keys, and Values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final linear projection
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs multi-head attention.

        Args:
            q (torch.Tensor): Queries.
            k (torch.Tensor): Keys.
            v (torch.Tensor): Values.
            mask (Optional[torch.Tensor], optional): Mask. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        batch_size = q.size(0)

        # 1. Linear projections and reshaping:
        # Original shape: (batch_size, seq_len, d_model)
        # Reshaped to: (batch_size, seq_len, n_heads, d_k)
        # Transposed to: (batch_size, n_heads, seq_len, d_k) for parallel attention over heads
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Apply scaled dot-product attention
        output, attn = self.attention(q, k, v, mask)

        # 3. Recombine heads:
        # Transpose back: (batch_size, seq_len, n_heads, d_k)
        # Contiguous aligns memory sequentially before reshaping.
        # View concatenates the heads: (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Final linear projection
        return self.fc(output), attn


class PositionWiseFeedForward(nn.Module):
    """Applies a fully connected feed-forward network to each position independently.

    Consists of two linear transformations with a ReLU activation in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """Initializes the PositionWiseFeedForward block.

        Args:
            d_model (int): Input and output dimension.
            d_ff (int): Dimensionality of the inner hidden layer.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes the feed-forward operations.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """A single layer of the Transformer Encoder.

    Consists of a Multi-Head Self-Attention mechanism and a Position-wise Feed-Forward network,
    both surrounded by residual connections and Layer Normalization.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        """Initializes the EncoderLayer.

        Args:
            d_model (int): Total dimension of the model.
            n_heads (int): Number of parallel attention heads.
            d_ff (int): Dimensionality of the inner feed-forward layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Executes the EncoderLayer.

        Args:
            x (torch.Tensor): Input sequence embeddings.
            mask (torch.Tensor): Source padding mask.

        Returns:
            torch.Tensor: Layer representations.
        """
        # Sub-layer 1: Self-Attention + Residual + LayerNorm
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Sub-layer 2: Feed-Forward + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


class DecoderLayer(nn.Module):
    """A single layer of the Transformer Decoder.

    Consists of three sub-layers: Masked Self-Attention, Cross-Attention (attending to
    the encoder's output), and a Position-wise Feed-Forward network.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        """Initializes the DecoderLayer.

        Args:
            d_model (int): Total dimension of the model.
            n_heads (int): Number of parallel attention heads.
            d_ff (int): Dimensionality of the inner feed-forward layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Executes the DecoderLayer.

        Args:
            x (torch.Tensor): Target sequence embeddings.
            enc_output (torch.Tensor): The final output representations from the Encoder.
            src_mask (torch.Tensor): Source padding mask (for cross-attention).
            tgt_mask (torch.Tensor): Target combined mask (for self-attention).

        Returns:
            torch.Tensor: Layer representations.
        """
        # Sub-layer 1: Masked Target Self-Attention
        # The mask prevents attending to future tokens.
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # Sub-layer 2: Cross-Attention / Encoder-Decoder Attention
        # Queries (q) come from the decoder's hidden states (x).
        # Keys (k) and Values (v) come from the encoder's output representations (enc_output).
        # This mechanism enforces alignment between the target being generated and the source.
        cross_attn_out, _ = self.cross_attn(q=x, k=enc_output, v=enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # Sub-layer 3: Feed-Forward block
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x


class Encoder(nn.Module):
    """The Transformer Encoder component.

    Stacks multiple EncoderLayers on top of TokenEmbeddings and PositionalEncodings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        max_len: int,
    ) -> None:
        """Initializes the Encoder.

        Args:
            vocab_size (int): Size of the source vocabulary.
            d_model (int): Dimensionality of the model.
            n_heads (int): Number of attention heads.
            d_ff (int): Feed-Forward inner dimension.
            num_layers (int): Number of EncoderLayers to stack.
            dropout (float): Dropout probability.
            max_len (int): Maximum expected sequence length.
        """
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encodes the source sequence.

        Args:
            src (torch.Tensor): Source sequence token indices.
            mask (torch.Tensor): Source padding mask.

        Returns:
            torch.Tensor: Encoded representations of the source sequence.
        """
        x = self.dropout(self.pos_emb(self.tok_emb(src)))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    """The Transformer Decoder component.

    Stacks multiple DecoderLayers and applies a final linear layer to project the
    hidden states back into vocabulary space.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        max_len: int,
    ) -> None:
        """Initializes the Decoder.

        Args:
            vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimensionality of the model.
            n_heads (int): Number of attention heads.
            d_ff (int): Feed-Forward inner dimension.
            num_layers (int): Number of DecoderLayers to stack.
            dropout (float): Dropout probability.
            max_len (int): Maximum expected sequence length.
        """
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Linear layer projecting from d_model directly to target vocabulary dimensions
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes the target sequence.

        Args:
            tgt (torch.Tensor): Target sequence token indices.
            enc_output (torch.Tensor): The final output representations from the Encoder.
            src_mask (torch.Tensor): Source padding mask.
            tgt_mask (torch.Tensor): Target combined mask.

        Returns:
            torch.Tensor: Logit predictions for the target vocabulary.
        """
        x = self.dropout(self.pos_emb(self.tok_emb(tgt)))
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.fc_out(x)


class Seq2SeqTransformer(nn.Module):
    """The complete Sequence-to-Sequence Transformer Model.

    Encapsulates both the Encoder and Decoder, providing a unified `forward` pass
    as well as standalone `encode` and `decode` methods for autoregressive inference.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        max_len: int,
    ) -> None:
        """Initializes the full Sequence-to-Sequence model.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimensionality of the model.
            n_heads (int): Number of attention heads.
            d_ff (int): Feed-Forward inner dimension.
            num_encoder_layers (int): Number of Encoder layers.
            num_decoder_layers (int): Number of Decoder layers.
            dropout (float): Dropout probability.
            max_len (int): Maximum expected sequence length.
        """
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, d_model, n_heads, d_ff, num_encoder_layers, dropout, max_len
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_heads, d_ff, num_decoder_layers, dropout, max_len
        )

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Executes a full forward pass during training (with Teacher Forcing).

        Args:
            src (torch.Tensor): Source sequence token indices.
            tgt (torch.Tensor): Target sequence token indices (shifted right).
            src_mask (torch.Tensor): Source padding mask.
            tgt_mask (torch.Tensor): Target combined padding and causal mask.

        Returns:
            torch.Tensor: Output logits.
        """
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return output

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Standalone encoder pass (useful during inference)."""
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Standalone decoder pass (useful during autoregressive inference)."""
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)
