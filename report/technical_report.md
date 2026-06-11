# Full Sequence-to-Sequence Encoder-Decoder Transformer for Neural Machine Translation

## Executive Summary
This project implements a complete Sequence-to-Sequence (Encoder-Decoder) Transformer model from scratch using PyTorch for Neural Machine Translation (NMT). The model translates English sentences into German. It is designed to be an educational yet robust implementation, strictly avoiding pre-trained models or the built-in `torch.nn.Transformer` class.

## Problem Statement
The goal is to build an end-to-end NMT system capable of learning the alignment and translation between English and German. This requires implementing the complex cross-attention mechanism that allows the decoder to condition its generation on the encoder's representation of the source text.

## Dataset
We utilized the `bentrevett/multi30k` dataset, a standard benchmark for small-scale machine translation tasks, containing aligned English-German sentence pairs.

## Preprocessing
The data pipeline includes Unicode normalization (NFC) to safely handle German characters without destruction. Both languages are lowercased, and punctuation is spaced to treat punctuation marks as separate tokens. Empty sentence pairs are filtered out.

## Tokenization Strategy and Justification
We employ separate Byte-Pair Encoding (BPE) tokenizers for English and German. While the languages share linguistic roots, they exhibit significant differences in morphology and compounding behavior (e.g., German's tendency for long compound words). Separate vocabularies allow each tokenizer to build an optimal subword representation tailored to its specific language, preventing the vocabulary from being dominated by one language or creating inefficient splits.

## Architecture Overview
The model follows the original Transformer architecture ("Attention is All You Need"):
- **Encoder**: Processes the source sequence using self-attention and feed-forward layers to create a rich contextual representation.
- **Decoder**: Generates the target sequence autoregressively.

### Cross-Attention Explanation
The critical component enabling translation is the cross-attention layer in the decoder. Here, the **queries** originate from the previous decoder hidden state, representing the current context of the generated translation. The **keys and values** come from the final output of the encoder, representing the source sentence. This mechanism computes an attention score that aligns the current target word being generated with relevant words in the source sentence.

### Masking Explanation
Three distinct masks are implemented:
1.  **Source Padding Mask**: Applied in the encoder self-attention and decoder cross-attention to prevent the model from attending to `<pad>` tokens, ensuring computational efficiency and model correctness.
2.  **Target Padding Mask**: Similar to the source mask, applied in the decoder self-attention.
3.  **Causal (Look-ahead) Mask**: Applied in the decoder self-attention. Since the decoder is autoregressive, it must not look at future tokens when predicting the next token during training. This mask zeros out future positions.

## Training Methodology
- **Teacher Forcing**: During training, the decoder receives the ground-truth shifted target sequence as input, rather than its own previous predictions. This stabilizes and accelerates training.
- **Optimizer & Scheduler**: We use the Adam optimizer coupled with the Noam learning rate scheduler, which incorporates a warmup period followed by an inverse square root decay, crucial for training Transformers from scratch.
- **Loss**: CrossEntropyLoss is used, explicitly ignoring the padding index to avoid penalizing predictions on padded regions.

## Evaluation Methodology
The model is evaluated using SacreBLEU on the test set. We implement both Greedy Decoding and Beam Search (width=3) for inference.

## Results
*Note: The following values will be populated after running the training and evaluation scripts.*
- **Actual Trainable Parameter Count**: [To be filled after training]
- **Final Training Loss**: [To be filled after training]
- **Best Validation Loss**: [To be filled after training]
- **BLEU Score (Greedy)**: [To be filled after evaluation]

## Qualitative Analysis
*Note: Sample translations will be appended here after evaluation.*

## Limitations
- The Multi30k dataset is relatively small, limiting the model's vocabulary and ability to generalize to complex sentence structures outside the training distribution.
- The implemented beam search is basic and could be optimized for better performance and speed.

## Future Work
- Scale the model and train on larger datasets like WMT.
- Implement label smoothing to improve generalization.
- Experiment with more advanced decoding strategies.

## References
1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.
