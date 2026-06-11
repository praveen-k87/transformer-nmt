# Full Sequence-to-Sequence (Encoder-Decoder) Transformer for Neural Machine Translation

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install notebook kernel:

```bash
python -m ipykernel install --user --name transformer-nmt --display-name "Transformer NMT"
```

---

## Project Description

**Building a Full Sequence-to-Sequence (Encoder-Decoder) Transformer for Neural Machine Translation**

### Objective
Develop a complete Transformer model from scratch featuring both Encoder and Decoder blocks. The system must perform Neural Machine Translation (NMT) by learning to map a source language sequence to a target language sequence using the Cross-Attention mechanism.

### Dataset
**Multi30k / IWSLT:** Suitable for small-scale translation tasks. Contains aligned sentence pairs in English and a target language (e.g., German or French).

**Source:** [https://huggingface.co/datasets/bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)

---

### Bilingual Data Processing & Tokenization

**Dual-Language Preprocessing**
Clean and normalize the bilingual corpus. Implement specific handling for language-specific punctuation and casing. Ensure the final dataset consists of parallel sentence pairs formatted for a seq2seq pipeline.

**Shared vs. Separate Tokenizer Training**
To Train subword tokenizers (BPE or WordPiece) for both source and target languages. Provide a technical justification for choosing either a single shared vocabulary or two independent vocabularies based on the linguistic distance between the languages.

---

### Full Transformer Architecture

**Implementation of Cross-Attention Mechanism**
Build the specific Cross-Attention (Encoder-Decoder Attention) layer where the decoder queries attend to the encoder’s output keys and values. Explain how this mechanism facilitates the alignment between source and target words.

**Positional & Padding Masking**
Implement the necessary masks: 
(a) Source Padding Mask for the Encoder 
(b) Combined Padding and Causal/Look-ahead Mask for the Decoder to prevent attending to future or padding tokens during training.

---

### Training & Translation Evaluation

**Training Loop with Teacher Forcing**
Execute the training process using the Teacher Forcing technique. Monitor the cross-entropy loss for the target sequence generation and implement a learning rate scheduler (e.g., Noam Scheduler) as used in the original 'Attention is All You Need' paper.

**Qualitative & Quantitative Translation Analysis**
Generate translations for at least five sample sentences from the test set using Greedy Search or Beam Search. Calculate the BLEU score and provide an analysis of translation quality, noting any errors in syntax or semantic preservation.