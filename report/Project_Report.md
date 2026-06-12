# Full Sequence-to-Sequence Encoder-Decoder Transformer for Neural Machine Translation

## 1. Cover Page

**Team Members:**

- Poonam Biswal (2024ac05803)
- Vikas Mahadev Hiremath (2023ad05081)
- Praveen Kanwar (2024ac05746)
- S Amina (2024ac05758)

## 2. Executive Summary

This report presents the implementation, training, and evaluation of a Neural Machine Translation (NMT) system built from scratch using the Transformer architecture in PyTorch. The model maps English source sequences to German target sequences using a full Encoder-Decoder paradigm with Cross-Attention.

## 3. Problem Statement

To build an autoregressive Transformer capable of learning an alignment and translation function between English and German without the use of pretrained translation libraries (e.g., Hugging Face Pipelines) or abstracted layers (e.g., `torch.nn.Transformer`).

## 4. Dataset Description

The `bentrevett/multi30k` dataset was utilized. This contains high-quality aligned English-German pairs describing images and physical scenes.

## 5. Module 1: Bilingual Data Processing and Tokenization

### Task 1: Dual-Language Preprocessing

Data pairs are normalized using Unicode NFC to preserve German umlauts (ä, ö, ü) and the Eszett (ß). Text is then lowercased and punctuation is separated by spaces to limit vocabulary explosion.

### Task 2: Shared vs Separate Tokenizer Training

Two distinct Word-Level Byte-Pair Encoding (BPE) tokenizers were trained. Separate tokenizers were chosen because German compounds words heavily and has distinct morphology from English, necessitating a specialized vocabulary sub-tree to minimize out-of-vocabulary (OOV) token shedding.

## 6. Module 2: Full Transformer Architecture

### Task 3: Cross-Attention Mechanism

The cross-attention mechanism acts as the conceptual bridge between the two languages.

- **Query (Q)**: Extracted from the partially generated German translation in the decoder.
- **Key (K) & Value (V)**: Extracted from the fully encoded English sentence.

### Task 4: Positional and Padding Masking

Dynamic masks were built to govern attention:

- **Source Padding Mask**: Hides `<pad>` tokens in the encoder.
- **Causal Mask**: An autoregressive lower-triangular boolean mask preventing the decoder from peaking at future tokens.
- **Combined Mask**: A logical AND operation merging target padding and the causal mask.

### Architecture Summary

- **Embedding Layer**: Vocabulary lookup + Sinusoidal Positional Encoding.
- **Encoder**: N stacks of Self-Attention + Position-Wise Feed-Forward.
- **Decoder**: N stacks of Masked Self-Attention + Cross-Attention + Position-Wise Feed-Forward.

## 7. Module 3: Training and Translation Evaluation

### Task 5: Training Loop with Teacher Forcing

The model was trained for 8 epochs using Adam optimizer and a Noam Learning Rate Scheduler. Teacher Forcing was utilized by shifting the target inputs by one position (`tgt[:, :-1]` for inputs, `tgt[:, 1:]` for loss labels).

**Hyperparameter Table:**
| Parameter | Value |
| --- | --- |
| D_MODEL | 256 |
| HEADS | 8 |
| ENCODER LAYERS | 3 |
| DECODER LAYERS | 3 |
| BATCH SIZE | 32 |
| MAX_LEN | 80 |

**Actual Trainable Parameter Count:** 10,105,664

**Training Performance:**

- Final Training Loss: 2.1141
- Final Validation Loss: 2.5044
- Best Validation Loss: 2.5044

*(Please refer to `outputs/training_curve.png` for the complete visual loss curve).*

### Task 6: Qualitative and Quantitative Translation Analysis

**Quantitative BLEU Score Table:**
| Decoding Strategy | BLEU Score |
| --- | --- |
| Greedy Decoding | 22.60 |
| Beam Search Decoding | 24.02 |

**Five Sample Translations & Qualitative Analysis:**
**Source (English):** three boys wearing green shirts and tan pants pose at the top of a slide .
**Reference (German):** drei jungen in grünen shirts und braunen hosen posieren auf dem oberteil einer rutsche .
**Greedy Translation:** drei jungen in hellbraunen oberteilen posieren am fuß einer rutsche .
**Beam Translation:** drei jungen in hellbraunen oberteilen posieren am fuß einer rutsche .
**Qualitative Analysis:** Acceptable translation: Partial semantic overlap; core meaning is somewhat captured but grammar may be flawed.

---

**Source (English):** a female performer with a violin plays on a street while a woman with a blue guitar looks on .
**Reference (German):** eine musikantin mit einer violine spielt auf der straße während eine frau mit einer blauen gitarre zusieht .
**Greedy Translation:** eine frau mit einer blauen gitarre spielt auf einer straße und schaut dabei zu .
**Beam Translation:** eine frau mit einer blauen gitarre spielt gitarre auf einer straße , während ein anderer gitarre spielt .
**Qualitative Analysis:** Acceptable translation: Partial semantic overlap; core meaning is somewhat captured but grammar may be flawed.

---

**Source (English):** a crowd gathered around a park water fountain in the rain .
**Reference (German):** eine menschenmenge hat sich im regen um einen springbrunnen im park versammelt .
**Greedy Translation:** eine menschenmenge versammelt sich im regen um einen brunnen versammelt .
**Beam Translation:** eine menschenmenge versammelt sich um einen brunnen im wasser versammelt .
**Qualitative Analysis:** Good translation: High lexical overlap; meaning is well preserved with minor syntactic differences.

---

**Source (English):** a girl jumping rope on a sidewalk near a parking garage .
**Reference (German):** ein mädchen beim seilhüpfen auf dem gehweg nahe einer garage .
**Greedy Translation:** ein mädchen auf einem parkplatz springt neben einem seil .
**Beam Translation:** ein mädchen auf einem parkplatz springt neben einem seil .
**Qualitative Analysis:** Poor translation: Limited lexical overlap; captures only fragmented concepts.

---

**Source (English):** a group of people are climbing in cold weather .
**Reference (German):** eine gruppe klettert bei kaltem wetter .
**Greedy Translation:** eine gruppe von menschen klettert in london hoch .
**Beam Translation:** eine gruppe von menschen klettert in london hoch .
**Qualitative Analysis:** Acceptable translation: Partial semantic overlap; core meaning is somewhat captured but grammar may be flawed.

---

## 8. Why Full Encoder-Decoder Architecture is Necessary for NMT

### Encoder-only vs Decoder-only vs Encoder-Decoder

| Architecture               | Information Flow               | Suitability for NMT                                                                                                                                      |
|----------------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Encoder-Only (BERT)**    | Bidirectional                  | Poor. Lacks the autoregressive capability to generate variable length sentences word-by-word.                                                            |
| **Decoder-Only (GPT)**     | Causal                         | Moderate. Prompts force translation, but the network loses the explicit mapping capability as the source context decays.                                 |
| **Encoder-Decoder (Ours)** | Bidirectional + Causal + Cross | **Excellent**. Encoder builds complete understanding. Decoder translates autoregressively while continuously referencing the source via Cross-Attention. |

## 9. Limitations

1. **Domain Restriction**: Highly restricted to image captioning. Degrades on conversational English.
2. **Computational Overhead**: Autoregressive decoding is slow without KV-Caching.

## 10. Future Improvements

1. **KV-Caching**: Implement matrix caching to drastically speed up inference generation.
2. **Larger Corpora**: Expand training corpus to WMT14 to broaden semantic capability.

## 11. Conclusion

The implementation successfully demonstrated the efficacy of the full Sequence-to-Sequence Transformer paradigm, achieving strong baseline BLEU scores using zero external pretrained weights. The model explicitly mapped English to German via proper cross-attention masking and Noam scheduled gradient descent.

## 12. References

1. Vaswani, A., et al. (2017). "Attention Is All You Need."
2. Sennrich, R., et al. (2015). "Neural Machine Translation of Rare Words with Subword Units."
