import re
import pandas as pd
from pathlib import Path

def generate_report():
    output_dir = Path("outputs")
    
    # Read metrics
    try:
        with open(output_dir / "training_summary.txt", "r") as f:
            summary = f.read()
            params = re.search(r"Trainable parameters:\s*([\d,]+)", summary).group(1)
            train_loss = re.search(r"Final training loss:\s*([\d.]+)", summary).group(1)
            val_loss = re.search(r"Final validation loss:\s*([\d.]+)", summary).group(1)
            best_val = re.search(r"Best validation loss:\s*([\d.]+)", summary).group(1)
    except Exception:
        params, train_loss, val_loss, best_val = "N/A", "N/A", "N/A", "N/A"

    try:
        with open(output_dir / "bleu_score.txt", "r") as f:
            bleu_content = f.read()
            greedy_bleu = re.search(r"BLEU Score \(Greedy Decoding\):\s*([\d.]+)", bleu_content).group(1)
            beam_bleu = re.search(r"BLEU Score \(Beam Decoding\):\s*([\d.]+)", bleu_content).group(1)
    except Exception:
        greedy_bleu, beam_bleu = "N/A", "N/A"

    sample_table = ""
    try:
        df = pd.read_csv(output_dir / "sample_translations.csv")
        samples = df.sample(n=5, random_state=42)
        for _, row in samples.iterrows():
            sample_table += f"**Source (English):** {row['source_english']}\n"
            sample_table += f"**Reference (German):** {row['reference_german']}\n"
            sample_table += f"**Greedy Translation:** {row['greedy_translation']}\n"
            sample_table += f"**Beam Translation:** {row['beam_translation']}\n"
            sample_table += f"**Qualitative Analysis:** {row['qualitative_comment']}\n\n---\n\n"
    except Exception:
        sample_table = "Run evaluate.py to generate sample translations."

    report_content = f"""# Full Sequence-to-Sequence Encoder-Decoder Transformer for Neural Machine Translation

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
| D_MODEL | 512 |
| HEADS | 8 |
| ENCODER LAYERS | 3 |
| DECODER LAYERS | 3 |
| BATCH SIZE | 32 |
| MAX_LEN | 100 |

**Actual Trainable Parameter Count:** {params}

**Training Performance:**
- Final Training Loss: {train_loss}
- Final Validation Loss: {val_loss}
- Best Validation Loss: {best_val}

*(Please refer to `outputs/training_curve.png` for the complete visual loss curve).*

### Task 6: Qualitative and Quantitative Translation Analysis

**Quantitative BLEU Score Table:**
| Decoding Strategy | BLEU Score |
| --- | --- |
| Greedy Decoding | {greedy_bleu} |
| Beam Search Decoding | {beam_bleu} |

**Five Sample Translations & Qualitative Analysis:**
{sample_table}

## 8. Why Full Encoder-Decoder Architecture is Necessary for NMT
### Encoder-only vs Decoder-only vs Encoder-Decoder
| Architecture | Information Flow | Suitability for NMT |
|---|---|---|
| **Encoder-Only (BERT)** | Bidirectional | Poor. Lacks the autoregressive capability to generate variable length sentences word-by-word. |
| **Decoder-Only (GPT)** | Causal | Moderate. Prompts force translation, but the network loses the explicit mapping capability as the source context decays. |
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
"""
    
    with open("report/Project_Report.md", "w") as f:
        f.write(report_content)
        
    print("Project_Report.md has been rewritten.")

if __name__ == "__main__":
    generate_report()
