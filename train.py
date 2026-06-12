# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Training pipeline for the Neural Machine Translation model.

This module orchestrates the entire training process: loading data, instantiating
tokenizers and the Sequence-to-Sequence Transformer, defining the loss function,
optimizer, and scheduler, and executing the training loop with validation checks.
"""

import torch
# pylint: disable=duplicate-code
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (ADAM_BETAS, ADAM_EPS, BATCH_SIZE, D_FF, D_MODEL,
                        DEVICE, DROPOUT, EPOCHS, GRADIENT_CLIPPING,
                        LEARNING_RATE, MAX_LEN, N_HEADS, NUM_DECODER_LAYERS,
                        NUM_ENCODER_LAYERS, OUTPUT_DIR, PAD_IDX, RANDOM_SEED,
                        WARMUP_STEPS)
from src.data import load_and_preprocess_data
from src.dataset import TranslationDataset, collate_fn
from src.masks import create_masks
from src.scheduler import NoamScheduler
from src.tokenizer_train import get_or_train_tokenizers
from src.transformer import Seq2SeqTransformer
from src.utils import (count_trainable_parameters, ensure_output_dirs,
                       save_checkpoint, save_loss_curve, set_seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: NoamScheduler,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Executes one complete training epoch.

    Args:
        model (nn.Module): The Transformer model.
        dataloader (DataLoader): The training DataLoader.
        optimizer (NoamScheduler): The custom learning rate scheduler wrapping the Adam optimizer.
        criterion (nn.Module): The loss function (CrossEntropyLoss).
        device (torch.device): The computation device.

    Returns:
        float: The average loss across the epoch.
    """
    model.train()
    total_loss = 0.0
    for src, tgt, _, _ in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)

        # Teacher Forcing mechanism:
        # The input to the decoder is the target sequence excluding the last token.
        tgt_input = tgt[:, :-1]

        # The labels we try to predict are the target sequence excluding the first token (BOS).
        tgt_output = tgt[:, 1:]

        # Generate the boolean masks required for attention mechanisms
        src_mask, tgt_mask = create_masks(src, tgt_input, device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Calculate loss. We flatten the outputs to match CrossEntropyLoss expectations.
        loss = criterion(
            output.contiguous().view(-1, output.shape[-1]), tgt_output.contiguous().view(-1)
        )

        # Backpropagation
        loss.backward()

        # Clip gradients to prevent the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING)

        # Update weights and learning rate
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """Executes a validation pass without calculating gradients.

    Args:
        model (nn.Module): The Transformer model.
        dataloader (DataLoader): The validation DataLoader.
        criterion (nn.Module): The loss function.
        device (torch.device): The computation device.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt, _, _ in tqdm(dataloader, desc="Validating"):
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask, tgt_mask = create_masks(src, tgt_input, device)

            output = model(src, tgt_input, src_mask, tgt_mask)

            loss = criterion(
                output.contiguous().view(-1, output.shape[-1]), tgt_output.contiguous().view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main() -> None:
    """The main entrypoint for the training script."""
    ensure_output_dirs()
    set_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")
    print("Loading and preprocessing data...")
    train_pairs, val_pairs, _ = load_and_preprocess_data()

    print("Training or loading tokenizers...")
    src_tokenizer, tgt_tokenizer = get_or_train_tokenizers(train_pairs)

    train_dataset = TranslationDataset(train_pairs, src_tokenizer, tgt_tokenizer)
    val_dataset = TranslationDataset(val_pairs, src_tokenizer, tgt_tokenizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model = Seq2SeqTransformer(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(DEVICE)

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPS
    )
    scheduler = NoamScheduler(optimizer, D_MODEL, WARMUP_STEPS)

    # The loss function explicitly ignores tokens that are padded,
    # to avoid penalizing the model for not predicting <pad>.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss = train_epoch(model, train_dataloader, scheduler, criterion, DEVICE)
        val_loss = validate_epoch(model, val_dataloader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Model Checkpointing logic: Save only the model with the lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, OUTPUT_DIR / "model.pt")
            print("Saved best model checkpoint.")

    # Save visual loss curves and final textual summaries
    save_loss_curve(train_losses, val_losses, OUTPUT_DIR / "training_curve.png")

    with open(OUTPUT_DIR / "training_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Trainable parameters: {count_trainable_parameters(model):,}\\n")
        f.write(f"Final training loss: {train_losses[-1]:.4f}\\n")
        f.write(f"Final validation loss: {val_losses[-1]:.4f}\\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\\n")

    # Export numeric history to CSV for programmatic loading in the notebook
    import pandas as pd
    history_df = pd.DataFrame({
        "epoch": range(1, EPOCHS + 1),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)


if __name__ == "__main__":
    main()
