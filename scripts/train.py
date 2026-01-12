#!/usr/bin/env python3
"""Training script for cuneiform NLP models."""

import argparse
import os
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nabu.tokenizers import StrokeTokenizer, SignTokenizer, HybridTokenizer
from nabu.datasets import CuneiformDataset
from nabu.dataloaders import build_dataloaders
from nabu.models import TransformerEncoder, TransformerDecoder, RNNEncoder


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_tokenizer(config: dict):
    """Build tokenizer based on configuration."""
    tokenizer_type = config["tokenizer"]["type"]
    paleocode_dir = config.get("paleocode_dir", None)

    if tokenizer_type == "stroke":
        tokenizer = StrokeTokenizer(paleocode_dir)
    elif tokenizer_type == "sign":
        tokenizer = SignTokenizer(paleocode_dir)
    elif tokenizer_type == "hybrid":
        tokenizer = HybridTokenizer(paleocode_dir)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    return tokenizer


def build_model(config: dict, vocab_size: int):
    """Build model based on configuration."""
    model_type = config["model"]["type"]
    model_config = config["model"]

    if model_type == "transformer_encoder":
        model = TransformerEncoder(
            vocab_size=vocab_size,
            embedding_dim=model_config.get("hidden_size", 256),
            num_layers=model_config.get("num_layers", 6),
            num_heads=model_config.get("num_heads", 8),
            feedforward_dim=model_config.get("feedforward_dim", 1024),
            dropout=model_config.get("dropout", 0.1),
        )
    elif model_type == "transformer_decoder":
        model = TransformerDecoder(
            vocab_size=vocab_size,
            embedding_dim=model_config.get("hidden_size", 256),
            num_layers=model_config.get("num_layers", 6),
            num_heads=model_config.get("num_heads", 8),
            feedforward_dim=model_config.get("feedforward_dim", 1024),
            dropout=model_config.get("dropout", 0.1),
        )
    elif model_type == "rnn":
        model = RNNEncoder(
            vocab_size=vocab_size,
            embedding_dim=model_config.get("embedding_dim", 256),
            hidden_dim=model_config.get("hidden_size", 512),
            num_layers=model_config.get("num_layers", 2),
            dropout=model_config.get("dropout", 0.1),
            bidirectional=model_config.get("bidirectional", True),
            rnn_type=model_config.get("rnn_type", "lstm"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]

        # Compute loss (language modeling)
        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train cuneiform NLP model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = build_tokenizer(config)

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    dataset = CuneiformDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_length=config["tokenizer"].get("max_length", 512),
    )

    # Build vocabulary
    print("Building vocabulary...")
    tokenizer.build_vocab(dataset.samples)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Save vocabulary
    vocab_path = output_dir / "vocab.json"
    tokenizer.save_vocabulary(str(vocab_path))
    print(f"Saved vocabulary to {vocab_path}")

    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = build_dataloaders(
        dataset,
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.1),
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 0),
        pad_token_id=tokenizer.pad_token_id,
        max_length=config["tokenizer"].get("max_length", 512),
    )

    # Build model
    print("Building model...")
    model = build_model(config, tokenizer.vocab_size)
    model = model.to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # TensorBoard
    writer = SummaryWriter(log_dir=output_dir / "logs")

    # Training loop
    num_epochs = config["training"]["epochs"]
    best_val_loss = float("inf")

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(
            model, dataloaders["train"], optimizer, criterion, device, epoch
        )

        # Evaluate
        val_loss = evaluate(model, dataloaders["val"], criterion, device)

        # Log
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"
            model.save_checkpoint(
                str(checkpoint_path),
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
            )
            print(f"Saved best model to {checkpoint_path}")

        # Save regular checkpoint
        if epoch % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            model.save_checkpoint(
                str(checkpoint_path),
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
            )

    print("\nTraining complete!")
    writer.close()


if __name__ == "__main__":
    main()
