#!/usr/bin/env python3
"""Evaluation script for cuneiform NLP models."""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nabu.tokenizers import StrokeTokenizer, SignTokenizer, HybridTokenizer
from nabu.datasets import CuneiformDataset
from nabu.dataloaders import build_dataloader
from nabu.models import TransformerEncoder, TransformerDecoder


def load_tokenizer(vocab_path: str, tokenizer_type: str, paleocode_dir=None):
    """Load tokenizer with vocabulary."""
    if tokenizer_type == "stroke":
        tokenizer = StrokeTokenizer(paleocode_dir)
    elif tokenizer_type == "sign":
        tokenizer = SignTokenizer(paleocode_dir)
    elif tokenizer_type == "hybrid":
        tokenizer = HybridTokenizer(paleocode_dir)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    tokenizer.load_vocabulary(vocab_path)
    return tokenizer


def compute_perplexity(model, dataloader, device):
    """Compute perplexity on dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Count non-padding tokens
            num_tokens = (shift_labels != 0).sum().item()

            total_loss += loss.item()
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity, avg_loss


def generate_samples(model, tokenizer, device, num_samples=5, max_length=100):
    """Generate sample texts from the model."""
    model.eval()
    samples = []

    for _ in range(num_samples):
        # Start with BOS token
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)

        generated_ids = []

        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(input_ids=input_ids)
                logits = outputs["logits"]

                # Get next token (greedy)
                next_token_id = logits[:, -1, :].argmax(dim=-1).item()

                # Stop if EOS
                if next_token_id == tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id)

                # Append to input
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token_id]], device=device)
                ], dim=1)

        # Decode
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        samples.append(text)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate cuneiform NLP model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation data")
    parser.add_argument("--tokenizer-type", type=str, default="sign", help="Tokenizer type")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--generate", action="store_true", help="Generate sample texts")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--output", type=str, help="Output file for results")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load tokenizer
    print(f"Loading tokenizer and vocabulary from {args.vocab}...")
    tokenizer = load_tokenizer(args.vocab, args.tokenizer_type)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Build model (from config in checkpoint)
    print("Building model...")
    config = checkpoint.get("config", {})
    model_type = config.get("model", {}).get("type", "transformer_encoder")

    if model_type == "transformer_encoder":
        model = TransformerEncoder(vocab_size=tokenizer.vocab_size)
    elif model_type == "transformer_decoder":
        model = TransformerDecoder(vocab_size=tokenizer.vocab_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded with {model.get_num_parameters():,} parameters")

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    dataset = CuneiformDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_length=512,
    )

    # Create dataloader
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collator_type="dynamic",
        pad_token_id=tokenizer.pad_token_id,
    )

    # Compute metrics
    print("\nEvaluating model...")
    perplexity, avg_loss = compute_perplexity(model, dataloader, device)

    results = {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "num_samples": len(dataset),
        "vocab_size": tokenizer.vocab_size,
    }

    print(f"\nResults:")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Average Loss: {avg_loss:.4f}")

    # Generate samples
    if args.generate:
        print(f"\nGenerating {args.num_samples} sample texts...")
        samples = generate_samples(
            model, tokenizer, device,
            num_samples=args.num_samples
        )

        print("\nGenerated samples:")
        for i, sample in enumerate(samples, 1):
            print(f"  {i}. {sample}")

        results["generated_samples"] = samples

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
