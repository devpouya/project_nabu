#!/usr/bin/env python3
"""Data preprocessing script for cuneiform corpus."""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nabu.tokenizers import PaleoCodeConverter


def convert_text_to_paleocode(input_path: str, output_path: str):
    """
    Convert cuneiform text file to PaleoCode representation.

    Args:
        input_path: Path to input text file (UTF-8 cuneiform)
        output_path: Path to output file
    """
    converter = PaleoCodeConverter()

    with open(input_path, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    converted_lines = []
    print(f"Converting {len(lines)} lines to PaleoCode...")

    for line in tqdm(lines):
        line = line.strip()
        if not line:
            continue

        # Convert each character to PaleoCode
        paleocodes = converter.text_to_paleocode(line)

        # Filter out None values and join
        paleocodes = [pc for pc in paleocodes if pc is not None]
        converted_line = " ".join(paleocodes)

        if converted_line:
            converted_lines.append(converted_line)

    # Write output
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(converted_lines))

    print(f"Converted {len(converted_lines)} lines")
    print(f"Output saved to {output_path}")


def build_sign_statistics(data_path: str, output_path: str):
    """
    Analyze cuneiform corpus and generate sign statistics.

    Args:
        data_path: Path to cuneiform text file
        output_path: Path to output JSON file
    """
    converter = PaleoCodeConverter()

    # Count sign occurrences
    sign_counts = {}
    paleocode_counts = {}
    total_chars = 0

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Analyzing {len(lines)} lines...")

    for line in tqdm(lines):
        for char in line.strip():
            if not char.strip():
                continue

            total_chars += 1

            # Count Unicode sign
            sign_counts[char] = sign_counts.get(char, 0) + 1

            # Count PaleoCode
            paleocode = converter.unicode_to_paleocode(char)
            if paleocode:
                paleocode_counts[paleocode] = paleocode_counts.get(paleocode, 0) + 1

    # Sort by frequency
    sign_counts = dict(sorted(sign_counts.items(), key=lambda x: x[1], reverse=True))
    paleocode_counts = dict(sorted(paleocode_counts.items(), key=lambda x: x[1], reverse=True))

    # Create statistics
    stats = {
        "total_characters": total_chars,
        "unique_signs": len(sign_counts),
        "unique_paleocodes": len(paleocode_counts),
        "top_signs": dict(list(sign_counts.items())[:20]),
        "top_paleocodes": dict(list(paleocode_counts.items())[:20]),
        "all_sign_counts": sign_counts,
        "all_paleocode_counts": paleocode_counts,
    }

    # Save statistics
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nStatistics:")
    print(f"  Total characters: {stats['total_characters']:,}")
    print(f"  Unique signs: {stats['unique_signs']}")
    print(f"  Unique PaleoCodes: {stats['unique_paleocodes']}")
    print(f"\nStatistics saved to {output_path}")


def split_dataset(input_path: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Split dataset into train/val/test sets.

    Args:
        input_path: Path to input data file
        output_dir: Output directory for splits
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    """
    # Read all lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total = len(lines)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[train_size + val_size:]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write splits
    splits = {
        "train.txt": train_lines,
        "val.txt": val_lines,
        "test.txt": test_lines,
    }

    for filename, data in splits.items():
        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(data))
        print(f"Wrote {len(data)} lines to {filepath}")

    print(f"\nDataset split complete:")
    print(f"  Train: {len(train_lines)} ({train_ratio*100:.0f}%)")
    print(f"  Val: {len(val_lines)} ({val_ratio*100:.0f}%)")
    print(f"  Test: {len(test_lines)} ({(1-train_ratio-val_ratio)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess cuneiform corpus")
    parser.add_argument("--input", type=str, required=True, help="Input data path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["convert", "stats", "split"],
        required=True,
        help="Preprocessing mode"
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation data ratio")
    args = parser.parse_args()

    if args.mode == "convert":
        convert_text_to_paleocode(args.input, args.output)
    elif args.mode == "stats":
        build_sign_statistics(args.input, args.output)
    elif args.mode == "split":
        split_dataset(args.input, args.output, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
