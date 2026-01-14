#!/usr/bin/env python3
"""
Convert ORACC transliterations to PaleoCode format.

This script loads ORACC data and converts it to PaleoCode encoding
for training neural models.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nabu.datasets import OraccDataset
from nabu.tokenizers.paleocode import PaleoCodeConverter


class TransliterationMapper:
    """Maps ORACC ATF transliterations to Unicode cuneiform."""

    def __init__(self, paleocode_converter: PaleoCodeConverter):
        """
        Initialize the mapper.

        Args:
            paleocode_converter: PaleoCode converter instance
        """
        self.converter = paleocode_converter
        self._build_transliteration_map()

    def _build_transliteration_map(self) -> None:
        """Build mapping from transliteration to Unicode."""
        self.translit_to_unicode: Dict[str, str] = {}
        self.unicode_to_translit: Dict[str, str] = {}

        # Get all signs from paleocode data
        for entry in self.converter.converter.get_all_signs():
            sign = entry.get("Sign")
            translit = entry.get("Transliteration", "").strip()

            if sign and translit:
                # Store both directions
                self.translit_to_unicode[translit.upper()] = sign
                self.unicode_to_translit[sign] = translit

                # Also store lowercase version
                self.translit_to_unicode[translit.lower()] = sign

        print(f"Built transliteration map with {len(self.translit_to_unicode)} entries")

    def transliteration_to_unicode(self, translit: str) -> Optional[str]:
        """
        Convert transliteration to Unicode cuneiform.

        Args:
            translit: Transliteration string (e.g., "A", "GAL")

        Returns:
            Unicode cuneiform character or None if not found
        """
        # Try direct lookup first
        if translit in self.translit_to_unicode:
            return self.translit_to_unicode[translit]

        # Try uppercase
        if translit.upper() in self.translit_to_unicode:
            return self.translit_to_unicode[translit.upper()]

        return None

    def process_oracc_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Convert ORACC text to Unicode cuneiform and PaleoCode.

        Args:
            text: ORACC text line (e.g., "1(N01) , GAL~a UMUN2")

        Returns:
            Tuple of (unicode_text, paleocode_list)
        """
        # Split by common separators
        tokens = text.replace(",", " ").replace(".", " ").split()

        unicode_chars = []
        paleocodes = []

        for token in tokens:
            # Skip brackets and special markers
            if token in ["[...]", "[", "]", "X", "#", "?"]:
                continue

            # Remove brackets, hash marks, tildes, etc.
            clean_token = token.replace("[", "").replace("]", "")
            clean_token = clean_token.replace("#", "").replace("?", "")
            clean_token = clean_token.replace("~a", "").replace("~b", "")
            clean_token = clean_token.replace("~c", "").replace("~d", "")

            # Skip if empty after cleaning
            if not clean_token:
                continue

            # Try to convert to Unicode
            unicode_char = self.transliteration_to_unicode(clean_token)

            if unicode_char:
                unicode_chars.append(unicode_char)

                # Convert to PaleoCode
                paleocode = self.converter.unicode_to_paleocode(unicode_char)
                if paleocode:
                    paleocodes.append(paleocode)

        unicode_text = "".join(unicode_chars)
        return unicode_text, paleocodes


def analyze_dataset(dataset: OraccDataset) -> Dict:
    """Analyze ORACC dataset statistics."""
    stats = dataset.get_stats()

    print("\n" + "=" * 60)
    print("ORACC DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {stats['num_samples']}")
    print(f"Text length - Min: {stats['min_length']}, Max: {stats['max_length']}, "
          f"Avg: {stats['avg_length']:.1f}")

    print("\nLanguage Distribution:")
    for lang, count in sorted(stats['language_distribution'].items(),
                              key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {lang}: {count}")

    print("\nGenre Distribution:")
    for genre, count in sorted(stats['genre_distribution'].items(),
                               key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {genre}: {count}")

    print("\nPeriod Distribution:")
    for period, count in sorted(stats['period_distribution'].items(),
                                key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {period}: {count}")

    return stats


def convert_dataset(
    dataset: OraccDataset,
    mapper: TransliterationMapper,
    output_file: str,
    format: str = "txt",
    max_samples: Optional[int] = None,
) -> None:
    """
    Convert ORACC dataset to PaleoCode format.

    Args:
        dataset: ORACC dataset
        mapper: Transliteration mapper
        output_file: Output file path
        format: Output format ('txt', 'json', 'jsonl')
        max_samples: Maximum number of samples to convert
    """
    print(f"\nConverting dataset to PaleoCode format...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    converted_samples = []
    failed_count = 0

    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")

        sample = dataset[i]
        text = sample["text"]
        metadata = sample["metadata"]

        # Convert to Unicode and PaleoCode
        unicode_text, paleocodes = mapper.process_oracc_text(text)

        # Skip if conversion failed
        if not unicode_text or not paleocodes:
            failed_count += 1
            continue

        converted_sample = {
            "original": text,
            "unicode": unicode_text,
            "paleocode": " ".join(paleocodes),
            "paleocode_list": paleocodes,
            "metadata": metadata,
        }

        converted_samples.append(converted_sample)

    print(f"\nSuccessfully converted {len(converted_samples)} samples")
    print(f"Failed conversions: {failed_count}")

    # Write output
    if format == "txt":
        # Write PaleoCode sequences, one per line
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in converted_samples:
                f.write(sample["paleocode"] + "\n")

    elif format == "json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(converted_samples, f, ensure_ascii=False, indent=2)

    elif format == "jsonl":
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in converted_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ORACC transliterations to PaleoCode"
    )
    parser.add_argument(
        "--oracc-json",
        type=str,
        default="data/external/oracc/oracc_pubs.json",
        help="Path to ORACC JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/oracc_paleocode.json",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["txt", "json", "jsonl"],
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--extract-mode",
        type=str,
        choices=["lines", "paragraphs", "full_text"],
        default="lines",
        help="How to extract text from documents"
    )
    parser.add_argument(
        "--filter-language",
        type=str,
        help="Filter by language code"
    )
    parser.add_argument(
        "--filter-genre",
        type=str,
        help="Filter by genre"
    )
    parser.add_argument(
        "--filter-period",
        type=str,
        help="Filter by period"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to convert"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze dataset statistics, don't convert"
    )

    args = parser.parse_args()

    # Initialize PaleoCode converter
    print("Initializing PaleoCode converter...")
    converter = PaleoCodeConverter()
    print(f"Loaded {converter.num_signs} cuneiform signs")

    # Load ORACC dataset
    print(f"\nLoading ORACC dataset from {args.oracc_json}...")
    dataset = OraccDataset(
        json_path=args.oracc_json,
        extract_mode=args.extract_mode,
        filter_language=args.filter_language,
        filter_genre=args.filter_genre,
        filter_period=args.filter_period,
    )

    # Analyze dataset
    analyze_dataset(dataset)

    if not args.analyze_only:
        # Initialize mapper
        mapper = TransliterationMapper(converter)

        # Convert dataset
        convert_dataset(
            dataset=dataset,
            mapper=mapper,
            output_file=args.output,
            format=args.format,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
