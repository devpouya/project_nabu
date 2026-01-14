"""
Example: Using the ORACC dataset and converting to PaleoCode.

This example demonstrates:
1. Loading the ORACC dataset
2. Exploring dataset statistics
3. Converting transliterations to PaleoCode
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nabu.datasets import OraccDataset
from nabu.tokenizers.paleocode import PaleoCodeConverter


def explore_oracc_dataset():
    """Explore the ORACC dataset structure and content."""
    print("=" * 60)
    print("ORACC Dataset Exploration")
    print("=" * 60)

    # Load dataset
    dataset = OraccDataset(
        json_path="data/external/oracc/oracc_pubs.json",
        extract_mode="lines",  # Extract individual lines
        min_text_length=5,
    )

    print(f"\nTotal samples: {len(dataset)}")

    # Get statistics
    stats = dataset.get_stats()
    print(f"\nText length statistics:")
    print(f"  Min: {stats['min_length']}")
    print(f"  Max: {stats['max_length']}")
    print(f"  Avg: {stats['avg_length']:.1f}")

    # Show language distribution
    print(f"\nLanguages in dataset:")
    for lang in sorted(dataset.get_unique_languages()):
        count = stats['language_distribution'].get(lang, 0)
        print(f"  {lang}: {count} samples")

    # Show some sample texts
    print("\n" + "=" * 60)
    print("Sample Texts")
    print("=" * 60)

    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Text: {sample['text']}")
        print(f"  Doc ID: {sample['metadata']['doc_id']}")
        print(f"  Language: {sample['metadata']['language']}")
        print(f"  Genre: {sample['metadata']['genre']}")
        print(f"  Period: {sample['metadata']['period']}")


def convert_to_paleocode():
    """Demonstrate converting ORACC text to PaleoCode."""
    print("\n" + "=" * 60)
    print("PaleoCode Conversion")
    print("=" * 60)

    # Initialize PaleoCode converter
    converter = PaleoCodeConverter()
    print(f"\nLoaded {converter.num_signs} cuneiform signs")

    # Load some sample data
    dataset = OraccDataset(
        json_path="data/external/oracc/oracc_pubs.json",
        extract_mode="lines",
        min_text_length=10,
    )

    # Build a simple transliteration map
    # Get all signs from paleocode data
    translit_map = {}
    for entry in converter.converter.get_all_signs():
        sign = entry.get("Sign")
        translit = entry.get("Transliteration", "").strip()
        if sign and translit:
            translit_map[translit.upper()] = sign

    print(f"\nBuilt transliteration map with {len(translit_map)} entries")

    # Try converting some samples
    print("\n" + "=" * 60)
    print("Conversion Examples")
    print("=" * 60)

    successful = 0
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        text = sample['text']

        # Simple tokenization and conversion
        tokens = text.replace(",", " ").split()
        unicode_chars = []
        paleocodes = []

        for token in tokens[:10]:  # Limit to first 10 tokens
            # Clean token
            clean = token.replace("[", "").replace("]", "")
            clean = clean.replace("#", "").replace("?", "")
            clean = clean.replace("~a", "").replace("~b", "")

            if clean in translit_map:
                unicode_char = translit_map[clean]
                unicode_chars.append(unicode_char)

                # Get PaleoCode
                paleocode = converter.unicode_to_paleocode(unicode_char)
                if paleocode:
                    paleocodes.append(paleocode)

        if paleocodes:
            successful += 1
            print(f"\nExample {successful}:")
            print(f"  Original: {text[:80]}...")
            print(f"  Unicode: {''.join(unicode_chars)}")
            print(f"  PaleoCode: {' '.join(paleocodes[:5])}...")

            if successful >= 3:
                break

    print(f"\n\nSuccessfully converted {successful} examples")


def filter_by_metadata():
    """Demonstrate filtering by metadata."""
    print("\n" + "=" * 60)
    print("Filtering by Metadata")
    print("=" * 60)

    # Load full dataset
    dataset = OraccDataset(
        json_path="data/external/oracc/oracc_pubs.json",
        extract_mode="lines",
    )

    print(f"\nFull dataset size: {len(dataset)}")

    # Filter by genre
    if dataset.get_unique_genres():
        genre = sorted(dataset.get_unique_genres())[0]
        filtered = dataset.filter_by_metadata(genre=genre)
        print(f"\nFiltered by genre '{genre}': {len(filtered)} samples")

    # Filter by language
    if dataset.get_unique_languages():
        lang = sorted(dataset.get_unique_languages())[0]
        filtered = dataset.filter_by_metadata(language=lang)
        print(f"Filtered by language '{lang}': {len(filtered)} samples")


if __name__ == "__main__":
    # Run all examples
    explore_oracc_dataset()
    convert_to_paleocode()
    filter_by_metadata()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python scripts/oracc_to_paleocode.py --analyze-only")
    print("2. Run: python scripts/oracc_to_paleocode.py --output data/processed/oracc.json")
    print("3. Train a model using the converted PaleoCode data")
