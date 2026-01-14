"""
Generate encodings for all signs in cuneiform_signs.csv and save to database.

This script processes all Unicode cuneiform characters and generates
Hantatallas encodings using the automatic encoding generation pipeline.

Output: data/encodings/generated_encodings.json
"""

import sys
from pathlib import Path
import csv
import json
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nabu.encoding.encoding_generator import generate_encoding_from_unicode


def load_cuneiform_signs(csv_path: Path) -> List[Dict]:
    """
    Load all signs from cuneiform_signs.csv.

    Returns:
        List of sign dictionaries with metadata
    """
    signs = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract Unicode character from first column
            ur_iii = row.get('Ur III', '').strip()
            neo_assyrian = row.get('Neo-Assyrian', '').strip()

            # Prefer Ur III, fallback to Neo-Assyrian
            unicode_char = None
            if ur_iii and len(ur_iii) == 1 and ord(ur_iii) >= 0x12000:
                unicode_char = ur_iii
            elif neo_assyrian and len(neo_assyrian) == 1 and ord(neo_assyrian) >= 0x12000:
                unicode_char = neo_assyrian

            if unicode_char:
                signs.append({
                    'unicode': unicode_char,
                    'code_point': f"U+{ord(unicode_char):04X}",
                    'phonetic': row.get('Phonetic value(s)', '').strip(),
                    'modsl': row.get('ModSL', '').strip(),
                    'meszl': row.get('Links/ABZ', '').strip(),
                    'colligation': row.get('Colligation', '').strip(),
                    'ur_iii': ur_iii,
                    'neo_assyrian': neo_assyrian,
                })

    return signs


def generate_encodings_batch(
    signs: List[Dict],
    skip_existing: bool = True,
    existing_encodings: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    Generate encodings for a batch of signs.

    Args:
        signs: List of sign dictionaries
        skip_existing: Skip signs that already have encodings
        existing_encodings: Dict of existing encodings to preserve

    Returns:
        Dictionary: code_point -> encoding data
    """
    encodings = existing_encodings.copy() if existing_encodings else {}

    total = len(signs)
    successful = 0
    failed = 0
    skipped = 0

    print(f"Processing {total} signs...")
    print("="*70)

    for i, sign in enumerate(signs, 1):
        unicode_char = sign['unicode']
        code_point = sign['code_point']

        # Skip if already exists
        if skip_existing and code_point in encodings:
            skipped += 1
            if i % 50 == 0:
                print(f"Progress: {i}/{total} (✓ {successful}, ✗ {failed}, ⊝ {skipped})")
            continue

        # Progress indicator
        if i % 10 == 0 or i == total:
            print(f"Progress: {i}/{total} - {unicode_char} ({code_point})")

        try:
            # Generate encoding
            encoding = generate_encoding_from_unicode(unicode_char)

            if encoding:
                encodings[code_point] = {
                    'unicode': unicode_char,
                    'encoding': encoding,
                    'phonetic': sign['phonetic'],
                    'modsl': sign['modsl'],
                    'generated_at': datetime.now().isoformat(),
                    'method': 'automatic_cv',
                    'status': 'generated'
                }
                successful += 1
            else:
                # Failed to generate
                encodings[code_point] = {
                    'unicode': unicode_char,
                    'encoding': None,
                    'phonetic': sign['phonetic'],
                    'modsl': sign['modsl'],
                    'generated_at': datetime.now().isoformat(),
                    'method': 'automatic_cv',
                    'status': 'failed',
                    'error': 'No encoding generated'
                }
                failed += 1

        except Exception as e:
            # Error during generation
            encodings[code_point] = {
                'unicode': unicode_char,
                'encoding': None,
                'phonetic': sign['phonetic'],
                'modsl': sign['modsl'],
                'generated_at': datetime.now().isoformat(),
                'method': 'automatic_cv',
                'status': 'error',
                'error': str(e)
            }
            failed += 1

    print()
    print("="*70)
    print(f"Generation complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total in database: {len(encodings)}")

    return encodings


def save_encodings(encodings: Dict, output_path: Path):
    """Save encodings to JSON file."""
    output_data = {
        'format_version': '1.0',
        'generated_at': datetime.now().isoformat(),
        'total_signs': len(encodings),
        'successful': sum(1 for e in encodings.values() if e.get('status') == 'generated'),
        'failed': sum(1 for e in encodings.values() if e.get('status') in ['failed', 'error']),
        'encodings': encodings
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nEncodings saved to: {output_path}")


def load_existing_encodings(path: Path) -> Dict:
    """Load existing encodings from file."""
    if not path.exists():
        return {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('encodings', {})
    except Exception as e:
        print(f"Warning: Could not load existing encodings: {e}")
        return {}


def print_statistics(encodings: Dict):
    """Print detailed statistics about encodings."""
    print("\n" + "="*70)
    print("ENCODING STATISTICS")
    print("="*70)

    total = len(encodings)
    successful = [e for e in encodings.values() if e.get('status') == 'generated']
    failed = [e for e in encodings.values() if e.get('status') in ['failed', 'error']]

    print(f"Total signs: {total}")
    print(f"Successfully encoded: {len(successful)} ({len(successful)/total*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/total*100:.1f}%)")
    print()

    # Encoding pattern distribution
    if successful:
        patterns = defaultdict(int)
        for e in successful:
            encoding = e.get('encoding', '')

            # Categorize by pattern
            if encoding and len(encoding) <= 2:
                if encoding[-1].isdigit():
                    patterns['repetition (h2, h3, etc.)'] += 1
                else:
                    patterns['simple (single stroke)'] += 1
            elif '[' in encoding:
                patterns['horizontal stacking [...]'] += 1
            elif '{' in encoding:
                patterns['vertical stacking {...}'] += 1
            elif '(' in encoding:
                patterns['superposition (...)'] += 1
            else:
                patterns['other'] += 1

        print("Encoding patterns:")
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count} ({count/len(successful)*100:.1f}%)")

    # Failure reasons
    if failed:
        print("\nFailure analysis:")
        errors = defaultdict(int)
        for e in failed:
            error = e.get('error', 'Unknown')
            errors[error] += 1

        for error, count in sorted(errors.items(), key=lambda x: -x[1])[:5]:
            print(f"  {error}: {count}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate encodings for all cuneiform signs')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/reference/cuneiform_signs.csv'),
        help='Path to cuneiform_signs.csv'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/encodings/generated_encodings.json'),
        help='Output path for encodings'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip signs that already have encodings'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate all encodings (ignore existing)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Only process first N signs (for testing)'
    )

    args = parser.parse_args()

    # Resolve paths
    input_path = project_root / args.input
    output_path = project_root / args.output

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print("="*70)
    print("AUTOMATIC ENCODING GENERATION")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    # Load signs
    print("Loading cuneiform signs...")
    signs = load_cuneiform_signs(input_path)
    print(f"Loaded {len(signs)} unique Unicode signs")
    print()

    # Limit for testing
    if args.limit:
        signs = signs[:args.limit]
        print(f"Limited to first {args.limit} signs for testing")
        print()

    # Load existing encodings
    existing = {} if args.force else load_existing_encodings(output_path)
    if existing and not args.force:
        print(f"Loaded {len(existing)} existing encodings")
        print()

    # Generate encodings
    encodings = generate_encodings_batch(
        signs,
        skip_existing=not args.force,
        existing_encodings=existing
    )

    # Save
    save_encodings(encodings, output_path)

    # Statistics
    print_statistics(encodings)

    return 0


if __name__ == "__main__":
    sys.exit(main())
