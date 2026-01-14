#!/usr/bin/env python3
"""
Validate Hantatallas coverage against cuneiform signs glossary.

This script checks how many signs from the cuneiform_signs.csv glossary
can be encoded using the Hantatallas stroke-based encoding system.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nabu.tokenizers.hantatallas import HantatalasConverter


def extract_unicode_signs(text: str) -> List[str]:
    """
    Extract individual Unicode cuneiform characters from text.

    Args:
        text: String that may contain cuneiform signs

    Returns:
        List of individual cuneiform Unicode characters
    """
    if not text or text == "-----":
        return []

    signs = []
    for char in text:
        # Check if character is in cuneiform Unicode blocks
        # U+12000–U+123FF (Cuneiform)
        # U+12400–U+1247F (Cuneiform Numbers and Punctuation)
        # U+12480–U+1254F (Early Dynastic Cuneiform)
        code_point = ord(char)
        if (0x12000 <= code_point <= 0x123FF or
            0x12400 <= code_point <= 0x1247F or
            0x12480 <= code_point <= 0x1254F):
            signs.append(char)

    return signs


def load_glossary_signs(csv_path: str) -> Tuple[Dict[str, Dict], Set[str]]:
    """
    Load signs from the cuneiform glossary CSV.

    Args:
        csv_path: Path to cuneiform_signs.csv

    Returns:
        Tuple of (sign_info_dict, unique_signs_set)
    """
    sign_info = {}
    all_signs = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader, start=2):  # Start at 2 (after header)
            # Extract signs from Ur III column
            ur_iii = row.get('Ur III', '').strip()
            neo_assyrian = row.get('Neo-Assyrian', '').strip()
            colligation = row.get('Colligation', '').strip()
            phonetic = row.get('Phonetic value(s)', '').strip()

            # Extract Unicode signs
            ur_iii_signs = extract_unicode_signs(ur_iii)
            neo_assyrian_signs = extract_unicode_signs(neo_assyrian)

            # Get all unique signs from this row
            row_signs = set(ur_iii_signs + neo_assyrian_signs)

            # Store info for each sign
            for sign in row_signs:
                if sign not in sign_info:
                    # Track which periods this sign appears in
                    periods = []
                    if sign in ur_iii_signs:
                        periods.append('Ur III')
                    if sign in neo_assyrian_signs:
                        periods.append('Neo-Assyrian')

                    sign_info[sign] = {
                        'phonetic': phonetic[:100] if phonetic else '',  # Truncate long values
                        'colligation': colligation,
                        'first_seen_row': row_idx,
                        'ur_iii': ur_iii,
                        'neo_assyrian': neo_assyrian,
                        'periods': periods,  # Track which periods
                    }

                all_signs.add(sign)

    return sign_info, all_signs


def analyze_missing_distribution(missing_signs: Set[str], sign_info: Dict[str, Dict]) -> Dict:
    """
    Analyze the distribution of missing signs by period/language.

    Args:
        missing_signs: Set of missing Unicode signs
        sign_info: Dictionary with sign information

    Returns:
        Dictionary with distribution statistics
    """
    distribution = {
        'ur_iii_only': 0,
        'neo_assyrian_only': 0,
        'both_periods': 0,
        'unknown': 0
    }

    for sign in missing_signs:
        info = sign_info.get(sign, {})
        periods = info.get('periods', [])

        if not periods:
            distribution['unknown'] += 1
        elif len(periods) == 2:
            distribution['both_periods'] += 1
        elif 'Ur III' in periods:
            distribution['ur_iii_only'] += 1
        elif 'Neo-Assyrian' in periods:
            distribution['neo_assyrian_only'] += 1

    return distribution


def validate_hantatallas_coverage(
    glossary_path: str,
    output_file: str = None,
    verbose: bool = False
) -> Dict:
    """
    Validate Hantatallas coverage against glossary.

    Args:
        glossary_path: Path to cuneiform_signs.csv
        output_file: Optional path to save detailed report
        verbose: Print detailed information

    Returns:
        Dictionary with validation results
    """
    print("=" * 70)
    print("HANTATALLAS COVERAGE VALIDATION")
    print("=" * 70)

    # Load Hantatallas converter
    print("\nLoading Hantatallas database...")
    try:
        converter = HantatalasConverter()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure the hantatallas repository is available at:")
        print("  ~/projects/cuneiform/hantatallas/")
        print("\nOr specify custom paths when creating the converter.")
        return {}

    # Get all signs in Hantatallas
    hantatallas_signs = converter.get_all_unicode_signs()
    print(f"Hantatallas database contains {len(hantatallas_signs)} signs")

    # Load glossary
    print(f"\nLoading glossary from {glossary_path}...")
    sign_info, glossary_signs = load_glossary_signs(glossary_path)
    print(f"Glossary contains {len(glossary_signs)} unique signs")

    # Find coverage
    covered_signs = glossary_signs & hantatallas_signs
    missing_signs = glossary_signs - hantatallas_signs
    extra_signs = hantatallas_signs - glossary_signs

    # Calculate statistics
    coverage_percent = (len(covered_signs) / len(glossary_signs) * 100) if glossary_signs else 0

    # Prepare results
    results = {
        'glossary_total': len(glossary_signs),
        'hantatallas_total': len(hantatallas_signs),
        'covered': len(covered_signs),
        'missing': len(missing_signs),
        'extra': len(extra_signs),
        'coverage_percent': coverage_percent,
        'covered_signs': sorted(list(covered_signs)),
        'missing_signs': sorted(list(missing_signs)),
        'extra_signs': sorted(list(extra_signs)),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY")
    print("=" * 70)
    print(f"Total signs in glossary:          {results['glossary_total']:>6}")
    print(f"Total signs in Hantatallas:       {results['hantatallas_total']:>6}")
    print(f"Signs covered by Hantatallas:     {results['covered']:>6}")
    print(f"Signs missing from Hantatallas:   {results['missing']:>6}")
    print(f"Extra signs in Hantatallas:       {results['extra']:>6}")
    print(f"\nCoverage: {coverage_percent:.2f}%")

    # Analyze distribution of missing signs
    if missing_signs:
        distribution = analyze_missing_distribution(missing_signs, sign_info)
        print("\n" + "=" * 70)
        print("MISSING SIGNS DISTRIBUTION BY PERIOD")
        print("=" * 70)
        print(f"Ur III only:                      {distribution['ur_iii_only']:>6} signs")
        print(f"Neo-Assyrian only:                {distribution['neo_assyrian_only']:>6} signs")
        print(f"Both periods:                     {distribution['both_periods']:>6} signs")
        if distribution['unknown'] > 0:
            print(f"Unknown period:                   {distribution['unknown']:>6} signs")

    # Print missing signs details
    if missing_signs:
        print("\n" + "=" * 70)
        print(f"MISSING SIGNS ({len(missing_signs)} signs)")
        print("=" * 70)
        print("(Signs in glossary but not in Hantatallas database)")
        print()

        # Sort by row number for easier reference
        missing_with_info = [
            (sign, sign_info.get(sign, {})) for sign in missing_signs
        ]
        missing_with_info.sort(key=lambda x: x[1].get('first_seen_row', 9999))

        for idx, (sign, info) in enumerate(missing_with_info, 1):
            code_point = f"U+{ord(sign):04X}"
            phonetic = info.get('phonetic', '')[:60]  # Truncate long values
            row = info.get('first_seen_row', '?')

            if verbose:
                print(f"{idx:4}. {sign}  {code_point:>8}  Row {row:>4}  {phonetic}")
            else:
                # Compact format
                if idx <= 50:  # Show first 50
                    print(f"{idx:4}. {sign}  {code_point:>8}  {phonetic[:40]}")

        if not verbose and len(missing_signs) > 50:
            print(f"\n... and {len(missing_signs) - 50} more (use --verbose to see all)")

    # Print extra signs summary
    if extra_signs:
        print("\n" + "=" * 70)
        print(f"EXTRA SIGNS IN HANTATALLAS ({len(extra_signs)} signs)")
        print("=" * 70)
        print("(Signs in Hantatallas but not in glossary)")
        print()

        if verbose:
            for idx, sign in enumerate(sorted(extra_signs), 1):
                code_point = f"U+{ord(sign):04X}"
                encoding = converter.unicode_to_hantatallas(sign)
                print(f"{idx:4}. {sign}  {code_point:>8}  Encoding: {encoding}")
        else:
            # Just show first 20
            for idx, sign in enumerate(sorted(list(extra_signs)[:20]), 1):
                code_point = f"U+{ord(sign):04X}"
                print(f"{idx:4}. {sign}  {code_point:>8}")

            if len(extra_signs) > 20:
                print(f"\n... and {len(extra_signs) - 20} more (use --verbose to see all)")

    # Example conversions for covered signs
    print("\n" + "=" * 70)
    print("EXAMPLE CONVERSIONS (First 10 covered signs)")
    print("=" * 70)

    for idx, sign in enumerate(sorted(list(covered_signs)[:10]), 1):
        code_point = f"U+{ord(sign):04X}"
        encoding = converter.unicode_to_hantatallas(sign)
        info = sign_info.get(sign, {})
        phonetic = info.get('phonetic', '')[:40]
        print(f"{idx:2}. {sign}  {code_point:>8}  →  {encoding:20}  ({phonetic})")

    # Save detailed report if requested
    if output_file:
        print(f"\n" + "=" * 70)
        print(f"Saving detailed report to {output_file}...")

        # Prepare detailed report
        detailed_report = {
            'summary': {
                'glossary_total': results['glossary_total'],
                'hantatallas_total': results['hantatallas_total'],
                'covered': results['covered'],
                'missing': results['missing'],
                'extra': results['extra'],
                'coverage_percent': results['coverage_percent'],
            },
            'missing_signs': [
                {
                    'sign': sign,
                    'code_point': f"U+{ord(sign):04X}",
                    'info': sign_info.get(sign, {})
                }
                for sign in sorted(missing_signs)
            ],
            'covered_signs': [
                {
                    'sign': sign,
                    'code_point': f"U+{ord(sign):04X}",
                    'encoding': converter.unicode_to_hantatallas(sign),
                    'info': sign_info.get(sign, {})
                }
                for sign in sorted(covered_signs)
            ],
            'extra_signs': [
                {
                    'sign': sign,
                    'code_point': f"U+{ord(sign):04X}",
                    'encoding': converter.unicode_to_hantatallas(sign)
                }
                for sign in sorted(extra_signs)
            ]
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, ensure_ascii=False, indent=2)

        print(f"Detailed report saved to {output_file}")

    print("\n" + "=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate Hantatallas coverage against cuneiform glossary"
    )
    parser.add_argument(
        "--glossary",
        type=str,
        default="data/reference/cuneiform_signs.csv",
        help="Path to cuneiform signs CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save detailed report to JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for all signs"
    )

    args = parser.parse_args()

    # Run validation
    results = validate_hantatallas_coverage(
        glossary_path=args.glossary,
        output_file=args.output,
        verbose=args.verbose
    )

    if not results:
        print("\nValidation failed due to missing data files.")
        return 1

    # Exit with status based on coverage
    if results['missing'] > 0:
        print(f"\n⚠️  Warning: {results['missing']} signs from glossary are not in Hantatallas")
        print(f"Coverage: {results['coverage_percent']:.2f}%")
    else:
        print(f"\n✓ All glossary signs are covered by Hantatallas!")

    print("\nValidation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
