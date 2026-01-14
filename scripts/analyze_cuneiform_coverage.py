#!/usr/bin/env python3
"""
Analyze cuneiform_signs.csv to understand language coverage, lexicon systems,
and identify gaps for extending the Hantatallas parser.

This script provides comprehensive statistics to guide data collection and
parser extension for Akkadian, Sumerian, and other cuneiform languages.
"""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def extract_unicode_signs(text: str) -> List[str]:
    """Extract individual Unicode cuneiform characters from text."""
    if not text or text == "-----":
        return []

    signs = []
    for char in text:
        code_point = ord(char)
        if (0x12000 <= code_point <= 0x123FF or
            0x12400 <= code_point <= 0x1247F or
            0x12480 <= code_point <= 0x1254F):
            signs.append(char)
    return signs


def analyze_lexicon_systems(csv_path: str) -> Dict:
    """
    Analyze which lexicon ID systems are used and their coverage.

    Returns statistics on MesZL, HethZL (HZL), aBZL, ModSL, etc.
    """
    lexicon_stats = {
        'MesZL': {'present': 0, 'empty': 0, 'signs': set()},
        'SL/HA': {'present': 0, 'empty': 0, 'signs': set()},
        'aBZL': {'present': 0, 'empty': 0, 'signs': set()},
        'HethZL': {'present': 0, 'empty': 0, 'signs': set()},
        'ModSL': {'present': 0, 'empty': 0, 'signs': set()},
    }

    total_rows = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1

            # Get all signs from this row
            ur_iii = row.get('Ur III', '').strip()
            neo_assyrian = row.get('Neo-Assyrian', '').strip()
            signs = set(extract_unicode_signs(ur_iii) + extract_unicode_signs(neo_assyrian))

            # Check each lexicon system
            for field, key in [
                ('MesZL', 'MesZL'),
                ('ŠL/HA', 'SL/HA'),
                ('aBZL', 'aBZL'),
                ('HethZL', 'HethZL'),
                ('ModSL', 'ModSL')
            ]:
                value = row.get(field, '').strip()
                # Check if truly empty (not just whitespace or special chars)
                if value and value != ' ' and value not in ['', '\xa0']:
                    lexicon_stats[key]['present'] += 1
                    lexicon_stats[key]['signs'].update(signs)
                else:
                    lexicon_stats[key]['empty'] += 1

    # Convert sets to counts
    for key in lexicon_stats:
        lexicon_stats[key]['unique_signs'] = len(lexicon_stats[key]['signs'])
        lexicon_stats[key]['signs'] = sorted(list(lexicon_stats[key]['signs']))

    lexicon_stats['total_rows'] = total_rows

    return lexicon_stats


def analyze_periods_languages(csv_path: str) -> Dict:
    """
    Analyze which historical periods and languages are represented.

    Examines Ur III, Neo-Assyrian columns and phonetic value annotations.
    """
    period_stats = {
        'ur_iii_only': set(),
        'neo_assyrian_only': set(),
        'both_periods': set(),
        'neither_period': set(),
    }

    language_mentions = Counter()
    sign_names = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            ur_iii = row.get('Ur III', '').strip()
            neo_assyrian = row.get('Neo-Assyrian', '').strip()
            phonetic = row.get('Phonetic value(s)', '').strip()

            ur_iii_signs = set(extract_unicode_signs(ur_iii))
            neo_assyrian_signs = set(extract_unicode_signs(neo_assyrian))
            all_signs = ur_iii_signs | neo_assyrian_signs

            # Categorize by period
            if ur_iii_signs and neo_assyrian_signs:
                period_stats['both_periods'].update(all_signs)
            elif ur_iii_signs:
                period_stats['ur_iii_only'].update(ur_iii_signs)
            elif neo_assyrian_signs:
                period_stats['neo_assyrian_only'].update(neo_assyrian_signs)
            else:
                # No actual signs, might be compound notation
                period_stats['neither_period'].add(f"[Row with no signs: {phonetic[:50]}]")

            # Extract language mentions from phonetic field
            # Look for patterns like "SUM", "AKK", "HIT", etc.
            if phonetic:
                # Common patterns: "SUM", "AKK", "HIT", "HURR", "DET"
                for lang in ['SUM', 'AKK', 'HIT', 'HURR', 'DET', 'Akkadian', 'Sumerian', 'Hittite']:
                    if lang in phonetic.upper():
                        language_mentions[lang.upper()] += 1

                sign_names.append(phonetic[:100])

    # Convert sets to sorted lists with counts
    result = {
        'periods': {
            'ur_iii_only': {
                'count': len(period_stats['ur_iii_only']),
                'signs': sorted(list(period_stats['ur_iii_only']))
            },
            'neo_assyrian_only': {
                'count': len(period_stats['neo_assyrian_only']),
                'signs': sorted(list(period_stats['neo_assyrian_only']))
            },
            'both_periods': {
                'count': len(period_stats['both_periods']),
                'signs': sorted(list(period_stats['both_periods']))
            },
            'neither_period': {
                'count': len(period_stats['neither_period']),
                'examples': list(period_stats['neither_period'])[:10]
            }
        },
        'language_mentions': dict(language_mentions.most_common()),
        'sample_sign_names': sign_names[:20]
    }

    return result


def analyze_sign_types(csv_path: str) -> Dict:
    """
    Categorize signs by type: simple, compound, numeric, etc.
    """
    sign_types = {
        'simple': [],           # Single Unicode character
        'compound': [],         # Multiple characters (ligatures)
        'numeric': [],          # Numeric signs
        'punctuation': [],      # Punctuation marks
        'missing': [],          # No Unicode representation
    }

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader, start=2):
            ur_iii = row.get('Ur III', '').strip()
            neo_assyrian = row.get('Neo-Assyrian', '').strip()
            colligation = row.get('Colligation', '').strip()
            phonetic = row.get('Phonetic value(s)', '').strip()

            ur_iii_signs = extract_unicode_signs(ur_iii)
            neo_assyrian_signs = extract_unicode_signs(neo_assyrian)
            all_signs = ur_iii_signs + neo_assyrian_signs

            if not all_signs:
                sign_types['missing'].append({
                    'row': row_idx,
                    'phonetic': phonetic[:50],
                    'colligation': colligation
                })
                continue

            # Check if numeric
            if 'NUMERIC' in colligation.upper() or any(c in phonetic.upper() for c in ['NUMERIC', 'NUMBER']):
                sign_types['numeric'].append({
                    'signs': all_signs,
                    'phonetic': phonetic[:50],
                    'colligation': colligation
                })
            # Check if punctuation
            elif 'PUNCTUATION' in colligation.upper():
                sign_types['punctuation'].append({
                    'signs': all_signs,
                    'phonetic': phonetic[:50],
                    'colligation': colligation
                })
            # Compound/ligature (multiple signs or contains operators like &, +)
            elif len(set(all_signs)) > 1 or '&' in colligation or '+' in colligation:
                sign_types['compound'].append({
                    'signs': list(set(all_signs)),
                    'phonetic': phonetic[:50],
                    'colligation': colligation
                })
            # Simple sign
            else:
                sign_types['simple'].append({
                    'signs': list(set(all_signs)),
                    'phonetic': phonetic[:50],
                    'colligation': colligation
                })

    # Summarize counts
    summary = {
        'simple': len(sign_types['simple']),
        'compound': len(sign_types['compound']),
        'numeric': len(sign_types['numeric']),
        'punctuation': len(sign_types['punctuation']),
        'missing': len(sign_types['missing'])
    }

    # Keep only samples for output
    for key in sign_types:
        sign_types[key] = sign_types[key][:10]  # First 10 examples

    return {
        'summary': summary,
        'examples': sign_types
    }


def map_signs_to_lexicons(csv_path: str) -> Dict:
    """
    For each unique Unicode sign, determine which lexicon systems have IDs for it.

    This helps identify alternative mapping strategies beyond HethZL.
    """
    sign_to_lexicons = defaultdict(lambda: {
        'MesZL': None,
        'SL/HA': None,
        'aBZL': None,
        'HethZL': None,
        'ModSL': None,
        'phonetic': None,
        'colligation': None
    })

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            ur_iii = row.get('Ur III', '').strip()
            neo_assyrian = row.get('Neo-Assyrian', '').strip()
            all_signs = set(extract_unicode_signs(ur_iii) + extract_unicode_signs(neo_assyrian))

            for sign in all_signs:
                # Record which lexicons have IDs for this sign
                for field, key in [
                    ('MesZL', 'MesZL'),
                    ('ŠL/HA', 'SL/HA'),
                    ('aBZL', 'aBZL'),
                    ('HethZL', 'HethZL'),
                    ('ModSL', 'ModSL')
                ]:
                    value = row.get(field, '').strip()
                    if value and value != ' ' and value not in ['', '\xa0']:
                        if sign_to_lexicons[sign][key] is None:
                            sign_to_lexicons[sign][key] = value

                # Also record descriptive info
                if sign_to_lexicons[sign]['phonetic'] is None:
                    sign_to_lexicons[sign]['phonetic'] = row.get('Phonetic value(s)', '').strip()[:50]
                if sign_to_lexicons[sign]['colligation'] is None:
                    sign_to_lexicons[sign]['colligation'] = row.get('Colligation', '').strip()[:50]

    # Analyze coverage patterns
    coverage_patterns = Counter()
    for sign, lexicons in sign_to_lexicons.items():
        available = tuple(sorted([k for k, v in lexicons.items() if v and k not in ['phonetic', 'colligation']]))
        coverage_patterns[available] += 1

    # Find signs with no lexicon IDs
    no_lexicon_signs = [
        {
            'sign': sign,
            'code_point': f"U+{ord(sign):04X}",
            'phonetic': data['phonetic'],
            'colligation': data['colligation']
        }
        for sign, data in sign_to_lexicons.items()
        if not any(data[k] for k in ['MesZL', 'SL/HA', 'aBZL', 'HethZL', 'ModSL'])
    ]

    return {
        'total_unique_signs': len(sign_to_lexicons),
        'coverage_patterns': dict(coverage_patterns.most_common(10)),
        'no_lexicon_signs': no_lexicon_signs[:20],
        'sample_mappings': {
            sign: lexicons
            for sign, lexicons in list(sign_to_lexicons.items())[:10]
        }
    }


def main():
    csv_path = Path(__file__).parent.parent / "data/reference/cuneiform_signs.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    print("=" * 80)
    print("CUNEIFORM SIGNS DATABASE ANALYSIS")
    print("=" * 80)
    print(f"Analyzing: {csv_path}")
    print()

    # 1. Lexicon systems analysis
    print("1. ANALYZING LEXICON SYSTEMS...")
    lexicon_stats = analyze_lexicon_systems(csv_path)
    print(f"   Total rows: {lexicon_stats['total_rows']}")
    print()
    print("   Coverage by lexicon system:")
    for system in ['MesZL', 'SL/HA', 'aBZL', 'HethZL', 'ModSL']:
        stats = lexicon_stats[system]
        total = stats['present'] + stats['empty']
        coverage_pct = (stats['present'] / total * 100) if total > 0 else 0
        print(f"   {system:12} {stats['present']:5} rows ({coverage_pct:5.1f}%), "
              f"{stats['unique_signs']:5} unique signs")
    print()

    # 2. Period/language analysis
    print("2. ANALYZING PERIODS AND LANGUAGES...")
    period_stats = analyze_periods_languages(csv_path)
    print("   Signs by period:")
    print(f"   Ur III only:        {period_stats['periods']['ur_iii_only']['count']:5} signs")
    print(f"   Neo-Assyrian only:  {period_stats['periods']['neo_assyrian_only']['count']:5} signs")
    print(f"   Both periods:       {period_stats['periods']['both_periods']['count']:5} signs")
    print(f"   Neither period:     {period_stats['periods']['neither_period']['count']:5} entries")
    print()
    print("   Language mentions in phonetic field:")
    for lang, count in sorted(period_stats['language_mentions'].items(), key=lambda x: -x[1]):
        print(f"   {lang:12} {count:5} mentions")
    print()

    # 3. Sign types analysis
    print("3. ANALYZING SIGN TYPES...")
    sign_types = analyze_sign_types(csv_path)
    print("   Sign type distribution:")
    print(f"   Simple signs:       {sign_types['summary']['simple']:5}")
    print(f"   Compound/ligatures: {sign_types['summary']['compound']:5}")
    print(f"   Numeric signs:      {sign_types['summary']['numeric']:5}")
    print(f"   Punctuation:        {sign_types['summary']['punctuation']:5}")
    print(f"   Missing Unicode:    {sign_types['summary']['missing']:5}")
    print()

    # 4. Sign-to-lexicon mapping
    print("4. ANALYZING SIGN-TO-LEXICON MAPPINGS...")
    mapping_stats = map_signs_to_lexicons(csv_path)
    print(f"   Total unique Unicode signs: {mapping_stats['total_unique_signs']}")
    print(f"   Signs with NO lexicon IDs:  {len(mapping_stats['no_lexicon_signs'])}")
    print()
    print("   Most common coverage patterns:")
    for pattern, count in list(mapping_stats['coverage_patterns'].items())[:5]:
        pattern_str = ', '.join(pattern) if pattern else '(none)'
        print(f"   {pattern_str:40} {count:5} signs")
    print()

    # Save detailed results
    output_file = Path(__file__).parent.parent / "results/cuneiform_coverage_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'lexicon_systems': lexicon_stats,
        'periods_languages': period_stats,
        'sign_types': sign_types,
        'sign_to_lexicon_mapping': mapping_stats
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to: {output_file}")
    print()

    # Key recommendations
    print("=" * 80)
    print("KEY FINDINGS FOR PARSER EXTENSION:")
    print("=" * 80)

    # Recommend best lexicon to use as primary ID
    best_lexicon = max(
        [(name, stats['present']) for name, stats in lexicon_stats.items() if name != 'total_rows'],
        key=lambda x: x[1]
    )
    print(f"1. Best primary lexicon: {best_lexicon[0]} ({best_lexicon[1]} rows)")
    print()

    # Show alternative lexicons
    alternatives = sorted(
        [(name, stats['present']) for name, stats in lexicon_stats.items()
         if name != 'total_rows' and name != best_lexicon[0]],
        key=lambda x: -x[1]
    )
    print("2. Alternative lexicons (for fallback):")
    for name, count in alternatives[:3]:
        print(f"   - {name}: {count} rows")
    print()

    print("3. Coverage gaps to address:")
    print(f"   - {len(mapping_stats['no_lexicon_signs'])} signs have NO lexicon IDs")
    print(f"   - {sign_types['summary']['numeric']} numeric signs (often missing from HZL)")
    print(f"   - {sign_types['summary']['compound']} compound signs need decomposition logic")
    print()

    print("4. Recommended approach:")
    print("   - Use MesZL as primary ID system (most complete)")
    print("   - Fall back to ModSL, aBZL for signs missing MesZL")
    print("   - Create custom encoding for signs with no lexicon IDs")
    print("   - Build compound sign decomposition from Colligation field")
    print()


if __name__ == "__main__":
    main()
