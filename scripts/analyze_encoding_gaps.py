#!/usr/bin/env python3
"""
Cross-reference cuneiform_signs.csv with Hantatallas databases to identify
exactly which signs need new encodings.

This script produces a prioritized list of signs to encode, categorized by:
- Language/period (Sumerian, Akkadian, Ur III, Neo-Assyrian)
- Sign type (simple, compound, numeric)
- Availability in different lexicon systems
"""

import csv
import json
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


def load_hantatallas_coverage(hantatallas_path: Path) -> Set[str]:
    """
    Load all Unicode signs that have encodings in Hantatallas HZL database.
    """
    hzl_to_encodings = {}
    unicode_to_hzl = {}

    # Parse hzl.dat
    hzl_dat = hantatallas_path / "hantatallas" / "data" / "hzl.dat"
    if hzl_dat.exists():
        with open(hzl_dat, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        current_id = None
        in_form_section = False

        for line in lines:
            tabs = len(line) - len(line.lstrip('\t'))
            content = line.strip()

            if not content:
                continue

            if tabs == 0 and content.isdigit():
                current_id = content
                in_form_section = False
                if current_id not in hzl_to_encodings:
                    hzl_to_encodings[current_id] = []

            elif tabs == 1:
                in_form_section = (content == 'FORM')

            elif tabs == 2 and in_form_section and current_id:
                parts = content.split()
                if parts and any(c in parts[0] for c in 'hvudcHVUDC[]{}()'):
                    hzl_to_encodings[current_id].append(parts[0])

    # Parse unicode_cleaned.csv
    unicode_csv = hantatallas_path / "hantatallas" / "data" / "unicode_cleaned.csv"
    if unicode_csv.exists():
        with open(unicode_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                if len(row) < 6:
                    continue

                hzl_id = row[3].strip()
                if not hzl_id or hzl_id in [' ', '\xa0']:
                    continue

                # Extract Unicode characters
                unicode_col = row[5]
                for char in unicode_col:
                    code_point = ord(char)
                    if 0x12000 <= code_point <= 0x123FF or \
                       0x12400 <= code_point <= 0x1247F or \
                       0x12480 <= code_point <= 0x1254F:
                        if hzl_id in hzl_to_encodings and hzl_to_encodings[hzl_id]:
                            unicode_to_hzl[char] = hzl_id

    return set(unicode_to_hzl.keys())


def analyze_gaps(glossary_path: Path, hantatallas_path: Path) -> Dict:
    """
    Compare cuneiform_signs.csv against Hantatallas to find encoding gaps.
    """
    # Load signs with encodings
    encoded_signs = load_hantatallas_coverage(hantatallas_path)
    print(f"   Loaded {len(encoded_signs)} signs with Hantatallas encodings")

    # Analyze glossary
    missing_signs = []
    covered_signs = []

    with open(glossary_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader, start=2):
            ur_iii = row.get('Ur III', '').strip()
            neo_assyrian = row.get('Neo-Assyrian', '').strip()
            phonetic = row.get('Phonetic value(s)', '').strip()
            colligation = row.get('Colligation', '').strip()

            ur_iii_signs = extract_unicode_signs(ur_iii)
            neo_assyrian_signs = extract_unicode_signs(neo_assyrian)
            all_signs = list(set(ur_iii_signs + neo_assyrian_signs))

            if not all_signs:
                continue

            # Determine periods
            periods = []
            if ur_iii_signs:
                periods.append('Ur III')
            if neo_assyrian_signs:
                periods.append('Neo-Assyrian')

            # Check lexicon coverage
            lexicons = {
                'MesZL': row.get('MesZL', '').strip(),
                'SL/HA': row.get('ŠL/HA', '').strip(),
                'aBZL': row.get('aBZL', '').strip(),
                'HethZL': row.get('HethZL', '').strip(),
                'ModSL': row.get('ModSL', '').strip()
            }
            available_lexicons = [k for k, v in lexicons.items() if v and v not in [' ', '\xa0', '']]

            # Categorize sign type
            sign_type = 'simple'
            if 'NUMERIC' in colligation.upper() or 'NUMBER' in phonetic.upper():
                sign_type = 'numeric'
            elif 'PUNCTUATION' in colligation.upper():
                sign_type = 'punctuation'
            elif len(all_signs) > 1 or '&' in colligation or '+' in colligation:
                sign_type = 'compound'

            # Check each sign
            for sign in all_signs:
                sign_info = {
                    'sign': sign,
                    'code_point': f"U+{ord(sign):04X}",
                    'row': row_idx,
                    'phonetic': phonetic[:80],
                    'colligation': colligation[:80],
                    'periods': periods,
                    'lexicons': available_lexicons,
                    'sign_type': sign_type
                }

                if sign in encoded_signs:
                    covered_signs.append(sign_info)
                else:
                    missing_signs.append(sign_info)

    return {
        'total_encoded': len(encoded_signs),
        'covered_signs': covered_signs,
        'missing_signs': missing_signs,
        'covered_count': len(covered_signs),
        'missing_count': len(missing_signs)
    }


def categorize_missing_signs(missing_signs: List[Dict]) -> Dict:
    """
    Categorize missing signs by various attributes for prioritization.
    """
    categories = {
        'by_type': Counter(),
        'by_period': Counter(),
        'by_lexicon_availability': Counter(),
        'priority_groups': {
            'high_priority': [],    # Simple signs with lexicon IDs
            'medium_priority': [],  # Compound signs with lexicon IDs
            'low_priority': [],     # Signs with no lexicon IDs
            'numeric': [],          # Numeric signs (special handling)
        }
    }

    for sign_info in missing_signs:
        # Count by type
        categories['by_type'][sign_info['sign_type']] += 1

        # Count by period
        for period in sign_info['periods']:
            categories['by_period'][period] += 1

        # Count by lexicon availability
        lex_count = len(sign_info['lexicons'])
        if lex_count == 0:
            categories['by_lexicon_availability']['No lexicon IDs'] += 1
        elif lex_count == 1:
            categories['by_lexicon_availability']['1 lexicon'] += 1
        elif lex_count >= 2:
            categories['by_lexicon_availability']['2+ lexicons'] += 1

        # Prioritize
        if sign_info['sign_type'] == 'numeric':
            categories['priority_groups']['numeric'].append(sign_info)
        elif not sign_info['lexicons']:
            categories['priority_groups']['low_priority'].append(sign_info)
        elif sign_info['sign_type'] == 'simple':
            categories['priority_groups']['high_priority'].append(sign_info)
        else:
            categories['priority_groups']['medium_priority'].append(sign_info)

    return categories


def main():
    print("=" * 80)
    print("ENCODING GAP ANALYSIS")
    print("=" * 80)
    print("Cross-referencing cuneiform_signs.csv with Hantatallas databases...")
    print()

    glossary_path = Path(__file__).parent.parent / "data/reference/cuneiform_signs.csv"
    hantatallas_path = Path.home() / "projects/cuneiform/hantatallas"

    if not glossary_path.exists():
        print(f"Error: {glossary_path} not found")
        return

    if not hantatallas_path.exists():
        print(f"Error: {hantatallas_path} not found")
        return

    # Analyze gaps
    print("1. LOADING DATA...")
    gap_analysis = analyze_gaps(glossary_path, hantatallas_path)

    total_unique_signs = gap_analysis['covered_count'] + gap_analysis['missing_count']
    coverage_pct = (gap_analysis['covered_count'] / total_unique_signs * 100) if total_unique_signs > 0 else 0

    print()
    print("2. COVERAGE SUMMARY:")
    print(f"   Total unique signs:     {total_unique_signs:5}")
    print(f"   Signs with encodings:   {gap_analysis['covered_count']:5} ({coverage_pct:.1f}%)")
    print(f"   Signs missing encodings:{gap_analysis['missing_count']:5} ({100-coverage_pct:.1f}%)")
    print()

    # Categorize missing signs
    print("3. ANALYZING MISSING SIGNS...")
    categories = categorize_missing_signs(gap_analysis['missing_signs'])

    print()
    print("   Missing signs by type:")
    for sign_type, count in categories['by_type'].most_common():
        print(f"   {sign_type:15} {count:5} signs")

    print()
    print("   Missing signs by period:")
    for period, count in categories['by_period'].most_common():
        print(f"   {period:15} {count:5} signs")

    print()
    print("   Missing signs by lexicon availability:")
    for lex_status, count in sorted(categories['by_lexicon_availability'].items()):
        print(f"   {lex_status:15} {count:5} signs")

    print()
    print("4. PRIORITIZED ENCODING TASKS:")
    print()
    print(f"   HIGH PRIORITY (simple signs with lexicon IDs):")
    print(f"   {len(categories['priority_groups']['high_priority'])} signs")
    for sign_info in categories['priority_groups']['high_priority'][:5]:
        print(f"   - {sign_info['sign']} {sign_info['code_point']} "
              f"[{', '.join(sign_info['lexicons'][:2])}] {sign_info['phonetic'][:40]}")
    if len(categories['priority_groups']['high_priority']) > 5:
        print(f"   ... and {len(categories['priority_groups']['high_priority']) - 5} more")

    print()
    print(f"   MEDIUM PRIORITY (compound signs with lexicon IDs):")
    print(f"   {len(categories['priority_groups']['medium_priority'])} signs")
    for sign_info in categories['priority_groups']['medium_priority'][:5]:
        print(f"   - {sign_info['sign']} {sign_info['code_point']} "
              f"[{', '.join(sign_info['lexicons'][:2])}] {sign_info['phonetic'][:40]}")
    if len(categories['priority_groups']['medium_priority']) > 5:
        print(f"   ... and {len(categories['priority_groups']['medium_priority']) - 5} more")

    print()
    print(f"   NUMERIC SIGNS (special handling needed):")
    print(f"   {len(categories['priority_groups']['numeric'])} signs")
    for sign_info in categories['priority_groups']['numeric'][:5]:
        print(f"   - {sign_info['sign']} {sign_info['code_point']} {sign_info['phonetic'][:40]}")
    if len(categories['priority_groups']['numeric']) > 5:
        print(f"   ... and {len(categories['priority_groups']['numeric']) - 5} more")

    print()
    print(f"   LOW PRIORITY (no lexicon IDs - need manual research):")
    print(f"   {len(categories['priority_groups']['low_priority'])} signs")

    # Save detailed results
    output_file = Path(__file__).parent.parent / "results/encoding_gap_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'summary': {
            'total_signs': total_unique_signs,
            'covered': gap_analysis['covered_count'],
            'missing': gap_analysis['missing_count'],
            'coverage_percent': coverage_pct
        },
        'missing_signs_by_priority': {
            'high': categories['priority_groups']['high_priority'],
            'medium': categories['priority_groups']['medium_priority'],
            'numeric': categories['priority_groups']['numeric'],
            'low': categories['priority_groups']['low_priority']
        },
        'categorization': {
            'by_type': dict(categories['by_type']),
            'by_period': dict(categories['by_period']),
            'by_lexicon': dict(categories['by_lexicon_availability'])
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print(f"Detailed gap analysis saved to: {output_file}")
    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS FOR IMPLEMENTATION:")
    print("=" * 80)
    print()
    print("Phase 1: Infrastructure (remove HethZL dependency)")
    print("  - Create unified sign ID system using MesZL as primary")
    print("  - Build multi-database resolver (HZL, UGA, custom)")
    print("  - Implement fallback chain for sign lookup")
    print()
    print("Phase 2: High-priority simple signs")
    print(f"  - {len(categories['priority_groups']['high_priority'])} simple signs with lexicon IDs")
    print("  - Can use visual decomposition or existing similar signs as templates")
    print()
    print("Phase 3: Numeric signs system")
    print(f"  - {len(categories['priority_groups']['numeric'])} numeric signs")
    print("  - Design systematic encoding for Sumerian/Akkadian numerals")
    print("  - Pattern: h=1, h2=2, h3=3, etc.")
    print()
    print("Phase 4: Compound signs")
    print(f"  - {len(categories['priority_groups']['medium_priority'])} compound/ligature signs")
    print("  - Parse Colligation field to decompose into constituent signs")
    print("  - Use existing sign encodings + composition operators")
    print()


if __name__ == "__main__":
    main()
