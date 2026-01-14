#!/usr/bin/env python3
"""
Generate a template file for manually adding stroke encodings for signs
that are missing from the Hantatallas databases.

This creates a structured JSON/YAML template that makes it easy to:
1. See which signs need encodings
2. Add encodings in the Hantatallas stroke format
3. Import the encodings into our extended database
"""

import csv
import json
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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


def load_existing_encodings(hantatallas_path: Path) -> Dict[str, str]:
    """Load existing Unicode -> encoding mappings from Hantatallas."""
    try:
        from nabu.tokenizers.hantatallas import HantatalasConverter
        converter = HantatalasConverter()

        encodings = {}
        for sign in converter.get_all_unicode_signs():
            encoding = converter.unicode_to_hantatallas(sign)
            if encoding:
                encodings[sign] = encoding

        return encodings
    except Exception as e:
        print(f"Warning: Could not load Hantatallas converter: {e}")
        return {}


def generate_encoding_template(glossary_path: Path, output_path: Path, hantatallas_path: Path):
    """
    Generate a JSON template file for missing encodings.
    """
    # Load existing encodings
    existing_encodings = load_existing_encodings(hantatallas_path)
    print(f"Loaded {len(existing_encodings)} existing encodings from Hantatallas")

    # Analyze glossary for missing signs
    template_entries = []

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

            # Get lexicon IDs
            lexicons = {
                'MesZL': row.get('MesZL', '').strip(),
                'SL_HA': row.get('ŠL/HA', '').strip(),
                'aBZL': row.get('aBZL', '').strip(),
                'HethZL': row.get('HethZL', '').strip(),
                'ModSL': row.get('ModSL', '').strip()
            }
            # Remove empty values
            lexicons = {k: v for k, v in lexicons.items() if v and v not in [' ', '\xa0', '']}

            # Determine primary ID (prefer MesZL)
            primary_id = lexicons.get('MesZL') or lexicons.get('ModSL') or lexicons.get('aBZL') or 'UNKNOWN'

            # Categorize
            sign_type = 'simple'
            if 'NUMERIC' in colligation.upper():
                sign_type = 'numeric'
            elif 'PUNCTUATION' in colligation.upper():
                sign_type = 'punctuation'
            elif len(all_signs) > 1 or '&' in colligation or '+' in colligation:
                sign_type = 'compound'

            # Determine periods
            periods = []
            if ur_iii_signs:
                periods.append('Ur_III')
            if neo_assyrian_signs:
                periods.append('Neo_Assyrian')

            # Check each sign
            for sign in all_signs:
                if sign not in existing_encodings:
                    # This sign needs an encoding
                    entry = {
                        'sign': sign,
                        'unicode': f"U+{ord(sign):04X}",
                        'primary_id': primary_id,
                        'lexicons': lexicons,
                        'phonetic': phonetic[:100],
                        'colligation': colligation[:100],
                        'periods': periods,
                        'type': sign_type,
                        'csv_row': row_idx,

                        # Fields to fill in
                        'encoding': '',  # TO BE FILLED: Hantatallas stroke encoding
                        'notes': '',     # TO BE FILLED: Any notes about the encoding
                        'similar_to': '',  # TO BE FILLED: Reference to similar sign
                        'status': 'pending'  # pending | encoded | needs_research
                    }
                    template_entries.append(entry)

    # Sort by priority
    def priority_key(entry):
        # Priority: simple > numeric > compound
        type_priority = {'simple': 0, 'numeric': 1, 'compound': 2, 'punctuation': 3}
        # More lexicons = higher priority
        lex_priority = -len(entry['lexicons'])
        return (type_priority.get(entry['type'], 4), lex_priority, entry['sign'])

    template_entries.sort(key=priority_key)

    # Generate template
    template = {
        'metadata': {
            'description': 'Template for adding Hantatallas stroke encodings',
            'format_version': '1.0',
            'total_missing_signs': len(template_entries),
            'encoding_format': {
                'strokes': {
                    'h': 'horizontal',
                    'v': 'vertical',
                    'u': 'upward diagonal',
                    'd': 'downward diagonal',
                    'c': 'winkelhaken (angular hook)',
                    '0': 'void/empty space'
                },
                'containers': {
                    '[...]': 'horizontal stacking (side by side)',
                    '{...}': 'vertical stacking (top to bottom)',
                    '(...)': 'superposition (overlapping)'
                },
                'modifiers': {
                    "'": 'shorten at head',
                    '"': 'shorten at tail',
                    '2': 'double stroke',
                    '3': 'triple stroke'
                },
                'canvas_shapes': {
                    'S': 'square (default)',
                    'L': 'landscape',
                    'P': 'portrait',
                    'W': 'wide',
                    'N': 'narrow'
                },
                'examples': {
                    'h': 'single horizontal (sign 𒀸 AŠ)',
                    'h2': 'double horizontal (sign 𒐀 AŠ.AŠ)',
                    'h3': 'triple horizontal (numeric 3)',
                    '{hv}': 'horizontal above vertical',
                    '[hv]': 'horizontal beside vertical',
                    '(h2[00v0])': 'AN sign - double horizontal over 4-element grid'
                }
            }
        },
        'signs_to_encode': template_entries
    }

    # Save template
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print()
    print(f"Generated encoding template with {len(template_entries)} signs")
    print(f"Saved to: {output_path}")
    print()

    # Print summary
    from collections import Counter
    type_counts = Counter(e['type'] for e in template_entries)
    print("Signs by type:")
    for sign_type, count in type_counts.most_common():
        print(f"  {sign_type:12} {count:5} signs")
    print()

    print("Next steps:")
    print("  1. Open the template file in a text editor")
    print("  2. For each sign, add the 'encoding' field with Hantatallas notation")
    print("  3. Use the 'similar_to' field to reference existing encoded signs")
    print("  4. Update 'status' to 'encoded' when complete")
    print("  5. Use import_encodings.py to add them to the database")
    print()


def main():
    glossary_path = Path(__file__).parent.parent / "data/reference/cuneiform_signs.csv"
    hantatallas_path = Path.home() / "projects/cuneiform/hantatallas"
    output_path = Path(__file__).parent.parent / "data/encoding_templates/missing_encodings.json"

    if not glossary_path.exists():
        print(f"Error: {glossary_path} not found")
        return

    print("=" * 80)
    print("GENERATING ENCODING TEMPLATE")
    print("=" * 80)
    print()

    generate_encoding_template(glossary_path, output_path, hantatallas_path)


if __name__ == "__main__":
    main()
