#!/usr/bin/env python3
"""
Analyze the Hantatallas repository to discover all available sign databases
beyond HZL (Hittite). This helps identify existing stroke encodings for
Akkadian, Sumerian, Ugaritic, and other cuneiform systems.
"""

import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set


def count_encodings_in_dat_file(dat_path: Path) -> Dict:
    """
    Parse a .dat file and count how many signs have stroke encodings.
    """
    if not dat_path.exists():
        return {'error': 'File not found'}

    with open(dat_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    sign_count = 0
    encoding_count = 0
    current_sign_id = None
    in_form_section = False
    encodings = []
    sign_ids = []

    for line in lines:
        tabs = len(line) - len(line.lstrip('\t'))
        stripped = line.strip()

        if not stripped:
            continue

        # Level 0: Sign ID
        if tabs == 0:
            if stripped.replace('.', '').replace('A', '').replace('B', '').replace('C', '').isdigit() or \
               (len(stripped) <= 5 and any(c.isdigit() for c in stripped)):
                sign_count += 1
                current_sign_id = stripped
                sign_ids.append(stripped)
                in_form_section = False

        # Level 1: Section headers
        elif tabs == 1:
            in_form_section = (stripped == 'FORM')

        # Level 2: Encoding forms
        elif tabs == 2 and in_form_section and current_sign_id:
            # Extract just the encoding (before any tags)
            parts = stripped.split()
            if parts:
                encoding = parts[0]
                # Check if it looks like a stroke encoding
                if any(c in encoding for c in 'hvudcHVUDC[]{}()'):
                    encoding_count += 1
                    encodings.append(encoding)

    return {
        'file': dat_path.name,
        'total_signs': sign_count,
        'signs_with_encodings': encoding_count,
        'coverage_percent': (encoding_count / sign_count * 100) if sign_count > 0 else 0,
        'sample_sign_ids': sign_ids[:10],
        'sample_encodings': encodings[:10]
    }


def analyze_unicode_mapping_csv(csv_path: Path, dat_files: List[str]) -> Dict:
    """
    Analyze unicode_cleaned.csv to see which Unicode signs map to which databases.
    """
    if not csv_path.exists():
        return {'error': 'File not found'}

    import csv

    # Map database name to column index/name
    # MesZL [0], ŠL/HA [1], aBZL [2], HethZL [3]
    database_coverage = {
        'MesZL': set(),
        'SL/HA': set(),
        'aBZL': set(),
        'HethZL': set(),
        'total_unicode_signs': set()
    }

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if len(row) < 6:
                continue

            # Extract Unicode characters from Unicode Glyph column [5]
            unicode_col = row[5]
            for char in unicode_col:
                code_point = ord(char)
                if 0x12000 <= code_point <= 0x123FF or \
                   0x12400 <= code_point <= 0x1247F or \
                   0x12480 <= code_point <= 0x1254F:
                    database_coverage['total_unicode_signs'].add(char)

                    # Check which databases have IDs for this sign
                    if row[0].strip() and row[0].strip() not in [' ', '\xa0']:  # MesZL
                        database_coverage['MesZL'].add(char)
                    if row[1].strip() and row[1].strip() not in [' ', '\xa0']:  # ŠL/HA
                        database_coverage['SL/HA'].add(char)
                    if row[2].strip() and row[2].strip() not in [' ', '\xa0']:  # aBZL
                        database_coverage['aBZL'].add(char)
                    if row[3].strip() and row[3].strip() not in [' ', '\xa0']:  # HethZL
                        database_coverage['HethZL'].add(char)

    return {
        'total_unicode_signs': len(database_coverage['total_unicode_signs']),
        'MesZL_coverage': len(database_coverage['MesZL']),
        'SL_HA_coverage': len(database_coverage['SL/HA']),
        'aBZL_coverage': len(database_coverage['aBZL']),
        'HethZL_coverage': len(database_coverage['HethZL'])
    }


def discover_additional_encodings(hantatallas_path: Path) -> Dict:
    """
    Look for any additional encoding sources in the Hantatallas repo.
    """
    # Check for other potential data sources
    data_dir = hantatallas_path / "hantatallas" / "data"

    findings = {
        'oracc_support': False,
        'font_generation': False,
        'unicode_mapping': False,
        'additional_databases': []
    }

    # Check for ORACC integration
    oracc_file = hantatallas_path / "hantatallas" / "oracc.py"
    if oracc_file.exists():
        findings['oracc_support'] = True
        with open(oracc_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'sl.xml' in content:
                findings['oracc_xml_source'] = 'Uses ORACC Sign List XML'

    # Check for font generation (may contain additional sign definitions)
    fonts_file = hantatallas_path / "hantatallas" / "fonts.py"
    if fonts_file.exists():
        findings['font_generation'] = True

    # Check for Unicode mapping CSV
    unicode_csv = data_dir / "unicode_cleaned.csv"
    if unicode_csv.exists():
        findings['unicode_mapping'] = True

    # List all .dat files
    if data_dir.exists():
        dat_files = list(data_dir.glob("*.dat"))
        findings['dat_files'] = [f.name for f in dat_files]

        # Check for documentation about other databases
        for dat_file in dat_files:
            if 'uga' in dat_file.name.lower():
                findings['additional_databases'].append('Ugaritic (uga.dat)')
            elif 'huehnergard' in dat_file.name.lower():
                findings['additional_databases'].append('Huehnergard (alternative spellings)')
            elif 'cleanup' in dat_file.name.lower():
                findings['additional_databases'].append('Cleanup/normalization rules')
            elif 'replacements' in dat_file.name.lower():
                findings['additional_databases'].append('Replacement/variant mappings')

    return findings


def main():
    print("=" * 80)
    print("HANTATALLAS DATABASE ANALYSIS")
    print("=" * 80)
    print()

    hantatallas_path = Path.home() / "projects/cuneiform/hantatallas"
    data_dir = hantatallas_path / "hantatallas" / "data"

    if not hantatallas_path.exists():
        print(f"Error: Hantatallas repository not found at {hantatallas_path}")
        return

    # 1. Discover what's available
    print("1. DISCOVERING AVAILABLE DATABASES...")
    discoveries = discover_additional_encodings(hantatallas_path)
    print(f"   ORACC integration:      {discoveries['oracc_support']}")
    print(f"   Font generation:        {discoveries['font_generation']}")
    print(f"   Unicode mapping CSV:    {discoveries['unicode_mapping']}")
    print()

    if 'dat_files' in discoveries:
        print(f"   Found {len(discoveries['dat_files'])} .dat files:")
        for dat_file in discoveries['dat_files']:
            print(f"   - {dat_file}")
    print()

    if discoveries['additional_databases']:
        print("   Additional databases identified:")
        for db in discoveries['additional_databases']:
            print(f"   - {db}")
    print()

    # 2. Analyze each .dat file
    print("2. ANALYZING .DAT FILES...")
    dat_stats = []

    if data_dir.exists():
        dat_files = sorted(data_dir.glob("*.dat"))
        for dat_file in dat_files:
            print(f"   Analyzing {dat_file.name}...")
            stats = count_encodings_in_dat_file(dat_file)
            dat_stats.append(stats)

            if 'error' in stats:
                print(f"     Error: {stats['error']}")
            else:
                print(f"     Total signs: {stats['total_signs']}")
                print(f"     Signs with encodings: {stats['signs_with_encodings']}")
                print(f"     Coverage: {stats['coverage_percent']:.1f}%")
                if stats['sample_encodings']:
                    print(f"     Sample encodings: {', '.join(stats['sample_encodings'][:3])}")
            print()

    # 3. Analyze Unicode mapping
    print("3. ANALYZING UNICODE MAPPING...")
    unicode_csv = data_dir / "unicode_cleaned.csv"
    if unicode_csv.exists():
        mapping_stats = analyze_unicode_mapping_csv(unicode_csv, discoveries.get('dat_files', []))
        print(f"   Total Unicode signs in CSV: {mapping_stats['total_unicode_signs']}")
        print(f"   Signs with MesZL IDs:       {mapping_stats['MesZL_coverage']} "
              f"({mapping_stats['MesZL_coverage']/mapping_stats['total_unicode_signs']*100:.1f}%)")
        print(f"   Signs with SL/HA IDs:       {mapping_stats['SL_HA_coverage']} "
              f"({mapping_stats['SL_HA_coverage']/mapping_stats['total_unicode_signs']*100:.1f}%)")
        print(f"   Signs with aBZL IDs:        {mapping_stats['aBZL_coverage']} "
              f"({mapping_stats['aBZL_coverage']/mapping_stats['total_unicode_signs']*100:.1f}%)")
        print(f"   Signs with HethZL IDs:      {mapping_stats['HethZL_coverage']} "
              f"({mapping_stats['HethZL_coverage']/mapping_stats['total_unicode_signs']*100:.1f}%)")
    else:
        print("   Unicode mapping CSV not found")
    print()

    # 4. Summary and recommendations
    print("=" * 80)
    print("FINDINGS FOR PARSER EXTENSION:")
    print("=" * 80)
    print()

    # Find database with most encodings
    if dat_stats:
        best_db = max(dat_stats, key=lambda x: x.get('signs_with_encodings', 0))
        print(f"1. Primary encoding source: {best_db['file']}")
        print(f"   - {best_db['signs_with_encodings']} signs with stroke encodings")
        print()

    print("2. Recommended multi-database strategy:")
    print("   a) Use MesZL as primary ID system (best Unicode coverage)")
    print("   b) Map MesZL → HZL for signs that have HZL encodings")
    print("   c) Create new encodings for signs missing from all databases")
    print()

    print("3. Additional data sources to leverage:")
    if discoveries['oracc_support']:
        print("   - ORACC Sign List (oracc.py) - comprehensive sign database")
    print("   - unicode_cleaned.csv - cross-reference for multiple ID systems")
    if 'Ugaritic (uga.dat)' in discoveries.get('additional_databases', []):
        print("   - uga.dat - Ugaritic cuneiform variant")
    print()

    print("4. Architecture recommendations:")
    print("   - Create unified sign ID system (prefer MesZL)")
    print("   - Build multi-source encoding resolver:")
    print("     1. Check HZL database for existing encoding")
    print("     2. Check UGA database for Ugaritic signs")
    print("     3. Synthesize encoding from compound notation (Colligation field)")
    print("     4. Manual encoding for remaining signs")
    print()


if __name__ == "__main__":
    main()
