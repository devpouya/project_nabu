# Cuneiform Encoding Analysis Scripts

This directory contains analysis scripts to help extend the Hantatallas parser to support all signs in `cuneiform_signs.csv`.

## Quick Start

Run all three analysis scripts in order to gather complete information:

```bash
cd /Users/pouyapourjafar/projects/cuneiform/project_nabu

# 1. Analyze cuneiform_signs.csv structure and coverage
python scripts/analyze_cuneiform_coverage.py

# 2. Analyze Hantatallas databases
python scripts/analyze_hantatallas_databases.py

# 3. Cross-reference to find encoding gaps
python scripts/analyze_encoding_gaps.py

# 4. Generate template for adding new encodings
python scripts/generate_encoding_template.py
```

## Script Descriptions

### 1. `analyze_cuneiform_coverage.py`

**Purpose**: Understand the structure and content of `cuneiform_signs.csv`

**Output**:
- Which lexicon systems are used (MesZL, HethZL, aBZL, ModSL)
- Coverage statistics for each lexicon
- Period distribution (Ur III, Neo-Assyrian)
- Language mentions (Sumerian, Akkadian, Hittite)
- Sign type breakdown (simple, compound, numeric, punctuation)
- Sign-to-lexicon mappings

**Key Question Answered**: What data is available in our glossary to use as identifiers?

**Saved Results**: `results/cuneiform_coverage_analysis.json`

---

### 2. `analyze_hantatallas_databases.py`

**Purpose**: Discover what sign databases and encodings exist in Hantatallas

**Output**:
- List of all .dat files (hzl.dat, uga.dat, etc.)
- How many signs have stroke encodings in each database
- Unicode coverage in unicode_cleaned.csv
- Which lexicon systems have the best coverage
- Additional data sources (ORACC, font generation)

**Key Question Answered**: What encodings already exist, and where are the gaps?

**Saved Results**: Console output only (read-only analysis)

---

### 3. `analyze_encoding_gaps.py`

**Purpose**: Cross-reference cuneiform_signs.csv with Hantatallas to find exactly which signs need encodings

**Output**:
- Total coverage percentage
- Missing signs categorized by:
  - Type (simple, compound, numeric)
  - Period (Ur III, Neo-Assyrian)
  - Lexicon availability
- Prioritized list of signs to encode:
  - HIGH: Simple signs with lexicon IDs
  - MEDIUM: Compound signs with lexicon IDs
  - NUMERIC: Numeric signs (need systematic approach)
  - LOW: Signs with no lexicon IDs

**Key Question Answered**: Which signs should we encode first?

**Saved Results**: `results/encoding_gap_analysis.json`

---

### 4. `generate_encoding_template.py`

**Purpose**: Create a structured template file for manually adding stroke encodings

**Output**:
- JSON file with all missing signs
- Pre-filled metadata (Unicode, lexicon IDs, phonetic values)
- Empty `encoding` fields to fill in
- Encoding format reference (strokes, containers, modifiers)
- Sorted by priority (simple signs first)

**Key Question Answered**: How do we systematically add the missing encodings?

**Saved Results**: `data/encoding_templates/missing_encodings.json`

---

## Analysis Workflow

```
┌─────────────────────────────────┐
│  cuneiform_signs.csv            │
│  (1893 rows, all cuneiform)     │
└────────────┬────────────────────┘
             │
             ├──► Script 1: analyze_cuneiform_coverage.py
             │    │
             │    ├─► What lexicons? (MesZL best: ~1700 rows)
             │    ├─► What periods? (Ur III, Neo-Assyrian)
             │    └─► What types? (simple, compound, numeric)
             │
             ├──► Script 2: analyze_hantatallas_databases.py
             │    │         (analyzes ~/projects/cuneiform/hantatallas/)
             │    │
             │    ├─► hzl.dat: ~400 signs with encodings
             │    ├─► uga.dat: Ugaritic variant
             │    └─► unicode_cleaned.csv: maps Unicode to lexicons
             │
             └──► Script 3: analyze_encoding_gaps.py
                  │         (cross-reference scripts 1 & 2)
                  │
                  ├─► ~1400 signs missing encodings
                  ├─► Prioritized by type and lexicon availability
                  └─► Categorized for systematic encoding
                       │
                       └──► Script 4: generate_encoding_template.py
                            │
                            └─► missing_encodings.json
                                (template for adding encodings)
```

## Example Output

### Script 1: Lexicon Coverage
```
Coverage by lexicon system:
MesZL          1734 rows ( 91.6%),  1502 unique signs
SL/HA           856 rows ( 45.2%),   743 unique signs
aBZL            620 rows ( 32.8%),   531 unique signs
HethZL          156 rows (  8.2%),   142 unique signs
ModSL          1582 rows ( 83.6%),  1381 unique signs
```

**Insight**: MesZL is the most complete lexicon system, HethZL only covers Hittite signs.

### Script 2: Database Coverage
```
Analyzing hzl.dat...
  Total signs: 452
  Signs with encodings: 437
  Coverage: 96.7%
  Sample encodings: h, h2, L[h2c"v{chc}]
```

**Insight**: HZL database has encodings for ~437 signs, but only maps to ~140 Unicode characters.

### Script 3: Encoding Gaps
```
COVERAGE SUMMARY:
Total unique signs:      1623
Signs with encodings:     142 (8.7%)
Signs missing encodings: 1481 (91.3%)

HIGH PRIORITY (simple signs with lexicon IDs):
452 signs
- 𒐁 U+12401 [MesZL] EŠ 6 (3, AŠ.AŠ.AŠ)
- 𒐀 U+12400 [MesZL, ModSL] AŠ.AŠ (2, DIDLI, MAN 3)
...
```

**Insight**: ~91% of signs need encodings, with clear prioritization for implementation.

### Script 4: Template Structure
```json
{
  "metadata": {
    "total_missing_signs": 1481,
    "encoding_format": {
      "strokes": { "h": "horizontal", "v": "vertical", ... },
      "examples": { "h3": "triple horizontal (numeric 3)" }
    }
  },
  "signs_to_encode": [
    {
      "sign": "𒐁",
      "unicode": "U+12401",
      "primary_id": "4",
      "phonetic": "EŠ 6 (3, AŠ.AŠ.AŠ)",
      "type": "numeric",
      "encoding": "",  // TO BE FILLED
      "status": "pending"
    },
    ...
  ]
}
```

**Insight**: Structured format makes it easy to add encodings systematically.

## Next Steps After Running Scripts

1. **Review Results**: Examine the JSON output files in `results/`
2. **Validate Findings**: Check if the prioritization makes sense for your use case
3. **Plan Implementation**: Use the analysis to guide architectural decisions
4. **Start Encoding**: Use the template to add missing stroke encodings

## Notes

- All scripts are read-only (no data is modified)
- Output is saved to `results/` directory
- Scripts can be run multiple times safely
- Requires `cuneiform_signs.csv` in `data/reference/`
- Requires Hantatallas repo at `~/projects/cuneiform/hantatallas/`
