# Automatic Encoding Generation - Complete Workflow

This document explains the complete workflow for generating and using automatic cuneiform encodings.

## Quick Start

### 1. Install Dependencies

```bash
pip install Pillow opencv-python numpy
```

### 2. Test the System

Test with the previously problematic sign (𒐁 THREE_ASH):

```bash
cd src/nabu/encoding
python quick_test.py 𒐁
```

Expected output: `h3` (three horizontal strokes)

### 3. Generate All Encodings

Generate encodings for all signs in your database:

```bash
# Start with a small test batch
python scripts/generate_all_encodings.py --limit 100

# Review results, then generate all
python scripts/generate_all_encodings.py
```

This creates: `data/encodings/generated_encodings.json`

### 4. Query Encodings

```bash
# Query by Unicode character
python scripts/query_encoding.py 𒐁

# Query by code point
python scripts/query_encoding.py U+12401

# Search by phonetic value
python scripts/query_encoding.py --phonetic ASH

# View statistics
python scripts/query_encoding.py --stats
```

## Detailed Workflow

### Phase 1: Understanding the Pipeline

The system uses computer vision to automatically generate encodings:

```
Unicode → Render Glyph → Detect Strokes → Analyze Spatial Structure → Generate Encoding
   𒐁         [image]         h h h            repetition(3)              h3
```

**Components:**
1. **GlyphRenderer** - Renders Unicode using cuneiform fonts
2. **StrokeDetector** - Detects lines using Hough transform
3. **SpatialAnalyzer** - Sweep-line algorithm for spatial relationships
4. **EncodingGenerator** - Generates compositional encoding

### Phase 2: Testing & Validation

Before generating all encodings, validate the system:

#### Test Individual Signs

```bash
cd src/nabu/encoding

# Test simple signs
python quick_test.py 𒀸  # ASH - single horizontal
python quick_test.py 𒁹  # DIŠ - single vertical

# Test repetitions
python quick_test.py 𒐀  # TWO_ASH - h2
python quick_test.py 𒐁  # THREE_ASH - h3
python quick_test.py 𒐂  # FOUR_ASH - h4

# Test complex signs
python quick_test.py 𒀭  # AN - complex composition
```

#### Test Multiple Signs

```bash
python quick_test.py --list
```

#### Validate Against HZL Ground Truth

HZL has ~437 Hittite signs with known correct encodings:

```bash
python hzl_validator.py
```

Review:
- **Exact match rate**: How many exactly match HZL?
- **Similarity scores**: Are close matches reasonable?
- **Worst cases**: Which signs fail? Why?

#### Create Visualizations

```bash
python visualizer.py
```

Opens: `results/visualizations/` with multi-panel debug views showing:
- Original glyph
- Detected strokes with classifications
- Spatial groupings
- Generated encoding

### Phase 3: Generating All Encodings

Once validated, generate encodings for all signs:

#### Small Test Batch First

```bash
python scripts/generate_all_encodings.py --limit 100
```

Review output:
- How many succeeded?
- What's the failure rate?
- Are errors consistent?

#### Generate All Encodings

```bash
python scripts/generate_all_encodings.py
```

This will:
- Process all ~1893 signs in `cuneiform_signs.csv`
- Generate encodings using computer vision
- Save to `data/encodings/generated_encodings.json`
- Report statistics

Expected results:
- **Success rate**: ~70-90% (depends on font quality, sign complexity)
- **Time**: ~5-15 minutes for all signs
- **Output size**: ~500KB JSON file

#### Incremental Updates

The system skips already-encoded signs by default:

```bash
# Add new signs without re-processing existing
python scripts/generate_all_encodings.py

# Force regenerate all
python scripts/generate_all_encodings.py --force
```

### Phase 4: Using Generated Encodings

#### In Python Code

```python
from nabu.encoding.encoding_database import EncodingDatabase

# Load database
db = EncodingDatabase()

# Get encoding
encoding = db.get_encoding('𒐁')
print(encoding)  # 'h3'

# Check if available
if db.has_encoding('𒐁'):
    print("Encoded!")

# Get full metadata
entry = db.get_entry('𒐁')
print(f"Phonetic: {entry.phonetic}")
print(f"ModSL: {entry.modsl}")
print(f"Status: {entry.status}")

# Search by phonetic
for entry in db.search_by_phonetic("ASH"):
    print(f"{entry.unicode}: {entry.encoding}")

# Get statistics
stats = db.get_statistics()
print(f"Coverage: {stats['coverage']*100:.1f}%")
```

#### From Command Line

```bash
# Query individual signs
python scripts/query_encoding.py 𒐁
python scripts/query_encoding.py U+12401

# Search
python scripts/query_encoding.py --phonetic ASH
python scripts/query_encoding.py --modsl 002a

# Statistics
python scripts/query_encoding.py --stats
```

### Phase 5: Quality Control & Refinement

#### Review Failed Signs

```python
from nabu.encoding.encoding_database import EncodingDatabase

db = EncodingDatabase()
failed = db.get_all_failed()

print(f"Failed: {len(failed)} signs")
for entry in failed[:10]:
    print(f"{entry.unicode} ({entry.code_point}): {entry.error}")
```

#### Common Failure Patterns

1. **No strokes detected**
   - Font doesn't render properly
   - Glyph too small/faint
   - Solution: Try different font, adjust rendering size

2. **Wrong stroke classification**
   - Angles at threshold boundaries
   - Solution: Adjust angle thresholds in `stroke_detector.py`

3. **Incorrect spatial grouping**
   - Strokes too close/far
   - Solution: Tune `ALIGNMENT_THRESHOLD`, `GAP_THRESHOLD` in `spatial_analyzer.py`

4. **Complex nested structures**
   - System picks wrong primary structure
   - Solution: May need manual encoding or improved heuristics

#### Iterative Improvement

1. **Identify failure patterns** from validation results
2. **Adjust thresholds** in detector/analyzer
3. **Regenerate** affected signs: `python scripts/generate_all_encodings.py --force`
4. **Validate** improvement: `python hzl_validator.py`
5. **Repeat** until satisfied with coverage

### Phase 6: Integration with PaleoCodage

Once you have good coverage, integrate with your main tokenizer:

```python
# In src/nabu/tokenizers/hantatallas.py

from nabu.encoding.encoding_database import EncodingDatabase

class HantatalasConverter:
    def __init__(self):
        # Load generated encodings
        self.encoding_db = EncodingDatabase()

        # Fallback to HZL for Hittite signs
        self.load_hzl_mappings()

    def unicode_to_hantatallas(self, unicode_char: str) -> Optional[str]:
        """Convert Unicode to Hantatallas encoding."""

        # Try generated encoding first
        encoding = self.encoding_db.get_encoding(unicode_char)
        if encoding:
            return encoding

        # Fallback to HZL
        return self.hzl_mappings.get(unicode_char)
```

## Troubleshooting

### Problem: No encodings generated

**Check:**
1. Are dependencies installed? `pip list | grep -E "Pillow|opencv|numpy"`
2. Is cuneiform font available? Test with `python src/nabu/encoding/glyph_renderer.py`
3. Are glyphs rendering? Check `results/rendered_glyphs/`

### Problem: Low accuracy on validation

**Check:**
1. What's the failure pattern? Run `python src/nabu/encoding/hzl_validator.py`
2. Review worst cases - are they genuinely hard signs?
3. Check visualizations - are strokes detected correctly?

**Tune:**
- Stroke angle thresholds in `stroke_detector.py`
- Spatial grouping thresholds in `spatial_analyzer.py`

### Problem: Specific sign fails

**Debug:**
```bash
# Create detailed visualization
python src/nabu/encoding/quick_test.py <sign>

# Check outputs in results/quick_tests/
```

Look at:
- Are all strokes detected?
- Are angles classified correctly?
- Is spatial grouping wrong?

## Expected Results

### Coverage

Based on sign complexity:
- **Simple signs** (single/few strokes): ~95% accuracy
- **Repetitions** (h2, h3): ~90% accuracy
- **Complex compositions**: ~70% accuracy
- **Overall**: ~80-85% coverage

### Common Gaps

1. **Winkelhaken (hooks)**: Current system classifies by angle, may miss angular features
2. **Very complex nested signs**: May pick wrong primary structure
3. **Rare variant forms**: May not match standard HZL encodings
4. **Damaged/partial signs**: No support for damage markers

These gaps can be filled with:
- Manual encodings (add to JSON)
- Improved detection algorithms
- Compound sign decomposition

## Summary

**Workflow:**
1. ✅ Test individual signs → Validate pipeline works
2. ✅ Validate against HZL → Measure accuracy
3. ✅ Generate all encodings → Create database
4. ✅ Query and use → Integrate with your code
5. ✅ Refine and improve → Iterative quality improvement

**Key Files:**
- `data/encodings/generated_encodings.json` - Main encoding database
- `scripts/generate_all_encodings.py` - Generation script
- `scripts/query_encoding.py` - Query tool
- `src/nabu/encoding/encoding_database.py` - Python API

**Next Steps:**
1. Run validation: `python src/nabu/encoding/hzl_validator.py`
2. Generate test batch: `python scripts/generate_all_encodings.py --limit 100`
3. Review results and tune if needed
4. Generate all: `python scripts/generate_all_encodings.py`
5. Integrate with PaleoCodage tokenizer
