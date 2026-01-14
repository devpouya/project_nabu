# Automatic Cuneiform Encoding Generation

This module implements an automatic system for generating Hantatallas stroke-based encodings from Unicode cuneiform characters using computer vision techniques.

## Overview

The system uses a multi-stage pipeline to analyze cuneiform glyphs and generate compositional encodings:

```
Unicode Character → Glyph Rendering → Stroke Detection → Spatial Analysis → Encoding Generation
```

## Architecture

### Components

1. **GlyphRenderer** (`glyph_renderer.py`)
   - Renders Unicode cuneiform characters to binary images
   - Uses system cuneiform fonts (NotoSansCuneiform)
   - Outputs: Binary numpy arrays (1 = stroke, 0 = background)

2. **StrokeDetector** (`stroke_detector.py`)
   - Detects individual strokes using Hough line detection
   - Classifies strokes by angle: h, v, u, d, c
   - Merges similar strokes to handle thick lines
   - Outputs: List of Stroke objects with type, position, angle, length

3. **SpatialAnalyzer** (`spatial_analyzer.py`)
   - Implements sweep-line algorithm for spatial relationship detection
   - Detects: horizontal stacking [...], vertical stacking {...}, superposition (...), repetition (h2, h3)
   - Outputs: Spatial groups by composition type

4. **EncodingGenerator** (`encoding_generator.py`)
   - Generates Hantatallas encoding from strokes and spatial structure
   - Applies composition operators based on primary structure
   - Compresses repetitions (e.g., h h h → h3)
   - Outputs: Hantatallas encoding string

5. **HZLValidator** (`hzl_validator.py`)
   - Validates automatic generation against HZL ground truth (~437 signs)
   - Computes accuracy metrics and similarity scores
   - Identifies best/worst cases for algorithm refinement

### Testing & Visualization

1. **EncodingPipelineTester** (`test_pipeline.py`)
   - Comprehensive test framework
   - Runs test suites with expected encodings
   - Generates JSON reports with statistics

2. **EncodingVisualizer** (`visualizer.py`)
   - Creates multi-panel debug visualizations
   - Shows: original glyph, detected strokes, spatial groups, summary
   - Generates comparison grids and legends

3. **QuickTest** (`quick_test.py`)
   - Command-line utility for testing individual signs
   - Usage: `python quick_test.py 𒐁` or `python quick_test.py U+12401`

## Usage

### Basic Usage

```python
from encoding_generator import generate_encoding_from_unicode

# Generate encoding for a sign (on-demand)
encoding = generate_encoding_from_unicode('𒐁')
print(encoding)  # Output: 'h3'
```

### Using Saved Encodings

```python
from encoding_database import EncodingDatabase

# Load pre-generated encodings
db = EncodingDatabase()

# Get encoding for a sign
encoding = db.get_encoding('𒐁')
print(encoding)  # Output: 'h3'

# Check if sign has encoding
if db.has_encoding('𒐁'):
    print("Sign is encoded!")

# Get full entry with metadata
entry = db.get_entry('𒐁')
print(f"Phonetic: {entry.phonetic}")
print(f"ModSL: {entry.modsl}")
print(f"Generated at: {entry.generated_at}")

# Search by phonetic value
results = db.search_by_phonetic("ASH")
for entry in results:
    print(f"{entry.unicode}: {entry.encoding}")
```

### Step-by-Step Pipeline

```python
from glyph_renderer import GlyphRenderer
from stroke_detector import StrokeDetector
from spatial_analyzer import SpatialAnalyzer
from encoding_generator import EncodingGenerator

# Initialize components
renderer = GlyphRenderer(size=128)
detector = StrokeDetector()
analyzer = SpatialAnalyzer()
generator = EncodingGenerator()

# Process a sign
unicode_char = '𒐁'

# 1. Render
binary = renderer.render(unicode_char)

# 2. Detect strokes
strokes = detector.detect_strokes(binary)

# 3. Analyze spatial structure
spatial_groups = analyzer.analyze(strokes)
primary_structure = analyzer.determine_primary_structure(spatial_groups)

# 4. Generate encoding
encoding = generator.generate(strokes)
```

### Generating Encodings for All Signs

```bash
# From project root, generate encodings for all signs in cuneiform_signs.csv
python scripts/generate_all_encodings.py

# Options:
python scripts/generate_all_encodings.py --help

# Test with limited set first (recommended)
python scripts/generate_all_encodings.py --limit 100

# Force regenerate all (ignore existing)
python scripts/generate_all_encodings.py --force

# Custom input/output paths
python scripts/generate_all_encodings.py \
    --input data/reference/cuneiform_signs.csv \
    --output data/encodings/generated_encodings.json
```

This will create `data/encodings/generated_encodings.json` containing:
- All generated encodings
- Metadata (phonetic values, ModSL IDs)
- Generation timestamps and status
- Error information for failed signs

### Testing

```bash
# Test individual sign
python quick_test.py 𒐁

# Test with Unicode code point
python quick_test.py U+12401

# Test predefined list
python quick_test.py --list

# Run full test suite
python test_pipeline.py

# Validate against HZL ground truth
python hzl_validator.py

# Create debug visualizations
python visualizer.py

# Check database statistics
cd src/nabu/encoding
python encoding_database.py
```

## Encoding Format

Hantatallas uses a compositional stroke-based encoding:

### Primitives
- `h` - Horizontal stroke
- `v` - Vertical stroke
- `u` - Upward diagonal (30-60°)
- `d` - Downward diagonal (120-150°)
- `c` - Winkelhaken (angular hook)

### Composition Operators
- `[...]` - Horizontal stacking (left-to-right)
- `{...}` - Vertical stacking (top-to-bottom)
- `(...)` - Superposition (overlapping)

### Modifiers
- `2`, `3`, `4` - Repetition count (e.g., `h3` = three horizontals)
- `'` - Shorten head
- `"` - Shorten tail
- `0` - Null/empty stroke

### Examples
- `h` - Single horizontal (ASH, 𒀸)
- `h3` - Three horizontals (THREE_ASH, 𒐁)
- `[hv]` - Horizontal then vertical side-by-side
- `{hv}` - Horizontal above vertical
- `(h2v)` - Two horizontals superposed with vertical
- `(h2[00v0])` - Complex nested composition (AN, 𒀭)

## Algorithm Details

### Stroke Detection
Uses OpenCV's Probabilistic Hough Transform:
- `rho=1` - Distance resolution
- `theta=π/180` - Angle resolution (1 degree)
- `threshold=30` - Minimum votes
- `minLineLength=10` - Minimum line length
- `maxLineGap=5` - Maximum gap in line

### Stroke Classification
Based on angle (normalized to [0, 180)):
- Horizontal: angle < 20° or angle > 160°
- Vertical: 70° < angle < 110°
- Upward diagonal: 30° ≤ angle ≤ 60°
- Downward diagonal: 120° ≤ angle ≤ 150°
- Winkelhaken: Currently classified as closest match (TODO: improve)

### Spatial Analysis
Sweep-line algorithm with thresholds:
- `ALIGNMENT_THRESHOLD = 15px` - Strokes within this are "aligned"
- `GAP_THRESHOLD = 20px` - Maximum gap for grouping
- `OVERLAP_THRESHOLD = 0.3` - 30% IoU for superposition

**Horizontal stacking**: Left-to-right scan, group if:
- `x_gap > 0` (to the right)
- `x_gap < GAP_THRESHOLD` (not too far)
- `y_diff < ALIGNMENT_THRESHOLD` (vertically aligned)

**Vertical stacking**: Top-to-bottom scan, group if:
- `y_gap > 0` (below)
- `y_gap < GAP_THRESHOLD` (not too far)
- `x_diff < ALIGNMENT_THRESHOLD` (horizontally aligned)

**Superposition**: Overlap detection using IoU > 0.3

**Repetition**: Same stroke type, close together (radius < 40px)

### Primary Structure Detection
Priority order:
1. Superposition (affects everything)
2. Repetition (if no other structure)
3. Vertical stacking
4. Horizontal stacking
5. Simple (single/few strokes)

## Dependencies

```bash
pip install Pillow opencv-python numpy
```

- **Pillow (PIL)**: Glyph rendering with cuneiform fonts
- **OpenCV**: Line detection (HoughLinesP)
- **NumPy**: Image processing

## Output Files

When running the pipeline, the following files are created:

```
project_nabu/
├── data/encodings/
│   └── generated_encodings.json  # MAIN DATABASE: All generated encodings
│
└── results/
    ├── encoding_tests/          # Test pipeline outputs
    │   ├── test_report.json     # JSON test results
    │   └── *_viz.png            # Individual test visualizations
    ├── visualizations/          # Debug visualizations
    │   ├── U+*_debug.png        # Multi-panel debug views
    │   └── legend.png           # Color coding legend
    ├── quick_tests/             # Quick test outputs
    │   └── U+*_*.png            # Individual stroke visualizations
    └── hzl_validation_results.txt  # HZL validation report
```

### Database Format

`generated_encodings.json` structure:
```json
{
  "format_version": "1.0",
  "generated_at": "2025-01-13T12:00:00",
  "total_signs": 1893,
  "successful": 1600,
  "failed": 293,
  "encodings": {
    "U+12401": {
      "unicode": "𒐁",
      "encoding": "h3",
      "phonetic": "EŠ 6 (3, AŠ.AŠ.AŠ)",
      "modsl": "002a",
      "generated_at": "2025-01-13T12:00:00",
      "method": "automatic_cv",
      "status": "generated"
    }
  }
}
```

## Validation

The system is validated against HZL's ~437 Hittite signs with known encodings:

```python
from hzl_validator import HZLValidator

validator = HZLValidator()
results = validator.validate(limit=50)  # Test first 50 signs
validator.print_summary(results)
```

Metrics:
- **Exact matches**: Generated encoding exactly matches HZL
- **Partial matches**: Similarity > 50%
- **Average similarity**: Based on character overlap ratio
- **Failures**: Unable to generate encoding

## Troubleshooting

### No strokes detected
- Check if cuneiform font is installed correctly
- Verify glyph renders (check rendered_glyphs/ output)
- Try adjusting Hough transform thresholds in StrokeDetector

### Wrong stroke classification
- Review stroke angle thresholds in `_classify_stroke_type()`
- Check visualizations to see detected angles
- May need font-specific tuning

### Incorrect spatial grouping
- Adjust spatial analyzer thresholds (ALIGNMENT_THRESHOLD, GAP_THRESHOLD)
- Check overlap threshold for superposition
- Review visualization to see how strokes are grouped

### Missing winkelhaken detection
- Current implementation classifies by angle only
- TODO: Implement dedicated winkelhaken detector using corner detection

## Future Improvements

1. **Winkelhaken Detection**: Use corner/angle detection instead of line detection
2. **Nested Structures**: Better handling of complex nested compositions
3. **Modifier Detection**: Automatic detection of shortened strokes (', ")
4. **Font Adaptation**: Auto-tune thresholds for different fonts
5. **Ligature Decomposition**: Use Colligation field to decompose compound signs
6. **Full Coverage**: Extend to all ~1600 signs in cuneiform_signs.csv

## References

- **Hantatallas**: [https://github.com/jcuenod/hantatallas](https://github.com/jcuenod/hantatallas)
- **HZL (Hethitisches Zeichenlexikon)**: Rüster & Neu's Hittite sign catalog
- **ModSL**: Modern Sign List - universal cuneiform sign identifier
- **OpenCV Hough Transform**: [https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)
