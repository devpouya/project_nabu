---
layout: default
title: Encoding Pipeline
---

# Automatic Encoding Generation Pipeline

One of the key innovations in Project Nabu is the automatic generation of stroke-based encodings (PaleoCode/Hantatallas) from Unicode cuneiform characters using computer vision techniques.

## Overview

The pipeline transforms Unicode cuneiform characters into structured stroke encodings through visual analysis:

```
Unicode Character → Rendered Glyph → Stroke Detection → Spatial Analysis → PaleoCode
       𒀀         →   [image]      →   [h,v,h,v]     →   [[h,v],[h,v]]  →  h-v:h-v
```

## Why Automatic Generation?

- **Coverage**: ~1,893 Unicode cuneiform signs exist, manual encoding is laborious
- **Consistency**: Algorithmic approach ensures systematic treatment
- **Validation**: Can be verified against existing lexicons (HZL)
- **Extensibility**: Easily applied to new fonts or sign variants

## Pipeline Components

### 1. Glyph Renderer

Renders Unicode cuneiform characters to binary images using the NotoSansCuneiform font.

```python
from nabu.encoding import GlyphRenderer

renderer = GlyphRenderer(font_size=200)
image = renderer.render("𒀀")
# Returns: Binary numpy array (1=stroke, 0=background)
```

**Parameters:**
- `font_size`: Rendering size (larger = more detail, default: 200)
- `font_path`: Path to .ttf font file (default: NotoSansCuneiform)

**Output:**
- Binary image where white pixels (1) represent stroke areas
- Automatically cropped to glyph bounding box

### 2. Stroke Detector

Detects individual strokes using Hough line transform and classifies them by angle.

```python
from nabu.encoding import StrokeDetector

detector = StrokeDetector()
strokes = detector.detect(binary_image)
# Returns: List of Stroke objects with type, angle, position
```

**Detection Process:**

1. **Skeletonization**: Reduce stroke regions to single-pixel lines
2. **Hough Transform**: Detect line segments
3. **Angle Classification**: Assign stroke type based on angle

**Stroke Classification:**

| Angle Range | Stroke Type | Symbol |
|-------------|-------------|--------|
| 0° - 30° | Horizontal | `h` |
| 30° - 60° | Upward Diagonal | `u` |
| 60° - 90° | Vertical | `v` |
| 120° - 150° | Downward Diagonal | `d` |
| Corner detection | Winkelhaken | `c` |

**Stroke Merging:**
Similar strokes (close in position and angle) are merged to handle thick lines that produce multiple detections.

### 3. Spatial Analyzer

Analyzes spatial relationships between detected strokes using a sweep-line algorithm.

```python
from nabu.encoding import SpatialAnalyzer

analyzer = SpatialAnalyzer()
groups = analyzer.analyze(strokes, image_shape)
# Returns: Hierarchical groupings by composition type
```

**Detected Relationships:**

| Relationship | Encoding | Detection Method |
|--------------|----------|------------------|
| Horizontal stacking | `[a-b]` | Strokes aligned vertically, ordered left-to-right |
| Vertical stacking | `{a-b}` | Strokes aligned horizontally, ordered top-to-bottom |
| Superposition | `(a-b)` | Overlapping bounding boxes |
| Repetition | `h3` | Three identical strokes in sequence |

**Algorithm:**
1. Sort strokes by x-coordinate (sweep line)
2. Group strokes by vertical alignment (horizontal stacking candidates)
3. Sort groups by y-coordinate
4. Identify vertical stacking between groups
5. Detect overlapping elements for superposition

### 4. Encoding Generator

Converts detected strokes and spatial relationships into PaleoCode notation.

```python
from nabu.encoding import EncodingGenerator

generator = EncodingGenerator()
paleocode = generator.generate(strokes, spatial_groups)
# Returns: String like "h-v:h-v"
```

**Encoding Rules:**

1. **Stroke Symbols**: Use primitives (h, v, u, d, c)
2. **Horizontal Groups**: Join with `-` (e.g., `h-v`)
3. **Vertical Groups**: Join with `:` (e.g., `h-v:h-v`)
4. **Superposition**: Use parentheses (e.g., `(h-v)`)
5. **Repetition**: Use count suffix (e.g., `h3` for three horizontals)

### 5. Encoding Database

Stores and queries generated encodings.

```python
from nabu.encoding import EncodingDatabase

db = EncodingDatabase("data/encodings/generated_encodings.json")

# Query by character
encoding = db.get_encoding("𒀀")

# Search by phonetic value
results = db.search_by_phonetic("an")

# Get statistics
stats = db.get_coverage_stats()
```

**Database Schema:**
```json
{
  "𒀀": {
    "encoding": "h-v:h-v",
    "phonetic": "an",
    "modsl_id": "AN",
    "generated_at": "2025-01-22T10:30:00",
    "status": "validated",
    "confidence": 0.95
  }
}
```

## Full Pipeline Usage

### Command Line

```bash
# Generate encodings for all Unicode cuneiform signs
python scripts/generate_all_encodings.py

# Generate with options
python scripts/generate_all_encodings.py \
    --output data/encodings/my_encodings.json \
    --font-size 300 \
    --skip-existing

# Test single character
python src/nabu/encoding/quick_test.py --char 𒀀
```

### Python API

```python
from nabu.encoding import (
    GlyphRenderer,
    StrokeDetector,
    SpatialAnalyzer,
    EncodingGenerator
)

def generate_encoding(char: str) -> str:
    # Step 1: Render glyph
    renderer = GlyphRenderer()
    image = renderer.render(char)

    # Step 2: Detect strokes
    detector = StrokeDetector()
    strokes = detector.detect(image)

    # Step 3: Analyze spatial relationships
    analyzer = SpatialAnalyzer()
    groups = analyzer.analyze(strokes, image.shape)

    # Step 4: Generate encoding
    generator = EncodingGenerator()
    encoding = generator.generate(strokes, groups)

    return encoding

# Example
encoding = generate_encoding("𒀀")
print(f"Encoding: {encoding}")  # h-v:h-v
```

## Validation

### HZL Ground Truth

The pipeline is validated against the Hethitisches Zeichenlexikon (HZL) which contains manually created encodings for ~437 Hittite cuneiform signs.

```python
from nabu.encoding import HZLValidator

validator = HZLValidator("data/reference/hzl_encodings.json")
results = validator.validate(generated_encodings)

print(f"Exact match: {results['exact_match_rate']:.1%}")
print(f"Similarity: {results['avg_similarity']:.1%}")
```

**Validation Metrics:**

| Metric | Description |
|--------|-------------|
| Exact Match Rate | Percentage of perfect matches |
| Similarity Score | Edit distance normalized by length |
| Stroke Accuracy | Correct stroke type identification |
| Composition Accuracy | Correct spatial relationship detection |

### Current Performance

| Metric | Value |
|--------|-------|
| Coverage | ~1,600 / 1,893 signs (85%) |
| Exact Match (HZL) | ~65% |
| High Similarity (>0.8) | ~82% |

## Visualization

Debug visualizations for pipeline analysis:

```python
from nabu.encoding import Visualizer

viz = Visualizer()

# Multi-panel debug view
viz.plot_pipeline(
    char="𒀀",
    save_path="debug_output.png"
)
```

**Visualization Panels:**
1. Original rendered glyph
2. Detected stroke lines overlaid
3. Spatial groupings highlighted
4. Final encoding with breakdown

## Configuration

### Tunable Parameters

```python
# Stroke detection
detector = StrokeDetector(
    hough_threshold=50,      # Minimum votes for line detection
    min_line_length=10,      # Minimum stroke length (pixels)
    max_line_gap=5,          # Maximum gap in line segments
    angle_tolerance=15,      # Degrees tolerance for classification
)

# Spatial analysis
analyzer = SpatialAnalyzer(
    horizontal_threshold=0.3,  # Vertical alignment tolerance
    vertical_threshold=0.3,    # Horizontal alignment tolerance
    overlap_threshold=0.5,     # IOU threshold for superposition
)
```

### Font Considerations

Different cuneiform fonts may require parameter tuning:

| Font | Recommended Settings |
|------|---------------------|
| NotoSansCuneiform | Default parameters |
| CuneiformComposite | Increase `font_size` to 300 |
| Assurbanipal | Reduce `angle_tolerance` to 10 |

## Limitations & Future Work

### Current Limitations

1. **Winkelhaken Detection**: Corner marks are challenging; currently uses heuristics
2. **Complex Compositions**: Deeply nested structures may not be fully captured
3. **Font Dependency**: Results vary with different fonts
4. **Ligatures**: Connected signs not automatically decomposed

### Planned Improvements

- [ ] Corner detection using Harris/Shi-Tomasi for winkelhaken
- [ ] Deep learning stroke classifier for robustness
- [ ] Multi-font ensemble for consistency
- [ ] Ligature decomposition using colligation data
- [ ] Confidence scoring for uncertain encodings

## Troubleshooting

### Common Issues

**No strokes detected:**
```python
# Increase font size for better resolution
renderer = GlyphRenderer(font_size=400)

# Lower Hough threshold
detector = StrokeDetector(hough_threshold=30)
```

**Too many strokes detected:**
```python
# Increase minimum line length
detector = StrokeDetector(min_line_length=20)

# Enable aggressive merging
detector = StrokeDetector(merge_similar=True, merge_threshold=10)
```

**Wrong spatial relationships:**
```python
# Adjust alignment thresholds
analyzer = SpatialAnalyzer(
    horizontal_threshold=0.4,  # More lenient
    vertical_threshold=0.4
)
```

## References

- [Hantatallas Encoding System](https://www.hethport.uni-wuerzburg.de/hantatallas/)
- [PaleoCode Documentation](https://github.com/situx/PaleoCodage)
- [HZL - Hethitisches Zeichenlexikon](https://www.harrassowitz-verlag.de/)
