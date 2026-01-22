---
layout: default
title: Project Nabu
---

# Project Nabu

**A Deep Learning Framework for Cuneiform Text Analysis**

Project Nabu is an advanced NLP framework for processing and analyzing cuneiform text—the world's oldest writing system used by ancient Sumerian and Akkadian civilizations. Unlike traditional approaches that work with transliterated text, Nabu operates directly on Unicode cuneiform characters, preserving the visual and structural information of the original writing system.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Cuneiform_script2.svg/400px-Cuneiform_script2.svg.png" alt="Cuneiform Script" width="300"/>
</p>

## Key Innovation: Stroke-Based Encoding

The project recognizes that cuneiform signs have **compositional structure**—each sign can be broken down into fundamental strokes and spatial relationships. This is similar to how Chinese characters are composed of radicals.

```
𒀀 (AN - sky/god) → PaleoCode: h-v:h-v
    ↓
Strokes: horizontal, vertical, horizontal, vertical
Composition: two horizontal-vertical pairs stacked
```

### The Five Stroke Primitives

| Stroke | Symbol | Description |
|--------|--------|-------------|
| Horizontal | `h` | Horizontal wedge (0-30°) |
| Vertical | `v` | Vertical wedge (60-90°) |
| Upward Diagonal | `u` | Upward diagonal (30-60°) |
| Downward Diagonal | `d` | Downward diagonal (120-150°) |
| Winkelhaken | `c` | Angular hook (corner mark) |

### Composition Operators

| Operator | Syntax | Meaning |
|----------|--------|---------|
| Horizontal Stack | `[a-b]` | Left-to-right arrangement |
| Vertical Stack | `{a-b}` | Top-to-bottom arrangement |
| Superposition | `(a-b)` | Overlapping elements |
| Repetition | `h3` | Three horizontal strokes |

## Features

### Three Tokenization Strategies

1. **Sign Tokenizer** - Each complete cuneiform sign as a token
2. **Stroke Tokenizer** - Individual strokes as tokens (finest granularity)
3. **Hybrid Tokenizer** - Both sign and stroke representations

### Automatic Encoding Generation

Computer vision pipeline that generates stroke encodings from Unicode characters:

```
Unicode Character → Rendered Glyph → Stroke Detection → Spatial Analysis → PaleoCode
```

### Line Detection for Tablet Images

CNN-based line segmentation with Hough transform post-processing for detecting text lines on cuneiform tablets.

### Transformer Models

BERT-style encoder and GPT-style decoder architectures adapted for cuneiform text.

## Quick Example

```python
from nabu.tokenizers import SignTokenizer
from nabu.paleocode import PaleoCodeConverter

# Initialize
converter = PaleoCodeConverter()
tokenizer = SignTokenizer()

# Convert Unicode cuneiform to strokes
text = "𒀀𒈾𒆠"  # "a-na ki" (to the place)
paleocodes = [converter.unicode_to_paleocode(ch) for ch in text]
# ['h-v:h-v', 'h-h:v', 'h-h-h:v-v']

# Tokenize for model input
tokens = tokenizer.encode(text)
```

## Project Structure

```
project_nabu/
├── src/nabu/
│   ├── tokenizers/      # Sign, Stroke, Hybrid tokenizers
│   ├── models/          # Transformer encoder/decoder
│   ├── datasets/        # ORACC, CuneiML, custom datasets
│   ├── encoding/        # Automatic encoding generation
│   ├── detection/       # Line detection from images
│   └── paleocode/       # PaleoCode conversion utilities
├── scripts/             # Training, evaluation, analysis
├── configs/             # YAML experiment configs
└── docs/                # This documentation
```

## Supported Datasets

| Dataset | Description | Format |
|---------|-------------|--------|
| **ORACC** | Open Richly Annotated Cuneiform Corpus | JSON/ATF |
| **CuneiML** | Machine Learning cuneiform dataset | JSON + Images |
| **CDLI** | Cuneiform Digital Library Initiative | Various |

## Research Goals

1. **Establish baselines** for cuneiform NLP tasks
2. **Compare tokenization approaches**: sign vs. stroke vs. hybrid
3. **Enable image-to-text**: OCR pipeline for tablet photographs
4. **Advance ancient language NLP**: Novel techniques for low-resource historical languages

## Documentation

- [Getting Started](getting-started) - Installation and first steps
- [Architecture](architecture) - System design and components
- [Encoding Pipeline](encoding-pipeline) - Automatic encoding generation
- [Line Detection](line-detection) - Tablet image processing
- [API Reference](api/) - Module documentation

## Citation

If you use Project Nabu in your research, please cite:

```bibtex
@software{project_nabu,
  title = {Project Nabu: A Deep Learning Framework for Cuneiform Text Analysis},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/project_nabu}
}
```

## License

MIT License - See [LICENSE](https://github.com/YOUR_USERNAME/project_nabu/blob/main/LICENSE) for details.

---

<p align="center">
  <em>Named after Nabu, the ancient Mesopotamian god of literacy, wisdom, and scribes.</em>
</p>
