---
layout: default
title: API Reference
---

# API Reference

This section provides documentation for the main modules in Project Nabu.

## Modules Overview

### Tokenizers (`nabu.tokenizers`)

| Class | Description |
|-------|-------------|
| [`SignTokenizer`](#signtokenizer) | Sign-level tokenization |
| [`StrokeTokenizer`](#stroketokenizer) | Stroke-level tokenization |
| [`HybridTokenizer`](#hybridtokenizer) | Combined sign + stroke tokenization |

### Models (`nabu.models`)

| Class | Description |
|-------|-------------|
| [`TransformerEncoder`](#transformerencoder) | BERT-style bidirectional encoder |
| [`TransformerDecoder`](#transformerdecoder) | GPT-style autoregressive decoder |

### Datasets (`nabu.datasets`)

| Class | Description |
|-------|-------------|
| [`OraccDataset`](#oraccdataset) | ORACC corpus loader |
| [`CuneiMLDataset`](#cuneimldataset) | CuneiML corpus loader |
| [`CuneiformLineDataset`](#cuneiformlinedataset) | Line detection training data |

### Encoding (`nabu.encoding`)

| Class | Description |
|-------|-------------|
| [`GlyphRenderer`](#glyphrenderer) | Render Unicode to images |
| [`StrokeDetector`](#strokedetector) | Detect strokes in glyphs |
| [`SpatialAnalyzer`](#spatialanalyzer) | Analyze spatial relationships |
| [`EncodingGenerator`](#encodinggenerator) | Generate PaleoCode encodings |

### Detection (`nabu.detection`)

| Class | Description |
|-------|-------------|
| [`LineNet`](#linenet) | Line classification CNN |
| [`LineNetFCN`](#linenetfcn) | Fully convolutional version |
| [`LineDetector`](#linedetector) | High-level detection interface |

---

## Tokenizers

### SignTokenizer

Tokenizes cuneiform text at the sign level.

```python
from nabu.tokenizers import SignTokenizer

tokenizer = SignTokenizer()
tokenizer.build_vocab(texts)

# Encode
ids = tokenizer.encode("𒀀𒈾𒆠")

# Decode
text = tokenizer.decode(ids)

# Batch operations
batch_ids = tokenizer.batch_encode(["𒀀𒈾", "𒆠𒀀"])
batch_texts = tokenizer.batch_decode(batch_ids)
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `build_vocab` | `texts: List[str]` | `None` | Build vocabulary from corpus |
| `encode` | `text: str` | `List[int]` | Convert text to token IDs |
| `decode` | `ids: List[int]` | `str` | Convert IDs to text |
| `batch_encode` | `texts: List[str]` | `List[List[int]]` | Batch encoding |
| `batch_decode` | `batch_ids` | `List[str]` | Batch decoding |

### StrokeTokenizer

Tokenizes at the stroke level for fine-grained analysis.

```python
from nabu.tokenizers import StrokeTokenizer

tokenizer = StrokeTokenizer()
tokenizer.build_vocab(texts)

ids = tokenizer.encode("𒀀")  # Returns stroke-level tokens
```

### HybridTokenizer

Provides both sign and stroke representations with alignment.

```python
from nabu.tokenizers import HybridTokenizer

tokenizer = HybridTokenizer()
result = tokenizer.encode("𒀀𒈾")

print(result.sign_ids)      # Sign-level tokens
print(result.stroke_ids)    # Stroke-level tokens
print(result.alignment)     # Sign-to-stroke mapping
```

---

## Models

### TransformerEncoder

BERT-style bidirectional transformer for classification and sequence labeling.

```python
from nabu.models import TransformerEncoder

model = TransformerEncoder(
    vocab_size=5000,
    embedding_dim=512,
    num_layers=6,
    num_heads=8,
    feedforward_dim=2048,
    dropout=0.1,
    max_len=512
)

# Forward pass
output = model(input_ids, attention_mask=mask)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | - | Size of vocabulary |
| `embedding_dim` | int | 512 | Hidden dimension |
| `num_layers` | int | 6 | Number of transformer layers |
| `num_heads` | int | 8 | Number of attention heads |
| `feedforward_dim` | int | 2048 | FFN intermediate dimension |
| `dropout` | float | 0.1 | Dropout probability |
| `max_len` | int | 5000 | Maximum sequence length |

### TransformerDecoder

GPT-style autoregressive transformer for text generation.

```python
from nabu.models import TransformerDecoder

model = TransformerDecoder(
    vocab_size=5000,
    embedding_dim=512,
    num_layers=6,
    num_heads=8
)

# Forward pass (with causal masking)
output = model(input_ids)
```

---

## Datasets

### OraccDataset

Load and process ORACC (Open Richly Annotated Cuneiform Corpus) data.

```python
from nabu.datasets import OraccDataset

dataset = OraccDataset(
    json_path="data/oracc_pubs.json",
    mode="lines",           # 'lines', 'paragraphs', 'full_text'
    language="akk",         # Filter by language
    min_length=5,           # Minimum text length
    max_length=512          # Maximum text length
)

# Iterate
for text, metadata in dataset:
    print(text, metadata['language'])
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `json_path` | str | - | Path to ORACC JSON file |
| `mode` | str | "lines" | Extraction mode |
| `language` | str | None | Filter by language code |
| `genre` | str | None | Filter by genre |
| `min_length` | int | 1 | Minimum text length |
| `max_length` | int | None | Maximum text length |

### CuneiformLineDataset

Dataset for training line segmentation models.

```python
from nabu.datasets import CuneiformLineDataset

dataset = CuneiformLineDataset(
    images_dir="data/images",
    annotations_path="data/annotations.csv",
    transform=train_transform,
    line_height=32,
    sample_radius=96
)

# Get sample
image, label = dataset[0]  # label: 0=background, 1=line
```

---

## Encoding

### GlyphRenderer

Render Unicode cuneiform to binary images.

```python
from nabu.encoding import GlyphRenderer

renderer = GlyphRenderer(font_size=200)
image = renderer.render("𒀀")
# Returns: numpy array (H, W), binary
```

### StrokeDetector

Detect strokes using Hough transform.

```python
from nabu.encoding import StrokeDetector

detector = StrokeDetector(
    hough_threshold=50,
    min_line_length=10,
    angle_tolerance=15
)

strokes = detector.detect(binary_image)
# Returns: List[Stroke] with type, angle, endpoints
```

### SpatialAnalyzer

Analyze spatial relationships between strokes.

```python
from nabu.encoding import SpatialAnalyzer

analyzer = SpatialAnalyzer(
    horizontal_threshold=0.3,
    vertical_threshold=0.3
)

groups = analyzer.analyze(strokes, image_shape)
# Returns: dict with 'horizontal', 'vertical', 'superposed' groups
```

### EncodingGenerator

Generate PaleoCode from strokes and spatial analysis.

```python
from nabu.encoding import EncodingGenerator

generator = EncodingGenerator()
paleocode = generator.generate(strokes, spatial_groups)
# Returns: str like "h-v:h-v"
```

---

## Detection

### LineNet

CNN classifier for line/non-line pixel classification.

```python
from nabu.detection import LineNet

model = LineNet(num_classes=2, input_channels=1)
output = model(input_tensor)  # (B, 2) logits
```

### LineNetFCN

Fully convolutional version for dense prediction.

```python
from nabu.detection import LineNetFCN

# Convert from trained classifier
model_fcn = LineNetFCN(trained_linenet, num_classes=2)

# Dense prediction
output = model_fcn(image)  # (B, 2, H', W') probabilities
```

### LineDetector

High-level interface for line detection.

```python
from nabu.detection import LineDetector

detector = LineDetector(
    model_path="weights/linenet_fcn.pth",
    device="cuda",
    confidence_threshold=0.7
)

result = detector.detect(
    image,
    scale=1.0,
    num_lines=15,
    use_hough=True
)

# Access results
print(result.line_hypos)        # DataFrame of line hypotheses
print(result.line_segs)         # List of line segments
print(result.segm_labels)       # Pixel-wise labels
print(result.dist_interline_median)  # Interline distance
```

---

## Utilities

### ExperimentTracker

Unified experiment tracking for TensorBoard and W&B.

```python
from nabu.utils import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="line_segmentation",
    run_name="baseline_v1",
    backend="wandb",  # or "tensorboard", "all"
    config={"epochs": 100, "batch_size": 128}
)

# Log metrics
tracker.log_metrics({"train/loss": 0.5}, step=epoch)

# Log model
tracker.log_model("model.pth", name="best_model")

# Finish
tracker.finish()
```

### PaleoCodeConverter

Convert between Unicode, PaleoCode, and strokes.

```python
from nabu.paleocode import PaleoCodeConverter

converter = PaleoCodeConverter()

# Unicode to PaleoCode
pc = converter.unicode_to_paleocode("𒀀")

# PaleoCode to strokes
strokes = converter.paleocode_to_strokes("h-v:h-v")

# Unicode to strokes (direct)
strokes = converter.unicode_to_strokes("𒀀")
```
