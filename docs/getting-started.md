---
layout: default
title: Getting Started
---

# Getting Started

This guide will help you set up Project Nabu and run your first experiments.

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/project_nabu.git
cd project_nabu
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "from nabu.tokenizers import SignTokenizer; print('Installation successful!')"
```

## Quick Start

### Working with Cuneiform Text

```python
from nabu.tokenizers import SignTokenizer, StrokeTokenizer
from nabu.paleocode import PaleoCodeConverter

# Initialize the PaleoCode converter
converter = PaleoCodeConverter()

# Example: Convert Unicode cuneiform to PaleoCode
sign = "𒀀"  # AN sign (sky/god)
paleocode = converter.unicode_to_paleocode(sign)
print(f"Unicode: {sign} → PaleoCode: {paleocode}")
# Output: Unicode: 𒀀 → PaleoCode: h-v:h-v

# Get individual strokes
strokes = converter.paleocode_to_strokes(paleocode)
print(f"Strokes: {strokes}")
# Output: Strokes: ['h', 'v', 'h', 'v']
```

### Tokenizing Text

```python
# Sign-level tokenization (each sign = one token)
sign_tokenizer = SignTokenizer()
sign_tokenizer.build_vocab(texts)  # Build from your corpus
tokens = sign_tokenizer.encode("𒀀𒈾𒆠")

# Stroke-level tokenization (each stroke = one token)
stroke_tokenizer = StrokeTokenizer()
stroke_tokenizer.build_vocab(texts)
tokens = stroke_tokenizer.encode("𒀀𒈾𒆠")
```

### Loading Datasets

```python
from nabu.datasets import OraccDataset

# Load ORACC corpus
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    mode="lines",
    min_length=5,
    language="akk"  # Akkadian
)

print(f"Loaded {len(dataset)} samples")

# Get a sample
text, metadata = dataset[0]
print(f"Text: {text}")
print(f"Language: {metadata['language']}")
```

## Training a Model

### 1. Prepare Configuration

Create or modify a config file in `configs/experiments/`:

```yaml
# configs/experiments/my_experiment.yaml
tokenizer:
  type: sign
  max_length: 512

model:
  type: transformer_encoder
  hidden_size: 256
  num_layers: 4
  num_heads: 8
  dropout: 0.1

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.0001
  device: cuda
```

### 2. Run Training

```bash
python scripts/train.py --config configs/experiments/my_experiment.yaml
```

### 3. Monitor with TensorBoard

```bash
tensorboard --logdir outputs/runs
```

## Line Detection on Tablet Images

### Generate Synthetic Annotations

```bash
python scripts/train_line_segmentation.py --generate-annotations
```

### Train Line Detection Model

```bash
python scripts/train_line_segmentation.py \
    --train \
    --epochs 50 \
    --batch-size 64 \
    --tracker tensorboard
```

### Run Detection on Images

```bash
python scripts/run_line_segmentation.py \
    --model-path outputs/line_training/weights/linenet_fcn.pth \
    --image data/external/cuneiml/images/P010054.jpg
```

## Automatic Encoding Generation

Generate stroke encodings for Unicode cuneiform characters:

```bash
# Generate encodings for all Unicode cuneiform signs
python scripts/generate_all_encodings.py

# Query a specific encoding
python scripts/query_encoding.py --char 𒀀

# Test the pipeline on a single sign
python src/nabu/encoding/quick_test.py --char 𒀀
```

## Project Structure

```
project_nabu/
├── configs/              # YAML configuration files
│   ├── default.yaml
│   └── experiments/
├── data/
│   ├── external/         # Downloaded datasets
│   ├── encodings/        # Generated encodings
│   └── reference/        # Reference data
├── docs/                 # Documentation (this site)
├── outputs/              # Training outputs
├── scripts/              # Runnable scripts
│   ├── train.py
│   ├── evaluate.py
│   └── ...
└── src/nabu/             # Main library
    ├── tokenizers/
    ├── models/
    ├── datasets/
    ├── encoding/
    └── detection/
```

## Next Steps

- Read the [Architecture](architecture) guide to understand the system design
- Explore the [Encoding Pipeline](encoding-pipeline) for automatic encoding generation
- Check the [Line Detection](line-detection) guide for tablet image processing
- See the [API Reference](api/) for detailed module documentation

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
python scripts/train.py --config config.yaml --batch-size 16
```

**Font Not Found (Encoding Pipeline)**
```bash
# Install NotoSansCuneiform font
# Download from Google Fonts and install system-wide
```

**Import Errors**
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Getting Help

- Open an issue on [GitHub](https://github.com/YOUR_USERNAME/project_nabu/issues)
- Check existing documentation in the `docs/` folder
- Review the research roadmap in `RESEARCH_ROADMAP.md`
