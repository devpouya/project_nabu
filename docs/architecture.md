---
layout: default
title: Architecture
---

# System Architecture

Project Nabu is built with a modular architecture that separates concerns into distinct components. This design enables flexibility in experimentation and easy extension of capabilities.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Project Nabu                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Datasets   │───▶│  Tokenizers  │───▶│    Models    │      │
│  │              │    │              │    │              │      │
│  │ • ORACC      │    │ • Sign       │    │ • Encoder    │      │
│  │ • CuneiML    │    │ • Stroke     │    │ • Decoder    │      │
│  │ • Custom     │    │ • Hybrid     │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│                    ┌────────▼────────┐                         │
│                    │    PaleoCode    │                         │
│                    │   Conversion    │                         │
│                    └────────┬────────┘                         │
│                             │                                   │
│         ┌───────────────────┴───────────────────┐              │
│         │                                       │               │
│  ┌──────▼──────┐                       ┌───────▼───────┐       │
│  │  Encoding   │                       │    Line       │       │
│  │  Pipeline   │                       │  Detection    │       │
│  │ (CV-based)  │                       │  (CNN+Hough)  │       │
│  └─────────────┘                       └───────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. PaleoCode System

The PaleoCode system is the foundation that converts between different representations of cuneiform text.

```
Unicode Cuneiform ←→ PaleoCode ←→ Strokes
      𒀀          ←→  h-v:h-v  ←→ [h,v,h,v]
```

**Key Classes:**
- `PaleoCodeConverter` - Handles all conversions
- Loads mappings from `paleocodes.json`

**Stroke Primitives:**
| Symbol | Angle Range | Description |
|--------|-------------|-------------|
| `h` | 0-30° | Horizontal |
| `v` | 60-90° | Vertical |
| `u` | 30-60° | Upward diagonal |
| `d` | 120-150° | Downward diagonal |
| `c` | Corner | Winkelhaken |

### 2. Tokenization Layer

Three tokenization strategies for different research needs:

#### Sign Tokenizer
```python
"𒀀𒈾𒆠" → [token_AN, token_NA, token_KI]
```
- One token per complete sign
- Best for semantic tasks
- Vocabulary: ~2,000 signs

#### Stroke Tokenizer
```python
"𒀀𒈾𒆠" → [h, v, h, v, h, h, v, h, h, h, v, v]
```
- One token per stroke
- Captures compositional structure
- Vocabulary: ~10 primitives

#### Hybrid Tokenizer
```python
"𒀀𒈾𒆠" → {
    signs: [token_AN, token_NA, token_KI],
    strokes: [h, v, h, v, ...],
    alignment: [(0, 0-4), (1, 4-7), (2, 7-12)]
}
```
- Both representations with alignment
- For hierarchical models

### 3. Dataset System

Unified interface for different cuneiform corpora:

```python
class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[str, Dict]: ...

    @abstractmethod
    def load_data(self) -> None: ...
```

**Implementations:**

| Dataset | Source | Features |
|---------|--------|----------|
| `OraccDataset` | ORACC JSON | Language filtering, metadata |
| `CuneiMLDataset` | CuneiML | Images + annotations |
| `CuneiformDataset` | Text files | Simple loading |

### 4. Model Architectures

#### Transformer Encoder (BERT-style)
```
Input → Embedding → Positional Encoding → N × [Self-Attention + FFN] → Output
```

**Configuration:**
```yaml
model:
  type: transformer_encoder
  vocab_size: auto
  embedding_dim: 512
  num_layers: 6
  num_heads: 8
  feedforward_dim: 2048
  dropout: 0.1
  max_len: 5000
```

#### Transformer Decoder (GPT-style)
```
Input → Embedding → Positional Encoding → N × [Masked Self-Attention + FFN] → Output
```

Uses causal masking to prevent attending to future tokens.

### 5. Encoding Generation Pipeline

Automatic generation of PaleoCode from visual analysis:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Glyph     │───▶│   Stroke    │───▶│   Spatial   │───▶│  Encoding   │
│  Renderer   │    │  Detector   │    │  Analyzer   │    │  Generator  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
   Binary Image    Detected Lines     Groupings         PaleoCode
   (font render)   (Hough transform)  (sweep-line)      (encoding)
```

**Components:**

1. **GlyphRenderer** - Renders Unicode to binary images using NotoSansCuneiform
2. **StrokeDetector** - Uses Hough lines to find strokes, classifies by angle
3. **SpatialAnalyzer** - Sweep-line algorithm for spatial relationships
4. **EncodingGenerator** - Converts to PaleoCode with operators

### 6. Line Detection System

Two-stage approach for tablet images:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Tablet    │───▶│  CNN Line   │───▶│   Hough     │
│   Image     │    │ Segmentation│    │  Transform  │
└─────────────┘    └─────────────┘    └─────────────┘
                          │                  │
                          ▼                  ▼
                   Binary Mask        Line Hypotheses
                   (pixel-wise)       (polar coords)
```

**LineNet Architecture:**
```
Conv(1→64, 11×11, s=4) → ReLU → MaxPool → LRN
→ Conv(64→256, 5×5) → ReLU → MaxPool → LRN
→ Conv(256→384, 3×3) → BN → ReLU
→ Conv(384→384, 3×3) → BN → ReLU
→ Conv(384→256, 3×3) → BN → ReLU → MaxPool
→ FC(9216→512) → ReLU → Dropout
→ FC(512→2) → Softmax
```

## Data Flow

### Training Pipeline

```
1. Load Config (YAML)
        │
2. Initialize Dataset
        │
3. Build Tokenizer Vocabulary
        │
4. Create DataLoaders
        │
5. Initialize Model
        │
6. Training Loop:
   │
   ├── Forward Pass
   ├── Compute Loss
   ├── Backward Pass
   ├── Update Weights
   ├── Log Metrics (TensorBoard/W&B)
   └── Save Checkpoints
        │
7. Evaluation
        │
8. Export Model
```

### Inference Pipeline

```
Input Text (Unicode)
        │
        ▼
   PaleoCode Conversion
        │
        ▼
   Tokenization (Sign/Stroke/Hybrid)
        │
        ▼
   Model Forward Pass
        │
        ▼
   Output (Classification/Generation)
```

## Configuration System

YAML-based configuration for reproducible experiments:

```yaml
# configs/experiments/example.yaml

# Data
data:
  train_path: data/train.txt
  val_path: data/val.txt
  test_path: data/test.txt

# Tokenizer
tokenizer:
  type: sign  # sign | stroke | hybrid
  max_length: 512
  vocab_min_freq: 2

# Model
model:
  type: transformer_encoder  # transformer_encoder | transformer_decoder
  hidden_size: 256
  num_layers: 6
  num_heads: 8
  feedforward_dim: 1024
  dropout: 0.1

# Training
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_steps: 1000
  gradient_clip: 1.0
  device: cuda

# Checkpointing
checkpointing:
  save_dir: outputs/checkpoints
  save_every: 10
  keep_last: 3

# Logging
logging:
  log_dir: outputs/logs
  log_every: 100
```

## Extension Points

### Adding a New Tokenizer

```python
from nabu.tokenizers.base import BaseTokenizer

class MyTokenizer(BaseTokenizer):
    def build_vocab(self, texts):
        # Build vocabulary from texts
        pass

    def encode(self, text):
        # Convert text to token IDs
        pass

    def decode(self, ids):
        # Convert token IDs to text
        pass
```

### Adding a New Dataset

```python
from nabu.datasets.base import BaseDataset

class MyDataset(BaseDataset):
    def load_data(self):
        # Load your data source
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.metadata[idx]
```

### Adding a New Model

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define layers

    def forward(self, x, attention_mask=None):
        # Forward pass
        return output
```

## Design Principles

1. **Modularity** - Components can be mixed and matched
2. **Configuration-Driven** - Experiments defined in YAML
3. **Extensibility** - Easy to add new tokenizers, datasets, models
4. **Reproducibility** - Checkpointing and logging built-in
5. **Multi-Scale** - Support for sign-level to stroke-level analysis

## File Organization

```
src/nabu/
├── __init__.py
├── tokenizers/
│   ├── __init__.py
│   ├── base.py           # Abstract base class
│   ├── sign_tokenizer.py
│   ├── stroke_tokenizer.py
│   └── hybrid_tokenizer.py
├── models/
│   ├── __init__.py
│   ├── transformer.py    # Encoder and Decoder
│   └── embeddings.py     # Token and positional embeddings
├── datasets/
│   ├── __init__.py
│   ├── base.py           # Abstract base class
│   ├── oracc_dataset.py
│   ├── cuneiml_dataset.py
│   └── line_dataset.py
├── encoding/
│   ├── __init__.py
│   ├── glyph_renderer.py
│   ├── stroke_detector.py
│   ├── spatial_analyzer.py
│   └── encoding_generator.py
├── detection/
│   ├── __init__.py
│   ├── linenet.py
│   └── line_detection.py
├── paleocode/
│   ├── __init__.py
│   └── paleocode.py
├── dataloaders/
│   ├── __init__.py
│   ├── builders.py
│   └── collate.py
└── utils/
    ├── __init__.py
    └── experiment_tracker.py
```
