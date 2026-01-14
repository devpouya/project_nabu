# Project Nabu

**NLP Training Framework for Cuneiform Corpus**

Project Nabu is a modular deep learning framework for training neural language models on cuneiform text using stroke-based PaleoCode encoding. Unlike approaches that work with transliterated text, Nabu operates directly on original cuneiform signs, preserving the visual and structural information encoded in the ancient writing system.

## Features

- **Multiple Tokenization Strategies**: Sign-level, stroke-level, and hybrid approaches
- **PaleoCode Integration**: Seamless integration with stroke-based cuneiform encoding
- **Flexible Model Architectures**: Transformer (BERT/GPT-style) models
- **Experiment Management**: YAML-based configuration system for reproducible experiments
- **Efficient Data Pipeline**: Custom datasets and dataloaders optimized for variable-length sequences
- **Training Utilities**: Built-in training scripts with TensorBoard logging and checkpointing

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
cd /Users/pouyapourjafar/projects/cuneiform/project_nabu
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Project Structure

```
project_nabu/
├── src/nabu/              # Main package
│   ├── tokenizers/        # Tokenization implementations
│   ├── datasets/          # Dataset classes
│   ├── dataloaders/       # DataLoader utilities
│   └── models/            # Neural network models
├── scripts/               # Training and evaluation scripts
├── configs/               # Configuration files
│   ├── default.yaml       # Default configuration
│   └── experiments/       # Experiment-specific configs
├── data/                  # Data directory (gitignored)
│   ├── raw/              # Raw cuneiform text
│   ├── processed/        # Preprocessed data
│   └── external/         # External datasets
└── notebooks/            # Jupyter notebooks for exploration
```

## Quick Start

### Option A: Using ORACC Dataset

The ORACC (Open Richly Annotated Cuneiform Corpus) dataset is included in `data/external/oracc/`:

```bash
# Analyze the ORACC dataset
python scripts/oracc_to_paleocode.py \
    --oracc-json data/external/oracc/oracc_pubs.json \
    --analyze-only

# Convert ORACC transliterations to PaleoCode
python scripts/oracc_to_paleocode.py \
    --oracc-json data/external/oracc/oracc_pubs.json \
    --output data/processed/oracc_paleocode.json \
    --format json

# Train on ORACC data
python scripts/train.py \
    --config configs/default.yaml \
    --data data/processed/oracc_paleocode.txt \
    --output-dir outputs/oracc_experiment
```

See `docs/ORACC_DATASET.md` for detailed documentation.

### Option B: Using Your Own Data

Place your cuneiform text files (UTF-8 encoded) in `data/raw/`. The text should contain actual cuneiform Unicode characters (𒀀, 𒀁, etc.).

### 2. Preprocess the Data

Split your dataset into train/val/test:
```bash
python scripts/preprocess.py \
    --input data/raw/cuneiform_corpus.txt \
    --output data/processed \
    --mode split \
    --train-ratio 0.8 \
    --val-ratio 0.1
```

Generate corpus statistics:
```bash
python scripts/preprocess.py \
    --input data/raw/cuneiform_corpus.txt \
    --output data/stats.json \
    --mode stats
```

### 3. Train a Model

Train with default configuration:
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data data/processed/train.txt \
    --output-dir outputs/experiment_1
```

Train with a specific experiment configuration:
```bash
python scripts/train.py \
    --config configs/experiments/sign_transformer.yaml \
    --data data/processed/train.txt \
    --output-dir outputs/sign_transformer
```

### 4. Evaluate the Model

```bash
python scripts/evaluate.py \
    --checkpoint outputs/sign_transformer/best_model.pt \
    --vocab outputs/sign_transformer/vocab.json \
    --data data/processed/test.txt \
    --tokenizer-type sign \
    --generate \
    --num-samples 10 \
    --output results/evaluation.json
```

## Tokenization Approaches

Project Nabu supports three tokenization strategies:

### Sign-Level Tokenization

Treats each complete cuneiform sign as a single token. Best for preserving sign-level semantics.

```python
from nabu.tokenizers import SignTokenizer

tokenizer = SignTokenizer()
tokenizer.build_vocab(texts)

# 𒀀 → "a-a:a" → token_id
```

**Use cases**: Sign prediction, sign classification, text generation

### Stroke-Level Tokenization

Breaks down each sign into its constituent strokes. Better for learning compositional structure.

```python
from nabu.tokenizers import StrokeTokenizer

tokenizer = StrokeTokenizer()
tokenizer.build_vocab(texts)

# 𒀀 → "a-a:a" → ['a', 'a', 'a'] → [token_id1, token_id2, token_id3]
```

**Use cases**: Stroke prediction, understanding sign composition, finer-grained modeling

### Hybrid Tokenization

Combines both sign and stroke representations for hierarchical modeling.

```python
from nabu.tokenizers import HybridTokenizer

tokenizer = HybridTokenizer()
tokenizer.build_vocab(texts)

# Provides both sign_ids and stroke_ids with alignment information
```

**Use cases**: Multi-granularity models, hierarchical architectures

## Model Architectures

### Transformer Encoder (BERT-style)

Bidirectional transformer for masked language modeling and classification:

```yaml
model:
  type: transformer_encoder
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1
```

### Transformer Decoder (GPT-style)

Autoregressive transformer for text generation:

```yaml
model:
  type: transformer_decoder
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1
```

## Configuration

All experiments are configured via YAML files in `configs/`. Key configuration options:

```yaml
# Tokenizer settings
tokenizer:
  type: sign              # 'sign', 'stroke', or 'hybrid'
  max_length: 512

# Model settings
model:
  type: transformer_encoder
  hidden_size: 256
  num_layers: 6
  num_heads: 8
  dropout: 0.1

# Training settings
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0001
  device: cuda
```

See `configs/default.yaml` for all available options.

## Example Experiments

### Experiment 1: Sign-Level Transformer

```bash
python scripts/train.py \
    --config configs/experiments/sign_transformer.yaml \
    --data data/processed/train.txt \
    --output-dir outputs/sign_transformer
```

### Experiment 2: Hybrid Model

```bash
python scripts/train.py \
    --config configs/experiments/hybrid_model.yaml \
    --data data/processed/train.txt \
    --output-dir outputs/hybrid
```

## Using the API

### Loading and Using a Tokenizer

```python
from nabu.tokenizers import SignTokenizer

# Initialize
tokenizer = SignTokenizer()

# Build vocabulary from texts
texts = ["𒀀𒀁𒀂", "𒋰𒀀𒀁"]
tokenizer.build_vocab(texts)

# Encode
token_ids = tokenizer.encode("𒀀𒀁", add_special_tokens=True)

# Decode
text = tokenizer.decode(token_ids)
```

### Creating a Dataset

#### Standard Cuneiform Dataset

```python
from nabu.datasets import CuneiformDataset

# Create dataset
dataset = CuneiformDataset(
    data_path="data/processed/train.txt",
    tokenizer=tokenizer,
    max_length=512
)

# Cache encodings for faster training
dataset.cache_encodings()

# Get a sample
sample = dataset[0]  # Returns {'input_ids': ..., 'attention_mask': ..., 'text': ...}
```

#### ORACC Dataset

```python
from nabu.datasets import OraccDataset

# Load ORACC corpus
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    extract_mode="lines",  # 'lines', 'paragraphs', or 'full_text'
    filter_language="akk",  # Optional: filter by language
    filter_genre="Lexical",  # Optional: filter by genre
    min_text_length=10
)

# Get statistics
stats = dataset.get_stats()
print(f"Languages: {stats['language_distribution']}")
print(f"Genres: {stats['genre_distribution']}")

# Get sample with metadata
sample = dataset[0]
print(sample["text"])
print(sample["metadata"])
```

### Training Loop

```python
from nabu.models import TransformerEncoder
from nabu.dataloaders import build_dataloader

# Build model
model = TransformerEncoder(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=512,
    num_layers=6,
    num_heads=8
)

# Create dataloader
dataloader = build_dataloader(
    dataset,
    batch_size=32,
    collator_type="dynamic",
    pad_token_id=tokenizer.pad_token_id
)

# Train
for batch in dataloader:
    input_ids = batch['input_ids']
    outputs = model(input_ids=input_ids)
    # ... training logic
```

## PaleoCode System

Project Nabu includes a bundled PaleoCode system for stroke-based encoding of cuneiform signs. PaleoCode represents each sign as a sequence of fundamental strokes, enabling models to learn compositional structure.

The PaleoCode data is included in `src/nabu/paleocode/` and is automatically loaded by the tokenizers.

### Example PaleoCode Mappings

- 𒀀 (U+12000, "A") → `a-a:a`
- 𒀁 (U+12001, "A x A") → `a-:sa-:sa:sa-a:a`
- 𒀂 (U+12002, "A x BAD") → `a-b-w-a:a`

## Contributing

This is a research project. Contributions and suggestions are welcome.

## Citation

If you use Project Nabu in your research, please cite:

```bibtex
@software{project_nabu,
  title={Project Nabu: NLP Training Framework for Cuneiform Corpus},
  author={Project Nabu Team},
  year={2026},
  url={https://github.com/yourusername/project_nabu}
}
```

## License

MIT License

## Acknowledgments

- PaleoCode system for stroke-based cuneiform encoding
- PyTorch and HuggingFace for deep learning infrastructure
