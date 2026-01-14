# ORACC Dataset Guide

This guide explains how to use the ORACC (Open Richly Annotated Cuneiform Corpus) dataset with Project Nabu and convert transliterations to PaleoCode format.

## Overview

The ORACC dataset contains cuneiform texts in ATF (ASCII Transliteration Format). This guide shows you how to:

1. Load the ORACC JSON data
2. Extract text samples with metadata
3. Convert transliterations to Unicode cuneiform
4. Convert to PaleoCode for training neural models

## Quick Start

### 1. Load ORACC Dataset

```python
from nabu.datasets import OraccDataset

# Load dataset
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    extract_mode="lines",  # Extract individual lines
    min_text_length=5,
)

print(f"Loaded {len(dataset)} samples")
```

### 2. Explore Dataset Statistics

```python
# Get statistics
stats = dataset.get_stats()

print(f"Languages: {dataset.get_unique_languages()}")
print(f"Genres: {dataset.get_unique_genres()}")
print(f"Periods: {dataset.get_unique_periods()}")

# Show distributions
print(stats['language_distribution'])
print(stats['genre_distribution'])
print(stats['period_distribution'])
```

### 3. Filter by Metadata

```python
# Filter by language
akkadian_dataset = dataset.filter_by_metadata(language="akk")

# Filter by genre
lexical_dataset = dataset.filter_by_metadata(genre="Lexical")

# Filter by multiple criteria
filtered = dataset.filter_by_metadata(
    language="sux",
    genre="Royal/Monumental",
    period="Ur III (ca. 2100-2000 BC)"
)
```

### 4. Convert to PaleoCode

Use the conversion script:

```bash
# Analyze dataset first
python scripts/oracc_to_paleocode.py \
    --oracc-json data/external/oracc/oracc_pubs.json \
    --analyze-only

# Convert to PaleoCode (JSON format)
python scripts/oracc_to_paleocode.py \
    --oracc-json data/external/oracc/oracc_pubs.json \
    --output data/processed/oracc_paleocode.json \
    --format json

# Convert to plain text (one PaleoCode sequence per line)
python scripts/oracc_to_paleocode.py \
    --oracc-json data/external/oracc/oracc_pubs.json \
    --output data/processed/oracc_paleocode.txt \
    --format txt

# Convert only specific language
python scripts/oracc_to_paleocode.py \
    --oracc-json data/external/oracc/oracc_pubs.json \
    --output data/processed/akkadian_paleocode.json \
    --filter-language akk
```

## Dataset Structure

### ORACC JSON Format

Each document in the ORACC JSON has:

```json
{
  "P000001": {
    "id": "P000001",
    "language": "akk",
    "text_areas": [
      {
        "name": "obverse",
        "lines": [
          {
            "number": "1.",
            "text": "GAL~a UMUN2",
            "languages": {}
          }
        ],
        "paragraphs": []
      }
    ],
    "genre": "Lexical",
    "period": "Uruk III (ca. 3200-3000 BC)",
    "object_type": "tablet"
  }
}
```

### OraccDataset Sample Format

When you load a sample from `OraccDataset`:

```python
sample = dataset[0]

{
  "text": "1(N01) , GAL~a UMUN2",
  "metadata": {
    "doc_id": "P000001",
    "area": "obverse",
    "line_number": "1.",
    "language": "akk",
    "genre": "Lexical",
    "period": "Uruk III (ca. 3200-3000 BC)",
    "object_type": "tablet"
  }
}
```

## Extract Modes

The `extract_mode` parameter controls how text is extracted:

### 1. Lines Mode (default)

Extracts each line as a separate sample:

```python
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    extract_mode="lines"
)
# Result: Many samples, each is a single line
```

### 2. Paragraphs Mode

Extracts paragraphs as samples:

```python
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    extract_mode="paragraphs"
)
# Result: Fewer samples, each is a paragraph
```

### 3. Full Text Mode

Extracts entire document as single sample:

```python
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    extract_mode="full_text"
)
# Result: One sample per document
```

## Conversion Process

### Transliteration to Unicode

The ORACC text uses ATF transliteration:
- `GAL~a` → Unicode cuneiform sign 𒃲
- `UMUN2` → Unicode cuneiform sign 𒌝

### Unicode to PaleoCode

Unicode signs are converted to stroke-based PaleoCode:
- 𒀀 → `a-a:a`
- 𒀁 → `a-:sa-:sa:sa-a:a`

### Example Conversion

```
Original (ATF):  "1(N01) , GAL~a UMUN2"
                     ↓
Unicode:         "𒃲𒌝"
                     ↓
PaleoCode:       "g-t-d:wa w-:w-a"
```

## Training with PaleoCode Data

After conversion, train a model:

```bash
# Train on converted PaleoCode data
python scripts/train.py \
    --config configs/default.yaml \
    --data data/processed/oracc_paleocode.txt \
    --output-dir outputs/oracc_experiment
```

## Filtering Options

Filter during dataset loading:

```python
# Only Akkadian texts
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    filter_language="akk"
)

# Only lexical texts
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    filter_genre="Lexical"
)

# Only specific period
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    filter_period="Ur III (ca. 2100-2000 BC)"
)

# Combine filters
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    filter_language="sux",
    filter_genre="Royal/Monumental",
    min_text_length=20
)
```

## Working with Tokenizers

Use the dataset with tokenizers:

```python
from nabu.tokenizers import SignTokenizer, StrokeTokenizer
from nabu.datasets import OraccDataset

# Initialize tokenizer
tokenizer = SignTokenizer()

# Load dataset with tokenizer
dataset = OraccDataset(
    json_path="data/external/oracc/oracc_pubs.json",
    tokenizer=tokenizer,
    max_length=512
)

# Build vocabulary
texts = [sample["text"] for sample in dataset.samples]
tokenizer.build_vocab(texts)

# Now dataset returns tokenized samples
sample = dataset[0]
print(sample["input_ids"])
print(sample["attention_mask"])
```

## Example Script

Run the example script:

```bash
python examples/oracc_example.py
```

This demonstrates:
- Loading the dataset
- Exploring statistics
- Converting to PaleoCode
- Filtering by metadata

## Common Issues

### Issue: Conversion Failures

**Problem**: Some transliterations don't convert to Unicode

**Solution**:
- ORACC uses ATF format which may not have 1:1 mapping to Unicode
- Some signs are damaged/uncertain (marked with `[...]`, `#`, `?`)
- Use `min_text_length` to filter out very short/problematic texts

### Issue: Unknown Characters

**Problem**: Characters not in PaleoCode database

**Solution**:
- The PaleoCode database contains ~1000 most common signs
- Rare or variant signs may not be included
- Consider filtering by quality or completeness

## Statistics Example

```python
stats = dataset.get_stats()

print(stats)
# Output:
# {
#   'num_samples': 45231,
#   'min_length': 5,
#   'max_length': 842,
#   'avg_length': 45.3,
#   'language_distribution': {
#     'akk': 15234,
#     'sux': 18901,
#     'qpc': 11096
#   },
#   'genre_distribution': {
#     'Lexical': 25431,
#     'Administrative': 12301,
#     'Royal/Monumental': 7499
#   },
#   ...
# }
```

## Next Steps

1. Explore the dataset with `examples/oracc_example.py`
2. Analyze statistics with `scripts/oracc_to_paleocode.py --analyze-only`
3. Convert to PaleoCode format
4. Train models on the converted data
5. Evaluate model performance on cuneiform text

## References

- ORACC Project: http://oracc.museum.upenn.edu/
- PaleoCode System: See `src/nabu/paleocode/`
- ATF Format: http://oracc.museum.upenn.edu/doc/help/editinginatf/
