"""Main dataset class for cuneiform text data."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import torch
from .base import BaseDataset


class CuneiformDataset(BaseDataset):
    """
    Dataset for cuneiform text.

    Supports:
    - Loading text from files (one sample per line or per file)
    - Pre-tokenized data loading
    - On-the-fly tokenization
    - Data transforms/augmentation
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        file_format: str = "txt",
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to data file or directory
            tokenizer: Tokenizer instance to use for encoding
            max_length: Maximum sequence length
            transform: Optional transform to apply to samples
            cache_dir: Directory to cache tokenized data
            file_format: Format of data files ('txt', 'json', 'jsonl')
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.cache_dir = cache_dir
        self.file_format = file_format

        self.samples: List[str] = []
        self._cached_encodings: Optional[List[Dict[str, Any]]] = None

        if data_path:
            self.load_data(data_path)

    def load_data(self, path: str) -> None:
        """
        Load cuneiform text data from file or directory.

        Supports:
        - .txt files: one sample per line
        - .json/.jsonl files: structured data
        - directories: loads all compatible files

        Args:
            path: Path to data file or directory
        """
        path_obj = Path(path)

        if path_obj.is_file():
            self._load_file(path)
        elif path_obj.is_dir():
            self._load_directory(path)
        else:
            raise ValueError(f"Path does not exist: {path}")

    def _load_file(self, file_path: str) -> None:
        """Load data from a single file."""
        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lower()

        if suffix == ".txt":
            self._load_txt_file(file_path)
        elif suffix == ".json":
            self._load_json_file(file_path)
        elif suffix == ".jsonl":
            self._load_jsonl_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_txt_file(self, file_path: str) -> None:
        """Load text file (one sample per line)."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(line)

    def _load_json_file(self, file_path: str) -> None:
        """Load JSON file containing list of samples."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.samples.extend([str(item) for item in data])
        elif isinstance(data, dict) and "text" in data:
            self.samples.extend(data["text"])
        else:
            raise ValueError("JSON file must contain a list or dict with 'text' key")

    def _load_jsonl_file(self, file_path: str) -> None:
        """Load JSONL file (one JSON object per line)."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        self.samples.append(obj)
                    elif isinstance(obj, dict) and "text" in obj:
                        self.samples.append(obj["text"])

    def _load_directory(self, dir_path: str) -> None:
        """Load all compatible files from directory."""
        dir_path_obj = Path(dir_path)

        for file_path in sorted(dir_path_obj.glob("*")):
            if file_path.is_file() and file_path.suffix.lower() in [".txt", ".json", ".jsonl"]:
                try:
                    self._load_file(str(file_path))
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.

        Returns:
            Dictionary with keys:
            - input_ids: Token IDs (if tokenizer provided)
            - attention_mask: Attention mask (if tokenizer provided)
            - text: Original text
        """
        # Check if we have cached encodings
        if self._cached_encodings is not None:
            return self._cached_encodings[idx]

        text = self.samples[idx]
        sample = {"text": text}

        # Tokenize if tokenizer is provided
        if self.tokenizer is not None:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding=True,
                truncation=True,
            )

            sample["input_ids"] = torch.tensor(encoded, dtype=torch.long)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0
                            for token_id in encoded]
            sample["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)

        # Apply transform if provided
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def cache_encodings(self) -> None:
        """
        Pre-compute and cache all encodings for faster training.
        Call this before training to speed up data loading.
        """
        if self.tokenizer is None:
            raise ValueError("Cannot cache encodings without a tokenizer")

        print(f"Caching {len(self)} samples...")
        self._cached_encodings = []

        for idx in range(len(self)):
            text = self.samples[idx]
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding=True,
                truncation=True,
            )

            attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0
                            for token_id in encoded]

            sample = {
                "text": text,
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

            self._cached_encodings.append(sample)

        print("Caching complete!")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        stats = super().get_stats()

        if self.samples:
            # Character-level statistics
            text_lengths = [len(text) for text in self.samples]
            stats.update({
                "min_length": min(text_lengths),
                "max_length": max(text_lengths),
                "avg_length": sum(text_lengths) / len(text_lengths),
            })

        if self.tokenizer is not None and self._cached_encodings:
            # Token-level statistics
            token_counts = [
                (sample["attention_mask"] == 1).sum().item()
                for sample in self._cached_encodings
            ]
            stats.update({
                "min_tokens": min(token_counts),
                "max_tokens": max(token_counts),
                "avg_tokens": sum(token_counts) / len(token_counts),
            })

        return stats

    def save_cache(self, cache_path: str) -> None:
        """
        Save cached encodings to disk.

        Args:
            cache_path: Path to save cache file
        """
        if self._cached_encodings is None:
            raise ValueError("No cached encodings to save. Call cache_encodings() first.")

        torch.save(self._cached_encodings, cache_path)
        print(f"Cache saved to {cache_path}")

    def load_cache(self, cache_path: str) -> None:
        """
        Load cached encodings from disk.

        Args:
            cache_path: Path to cache file
        """
        self._cached_encodings = torch.load(cache_path)
        print(f"Cache loaded from {cache_path}")
