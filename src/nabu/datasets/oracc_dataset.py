"""Dataset class for ORACC cuneiform corpus."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
import torch
from .base import BaseDataset


class OraccDataset(BaseDataset):
    """
    Dataset for ORACC cuneiform corpus.

    Loads ORACC JSON format and provides access to cuneiform texts with metadata.
    Supports conversion from transliteration to Unicode cuneiform and PaleoCode.
    """

    def __init__(
        self,
        json_path: str,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        transform: Optional[Callable] = None,
        filter_language: Optional[str] = None,
        filter_genre: Optional[str] = None,
        filter_period: Optional[str] = None,
        extract_mode: str = "lines",  # 'lines', 'paragraphs', 'full_text'
        min_text_length: int = 1,
    ):
        """
        Initialize the ORACC dataset.

        Args:
            json_path: Path to ORACC JSON file
            tokenizer: Tokenizer instance for encoding
            max_length: Maximum sequence length
            transform: Optional transform to apply
            filter_language: Filter by language code (e.g., 'akk', 'sux')
            filter_genre: Filter by genre (e.g., 'Lexical', 'Royal/Monumental')
            filter_period: Filter by period (e.g., 'Uruk III')
            extract_mode: How to extract text - 'lines', 'paragraphs', or 'full_text'
            min_text_length: Minimum text length to include
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.filter_language = filter_language
        self.filter_genre = filter_genre
        self.filter_period = filter_period
        self.extract_mode = extract_mode
        self.min_text_length = min_text_length

        self.samples: List[Dict[str, Any]] = []
        self.documents: Dict[str, Any] = {}

        self._load_oracc_json(json_path)
        self._extract_samples()

    def _load_oracc_json(self, json_path: str) -> None:
        """Load ORACC JSON file."""
        print(f"Loading ORACC data from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        print(f"Loaded {len(self.documents)} documents")

    def _extract_samples(self) -> None:
        """Extract text samples from ORACC documents."""
        print("Extracting samples...")

        for doc_id, doc in self.documents.items():
            # Apply filters
            if self.filter_language and doc.get("language") != self.filter_language:
                continue
            if self.filter_genre and doc.get("genre") != self.filter_genre:
                continue
            if self.filter_period and doc.get("period") != self.filter_period:
                continue

            # Extract text based on mode
            if self.extract_mode == "lines":
                self._extract_lines(doc_id, doc)
            elif self.extract_mode == "paragraphs":
                self._extract_paragraphs(doc_id, doc)
            elif self.extract_mode == "full_text":
                self._extract_full_text(doc_id, doc)
            else:
                raise ValueError(f"Unknown extract_mode: {self.extract_mode}")

        print(f"Extracted {len(self.samples)} samples")

    def _extract_lines(self, doc_id: str, doc: Dict[str, Any]) -> None:
        """Extract individual lines as samples."""
        text_areas = doc.get("text_areas", [])

        for area in text_areas:
            area_name = area.get("name", "")
            lines = area.get("lines", [])

            for line in lines:
                text = line.get("text", "").strip()
                line_number = line.get("number", "")

                # Skip empty or very short texts
                if len(text) < self.min_text_length:
                    continue

                # Skip lines with only [...] or X
                if text in ["[...]", "X", "[X]"]:
                    continue

                sample = {
                    "text": text,
                    "doc_id": doc_id,
                    "area": area_name,
                    "line_number": line_number,
                    "language": doc.get("language"),
                    "genre": doc.get("genre"),
                    "period": doc.get("period"),
                    "object_type": doc.get("object_type"),
                }

                self.samples.append(sample)

    def _extract_paragraphs(self, doc_id: str, doc: Dict[str, Any]) -> None:
        """Extract paragraphs as samples."""
        text_areas = doc.get("text_areas", [])

        for area in text_areas:
            area_name = area.get("name", "")
            paragraphs = area.get("paragraphs", [])

            for para_idx, para in enumerate(paragraphs):
                # Paragraphs might be strings or structured data
                if isinstance(para, str):
                    text = para.strip()
                elif isinstance(para, dict):
                    text = para.get("text", "").strip()
                else:
                    continue

                if len(text) < self.min_text_length:
                    continue

                sample = {
                    "text": text,
                    "doc_id": doc_id,
                    "area": area_name,
                    "paragraph_number": para_idx,
                    "language": doc.get("language"),
                    "genre": doc.get("genre"),
                    "period": doc.get("period"),
                    "object_type": doc.get("object_type"),
                }

                self.samples.append(sample)

    def _extract_full_text(self, doc_id: str, doc: Dict[str, Any]) -> None:
        """Extract full document text as single sample."""
        text_areas = doc.get("text_areas", [])
        all_lines = []

        for area in text_areas:
            lines = area.get("lines", [])
            for line in lines:
                text = line.get("text", "").strip()
                if text and text not in ["[...]", "X", "[X]"]:
                    all_lines.append(text)

        full_text = " ".join(all_lines)

        if len(full_text) < self.min_text_length:
            return

        sample = {
            "text": full_text,
            "doc_id": doc_id,
            "language": doc.get("language"),
            "genre": doc.get("genre"),
            "period": doc.get("period"),
            "object_type": doc.get("object_type"),
        }

        self.samples.append(sample)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.

        Returns:
            Dictionary with keys:
            - text: Original text (transliteration)
            - input_ids: Token IDs (if tokenizer provided)
            - attention_mask: Attention mask (if tokenizer provided)
            - metadata: Document metadata
        """
        sample = self.samples[idx].copy()
        text = sample["text"]

        # Store metadata separately
        metadata = {
            "doc_id": sample.pop("doc_id"),
            "language": sample.pop("language", None),
            "genre": sample.pop("genre", None),
            "period": sample.pop("period", None),
            "object_type": sample.pop("object_type", None),
        }

        # Add area and line info if available
        if "area" in sample:
            metadata["area"] = sample.pop("area")
        if "line_number" in sample:
            metadata["line_number"] = sample.pop("line_number")
        if "paragraph_number" in sample:
            metadata["paragraph_number"] = sample.pop("paragraph_number")

        result = {
            "text": text,
            "metadata": metadata,
        }

        # Tokenize if tokenizer is provided
        if self.tokenizer is not None:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding=True,
                truncation=True,
            )

            result["input_ids"] = torch.tensor(encoded, dtype=torch.long)

            # Create attention mask
            attention_mask = [
                1 if token_id != self.tokenizer.pad_token_id else 0
                for token_id in encoded
            ]
            result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)

        # Apply transform if provided
        if self.transform is not None:
            result = self.transform(result)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = super().get_stats()

        if not self.samples:
            return stats

        # Text length statistics
        text_lengths = [len(s["text"]) for s in self.samples]
        stats.update({
            "min_length": min(text_lengths),
            "max_length": max(text_lengths),
            "avg_length": sum(text_lengths) / len(text_lengths),
        })

        # Language distribution
        languages = {}
        for s in self.samples:
            lang = s.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
        stats["language_distribution"] = languages

        # Genre distribution
        genres = {}
        for s in self.samples:
            genre = s.get("genre", "unknown")
            genres[genre] = genres.get(genre, 0) + 1
        stats["genre_distribution"] = genres

        # Period distribution
        periods = {}
        for s in self.samples:
            period = s.get("period", "unknown")
            periods[period] = periods.get(period, 0) + 1
        stats["period_distribution"] = periods

        return stats

    def get_unique_languages(self) -> List[str]:
        """Get list of unique languages in dataset."""
        return list(set(s.get("language") for s in self.samples if s.get("language")))

    def get_unique_genres(self) -> List[str]:
        """Get list of unique genres in dataset."""
        return list(set(s.get("genre") for s in self.samples if s.get("genre")))

    def get_unique_periods(self) -> List[str]:
        """Get list of unique periods in dataset."""
        return list(set(s.get("period") for s in self.samples if s.get("period")))

    def filter_by_metadata(
        self,
        language: Optional[str] = None,
        genre: Optional[str] = None,
        period: Optional[str] = None,
    ) -> "OraccDataset":
        """
        Create a filtered copy of the dataset.

        Args:
            language: Filter by language
            genre: Filter by genre
            period: Filter by period

        Returns:
            New OraccDataset with filtered samples
        """
        filtered_dataset = OraccDataset.__new__(OraccDataset)
        filtered_dataset.tokenizer = self.tokenizer
        filtered_dataset.max_length = self.max_length
        filtered_dataset.transform = self.transform
        filtered_dataset.extract_mode = self.extract_mode
        filtered_dataset.min_text_length = self.min_text_length
        filtered_dataset.documents = self.documents

        # Filter samples
        filtered_samples = []
        for sample in self.samples:
            if language and sample.get("language") != language:
                continue
            if genre and sample.get("genre") != genre:
                continue
            if period and sample.get("period") != period:
                continue
            filtered_samples.append(sample)

        filtered_dataset.samples = filtered_samples

        return filtered_dataset
