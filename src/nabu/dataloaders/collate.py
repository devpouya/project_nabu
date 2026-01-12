"""Custom collate functions for batching cuneiform data."""

from typing import List, Dict, Any
import torch


class PaddingCollator:
    """
    Collator that pads sequences to a fixed maximum length.
    """

    def __init__(self, pad_token_id: int = 0, max_length: int = 512):
        """
        Initialize the collator.

        Args:
            pad_token_id: Token ID to use for padding
            max_length: Maximum sequence length
        """
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples with padding.

        Args:
            batch: List of sample dictionaries

        Returns:
            Dictionary with batched and padded tensors
        """
        # Extract keys from first sample
        keys = batch[0].keys()
        collated = {}

        for key in keys:
            if key == "text":
                # Keep text as list
                collated[key] = [sample[key] for sample in batch]
            elif key in ["input_ids", "attention_mask", "labels"]:
                # Pad sequences
                sequences = [sample[key] for sample in batch]
                padded = self._pad_sequences(sequences, key == "labels")
                collated[key] = padded
            else:
                # Try to stack other tensors
                try:
                    values = [sample[key] for sample in batch]
                    if isinstance(values[0], torch.Tensor):
                        collated[key] = torch.stack(values)
                    else:
                        collated[key] = values
                except:
                    collated[key] = [sample[key] for sample in batch]

        return collated

    def _pad_sequences(
        self, sequences: List[torch.Tensor], is_labels: bool = False
    ) -> torch.Tensor:
        """
        Pad sequences to the same length.

        Args:
            sequences: List of 1D tensors
            is_labels: Whether these are label sequences (use -100 for padding)

        Returns:
            2D tensor of shape (batch_size, max_length)
        """
        batch_size = len(sequences)
        max_len = min(max(seq.size(0) for seq in sequences), self.max_length)

        pad_value = -100 if is_labels else self.pad_token_id
        padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)

        for i, seq in enumerate(sequences):
            length = min(seq.size(0), max_len)
            padded[i, :length] = seq[:length]

        return padded


class DynamicPaddingCollator:
    """
    Collator that dynamically pads sequences to the longest sequence in each batch.
    More efficient than fixed padding when sequences vary in length.
    """

    def __init__(self, pad_token_id: int = 0):
        """
        Initialize the collator.

        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples with dynamic padding.

        Args:
            batch: List of sample dictionaries

        Returns:
            Dictionary with batched and padded tensors
        """
        # Extract keys from first sample
        keys = batch[0].keys()
        collated = {}

        for key in keys:
            if key == "text":
                # Keep text as list
                collated[key] = [sample[key] for sample in batch]
            elif key in ["input_ids", "attention_mask", "labels"]:
                # Pad sequences dynamically
                sequences = [sample[key] for sample in batch]
                padded = self._pad_sequences_dynamic(sequences, key == "labels")
                collated[key] = padded
            else:
                # Try to stack other tensors
                try:
                    values = [sample[key] for sample in batch]
                    if isinstance(values[0], torch.Tensor):
                        collated[key] = torch.stack(values)
                    else:
                        collated[key] = values
                except:
                    collated[key] = [sample[key] for sample in batch]

        return collated

    def _pad_sequences_dynamic(
        self, sequences: List[torch.Tensor], is_labels: bool = False
    ) -> torch.Tensor:
        """
        Pad sequences to the length of the longest sequence in the batch.

        Args:
            sequences: List of 1D tensors
            is_labels: Whether these are label sequences (use -100 for padding)

        Returns:
            2D tensor of shape (batch_size, max_length_in_batch)
        """
        batch_size = len(sequences)
        max_len = max(seq.size(0) for seq in sequences)

        pad_value = -100 if is_labels else self.pad_token_id
        padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)

        for i, seq in enumerate(sequences):
            padded[i, : seq.size(0)] = seq

        return padded


class HierarchicalCollator:
    """
    Collator for hierarchical tokenization (sign + stroke levels).
    Pads both sign and stroke sequences appropriately.
    """

    def __init__(self, pad_token_id: int = 0):
        """
        Initialize the collator.

        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch with hierarchical encodings.

        Args:
            batch: List of sample dictionaries with sign_ids and stroke_ids

        Returns:
            Dictionary with batched tensors for both levels
        """
        collated = {}

        # Handle text
        if "text" in batch[0]:
            collated["text"] = [sample["text"] for sample in batch]

        # Handle sign-level encodings
        if "sign_ids" in batch[0]:
            sign_sequences = [torch.tensor(sample["sign_ids"]) for sample in batch]
            collated["sign_ids"] = self._pad_sequences(sign_sequences)

        # Handle stroke-level encodings
        if "stroke_ids" in batch[0]:
            stroke_sequences = [torch.tensor(sample["stroke_ids"]) for sample in batch]
            collated["stroke_ids"] = self._pad_sequences(stroke_sequences)

        # Handle alignment information
        if "alignment" in batch[0]:
            collated["alignment"] = [sample["alignment"] for sample in batch]

        return collated

    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to the longest in batch."""
        batch_size = len(sequences)
        max_len = max(seq.size(0) for seq in sequences)

        padded = torch.full((batch_size, max_len), self.pad_token_id, dtype=sequences[0].dtype)

        for i, seq in enumerate(sequences):
            padded[i, : seq.size(0)] = seq

        # Create attention mask
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            attention_mask[i, : seq.size(0)] = 1

        return padded
