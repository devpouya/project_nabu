"""Data transforms and augmentation for cuneiform datasets."""

import random
from typing import Dict, Any, Optional
import torch


class CuneiformTransform:
    """Base class for cuneiform data transforms."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transform to a sample.

        Args:
            sample: Dictionary containing sample data

        Returns:
            Transformed sample
        """
        raise NotImplementedError


class MaskingTransform(CuneiformTransform):
    """
    Randomly mask tokens in the input sequence.
    Useful for masked language modeling (BERT-style pretraining).
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_token_id: int = 1,
        vocab_size: Optional[int] = None,
        random_token_prob: float = 0.1,
        keep_prob: float = 0.1,
    ):
        """
        Initialize masking transform.

        Args:
            mask_prob: Probability of masking a token
            mask_token_id: Token ID to use for masking
            vocab_size: Size of vocabulary (for random token replacement)
            random_token_prob: Probability of replacing with random token instead of mask
            keep_prob: Probability of keeping original token instead of masking
        """
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.random_token_prob = random_token_prob
        self.keep_prob = keep_prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply masking to input_ids."""
        if "input_ids" not in sample:
            return sample

        input_ids = sample["input_ids"].clone()
        labels = sample["input_ids"].clone()

        # Create mask for tokens to modify
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)

        # Don't mask special tokens (padding, BOS, EOS)
        if "attention_mask" in sample:
            special_tokens_mask = sample["attention_mask"] == 0
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time: replace with [MASK]
        mask_token_indices = masked_indices & (torch.rand(input_ids.shape) < 0.8)
        input_ids[mask_token_indices] = self.mask_token_id

        # 10% of the time: replace with random token
        if self.vocab_size is not None:
            random_token_indices = (
                masked_indices
                & ~mask_token_indices
                & (torch.rand(input_ids.shape) < self.random_token_prob / (1 - 0.8))
            )
            random_tokens = torch.randint(
                low=4,  # Start after special tokens
                high=self.vocab_size,
                size=input_ids.shape,
                dtype=input_ids.dtype,
            )
            input_ids[random_token_indices] = random_tokens[random_token_indices]

        # 10% of the time: keep original token (no change needed)

        # Only compute loss on masked tokens
        labels[~masked_indices] = -100

        sample["input_ids"] = input_ids
        sample["labels"] = labels

        return sample


class NoiseTransform(CuneiformTransform):
    """
    Add noise to input sequences by randomly swapping, deleting, or duplicating tokens.
    Useful for denoising autoencoder pretraining.
    """

    def __init__(
        self,
        swap_prob: float = 0.05,
        delete_prob: float = 0.05,
        duplicate_prob: float = 0.05,
    ):
        """
        Initialize noise transform.

        Args:
            swap_prob: Probability of swapping adjacent tokens
            delete_prob: Probability of deleting a token
            duplicate_prob: Probability of duplicating a token
        """
        self.swap_prob = swap_prob
        self.delete_prob = delete_prob
        self.duplicate_prob = duplicate_prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply noise to input_ids."""
        if "input_ids" not in sample:
            return sample

        input_ids = sample["input_ids"].clone()
        attention_mask = sample.get("attention_mask", torch.ones_like(input_ids))

        # Get actual sequence length (excluding padding)
        seq_len = attention_mask.sum().item()

        if seq_len <= 2:  # Don't modify very short sequences
            return sample

        # Convert to list for easier manipulation
        ids_list = input_ids[:seq_len].tolist()

        # Apply swaps
        i = 1  # Start at 1 to avoid swapping special tokens
        while i < len(ids_list) - 1:
            if random.random() < self.swap_prob:
                ids_list[i], ids_list[i + 1] = ids_list[i + 1], ids_list[i]
                i += 2  # Skip next token to avoid double-swapping
            else:
                i += 1

        # Apply deletions
        i = 1
        while i < len(ids_list) - 1:
            if random.random() < self.delete_prob:
                ids_list.pop(i)
            else:
                i += 1

        # Apply duplications
        i = 1
        while i < len(ids_list) - 1:
            if random.random() < self.duplicate_prob:
                ids_list.insert(i, ids_list[i])
                i += 2  # Skip the duplicated token
            else:
                i += 1

        # Convert back to tensor and pad if necessary
        new_len = len(ids_list)
        if new_len <= len(input_ids):
            input_ids[:new_len] = torch.tensor(ids_list, dtype=input_ids.dtype)
            input_ids[new_len:] = 0  # Pad with 0
            attention_mask[:new_len] = 1
            attention_mask[new_len:] = 0
        else:
            # Truncate if sequence grew too long
            input_ids = torch.tensor(ids_list[: len(input_ids)], dtype=input_ids.dtype)
            attention_mask = torch.ones_like(input_ids)

        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask

        return sample


class ComposeTransforms(CuneiformTransform):
    """Compose multiple transforms together."""

    def __init__(self, transforms: list):
        """
        Initialize composed transform.

        Args:
            transforms: List of transform instances
        """
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample
