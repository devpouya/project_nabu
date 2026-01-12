"""DataLoader builder functions for easy configuration."""

from typing import Optional, Dict, Any
from torch.utils.data import DataLoader, Dataset, random_split
from .collate import PaddingCollator, DynamicPaddingCollator, HierarchicalCollator


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    collator_type: str = "dynamic",
    pad_token_id: int = 0,
    max_length: Optional[int] = None,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader with appropriate collate function.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        collator_type: Type of collator ('fixed', 'dynamic', 'hierarchical')
        pad_token_id: Token ID for padding
        max_length: Maximum sequence length (for fixed collator)
        drop_last: Whether to drop incomplete last batch
        pin_memory: Whether to pin memory (faster GPU transfer)

    Returns:
        DataLoader instance
    """
    # Select collate function
    if collator_type == "fixed":
        if max_length is None:
            raise ValueError("max_length must be specified for fixed padding")
        collate_fn = PaddingCollator(pad_token_id=pad_token_id, max_length=max_length)
    elif collator_type == "dynamic":
        collate_fn = DynamicPaddingCollator(pad_token_id=pad_token_id)
    elif collator_type == "hierarchical":
        collate_fn = HierarchicalCollator(pad_token_id=pad_token_id)
    else:
        raise ValueError(f"Unknown collator type: {collator_type}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def build_dataloaders(
    dataset: Dataset,
    train_split: float = 0.8,
    val_split: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0,
    collator_type: str = "dynamic",
    pad_token_id: int = 0,
    max_length: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from a single dataset.

    Args:
        dataset: Dataset instance
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        batch_size: Batch size
        num_workers: Number of worker processes
        collator_type: Type of collator
        pad_token_id: Token ID for padding
        max_length: Maximum sequence length
        seed: Random seed for splitting

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Build DataLoaders
    dataloaders = {
        "train": build_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collator_type=collator_type,
            pad_token_id=pad_token_id,
            max_length=max_length,
            drop_last=True,
        ),
        "val": build_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collator_type=collator_type,
            pad_token_id=pad_token_id,
            max_length=max_length,
            drop_last=False,
        ),
        "test": build_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collator_type=collator_type,
            pad_token_id=pad_token_id,
            max_length=max_length,
            drop_last=False,
        ),
    }

    return dataloaders


# Need to import torch for random_split
import torch
