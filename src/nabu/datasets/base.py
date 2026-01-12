"""Base dataset class for cuneiform datasets."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for cuneiform datasets."""

    def __init__(self):
        """Initialize the dataset."""
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing the sample data
        """
        pass

    @abstractmethod
    def load_data(self, path: str) -> None:
        """
        Load data from the specified path.

        Args:
            path: Path to the data file or directory
        """
        pass

    def get_sample(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index (alias for __getitem__).

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing the sample data
        """
        return self.__getitem__(idx)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            "num_samples": len(self),
        }
