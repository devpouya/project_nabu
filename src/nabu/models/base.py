"""Base model class for all neural network models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""

    def __init__(self):
        """Initialize the model."""
        super().__init__()

    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Returns:
            Dictionary containing model outputs
        """
        pass

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get the number of parameters in the model.

        Args:
            trainable_only: If True, only count trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str, **kwargs) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            **kwargs: Additional information to save
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            **kwargs,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce key matching

        Returns:
            Dictionary with additional checkpoint information
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Return other info from checkpoint
        info = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        return info

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Dictionary with model configuration
        """
        return {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseModel":
        """
        Create model from configuration dictionary.

        Args:
            config: Model configuration

        Returns:
            Model instance
        """
        return cls(**config)
