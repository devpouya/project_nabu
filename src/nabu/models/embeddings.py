"""Embedding layers for cuneiform models."""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional scaling.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0,
        scale_embeddings: bool = True,
    ):
        """
        Initialize token embeddings.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            padding_idx: Index for padding token
            scale_embeddings: Whether to scale embeddings by sqrt(embedding_dim)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale_embeddings = scale_embeddings

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        # Initialize with normal distribution
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)
        if padding_idx is not None:
            nn.init.constant_(self.embedding.weight[padding_idx], 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        embeddings = self.embedding(x)
        if self.scale_embeddings:
            embeddings = embeddings * math.sqrt(self.embedding_dim)
        return embeddings


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as used in "Attention is All You Need".
    """

    def __init__(self, embedding_dim: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            embedding_dim: Dimension of embeddings
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, embedding_dim)

        Returns:
            Embeddings with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings as an alternative to sinusoidal encoding.
    """

    def __init__(self, max_len: int, embedding_dim: int, dropout: float = 0.1):
        """
        Initialize learned positional embeddings.

        Args:
            max_len: Maximum sequence length
            embedding_dim: Dimension of embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Initialize
        nn.init.normal_(self.positional_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings to input.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, embedding_dim)

        Returns:
            Embeddings with positional encoding added
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.positional_embedding(positions)
        x = x + pos_embeddings
        return self.dropout(x)


class HierarchicalEmbedding(nn.Module):
    """
    Hierarchical embedding that combines sign-level and stroke-level embeddings.
    Useful for hybrid tokenization approaches.
    """

    def __init__(
        self,
        sign_vocab_size: int,
        stroke_vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0,
        combination: str = "concat",
    ):
        """
        Initialize hierarchical embeddings.

        Args:
            sign_vocab_size: Size of sign vocabulary
            stroke_vocab_size: Size of stroke vocabulary
            embedding_dim: Dimension of final embeddings
            padding_idx: Index for padding token
            combination: How to combine sign and stroke embeddings ('concat', 'add', 'project')
        """
        super().__init__()
        self.combination = combination

        if combination == "concat":
            # Each level gets half the embedding dimension
            sign_dim = embedding_dim // 2
            stroke_dim = embedding_dim - sign_dim
            self.sign_embedding = TokenEmbedding(
                sign_vocab_size, sign_dim, padding_idx, scale_embeddings=False
            )
            self.stroke_embedding = TokenEmbedding(
                stroke_vocab_size, stroke_dim, padding_idx, scale_embeddings=False
            )
        elif combination in ["add", "project"]:
            # Both levels use full embedding dimension
            self.sign_embedding = TokenEmbedding(
                sign_vocab_size, embedding_dim, padding_idx, scale_embeddings=False
            )
            self.stroke_embedding = TokenEmbedding(
                stroke_vocab_size, embedding_dim, padding_idx, scale_embeddings=False
            )
            if combination == "project":
                self.projection = nn.Linear(embedding_dim * 2, embedding_dim)
        else:
            raise ValueError(f"Unknown combination method: {combination}")

    def forward(
        self, sign_ids: torch.Tensor, stroke_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with both sign and stroke IDs.

        Args:
            sign_ids: Sign token IDs of shape (batch_size, seq_len)
            stroke_ids: Stroke token IDs of shape (batch_size, seq_len)

        Returns:
            Combined embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        sign_emb = self.sign_embedding(sign_ids)
        stroke_emb = self.stroke_embedding(stroke_ids)

        if self.combination == "concat":
            return torch.cat([sign_emb, stroke_emb], dim=-1)
        elif self.combination == "add":
            return sign_emb + stroke_emb
        elif self.combination == "project":
            combined = torch.cat([sign_emb, stroke_emb], dim=-1)
            return self.projection(combined)
