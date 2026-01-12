"""Transformer-based models for cuneiform NLP."""

from typing import Dict, Optional
import torch
import torch.nn as nn
from .base import BaseModel
from .embeddings import TokenEmbedding, PositionalEncoding


class TransformerEncoder(BaseModel):
    """
    Transformer encoder model (BERT-style).
    Suitable for classification, sequence tagging, and masked language modeling.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_idx: int = 0,
    ):
        """
        Initialize transformer encoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            feedforward_dim: Dimension of feedforward network
            dropout: Dropout rate
            max_len: Maximum sequence length
            padding_idx: Padding token index
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size, embedding_dim, padding_idx
        )
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_len, dropout
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layer (for language modeling)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Dictionary with 'hidden_states' and 'logits'
        """
        # Embed tokens
        embeddings = self.token_embedding(input_ids)
        embeddings = self.positional_encoding(embeddings)

        # Create padding mask for transformer
        if attention_mask is not None:
            # PyTorch transformer expects True for positions to mask
            padding_mask = attention_mask == 0
        else:
            padding_mask = None

        # Apply transformer
        hidden_states = self.transformer(embeddings, src_key_padding_mask=padding_mask)

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return {
            "hidden_states": hidden_states,
            "logits": logits,
        }

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "vocab_size": self.token_embedding.embedding.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "num_layers": len(self.transformer.layers),
            "padding_idx": self.padding_idx,
        }


class TransformerDecoder(BaseModel):
    """
    Transformer decoder model (GPT-style).
    Suitable for autoregressive language modeling.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_idx: int = 0,
    ):
        """
        Initialize transformer decoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            feedforward_dim: Dimension of feedforward network
            dropout: Dropout rate
            max_len: Maximum sequence length
            padding_idx: Padding token index
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size, embedding_dim, padding_idx
        )
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_len, dropout
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output layer
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with causal masking.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Dictionary with 'hidden_states' and 'logits'
        """
        batch_size, seq_len = input_ids.size()

        # Embed tokens
        embeddings = self.token_embedding(input_ids)
        embeddings = self.positional_encoding(embeddings)

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        )

        # Create padding mask
        if attention_mask is not None:
            padding_mask = attention_mask == 0
        else:
            padding_mask = None

        # Create dummy memory for decoder (decoder-only, so memory = embeddings)
        memory = torch.zeros_like(embeddings[:, :1, :])  # Dummy memory

        # Apply transformer decoder
        hidden_states = self.transformer(
            embeddings,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return {
            "hidden_states": hidden_states,
            "logits": logits,
        }

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "vocab_size": self.token_embedding.embedding.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "num_layers": len(self.transformer.layers),
            "padding_idx": self.padding_idx,
        }


class TransformerEncoderDecoder(BaseModel):
    """
    Transformer encoder-decoder model (T5-style).
    Suitable for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_idx: int = 0,
    ):
        """
        Initialize transformer encoder-decoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            feedforward_dim: Dimension of feedforward network
            dropout: Dropout rate
            max_len: Maximum sequence length
            padding_idx: Padding token index
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Shared embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size, embedding_dim, padding_idx
        )
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_len, dropout
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Output layer
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        tgt_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            src_input_ids: Source token IDs of shape (batch_size, src_len)
            tgt_input_ids: Target token IDs of shape (batch_size, tgt_len)
            src_attention_mask: Source attention mask
            tgt_attention_mask: Target attention mask

        Returns:
            Dictionary with 'hidden_states' and 'logits'
        """
        # Embed tokens
        src_embeddings = self.positional_encoding(self.token_embedding(src_input_ids))
        tgt_embeddings = self.positional_encoding(self.token_embedding(tgt_input_ids))

        # Create causal mask for target
        tgt_len = tgt_input_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt_input_ids.device
        )

        # Create padding masks
        src_padding_mask = src_attention_mask == 0 if src_attention_mask is not None else None
        tgt_padding_mask = tgt_attention_mask == 0 if tgt_attention_mask is not None else None

        # Apply transformer
        hidden_states = self.transformer(
            src_embeddings,
            tgt_embeddings,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return {
            "hidden_states": hidden_states,
            "logits": logits,
        }

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "vocab_size": self.token_embedding.embedding.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "num_encoder_layers": self.transformer.encoder.num_layers,
            "num_decoder_layers": self.transformer.decoder.num_layers,
            "padding_idx": self.padding_idx,
        }
