"""RNN-based models for cuneiform NLP."""

from typing import Dict, Optional, Literal
import torch
import torch.nn as nn
from .base import BaseModel
from .embeddings import TokenEmbedding


class RNNEncoder(BaseModel):
    """
    RNN encoder model using LSTM or GRU.
    Suitable for classification and sequence encoding tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        padding_idx: int = 0,
    ):
        """
        Initialize RNN encoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension of RNN
            num_layers: Number of RNN layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional RNN
            rnn_type: Type of RNN ('lstm' or 'gru')
            padding_idx: Padding token index
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.padding_idx = padding_idx

        # Embedding
        self.embedding = TokenEmbedding(
            vocab_size, embedding_dim, padding_idx, scale_embeddings=False
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # RNN
        rnn_class = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output layer
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_projection = nn.Linear(output_dim, vocab_size)

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
            Dictionary with 'hidden_states', 'logits', and 'final_hidden'
        """
        # Embed tokens
        embeddings = self.embedding(input_ids)
        embeddings = self.embedding_dropout(embeddings)

        # Pack padded sequences for efficiency
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            embeddings = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )

        # Apply RNN
        hidden_states, final_hidden = self.rnn(embeddings)

        # Unpack sequences
        if attention_mask is not None:
            hidden_states, _ = nn.utils.rnn.pad_packed_sequence(
                hidden_states, batch_first=True
            )

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return {
            "hidden_states": hidden_states,
            "logits": logits,
            "final_hidden": final_hidden,
        }

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "vocab_size": self.embedding.embedding.num_embeddings,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "rnn_type": self.rnn_type,
            "padding_idx": self.padding_idx,
        }


class RNNDecoder(BaseModel):
    """
    RNN decoder model for autoregressive generation.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        padding_idx: int = 0,
    ):
        """
        Initialize RNN decoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension of RNN
            num_layers: Number of RNN layers
            dropout: Dropout rate
            rnn_type: Type of RNN ('lstm' or 'gru')
            padding_idx: Padding token index
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.padding_idx = padding_idx

        # Embedding
        self.embedding = TokenEmbedding(
            vocab_size, embedding_dim, padding_idx, scale_embeddings=False
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # RNN
        rnn_class = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            hidden: Previous hidden state (optional)

        Returns:
            Dictionary with 'hidden_states', 'logits', and 'final_hidden'
        """
        # Embed tokens
        embeddings = self.embedding(input_ids)
        embeddings = self.embedding_dropout(embeddings)

        # Apply RNN
        if hidden is not None:
            hidden_states, final_hidden = self.rnn(embeddings, hidden)
        else:
            hidden_states, final_hidden = self.rnn(embeddings)

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return {
            "hidden_states": hidden_states,
            "logits": logits,
            "final_hidden": final_hidden,
        }

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "vocab_size": self.embedding.embedding.num_embeddings,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "padding_idx": self.padding_idx,
        }


class RNNEncoderDecoder(BaseModel):
    """
    RNN encoder-decoder model for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional_encoder: bool = True,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        padding_idx: int = 0,
    ):
        """
        Initialize RNN encoder-decoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension of RNN
            num_layers: Number of RNN layers
            dropout: Dropout rate
            bidirectional_encoder: Whether encoder should be bidirectional
            rnn_type: Type of RNN ('lstm' or 'gru')
            padding_idx: Padding token index
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional_encoder = bidirectional_encoder
        self.rnn_type = rnn_type
        self.padding_idx = padding_idx

        # Shared embedding
        self.embedding = TokenEmbedding(
            vocab_size, embedding_dim, padding_idx, scale_embeddings=False
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # Encoder
        rnn_class = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.encoder = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional_encoder,
            batch_first=True,
        )

        # Bridge to connect bidirectional encoder to decoder
        if bidirectional_encoder:
            self.bridge = nn.Linear(hidden_dim * 2, hidden_dim)

        # Decoder
        self.decoder = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            src_input_ids: Source token IDs of shape (batch_size, src_len)
            tgt_input_ids: Target token IDs of shape (batch_size, tgt_len)
            src_attention_mask: Source attention mask

        Returns:
            Dictionary with 'logits'
        """
        # Encode source
        src_embeddings = self.embedding_dropout(self.embedding(src_input_ids))

        if src_attention_mask is not None:
            src_lengths = src_attention_mask.sum(dim=1).cpu()
            src_embeddings = nn.utils.rnn.pack_padded_sequence(
                src_embeddings, src_lengths, batch_first=True, enforce_sorted=False
            )

        _, encoder_hidden = self.encoder(src_embeddings)

        # Bridge hidden state if encoder is bidirectional
        if self.bidirectional_encoder:
            if self.rnn_type == "lstm":
                h, c = encoder_hidden
                h = self._bridge_hidden(h)
                c = self._bridge_hidden(c)
                decoder_hidden = (h, c)
            else:
                decoder_hidden = self._bridge_hidden(encoder_hidden)
        else:
            decoder_hidden = encoder_hidden

        # Decode target
        tgt_embeddings = self.embedding_dropout(self.embedding(tgt_input_ids))
        hidden_states, _ = self.decoder(tgt_embeddings, decoder_hidden)

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        return {
            "logits": logits,
        }

    def _bridge_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Bridge bidirectional encoder hidden states to decoder.

        Args:
            hidden: Hidden state of shape (num_layers * 2, batch_size, hidden_dim)

        Returns:
            Bridged hidden state of shape (num_layers, batch_size, hidden_dim)
        """
        # Concatenate forward and backward hidden states
        batch_size = hidden.size(1)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)

        # Project to decoder hidden dimension
        return self.bridge(hidden)

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "vocab_size": self.embedding.embedding.num_embeddings,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional_encoder": self.bidirectional_encoder,
            "rnn_type": self.rnn_type,
            "padding_idx": self.padding_idx,
        }
