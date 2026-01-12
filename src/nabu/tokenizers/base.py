"""Base tokenizer interface for cuneiform text."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""

    def __init__(self):
        """Initialize the tokenizer."""
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    @abstractmethod
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of cuneiform text strings
        """
        pass

    @abstractmethod
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input cuneiform text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        pass

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of texts.

        Args:
            texts: List of input texts
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        encoded = [
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=False,
                truncation=truncation,
            )
            for text in texts
        ]

        if padding and max_length is not None:
            padded = []
            attention_masks = []
            for seq in encoded:
                pad_len = max_length - len(seq)
                if pad_len > 0:
                    padded.append(seq + [self.pad_token_id] * pad_len)
                    attention_masks.append([1] * len(seq) + [0] * pad_len)
                else:
                    padded.append(seq)
                    attention_masks.append([1] * len(seq))
            return {"input_ids": padded, "attention_mask": attention_masks}

        return {"input_ids": encoded, "attention_mask": [[1] * len(seq) for seq in encoded]}

    def batch_decode(
        self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            token_ids_batch: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded text strings
        """
        return [self.decode(token_ids, skip_special_tokens) for token_ids in token_ids_batch]

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._vocab)

    @property
    def vocab(self) -> Dict[str, int]:
        """Return vocabulary dictionary."""
        return self._vocab

    def save_vocabulary(self, path: str) -> None:
        """
        Save vocabulary to file.

        Args:
            path: File path to save vocabulary
        """
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

    def load_vocabulary(self, path: str) -> None:
        """
        Load vocabulary from file.

        Args:
            path: File path to load vocabulary from
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            self._vocab = json.load(f)
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}
