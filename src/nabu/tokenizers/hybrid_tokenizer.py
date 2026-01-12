"""Hybrid tokenizer combining sign-level and stroke-level representations."""

from typing import List, Optional, Dict, Tuple
from .base import BaseTokenizer
from .paleocode import PaleoCodeConverter
from .sign_tokenizer import SignTokenizer
from .stroke_tokenizer import StrokeTokenizer


class HybridTokenizer(BaseTokenizer):
    """
    Hybrid tokenizer that provides both sign-level and stroke-level encodings.
    Useful for hierarchical models or multi-granularity experiments.

    This tokenizer can:
    1. Encode at both sign and stroke levels simultaneously
    2. Switch between sign and stroke modes
    3. Provide hierarchical representations (sign + constituent strokes)
    """

    def __init__(self, paleocode_dir: Optional[str] = None):
        """
        Initialize hybrid tokenizer.

        Args:
            paleocode_dir: Path to PaleoCodage directory
        """
        super().__init__()
        self.converter = PaleoCodeConverter(paleocode_dir) if paleocode_dir else PaleoCodeConverter()

        # Initialize both sign and stroke tokenizers
        self.sign_tokenizer = SignTokenizer(paleocode_dir)
        self.stroke_tokenizer = StrokeTokenizer(paleocode_dir)

        # Use sign tokenizer's vocab as primary
        self._vocab = self.sign_tokenizer.vocab
        self._reverse_vocab = self.sign_tokenizer._reverse_vocab

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabularies for both sign and stroke levels.

        Args:
            texts: List of cuneiform text strings
        """
        self.sign_tokenizer.build_vocab(texts)
        self.stroke_tokenizer.build_vocab(texts)

        # Update primary vocab reference
        self._vocab = self.sign_tokenizer.vocab
        self._reverse_vocab = self.sign_tokenizer._reverse_vocab

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """
        Encode text using sign-level tokenization (default mode).

        Args:
            text: Input cuneiform text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            List of token IDs (sign-level)
        """
        return self.sign_tokenizer.encode(
            text, add_special_tokens, max_length, padding, truncation
        )

    def encode_hierarchical(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Encode text at both sign and stroke levels with alignment information.

        Returns a dictionary containing:
        - sign_ids: Sign-level token IDs
        - stroke_ids: Stroke-level token IDs
        - alignment: Mapping from sign indices to stroke index ranges

        Args:
            text: Input cuneiform text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length for signs

        Returns:
            Dictionary with hierarchical encoding information
        """
        # Get sign-level encoding
        sign_ids = self.sign_tokenizer.encode(
            text, add_special_tokens=False, max_length=max_length, truncation=True
        )

        # Get stroke-level encoding
        stroke_ids = self.stroke_tokenizer.encode(
            text, add_special_tokens=False
        )

        # Build alignment between signs and strokes
        paleocodes = self.converter.text_to_paleocode(text)
        alignment = []
        stroke_offset = 0

        for paleocode in paleocodes:
            if paleocode:
                strokes = self.stroke_tokenizer._parse_paleocode_to_strokes(paleocode)
                num_strokes = len(strokes)
                alignment.append((stroke_offset, stroke_offset + num_strokes))
                stroke_offset += num_strokes
            else:
                alignment.append((stroke_offset, stroke_offset))

        # Add special tokens if requested
        if add_special_tokens:
            sign_ids = [self.bos_token_id] + sign_ids + [self.eos_token_id]
            stroke_ids = [self.bos_token_id] + stroke_ids + [self.eos_token_id]
            # Adjust alignment for BOS token
            alignment = [(start + 1, end + 1) for start, end in alignment]

        return {
            "sign_ids": sign_ids,
            "stroke_ids": stroke_ids,
            "alignment": alignment,
        }

    def encode_signs(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """
        Encode text at sign level.

        Args:
            text: Input cuneiform text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            List of sign token IDs
        """
        return self.sign_tokenizer.encode(
            text, add_special_tokens, max_length, padding, truncation
        )

    def encode_strokes(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """
        Encode text at stroke level.

        Args:
            text: Input cuneiform text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            List of stroke token IDs
        """
        return self.stroke_tokenizer.encode(
            text, add_special_tokens, max_length, padding, truncation
        )

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode sign-level token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded cuneiform text
        """
        return self.sign_tokenizer.decode(token_ids, skip_special_tokens)

    def decode_strokes(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode stroke-level token IDs back to stroke sequence.

        Args:
            token_ids: List of stroke token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded stroke sequence
        """
        return self.stroke_tokenizer.decode(token_ids, skip_special_tokens)

    @property
    def sign_vocab_size(self) -> int:
        """Return sign-level vocabulary size."""
        return self.sign_tokenizer.vocab_size

    @property
    def stroke_vocab_size(self) -> int:
        """Return stroke-level vocabulary size."""
        return self.stroke_tokenizer.vocab_size

    def get_tokenizer(self, mode: str) -> BaseTokenizer:
        """
        Get the appropriate tokenizer for the specified mode.

        Args:
            mode: Either 'sign' or 'stroke'

        Returns:
            The requested tokenizer

        Raises:
            ValueError: If mode is not 'sign' or 'stroke'
        """
        if mode == "sign":
            return self.sign_tokenizer
        elif mode == "stroke":
            return self.stroke_tokenizer
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sign' or 'stroke'.")
