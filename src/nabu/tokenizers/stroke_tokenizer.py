"""Stroke-level tokenizer for cuneiform text."""

from typing import List, Optional, Set
from .base import BaseTokenizer
from .paleocode import PaleoCodeConverter


class StrokeTokenizer(BaseTokenizer):
    """
    Tokenizes cuneiform text at the stroke level.
    Breaks PaleoCode strings into individual stroke primitives.

    Example:
        PaleoCode "a-a:a" -> strokes ['a', 'a', 'a']
    """

    def __init__(self, paleocode_dir: Optional[str] = None):
        """
        Initialize stroke tokenizer.

        Args:
            paleocode_dir: Path to PaleoCodage directory
        """
        super().__init__()
        self.converter = PaleoCodeConverter(paleocode_dir) if paleocode_dir else PaleoCodeConverter()

        # Initialize vocabulary with special tokens
        self._vocab = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
        }
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}

    def _parse_paleocode_to_strokes(self, paleocode: str) -> List[str]:
        """
        Parse PaleoCode string into individual strokes.

        PaleoCode format uses '-' and ':' as separators.
        Example: "a-a:a" -> ['a', 'a', 'a']

        Args:
            paleocode: PaleoCode string

        Returns:
            List of stroke strings
        """
        if not paleocode:
            return []

        # Split by both '-' and ':' separators
        strokes = []
        for part in paleocode.replace(":", "-").split("-"):
            part = part.strip()
            if part:
                strokes.append(part)
        return strokes

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from cuneiform texts by extracting all unique strokes.

        Args:
            texts: List of cuneiform text strings
        """
        unique_strokes: Set[str] = set()

        for text in texts:
            # Convert text to PaleoCodes
            paleocodes = self.converter.text_to_paleocode(text)

            # Extract strokes from each PaleoCode
            for paleocode in paleocodes:
                if paleocode:
                    strokes = self._parse_paleocode_to_strokes(paleocode)
                    unique_strokes.update(strokes)

        # Add strokes to vocabulary (starting after special tokens)
        next_id = len(self._vocab)
        for stroke in sorted(unique_strokes):
            if stroke not in self._vocab:
                self._vocab[stroke] = next_id
                self._reverse_vocab[next_id] = stroke
                next_id += 1

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """
        Encode cuneiform text to stroke token IDs.

        Args:
            text: Input cuneiform text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            List of token IDs
        """
        # Convert text to PaleoCodes
        paleocodes = self.converter.text_to_paleocode(text)

        # Convert PaleoCodes to strokes
        all_strokes = []
        for paleocode in paleocodes:
            if paleocode:
                strokes = self._parse_paleocode_to_strokes(paleocode)
                all_strokes.extend(strokes)

        # Convert strokes to IDs
        token_ids = [
            self._vocab.get(stroke, self.unk_token_id)
            for stroke in all_strokes
        ]

        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        # Apply truncation
        if truncation and max_length is not None:
            if add_special_tokens:
                # Keep BOS, truncate middle, keep EOS
                if len(token_ids) > max_length:
                    token_ids = [token_ids[0]] + token_ids[1:max_length-1] + [token_ids[-1]]
            else:
                token_ids = token_ids[:max_length]

        # Apply padding
        if padding and max_length is not None:
            pad_len = max_length - len(token_ids)
            if pad_len > 0:
                token_ids = token_ids + [self.pad_token_id] * pad_len

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to strokes (PaleoCode format).

        Note: This returns strokes joined by '-', not original Unicode text,
        as stroke-to-sign mapping is not unique.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded stroke sequence as string
        """
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}

        strokes = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            stroke = self._reverse_vocab.get(token_id, self.unk_token)
            strokes.append(stroke)

        return "-".join(strokes)

    def get_stroke_vocab(self) -> List[str]:
        """
        Get list of all stroke tokens (excluding special tokens).

        Returns:
            List of stroke strings
        """
        special_tokens = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        return [token for token in self._vocab.keys() if token not in special_tokens]
