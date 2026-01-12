"""Sign-level tokenizer for cuneiform text."""

from typing import List, Optional, Set
from .base import BaseTokenizer
from .paleocode import PaleoCodeConverter


class SignTokenizer(BaseTokenizer):
    """
    Tokenizes cuneiform text at the sign level.
    Each complete PaleoCode is treated as a single token.

    Example:
        Unicode "𒀀" -> PaleoCode "a-a:a" -> token ID
    """

    def __init__(self, paleocode_dir: Optional[str] = None):
        """
        Initialize sign tokenizer.

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

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from cuneiform texts using complete PaleoCodes.

        Args:
            texts: List of cuneiform text strings
        """
        unique_paleocodes: Set[str] = set()

        for text in texts:
            # Convert text to PaleoCodes
            paleocodes = self.converter.text_to_paleocode(text)

            # Collect all unique PaleoCodes
            for paleocode in paleocodes:
                if paleocode:
                    unique_paleocodes.add(paleocode)

        # Add PaleoCodes to vocabulary (starting after special tokens)
        next_id = len(self._vocab)
        for paleocode in sorted(unique_paleocodes):
            if paleocode not in self._vocab:
                self._vocab[paleocode] = next_id
                self._reverse_vocab[next_id] = paleocode
                next_id += 1

    def build_vocab_from_all_signs(self) -> None:
        """
        Build vocabulary from all signs in the PaleoCode database.
        Useful for ensuring complete coverage without needing training data.
        """
        all_paleocodes = self.converter.get_all_paleocodes()

        # Add all PaleoCodes to vocabulary
        next_id = len(self._vocab)
        for paleocode in sorted(all_paleocodes):
            if paleocode not in self._vocab:
                self._vocab[paleocode] = next_id
                self._reverse_vocab[next_id] = paleocode
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
        Encode cuneiform text to sign token IDs.

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

        # Convert PaleoCodes to token IDs
        token_ids = []
        for paleocode in paleocodes:
            if paleocode:
                token_id = self._vocab.get(paleocode, self.unk_token_id)
                token_ids.append(token_id)

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
        Decode token IDs back to cuneiform text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded cuneiform text
        """
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}

        # Convert token IDs to PaleoCodes
        paleocodes = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            paleocode = self._reverse_vocab.get(token_id)
            if paleocode and paleocode != self.unk_token:
                paleocodes.append(paleocode)

        # Convert PaleoCodes back to Unicode text
        return self.converter.paleocode_to_text(paleocodes)

    def encode_paleocode(self, paleocode: str) -> int:
        """
        Encode a single PaleoCode to token ID.

        Args:
            paleocode: PaleoCode string

        Returns:
            Token ID
        """
        return self._vocab.get(paleocode, self.unk_token_id)

    def decode_paleocode(self, token_id: int) -> Optional[str]:
        """
        Decode a token ID to PaleoCode.

        Args:
            token_id: Token ID

        Returns:
            PaleoCode string or None
        """
        return self._reverse_vocab.get(token_id)

    def get_sign_vocab(self) -> List[str]:
        """
        Get list of all sign tokens (PaleoCodes) excluding special tokens.

        Returns:
            List of PaleoCode strings
        """
        special_tokens = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        return [token for token in self._vocab.keys() if token not in special_tokens]
