"""PaleoCode converter wrapper using the bundled PaleoCode module."""

from pathlib import Path
from typing import Optional, Dict, Any, List

# Import from the bundled paleocode module
from nabu.paleocode import CuneiformConverter


class PaleoCodeConverter:
    """
    Wrapper for CuneiformConverter from bundled PaleoCode module.
    Provides caching and batch conversion utilities.
    """

    def __init__(self, paleocode_dir: Optional[str] = None):
        """
        Initialize the converter.

        Args:
            paleocode_dir: Optional path to alternative PaleoCode directory.
                          If None, uses the bundled paleocode data.
        """
        if paleocode_dir is None:
            # Use bundled paleocode data
            module_dir = Path(__file__).parent.parent / "paleocode"
            json_path = module_dir / "paleocodes.json"
        else:
            # Use custom paleocode directory
            json_path = Path(paleocode_dir) / "paleocodes.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"PaleoCode data file not found at {json_path}. "
                f"Make sure paleocodes.json is available."
            )

        # Initialize the converter
        self.converter = CuneiformConverter(str(json_path))

        # Build lookup dictionaries for faster access
        self._unicode_to_paleocode: Dict[str, str] = {}
        self._paleocode_to_unicode: Dict[str, str] = {}
        self._build_lookup_dicts()

    def _build_lookup_dicts(self) -> None:
        """Build lookup dictionaries for bidirectional conversion."""
        for entry in self.converter.get_all_signs():
            sign = entry.get("Sign")
            paleocode = entry.get("PaleoCode")
            if sign and paleocode:
                self._unicode_to_paleocode[sign] = paleocode
                # Handle multiple signs mapping to same paleocode
                if paleocode not in self._paleocode_to_unicode:
                    self._paleocode_to_unicode[paleocode] = sign

    def unicode_to_paleocode(self, unicode_char: str) -> Optional[str]:
        """
        Convert Unicode cuneiform character to PaleoCode.

        Args:
            unicode_char: Unicode cuneiform character (e.g., "𒀀")

        Returns:
            PaleoCode string (e.g., "a-a:a") or None if not found
        """
        return self._unicode_to_paleocode.get(unicode_char)

    def paleocode_to_unicode(self, paleocode: str) -> Optional[str]:
        """
        Convert PaleoCode to Unicode cuneiform character.

        Args:
            paleocode: PaleoCode string (e.g., "a-a:a")

        Returns:
            Unicode cuneiform character or None if not found
        """
        return self._paleocode_to_unicode.get(paleocode)

    def text_to_paleocode(self, text: str) -> List[Optional[str]]:
        """
        Convert a text string to list of PaleoCodes.

        Args:
            text: String containing cuneiform characters

        Returns:
            List of PaleoCode strings (None for unknown characters)
        """
        return [self.unicode_to_paleocode(char) for char in text]

    def paleocode_to_text(self, paleocodes: List[str]) -> str:
        """
        Convert list of PaleoCodes back to text.

        Args:
            paleocodes: List of PaleoCode strings

        Returns:
            Text string with cuneiform characters
        """
        chars = []
        for pc in paleocodes:
            char = self.paleocode_to_unicode(pc)
            if char:
                chars.append(char)
        return "".join(chars)

    def get_sign_info(self, unicode_char: str) -> Optional[Dict[str, Any]]:
        """
        Get complete sign information.

        Args:
            unicode_char: Unicode cuneiform character

        Returns:
            Dictionary with sign information or None
        """
        return self.converter.unicode_to_sign_info(unicode_char)

    def get_all_paleocodes(self) -> List[str]:
        """
        Get list of all unique PaleoCodes in the database.

        Returns:
            List of PaleoCode strings
        """
        return list(self._paleocode_to_unicode.keys())

    def get_all_unicode_signs(self) -> List[str]:
        """
        Get list of all Unicode cuneiform signs in the database.

        Returns:
            List of Unicode characters
        """
        return list(self._unicode_to_paleocode.keys())

    @property
    def num_signs(self) -> int:
        """Return total number of signs in the database."""
        return len(self._unicode_to_paleocode)
