#!/usr/bin/env python3
"""
Unicode to PaleoCode Converter

This module provides functions to convert Unicode cuneiform characters
to their PaleoCode encoding representations.
"""

import json
from typing import Optional, Dict, Any


class CuneiformConverter:
    """Converter for Unicode cuneiform characters to PaleoCode encodings."""

    def __init__(self, json_path: str = 'paleocodes.json'):
        """
        Initialize the converter by loading the paleocodestore data.

        Args:
            json_path: Path to the JSON file containing paleocodestore data
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.paleocodestore = json.load(f)

        # Build a lookup dictionary for faster access
        self._unicode_to_entry = {
            entry['Sign']: entry
            for entry in self.paleocodestore
            if entry.get('Sign')
        }

    def unicode_to_paleocode(self, unicode_char: str) -> Optional[str]:
        """
        Convert a Unicode cuneiform character to its PaleoCode encoding.

        Args:
            unicode_char: The Unicode cuneiform character (e.g., "𒀀")

        Returns:
            The PaleoCode encoding (e.g., "a-a:a") or None if not found

        Examples:
            >>> converter = CuneiformConverter()
            >>> converter.unicode_to_paleocode("𒀀")
            'a-a:a'
            >>> converter.unicode_to_paleocode("𒋰")
            'b:b'
            >>> converter.unicode_to_paleocode("invalid")
            None
        """
        entry = self._unicode_to_entry.get(unicode_char)
        return entry.get('PaleoCode') if entry else None

    def unicode_to_sign_info(self, unicode_char: str) -> Optional[Dict[str, Any]]:
        """
        Get complete sign information for a Unicode cuneiform character.

        Args:
            unicode_char: The Unicode cuneiform character (e.g., "𒀀")

        Returns:
            Dictionary containing all sign information (PaleoCode, transliteration,
            Borger number, etc.) or None if not found

        Examples:
            >>> converter = CuneiformConverter()
            >>> info = converter.unicode_to_sign_info("𒀀")
            >>> info['Transliteration']
            'A'
            >>> info['PaleoCode']
            'a-a:a'
        """
        return self._unicode_to_entry.get(unicode_char)

    def transliteration_to_paleocode(self, transliteration: str) -> Optional[str]:
        """
        Convert a transliteration to its PaleoCode encoding.

        Args:
            transliteration: The transliteration (e.g., "A", "TAB", "LAL")

        Returns:
            The PaleoCode encoding or None if not found

        Examples:
            >>> converter = CuneiformConverter()
            >>> converter.transliteration_to_paleocode("A")
            'a-a:a'
            >>> converter.transliteration_to_paleocode("TAB")
            'b:b'
        """
        for entry in self.paleocodestore:
            if entry.get('Transliteration') == transliteration:
                return entry.get('PaleoCode')
        return None

    def codepoint_to_paleocode(self, codepoint: str) -> Optional[str]:
        """
        Convert a Unicode code point to its PaleoCode encoding.

        Args:
            codepoint: The Unicode code point (e.g., "U+12000")

        Returns:
            The PaleoCode encoding or None if not found

        Examples:
            >>> converter = CuneiformConverter()
            >>> converter.codepoint_to_paleocode("U+12000")
            'a-a:a'
        """
        for entry in self.paleocodestore:
            if entry.get('Code point') == codepoint:
                return entry.get('PaleoCode')
        return None

    def get_all_signs(self) -> list:
        """
        Get all cuneiform signs in the database.

        Returns:
            List of all sign entries
        """
        return self.paleocodestore


# Convenience function for simple usage
def unicode_to_paleocode(unicode_char: str, json_path: str = 'paleocodes.json') -> Optional[str]:
    """
    Standalone function to convert Unicode cuneiform to PaleoCode.

    Args:
        unicode_char: The Unicode cuneiform character (e.g., "𒀀")
        json_path: Path to the JSON file containing paleocodestore data

    Returns:
        The PaleoCode encoding (e.g., "a-a:a") or None if not found

    Examples:
        >>> unicode_to_paleocode("𒀀")
        'a-a:a'
        >>> unicode_to_paleocode("𒋰")
        'b:b'
    """
    converter = CuneiformConverter(json_path)
    return converter.unicode_to_paleocode(unicode_char)


if __name__ == '__main__':
    # Example usage
    converter = CuneiformConverter()

    # Test some conversions
    test_chars = ["𒀀", "𒋰", "𒇲", "𒀂"]

    print("Unicode to PaleoCode Conversion Examples:")
    print("-" * 60)

    for char in test_chars:
        info = converter.unicode_to_sign_info(char)
        if info:
            print(f"Character: {char}")
            print(f"  Unicode:        {info.get('Code point', 'N/A')}")
            print(f"  Transliteration: {info.get('Transliteration', 'N/A')}")
            print(f"  PaleoCode:      {info.get('PaleoCode', 'N/A')}")
            print(f"  Borger:         {info.get('Borger', 'N/A')}")
            if info.get('Comment'):
                print(f"  Comment:        {info['Comment']}")
            print()

    # Test transliteration lookup
    print("\nTransliteration to PaleoCode:")
    print("-" * 60)
    transliterations = ["A", "TAB", "LAL", "U2"]
    for trans in transliterations:
        paleocode = converter.transliteration_to_paleocode(trans)
        print(f"{trans:10} -> {paleocode}")

    print(f"\nTotal signs in database: {len(converter.get_all_signs())}")
