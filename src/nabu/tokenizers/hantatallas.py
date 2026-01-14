"""Hantatallas stroke-based encoding converter for cuneiform signs.

This module provides conversion between Unicode cuneiform characters and
Hantatallas stroke-based encoding. Hantatallas uses a compositional approach
where signs are represented as combinations of basic strokes and composition
operators.

Basic strokes:
    h - horizontal
    v - vertical
    u - upward diagonal
    d - downward diagonal
    c - winkelhaken (angular hook)

Composition operators:
    [...] - horizontal stacking
    {...} - vertical stacking
    (...) - superposition (overlapping)
"""

from pathlib import Path
from typing import Optional, Dict, List, Set
import csv
import re


class HantatalasConverter:
    """
    Converter for Hantatallas stroke-based encoding system.

    Provides bidirectional conversion between Unicode cuneiform characters
    and Hantatallas stroke encodings using data from the HZL (Hittite
    Zeichenlexikon) database.
    """

    def __init__(
        self,
        hzl_dat_path: Optional[str] = None,
        unicode_csv_path: Optional[str] = None
    ):
        """
        Initialize the Hantatallas converter.

        Args:
            hzl_dat_path: Path to hzl.dat file from hantatallas repo
            unicode_csv_path: Path to unicode_cleaned.csv file
        """
        # Use default paths if not provided
        if hzl_dat_path is None:
            hzl_dat_path = Path.home() / "projects/cuneiform/hantatallas/hantatallas/data/hzl.dat"
        if unicode_csv_path is None:
            unicode_csv_path = Path.home() / "projects/cuneiform/hantatallas/hantatallas/data/unicode_cleaned.csv"

        self.hzl_dat_path = Path(hzl_dat_path)
        self.unicode_csv_path = Path(unicode_csv_path)

        # Storage for mappings
        self._hzl_to_encodings: Dict[str, List[str]] = {}  # HZL ID -> list of stroke encodings
        self._unicode_to_hzl: Dict[str, str] = {}  # Unicode char -> HZL ID
        self._unicode_to_encoding: Dict[str, str] = {}  # Unicode char -> primary encoding
        self._encoding_to_unicode: Dict[str, str] = {}  # encoding -> Unicode char (first match)

        # Load data
        self._load_hzl_data()
        self._load_unicode_mappings()
        self._build_unicode_to_encoding()

    def _load_hzl_data(self) -> None:
        """Parse hzl.dat file to extract HZL ID -> stroke encoding mappings."""
        if not self.hzl_dat_path.exists():
            raise FileNotFoundError(f"HZL data file not found: {self.hzl_dat_path}")

        with open(self.hzl_dat_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        current_id = None
        in_form_section = False

        for line in lines:
            # Count tabs to determine indentation level
            tabs = len(line) - len(line.lstrip('\t'))
            content = line.strip()

            if not content:
                continue

            # Level 0: Sign ID
            if tabs == 0 and content.isdigit():
                current_id = content
                in_form_section = False
                if current_id not in self._hzl_to_encodings:
                    self._hzl_to_encodings[current_id] = []

            # Level 1: Section header
            elif tabs == 1:
                in_form_section = (content == 'FORM')

            # Level 2: Content within section
            elif tabs == 2 and in_form_section and current_id:
                # Extract the stroke encoding (before any tags like "old", "new", "variant")
                parts = content.split()
                if parts:
                    encoding = parts[0]
                    # Basic validation: should contain stroke characters or composition operators
                    if any(c in encoding for c in 'hvudcHVUDC[]{}()'):
                        self._hzl_to_encodings[current_id].append(encoding)

    def _load_unicode_mappings(self) -> None:
        """Parse unicode_cleaned.csv to extract Unicode -> HZL ID mappings."""
        if not self.unicode_csv_path.exists():
            raise FileNotFoundError(f"Unicode CSV file not found: {self.unicode_csv_path}")

        with open(self.unicode_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                if len(row) < 7:
                    continue

                # Column indices:
                # 0: MesZL, 1: ŠL/HA, 2: aBZL, 3: HethZL, 4: Sign Name,
                # 5: Unicode Glyph, 6: Unicode Name

                hzl_id = row[3].strip()
                unicode_col = row[5].strip()

                # Skip if no HZL ID
                if not hzl_id or hzl_id == ' ':
                    continue

                # Extract Unicode characters from column
                # Format can be: "U+12000 𒀀" or "U+12000 & U+12001" or just "U+12000"
                unicode_chars = self._extract_unicode_chars(unicode_col)

                # Map the first Unicode character to this HZL ID
                # (some entries have multiple characters for ligatures)
                if unicode_chars:
                    first_char = unicode_chars[0]
                    # Only map if we have encoding data for this HZL ID
                    if hzl_id in self._hzl_to_encodings:
                        self._unicode_to_hzl[first_char] = hzl_id

    def _extract_unicode_chars(self, unicode_col: str) -> List[str]:
        """
        Extract Unicode characters from the Unicode column.

        Args:
            unicode_col: String like "U+12000 𒀀" or "U+12000 & U+12001"

        Returns:
            List of Unicode characters
        """
        chars = []

        # Find all actual cuneiform characters in the string
        for char in unicode_col:
            code_point = ord(char)
            # Cuneiform Unicode blocks
            if (0x12000 <= code_point <= 0x123FF or
                0x12400 <= code_point <= 0x1247F or
                0x12480 <= code_point <= 0x1254F):
                if char not in chars:  # Avoid duplicates
                    chars.append(char)

        return chars

    def _build_unicode_to_encoding(self) -> None:
        """Build final Unicode -> encoding mapping by combining HZL mappings."""
        for unicode_char, hzl_id in self._unicode_to_hzl.items():
            encodings = self._hzl_to_encodings.get(hzl_id, [])
            if encodings:
                # Use the first encoding as primary (usually the main form)
                primary_encoding = encodings[0]
                self._unicode_to_encoding[unicode_char] = primary_encoding

                # Build reverse mapping (first occurrence wins)
                if primary_encoding not in self._encoding_to_unicode:
                    self._encoding_to_unicode[primary_encoding] = unicode_char

    def unicode_to_hantatallas(self, unicode_char: str) -> Optional[str]:
        """
        Convert Unicode cuneiform character to Hantatallas encoding.

        Args:
            unicode_char: Unicode cuneiform character (e.g., "𒀀")

        Returns:
            Hantatallas stroke encoding (e.g., "(h2[00v0])") or None if not found
        """
        return self._unicode_to_encoding.get(unicode_char)

    def hantatallas_to_unicode(self, encoding: str) -> Optional[str]:
        """
        Convert Hantatallas encoding to Unicode character.

        Args:
            encoding: Hantatallas stroke encoding (e.g., "(h2[00v0])")

        Returns:
            Unicode cuneiform character or None if not found
        """
        return self._encoding_to_unicode.get(encoding)

    def text_to_hantatallas(self, text: str) -> List[Optional[str]]:
        """
        Convert text string to list of Hantatallas encodings.

        Args:
            text: String containing cuneiform characters

        Returns:
            List of Hantatallas encodings (None for unknown characters)
        """
        return [self.unicode_to_hantatallas(char) for char in text]

    def get_all_unicode_signs(self) -> Set[str]:
        """
        Get all Unicode signs supported by this converter.

        Returns:
            Set of Unicode cuneiform characters
        """
        return set(self._unicode_to_encoding.keys())

    def get_all_encodings(self) -> Set[str]:
        """
        Get all Hantatallas encodings in the database.

        Returns:
            Set of stroke encodings
        """
        return set(self._encoding_to_unicode.keys())

    def get_sign_info(self, unicode_char: str) -> Optional[Dict]:
        """
        Get detailed information about a sign.

        Args:
            unicode_char: Unicode cuneiform character

        Returns:
            Dictionary with sign information or None if not found
        """
        hzl_id = self._unicode_to_hzl.get(unicode_char)
        if hzl_id is None:
            return None

        encodings = self._hzl_to_encodings.get(hzl_id, [])

        return {
            'unicode': unicode_char,
            'hzl_id': hzl_id,
            'code_point': f"U+{ord(unicode_char):04X}",
            'primary_encoding': encodings[0] if encodings else None,
            'all_encodings': encodings,
            'encoding_count': len(encodings)
        }
