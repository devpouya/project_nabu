"""
Database for loading and querying generated cuneiform encodings.

Provides easy access to automatically generated encodings saved by generate_all_encodings.py
"""

from pathlib import Path
import json
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EncodingEntry:
    """Single encoding entry."""
    unicode: str
    code_point: str
    encoding: Optional[str]
    phonetic: str
    modsl: str
    generated_at: str
    method: str
    status: str
    error: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if this is a valid encoding."""
        return self.status == 'generated' and self.encoding is not None


class EncodingDatabase:
    """
    Database for cuneiform encodings.

    Loads and provides access to automatically generated encodings.
    """

    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize encoding database.

        Args:
            database_path: Path to generated_encodings.json
                          If None, uses default location
        """
        if database_path is None:
            # Default location
            database_path = Path(__file__).parent.parent.parent.parent / "data/encodings/generated_encodings.json"

        self.database_path = Path(database_path)
        self.encodings: Dict[str, EncodingEntry] = {}
        self.metadata: Dict = {}

        if self.database_path.exists():
            self.load()

    def load(self):
        """Load encodings from database file."""
        with open(self.database_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.metadata = {
            'format_version': data.get('format_version'),
            'generated_at': data.get('generated_at'),
            'total_signs': data.get('total_signs', 0),
            'successful': data.get('successful', 0),
            'failed': data.get('failed', 0),
        }

        # Load encodings
        encodings_data = data.get('encodings', {})
        for code_point, entry_data in encodings_data.items():
            entry = EncodingEntry(
                unicode=entry_data['unicode'],
                code_point=code_point,
                encoding=entry_data.get('encoding'),
                phonetic=entry_data.get('phonetic', ''),
                modsl=entry_data.get('modsl', ''),
                generated_at=entry_data.get('generated_at', ''),
                method=entry_data.get('method', 'unknown'),
                status=entry_data.get('status', 'unknown'),
                error=entry_data.get('error')
            )
            self.encodings[code_point] = entry

    def get_encoding(self, unicode_char: str) -> Optional[str]:
        """
        Get encoding for a Unicode character.

        Args:
            unicode_char: Unicode cuneiform character

        Returns:
            Hantatallas encoding string or None if not found
        """
        code_point = f"U+{ord(unicode_char):04X}"
        entry = self.encodings.get(code_point)
        return entry.encoding if entry and entry.is_valid() else None

    def get_entry(self, unicode_char: str) -> Optional[EncodingEntry]:
        """
        Get full encoding entry for a Unicode character.

        Args:
            unicode_char: Unicode cuneiform character

        Returns:
            EncodingEntry or None if not found
        """
        code_point = f"U+{ord(unicode_char):04X}"
        return self.encodings.get(code_point)

    def has_encoding(self, unicode_char: str) -> bool:
        """Check if a character has a valid encoding."""
        entry = self.get_entry(unicode_char)
        return entry is not None and entry.is_valid()

    def get_by_code_point(self, code_point: str) -> Optional[str]:
        """
        Get encoding by Unicode code point.

        Args:
            code_point: Unicode code point (e.g., "U+12401")

        Returns:
            Encoding string or None
        """
        entry = self.encodings.get(code_point)
        return entry.encoding if entry and entry.is_valid() else None

    def get_by_modsl(self, modsl_id: str) -> List[EncodingEntry]:
        """
        Get all entries with a given ModSL ID.

        Args:
            modsl_id: ModSL identifier

        Returns:
            List of matching entries
        """
        return [
            entry for entry in self.encodings.values()
            if entry.modsl == modsl_id
        ]

    def search_by_phonetic(self, phonetic: str) -> List[EncodingEntry]:
        """
        Search for signs by phonetic value.

        Args:
            phonetic: Phonetic value to search for (case-insensitive)

        Returns:
            List of matching entries
        """
        phonetic_lower = phonetic.lower()
        return [
            entry for entry in self.encodings.values()
            if phonetic_lower in entry.phonetic.lower()
        ]

    def get_all_valid(self) -> List[EncodingEntry]:
        """Get all entries with valid encodings."""
        return [
            entry for entry in self.encodings.values()
            if entry.is_valid()
        ]

    def get_all_failed(self) -> List[EncodingEntry]:
        """Get all entries that failed to generate."""
        return [
            entry for entry in self.encodings.values()
            if entry.status in ['failed', 'error']
        ]

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        total = len(self.encodings)
        valid = len(self.get_all_valid())
        failed = len(self.get_all_failed())

        return {
            'total': total,
            'valid': valid,
            'failed': failed,
            'coverage': valid / total if total > 0 else 0,
            'metadata': self.metadata
        }

    def print_statistics(self):
        """Print database statistics."""
        stats = self.get_statistics()

        print("="*70)
        print("ENCODING DATABASE STATISTICS")
        print("="*70)
        print(f"Database: {self.database_path}")
        print(f"Generated: {stats['metadata'].get('generated_at', 'Unknown')}")
        print()
        print(f"Total signs: {stats['total']}")
        print(f"Valid encodings: {stats['valid']} ({stats['coverage']*100:.1f}%)")
        print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")


def get_default_database() -> EncodingDatabase:
    """Get the default encoding database instance (singleton)."""
    if not hasattr(get_default_database, '_instance'):
        get_default_database._instance = EncodingDatabase()
    return get_default_database._instance


def unicode_to_encoding(unicode_char: str) -> Optional[str]:
    """
    Convenience function: Convert Unicode character to encoding.

    Args:
        unicode_char: Unicode cuneiform character

    Returns:
        Hantatallas encoding or None
    """
    db = get_default_database()
    return db.get_encoding(unicode_char)


if __name__ == "__main__":
    # Test database
    import sys

    db = EncodingDatabase()

    if not db.encodings:
        print("No encodings found. Run generate_all_encodings.py first.")
        sys.exit(1)

    # Print statistics
    db.print_statistics()

    # Test some lookups
    print("\n" + "="*70)
    print("SAMPLE LOOKUPS")
    print("="*70)

    test_chars = ['𒀸', '𒐁', '𒀭', '𒁹']

    for char in test_chars:
        encoding = db.get_encoding(char)
        entry = db.get_entry(char)

        if entry:
            status = "✓" if entry.is_valid() else "✗"
            print(f"{status} {char} ({entry.code_point}): {encoding or 'FAILED'}")
            if entry.phonetic:
                print(f"   Phonetic: {entry.phonetic}")
        else:
            print(f"✗ {char}: Not in database")
