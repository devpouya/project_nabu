"""
Query the encoding database from command line.

Usage:
    python query_encoding.py 𒐁
    python query_encoding.py U+12401
    python query_encoding.py --phonetic ASH
    python query_encoding.py --modsl 002a
    python query_encoding.py --stats
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nabu.encoding.encoding_database import EncodingDatabase


def query_unicode(db: EncodingDatabase, unicode_char: str):
    """Query by Unicode character."""
    entry = db.get_entry(unicode_char)

    if entry:
        print(f"Unicode: {entry.unicode} ({entry.code_point})")
        print(f"Encoding: {entry.encoding or 'FAILED'}")
        print(f"Phonetic: {entry.phonetic or 'N/A'}")
        print(f"ModSL: {entry.modsl or 'N/A'}")
        print(f"Status: {entry.status}")
        print(f"Method: {entry.method}")
        print(f"Generated: {entry.generated_at}")

        if entry.error:
            print(f"Error: {entry.error}")
    else:
        print(f"Not found: {unicode_char}")


def query_code_point(db: EncodingDatabase, code_point: str):
    """Query by Unicode code point."""
    if not code_point.startswith('U+'):
        code_point = f"U+{code_point}"

    encoding = db.get_by_code_point(code_point)

    if encoding:
        entry = db.encodings[code_point]
        print(f"Unicode: {entry.unicode} ({code_point})")
        print(f"Encoding: {encoding}")
        print(f"Phonetic: {entry.phonetic or 'N/A'}")
        print(f"ModSL: {entry.modsl or 'N/A'}")
    else:
        print(f"Not found: {code_point}")


def query_phonetic(db: EncodingDatabase, phonetic: str):
    """Query by phonetic value."""
    results = db.search_by_phonetic(phonetic)

    if results:
        print(f"Found {len(results)} signs with phonetic value matching '{phonetic}':")
        print("="*70)

        for entry in results[:20]:  # Limit to 20
            status = "✓" if entry.is_valid() else "✗"
            print(f"{status} {entry.unicode} ({entry.code_point}): {entry.encoding or 'FAILED'}")
            print(f"   {entry.phonetic}")

        if len(results) > 20:
            print(f"\n... and {len(results) - 20} more")
    else:
        print(f"No signs found with phonetic value matching '{phonetic}'")


def query_modsl(db: EncodingDatabase, modsl_id: str):
    """Query by ModSL ID."""
    results = db.get_by_modsl(modsl_id)

    if results:
        print(f"Found {len(results)} signs with ModSL ID '{modsl_id}':")
        print("="*70)

        for entry in results:
            status = "✓" if entry.is_valid() else "✗"
            print(f"{status} {entry.unicode} ({entry.code_point}): {entry.encoding or 'FAILED'}")
            print(f"   {entry.phonetic}")
    else:
        print(f"No signs found with ModSL ID '{modsl_id}'")


def show_statistics(db: EncodingDatabase):
    """Show database statistics."""
    db.print_statistics()

    # Show some examples
    print("\n" + "="*70)
    print("SAMPLE ENTRIES")
    print("="*70)

    valid = db.get_all_valid()[:10]
    failed = db.get_all_failed()[:5]

    print("\nSuccessful encodings:")
    for entry in valid:
        print(f"  ✓ {entry.unicode} ({entry.code_point}): {entry.encoding}")

    if failed:
        print("\nFailed encodings:")
        for entry in failed:
            print(f"  ✗ {entry.unicode} ({entry.code_point}): {entry.error or 'No encoding'}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python query_encoding.py <unicode_char>")
        print("  python query_encoding.py U+12401")
        print("  python query_encoding.py --phonetic ASH")
        print("  python query_encoding.py --modsl 002a")
        print("  python query_encoding.py --stats")
        print()
        print("Examples:")
        print("  python query_encoding.py 𒐁")
        print("  python query_encoding.py U+12401")
        print("  python query_encoding.py --phonetic ASH")
        print("  python query_encoding.py --modsl 002a")
        print("  python query_encoding.py --stats")
        return 1

    # Load database
    db = EncodingDatabase()

    if not db.encodings:
        print("Error: No encodings found in database.")
        print("Run 'python scripts/generate_all_encodings.py' first.")
        return 1

    arg = sys.argv[1]

    if arg == "--stats":
        show_statistics(db)

    elif arg == "--phonetic":
        if len(sys.argv) < 3:
            print("Error: --phonetic requires a phonetic value")
            return 1
        query_phonetic(db, sys.argv[2])

    elif arg == "--modsl":
        if len(sys.argv) < 3:
            print("Error: --modsl requires a ModSL ID")
            return 1
        query_modsl(db, sys.argv[2])

    elif arg.startswith("U+") or arg.startswith("u+"):
        query_code_point(db, arg)

    else:
        # Direct Unicode character
        query_unicode(db, arg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
