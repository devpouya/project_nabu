"""
Simple test script to verify encoding generation works.

Usage:
    python scripts/test_encoding_single.py
    python scripts/test_encoding_single.py 𒐁
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nabu.encoding import generate_encoding_from_unicode


def test_sign(unicode_char: str):
    """Test encoding generation for a single sign."""
    print(f"Testing sign: {unicode_char} (U+{ord(unicode_char):04X})")
    print("="*60)

    try:
        encoding = generate_encoding_from_unicode(unicode_char)

        if encoding:
            print(f"✓ Successfully generated encoding: {encoding}")
            return True
        else:
            print("✗ Failed to generate encoding")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_default_signs():
    """Test a few default signs."""
    test_signs = [
        ('𒀸', 'ASH', 'h'),
        ('𒐁', 'THREE_ASH', 'h3'),
        ('𒐀', 'TWO_ASH', 'h2'),
        ('𒁹', 'DIŠ', 'v'),
    ]

    print("Testing default signs...")
    print("="*60)

    results = []
    for char, name, expected in test_signs:
        print(f"\n{name} ({char}):")
        print(f"  Expected: {expected}")

        encoding = generate_encoding_from_unicode(char)

        if encoding:
            match = encoding == expected
            status = "✓ MATCH" if match else "✗ DIFFERENT"
            print(f"  Generated: {encoding}")
            print(f"  {status}")
            results.append(match)
        else:
            print(f"  Generated: FAILED")
            print(f"  ✗ FAILED")
            results.append(False)

    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} passed")

    return all(results)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Test specific sign
        unicode_char = sys.argv[1]
        success = test_sign(unicode_char)
    else:
        # Test default signs
        success = test_default_signs()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
