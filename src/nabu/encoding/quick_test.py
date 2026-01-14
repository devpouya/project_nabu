"""
Quick test utility for encoding generation.

Usage:
    python quick_test.py 𒐁
    python quick_test.py U+12401
    python quick_test.py --list  # Test a predefined list
"""

import sys
from pathlib import Path
from typing import Optional

try:
    from .glyph_renderer import GlyphRenderer
    from .stroke_detector import StrokeDetector, visualize_strokes
    from .spatial_analyzer import SpatialAnalyzer
    from .encoding_generator import EncodingGenerator
except ImportError:
    from glyph_renderer import GlyphRenderer
    from stroke_detector import StrokeDetector, visualize_strokes
    from spatial_analyzer import SpatialAnalyzer
    from encoding_generator import EncodingGenerator


def test_single_sign(unicode_char: str, save_viz: bool = True) -> Optional[str]:
    """
    Quick test of a single sign.

    Args:
        unicode_char: Unicode character to test
        save_viz: Whether to save visualization

    Returns:
        Generated encoding or None
    """
    try:
        # Initialize components
        renderer = GlyphRenderer(size=128)
        detector = StrokeDetector()
        analyzer = SpatialAnalyzer()
        generator = EncodingGenerator()

        # Run pipeline
        print(f"Testing: {unicode_char} (U+{ord(unicode_char):04X})")
        print("="*60)

        # Step 1: Render
        print("1. Rendering glyph...")
        binary = renderer.render(unicode_char)
        print(f"   Image size: {binary.shape}, Non-zero pixels: {binary.sum()}")

        # Step 2: Detect strokes
        print("2. Detecting strokes...")
        strokes = detector.detect_strokes(binary)
        print(f"   Detected {len(strokes)} strokes:")
        for i, stroke in enumerate(strokes, 1):
            print(f"     {i}. {stroke.stroke_type} - "
                  f"angle: {stroke.angle:.1f}°, "
                  f"length: {stroke.length:.1f}px, "
                  f"center: ({stroke.center_x:.1f}, {stroke.center_y:.1f})")

        # Step 3: Spatial analysis
        print("3. Analyzing spatial structure...")
        spatial_groups = analyzer.analyze(strokes)
        primary = analyzer.determine_primary_structure(spatial_groups)
        print(f"   Primary structure: {primary or 'simple'}")

        for group_type, groups in spatial_groups.items():
            if groups:
                print(f"   {group_type}: {len(groups)} group(s)")
                for i, group in enumerate(groups, 1):
                    print(f"     Group {i}: {len(group.strokes)} strokes")

        # Step 4: Generate encoding
        print("4. Generating encoding...")
        encoding = generator.generate(strokes)
        print(f"   ✓ Encoding: {encoding or 'FAILED'}")

        # Save visualization
        if save_viz and encoding:
            output_dir = Path(__file__).parent.parent.parent.parent / "results/quick_tests"
            output_dir.mkdir(parents=True, exist_ok=True)

            code_point = f"U+{ord(unicode_char):04X}"
            viz_path = output_dir / f"{code_point}_{encoding.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('(', '').replace(')', '')}.png"
            visualize_strokes(binary, strokes, str(viz_path))
            print(f"\nVisualization saved to: {viz_path}")

        return encoding

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_unicode_list():
    """Test a predefined list of signs."""
    test_signs = [
        ('𒀸', 'U+12038', 'ASH', 'h'),
        ('𒁹', 'U+12079', 'DIŠ', 'v'),
        ('𒐀', 'U+12400', 'TWO_ASH', 'h2'),
        ('𒐁', 'U+12401', 'THREE_ASH', 'h3'),
        ('𒐂', 'U+12402', 'FOUR_ASH', 'h4'),
        ('𒀀', 'U+12000', 'A', None),
        ('𒀭', 'U+1202D', 'AN', None),
        ('𒂍', 'U+1208D', 'E2', None),
    ]

    print("="*70)
    print("QUICK TEST - MULTIPLE SIGNS")
    print("="*70)
    print()

    results = []
    for char, code, name, expected in test_signs:
        print(f"\n{name} ({code}): {char}")
        if expected:
            print(f"Expected: {expected}")

        encoding = test_single_sign(char, save_viz=False)

        match = False
        if expected and encoding:
            match = (encoding == expected or expected in encoding or encoding in expected)

        status = "✓" if match or (encoding and not expected) else "✗"
        print(f"{status} Result: {encoding or 'FAILED'}")

        results.append({
            'name': name,
            'char': char,
            'expected': expected,
            'generated': encoding,
            'match': match
        })

        print("-"*60)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful = sum(1 for r in results if r['generated'])
    total = len(results)
    print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")

    with_expected = [r for r in results if r['expected']]
    matches = sum(1 for r in with_expected if r['match'])
    if with_expected:
        print(f"Matches: {matches}/{len(with_expected)} ({matches/len(with_expected)*100:.1f}%)")

    print("\nResults:")
    for r in results:
        status = "✓" if (r['match'] or (r['generated'] and not r['expected'])) else "✗"
        print(f"  {status} {r['name']:15} {r['generated'] or 'FAILED':15} "
              f"(expected: {r['expected'] or 'N/A'})")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quick_test.py <unicode_char>")
        print("  python quick_test.py U+12401")
        print("  python quick_test.py --list")
        print()
        print("Examples:")
        print("  python quick_test.py 𒐁")
        print("  python quick_test.py U+12401")
        print("  python quick_test.py --list")
        return

    arg = sys.argv[1]

    if arg == "--list":
        test_unicode_list()
    elif arg.startswith("U+") or arg.startswith("u+"):
        # Parse Unicode code point
        code_point = int(arg[2:], 16)
        unicode_char = chr(code_point)
        test_single_sign(unicode_char)
    else:
        # Direct Unicode character
        test_single_sign(arg)


if __name__ == "__main__":
    main()
