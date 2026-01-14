"""
Encoding generation: Convert detected strokes and spatial structure to Hantatallas encoding.

Takes output from stroke detection and spatial analysis and generates
Hantatallas stroke-based encoding strings.
"""

from typing import List, Optional, Dict
from collections import Counter

try:
    from .stroke_detector import Stroke
    from .spatial_analyzer import SpatialGroup, SpatialAnalyzer
except ImportError:
    from stroke_detector import Stroke
    from spatial_analyzer import SpatialGroup, SpatialAnalyzer


class EncodingGenerator:
    """
    Generates Hantatallas encodings from stroke and spatial analysis.
    """

    def __init__(self):
        """Initialize encoding generator."""
        self.analyzer = SpatialAnalyzer()

    def generate(self, strokes: List[Stroke]) -> Optional[str]:
        """
        Generate Hantatallas encoding from strokes.

        Args:
            strokes: List of detected strokes

        Returns:
            Hantatallas encoding string or None if unable to encode
        """
        if not strokes:
            return None

        # Analyze spatial structure
        spatial_groups = self.analyzer.analyze(strokes)

        # Determine primary structure
        primary = self.analyzer.determine_primary_structure(spatial_groups)

        # Generate encoding based on structure
        if primary == 'repetition':
            return self._generate_repetition_encoding(spatial_groups['repetition'])
        elif primary == 'horizontal':
            return self._generate_horizontal_encoding(spatial_groups['horizontal'])
        elif primary == 'vertical':
            return self._generate_vertical_encoding(spatial_groups['vertical'])
        elif primary == 'superposed':
            return self._generate_superposition_encoding(spatial_groups['superposed'])
        else:
            # Simple case - single or few strokes
            return self._generate_simple_encoding(strokes)

    def _generate_simple_encoding(self, strokes: List[Stroke]) -> str:
        """
        Generate encoding for simple signs (single stroke or small number).

        Args:
            strokes: List of strokes

        Returns:
            Simple encoding string
        """
        if len(strokes) == 1:
            # Single stroke
            return strokes[0].stroke_type

        # Check for same type (could be repetition not caught by analyzer)
        stroke_types = [s.stroke_type for s in strokes]
        if len(set(stroke_types)) == 1:
            # All same type → try repetition notation
            count = len(strokes)
            if count <= 4:  # h2, h3, h4
                return f"{strokes[0].stroke_type}{count}"

        # Mixed types - default to horizontal stacking
        return '[' + ''.join(s.stroke_type for s in strokes) + ']'

    def _generate_repetition_encoding(self, repetition_groups: List[SpatialGroup]) -> str:
        """
        Generate encoding for repetitions (e.g., h2, h3).

        Args:
            repetition_groups: List of repetition groups

        Returns:
            Repetition encoding
        """
        if not repetition_groups:
            return None

        # Take the largest repetition group
        main_group = max(repetition_groups, key=lambda g: len(g.strokes))

        stroke_type = main_group.strokes[0].stroke_type
        count = len(main_group.strokes)

        if count <= 1:
            return stroke_type
        elif count <= 4:
            # Standard repetition notation: h2, h3, h4
            return f"{stroke_type}{count}"
        else:
            # Too many - probably not a simple repetition
            # Fall back to vertical/horizontal stacking
            return '{' + (stroke_type * count) + '}'

    def _generate_horizontal_encoding(self, horizontal_groups: List[SpatialGroup]) -> str:
        """
        Generate [...] encoding for horizontal stacking.

        Args:
            horizontal_groups: List of horizontal stacking groups

        Returns:
            Horizontal stacking encoding
        """
        if not horizontal_groups:
            return None

        # Take the largest horizontal group
        main_group = max(horizontal_groups, key=lambda g: len(g.strokes))

        # Sort strokes left-to-right
        sorted_strokes = sorted(main_group.strokes, key=lambda s: s.center_x)

        # Encode each stroke
        encoded_strokes = [s.stroke_type for s in sorted_strokes]

        # Check for repetitions within the group
        encoded_strokes = self._compress_repetitions(encoded_strokes)

        # Wrap in horizontal stacking operator
        return '[' + ''.join(encoded_strokes) + ']'

    def _generate_vertical_encoding(self, vertical_groups: List[SpatialGroup]) -> str:
        """
        Generate {...} encoding for vertical stacking.

        Args:
            vertical_groups: List of vertical stacking groups

        Returns:
            Vertical stacking encoding
        """
        if not vertical_groups:
            return None

        # Take the largest vertical group
        main_group = max(vertical_groups, key=lambda g: len(g.strokes))

        # Sort strokes top-to-bottom
        sorted_strokes = sorted(main_group.strokes, key=lambda s: s.center_y)

        # Encode each stroke
        encoded_strokes = [s.stroke_type for s in sorted_strokes]

        # Check for repetitions within the group
        encoded_strokes = self._compress_repetitions(encoded_strokes)

        # Wrap in vertical stacking operator
        return '{' + ''.join(encoded_strokes) + '}'

    def _generate_superposition_encoding(self, superposed_groups: List[SpatialGroup]) -> str:
        """
        Generate (...) encoding for superposition (overlapping strokes).

        Args:
            superposed_groups: List of superposition groups

        Returns:
            Superposition encoding
        """
        if not superposed_groups:
            return None

        # Take the largest superposed group
        main_group = max(superposed_groups, key=lambda g: len(g.strokes))

        # Encode each stroke
        encoded_strokes = [s.stroke_type for s in main_group.strokes]

        # Check for repetitions
        encoded_strokes = self._compress_repetitions(encoded_strokes)

        # Wrap in superposition operator
        return '(' + ''.join(encoded_strokes) + ')'

    def _compress_repetitions(self, encoded_strokes: List[str]) -> List[str]:
        """
        Compress consecutive repetitions in encoded strokes.

        Example: ['h', 'h', 'v'] → ['h2', 'v']

        Args:
            encoded_strokes: List of stroke encodings

        Returns:
            Compressed list
        """
        if not encoded_strokes:
            return []

        compressed = []
        current = encoded_strokes[0]
        count = 1

        for stroke in encoded_strokes[1:]:
            if stroke == current:
                count += 1
            else:
                # Add current group
                if count == 1:
                    compressed.append(current)
                elif count <= 4:
                    compressed.append(f"{current}{count}")
                else:
                    # Too many - keep separate
                    compressed.extend([current] * count)

                # Start new group
                current = stroke
                count = 1

        # Don't forget the last group
        if count == 1:
            compressed.append(current)
        elif count <= 4:
            compressed.append(f"{current}{count}")
        else:
            compressed.extend([current] * count)

        return compressed


def generate_encoding_from_unicode(unicode_char: str) -> Optional[str]:
    """
    Convenience function: Generate encoding directly from Unicode character.

    Args:
        unicode_char: Unicode cuneiform character

    Returns:
        Hantatallas encoding or None if unable
    """
    try:
        from .glyph_renderer import GlyphRenderer
        from .stroke_detector import StrokeDetector
    except ImportError:
        from glyph_renderer import GlyphRenderer
        from stroke_detector import StrokeDetector

    # Render
    renderer = GlyphRenderer(size=128)
    binary = renderer.render(unicode_char)

    # Detect strokes
    detector = StrokeDetector()
    strokes = detector.detect_strokes(binary)

    # Generate encoding
    generator = EncodingGenerator()
    encoding = generator.generate(strokes)

    return encoding


if __name__ == "__main__":
    # Test encoding generation
    from pathlib import Path

    test_chars = [
        ('𒀸', 'ASH', 'U+12038', 'h'),           # Expected: h
        ('𒐀', 'TWO_ASH', 'U+12400', 'h2'),      # Expected: h2
        ('𒐁', 'THREE_ASH', 'U+12401', 'h3'),    # Expected: h3
        ('𒀭', 'AN', 'U+1202D', '(h2[00v0])'),   # Expected: complex
    ]

    print("="*70)
    print("ENCODING GENERATION TEST")
    print("="*70)

    results = []

    for char, name, code, expected in test_chars:
        print(f"\n{name} ({code}): {char}")
        print(f"  Expected: {expected}")

        # Generate encoding
        generated = generate_encoding_from_unicode(char)

        print(f"  Generated: {generated}")

        if generated:
            match = (generated == expected) or (expected in generated) or (generated in expected)
            status = "✓ MATCH" if match else "✗ DIFFERENT"
            print(f"  Status: {status}")

            results.append({
                'name': name,
                'unicode': char,
                'code': code,
                'expected': expected,
                'generated': generated,
                'match': match
            })
        else:
            print("  Status: ✗ FAILED (no encoding generated)")
            results.append({
                'name': name,
                'unicode': char,
                'code': code,
                'expected': expected,
                'generated': None,
                'match': False
            })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    matches = sum(1 for r in results if r['match'])
    total = len(results)

    print(f"Matches: {matches}/{total} ({matches/total*100:.1f}%)")
    print()

    for r in results:
        status = "✓" if r['match'] else "✗"
        print(f"{status} {r['name']}: {r['generated']} (expected: {r['expected']})")
