"""
Spatial analysis using sweep-line algorithm for cuneiform signs.

Analyzes how strokes are arranged spatially to determine composition type:
- Horizontal stacking [...]
- Vertical stacking {...}
- Superposition (...)
- Simple repetition (h2, h3, etc.)
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import math

try:
    from .stroke_detector import Stroke
except ImportError:
    from stroke_detector import Stroke


@dataclass
class SpatialGroup:
    """Represents a group of spatially related strokes."""
    group_type: str  # 'horizontal', 'vertical', 'superposed', 'repetition'
    strokes: List[Stroke]
    bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


class SpatialAnalyzer:
    """
    Analyzes spatial relationships between strokes using sweep-line algorithm.
    """

    def __init__(self):
        """Initialize spatial analyzer."""
        # Thresholds for spatial relationships
        self.ALIGNMENT_THRESHOLD = 15  # pixels - strokes within this are "aligned"
        self.GAP_THRESHOLD = 20  # pixels - max gap for grouping
        self.OVERLAP_THRESHOLD = 0.3  # 30% overlap for superposition

    def analyze(self, strokes: List[Stroke]) -> Dict[str, List[SpatialGroup]]:
        """
        Analyze spatial structure of strokes.

        Args:
            strokes: List of detected strokes

        Returns:
            Dictionary with keys:
            - 'horizontal': List of horizontal stacking groups
            - 'vertical': List of vertical stacking groups
            - 'superposed': List of superposition groups
            - 'repetition': List of repetition groups (same type, close together)
        """
        if not strokes:
            return {
                'horizontal': [],
                'vertical': [],
                'superposed': [],
                'repetition': []
            }

        # Detect different composition types
        horizontal_groups = self._detect_horizontal_stacking(strokes)
        vertical_groups = self._detect_vertical_stacking(strokes)
        superposed_groups = self._detect_superposition(strokes)
        repetition_groups = self._detect_repetitions(strokes)

        return {
            'horizontal': horizontal_groups,
            'vertical': vertical_groups,
            'superposed': superposed_groups,
            'repetition': repetition_groups
        }

    def _detect_horizontal_stacking(self, strokes: List[Stroke]) -> List[SpatialGroup]:
        """
        Detect horizontal stacking [...] using sweep-line.

        Strokes are horizontally stacked if:
        - They are arranged left-to-right
        - They are vertically aligned (similar y-position)
        - They don't overlap in x-direction
        """
        # Sort by x-position (left to right)
        sorted_strokes = sorted(strokes, key=lambda s: s.center_x)

        groups = []
        current_group = []

        for stroke in sorted_strokes:
            if not current_group:
                current_group.append(stroke)
                continue

            prev_stroke = current_group[-1]

            # Check if horizontally adjacent and vertically aligned
            x_gap = stroke.center_x - prev_stroke.center_x
            y_diff = abs(stroke.center_y - prev_stroke.center_y)

            if (x_gap > 0 and  # To the right
                x_gap < self.GAP_THRESHOLD and  # Not too far
                y_diff < self.ALIGNMENT_THRESHOLD):  # Aligned vertically
                current_group.append(stroke)
            else:
                # Group ended
                if len(current_group) > 1:
                    groups.append(self._create_group('horizontal', current_group))
                current_group = [stroke]

        # Don't forget the last group
        if len(current_group) > 1:
            groups.append(self._create_group('horizontal', current_group))

        return groups

    def _detect_vertical_stacking(self, strokes: List[Stroke]) -> List[SpatialGroup]:
        """
        Detect vertical stacking {...} using sweep-line.

        Strokes are vertically stacked if:
        - They are arranged top-to-bottom
        - They are horizontally aligned (similar x-position)
        - They don't overlap in y-direction
        """
        # Sort by y-position (top to bottom)
        sorted_strokes = sorted(strokes, key=lambda s: s.center_y)

        groups = []
        current_group = []

        for stroke in sorted_strokes:
            if not current_group:
                current_group.append(stroke)
                continue

            prev_stroke = current_group[-1]

            # Check if vertically adjacent and horizontally aligned
            y_gap = stroke.center_y - prev_stroke.center_y
            x_diff = abs(stroke.center_x - prev_stroke.center_x)

            if (y_gap > 0 and  # Below
                y_gap < self.GAP_THRESHOLD and  # Not too far
                x_diff < self.ALIGNMENT_THRESHOLD):  # Aligned horizontally
                current_group.append(stroke)
            else:
                # Group ended
                if len(current_group) > 1:
                    groups.append(self._create_group('vertical', current_group))
                current_group = [stroke]

        # Don't forget the last group
        if len(current_group) > 1:
            groups.append(self._create_group('vertical', current_group))

        return groups

    def _detect_superposition(self, strokes: List[Stroke]) -> List[SpatialGroup]:
        """
        Detect overlapping strokes (superposition).

        Strokes are superposed if their bounding boxes overlap significantly.
        """
        groups = []
        used = set()

        for i, stroke1 in enumerate(strokes):
            if i in used:
                continue

            overlapping = [stroke1]

            for j, stroke2 in enumerate(strokes):
                if i == j or j in used:
                    continue

                if self._do_strokes_overlap(stroke1, stroke2):
                    overlapping.append(stroke2)
                    used.add(j)

            if len(overlapping) > 1:
                groups.append(self._create_group('superposed', overlapping))
                used.add(i)

        return groups

    def _detect_repetitions(self, strokes: List[Stroke]) -> List[SpatialGroup]:
        """
        Detect repetitions of the same stroke type (e.g., h2, h3).

        Repetitions are:
        - Same stroke type
        - Close together
        - Not part of other composition types
        """
        # Group by stroke type
        by_type = {}
        for stroke in strokes:
            if stroke.stroke_type not in by_type:
                by_type[stroke.stroke_type] = []
            by_type[stroke.stroke_type].append(stroke)

        groups = []

        for stroke_type, type_strokes in by_type.items():
            if len(type_strokes) < 2:
                continue

            # Check if they're close together (could be repetition)
            if self._are_strokes_close(type_strokes):
                groups.append(self._create_group('repetition', type_strokes))

        return groups

    def _do_strokes_overlap(self, s1: Stroke, s2: Stroke) -> bool:
        """
        Check if two strokes' bounding boxes overlap.

        Uses Intersection over Union (IoU) metric.
        """
        b1 = s1.bounds
        b2 = s2.bounds

        # Calculate intersection
        x_left = max(b1[0], b2[0])
        y_top = max(b1[1], b2[1])
        x_right = min(b1[2], b2[2])
        y_bottom = min(b1[3], b2[3])

        if x_right < x_left or y_bottom < y_top:
            return False  # No intersection

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

        # Calculate IoU
        union_area = area1 + area2 - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0

        return iou > self.OVERLAP_THRESHOLD

    def _are_strokes_close(self, strokes: List[Stroke]) -> bool:
        """
        Check if strokes are close together (for repetition detection).

        Strokes are close if the maximum pairwise distance is small.
        """
        if len(strokes) < 2:
            return False

        # Calculate average center
        avg_x = sum(s.center_x for s in strokes) / len(strokes)
        avg_y = sum(s.center_y for s in strokes) / len(strokes)

        # Check if all strokes are close to the average
        max_dist = 0
        for stroke in strokes:
            dist = math.sqrt(
                (stroke.center_x - avg_x)**2 +
                (stroke.center_y - avg_y)**2
            )
            max_dist = max(max_dist, dist)

        # Consider "close" if within a reasonable radius
        return max_dist < 40  # pixels

    def _create_group(self, group_type: str, strokes: List[Stroke]) -> SpatialGroup:
        """Create a SpatialGroup from strokes."""
        # Calculate bounding box
        min_x = min(s.bounds[0] for s in strokes)
        min_y = min(s.bounds[1] for s in strokes)
        max_x = max(s.bounds[2] for s in strokes)
        max_y = max(s.bounds[3] for s in strokes)

        return SpatialGroup(
            group_type=group_type,
            strokes=strokes,
            bounds=(min_x, min_y, max_x, max_y)
        )

    def determine_primary_structure(self, spatial_groups: Dict[str, List[SpatialGroup]]) -> Optional[str]:
        """
        Determine the primary (outermost) composition type.

        Priority:
        1. Superposition (highest priority - affects everything)
        2. Repetition (simple case)
        3. Vertical stacking
        4. Horizontal stacking

        Args:
            spatial_groups: Output from analyze()

        Returns:
            Primary structure type or None if unclear
        """
        # Check for superposition first
        if spatial_groups['superposed']:
            return 'superposed'

        # Check for repetition
        if spatial_groups['repetition']:
            # Only if that's the only structure
            if not spatial_groups['horizontal'] and not spatial_groups['vertical']:
                return 'repetition'

        # Check for vertical stacking
        if spatial_groups['vertical']:
            return 'vertical'

        # Check for horizontal stacking
        if spatial_groups['horizontal']:
            return 'horizontal'

        return None


if __name__ == "__main__":
    # Test spatial analysis
    from pathlib import Path
    try:
        from .glyph_renderer import GlyphRenderer
        from .stroke_detector import StrokeDetector
    except ImportError:
        from glyph_renderer import GlyphRenderer
        from stroke_detector import StrokeDetector

    renderer = GlyphRenderer(size=128)
    detector = StrokeDetector()
    analyzer = SpatialAnalyzer()

    test_chars = [
        ('𒀸', 'ASH', 'Single horizontal'),
        ('𒐁', 'THREE_ASH', 'Three horizontal (repetition)'),
        ('𒀭', 'AN', 'Complex composition'),
    ]

    for char, name, description in test_chars:
        print(f"\n{'='*60}")
        print(f"{name}: {char} - {description}")
        print('='*60)

        # Render and detect
        binary = renderer.render(char)
        strokes = detector.detect_strokes(binary)

        print(f"Detected {len(strokes)} strokes:")
        for i, stroke in enumerate(strokes, 1):
            print(f"  {i}. {stroke.stroke_type} at ({stroke.center_x:.0f}, {stroke.center_y:.0f})")

        # Analyze spatial structure
        spatial = analyzer.analyze(strokes)

        print(f"\nSpatial analysis:")
        for structure_type, groups in spatial.items():
            if groups:
                print(f"  {structure_type}: {len(groups)} group(s)")
                for i, group in enumerate(groups, 1):
                    print(f"    Group {i}: {len(group.strokes)} strokes")

        # Determine primary structure
        primary = analyzer.determine_primary_structure(spatial)
        print(f"\nPrimary structure: {primary}")
