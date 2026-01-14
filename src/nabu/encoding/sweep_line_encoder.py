"""
Sweep-line shift-reduce encoder for cuneiform signs.

Encodes cuneiform signs as sequences of strokes using a sweep-line algorithm
combined with a shift-reduce parser:

Stroke vocabulary:
- h: horizontal stroke
- v: vertical stroke
- u: upward diagonal
- d: downward diagonal
- c: winkelhaken (angular hook)

Algorithm:
1. Sweep line moves left to right across the sign
2. At the origin (left edge) of each stroke: SHIFT (push to stack)
3. At the end (right edge) of each stroke: POP from stack
4. If a new stroke is contained by or attached to a stroke on the stack: REDUCE
   (create dependency edge, represented with parentheses in output)
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
import math

try:
    from .stroke_detector import Stroke
except ImportError:
    from stroke_detector import Stroke


class EventType(Enum):
    """Types of events in the sweep line algorithm."""
    SHIFT = 1  # Stroke begins (push to stack)
    POP = 2    # Stroke ends (pop from stack)


@dataclass
class SweepEvent:
    """An event in the sweep line algorithm."""
    x: float           # X coordinate where event occurs
    event_type: EventType
    stroke: Stroke
    stroke_id: int     # Unique identifier for the stroke


@dataclass
class StrokeNode:
    """A node in the dependency tree."""
    stroke: Stroke
    stroke_id: int
    children: List['StrokeNode'] = field(default_factory=list)
    parent: Optional['StrokeNode'] = None

    @property
    def stroke_type(self) -> str:
        return self.stroke.stroke_type


class SweepLineEncoder:
    """
    Encodes cuneiform signs using sweep-line + shift-reduce parsing.

    The algorithm treats each sign as a set of line segments on a 2D plane.
    Dependencies between strokes are determined by physical attachment:
    - A stroke must physically touch or connect to another stroke
    - Parallel strokes (like multiple horizontals) are NOT attached
    """

    def __init__(self):
        """Initialize the encoder."""
        # Threshold for physical attachment (endpoint touching)
        self.ATTACHMENT_THRESHOLD = 8  # pixels - endpoints must be this close

    def encode(self, strokes: List[Stroke]) -> str:
        """
        Encode a list of strokes into a string representation.

        Args:
            strokes: List of detected strokes from StrokeDetector

        Returns:
            Encoded string representation of the sign
        """
        if not strokes:
            return ""

        # Deduplicate strokes (merge very similar strokes that are likely the same)
        strokes = self._deduplicate_strokes(strokes)

        if len(strokes) == 1:
            return strokes[0].stroke_type

        # Create events for sweep line
        events = self._create_events(strokes)

        # Run sweep line algorithm to build dependency tree
        roots = self._sweep_line_parse(events)

        # Generate encoding from dependency tree
        encoding = self._generate_encoding(roots)

        return encoding

    def _deduplicate_strokes(self, strokes: List[Stroke]) -> List[Stroke]:
        """
        Merge strokes that are essentially duplicates.

        The Hough transform can detect multiple line segments for a single
        thick stroke. This merges strokes that:
        - Are the same type
        - Are parallel
        - Have very close centers

        Args:
            strokes: List of strokes

        Returns:
            Deduplicated list of strokes
        """
        if len(strokes) <= 1:
            return strokes

        # Group strokes by type
        by_type: Dict[str, List[Stroke]] = {}
        for stroke in strokes:
            if stroke.stroke_type not in by_type:
                by_type[stroke.stroke_type] = []
            by_type[stroke.stroke_type].append(stroke)

        deduplicated = []

        for stroke_type, type_strokes in by_type.items():
            # Cluster similar strokes
            clusters = self._cluster_strokes(type_strokes)
            for cluster in clusters:
                # Take the stroke with median length from each cluster
                cluster.sort(key=lambda s: s.length)
                median_idx = len(cluster) // 2
                deduplicated.append(cluster[median_idx])

        return deduplicated

    def _cluster_strokes(self, strokes: List[Stroke]) -> List[List[Stroke]]:
        """
        Cluster strokes that are essentially the same stroke.

        Args:
            strokes: List of strokes of the same type

        Returns:
            List of clusters, where each cluster is a list of similar strokes
        """
        if not strokes:
            return []

        CENTER_THRESHOLD = 12  # pixels - centers must be this close
        ANGLE_THRESHOLD = 20  # degrees

        clusters: List[List[Stroke]] = []
        used = set()

        for i, stroke in enumerate(strokes):
            if i in used:
                continue

            cluster = [stroke]
            used.add(i)

            for j, other in enumerate(strokes):
                if j in used:
                    continue

                # Check if same cluster
                center_dist = math.sqrt(
                    (stroke.center_x - other.center_x)**2 +
                    (stroke.center_y - other.center_y)**2
                )
                angle_diff = abs(stroke.angle - other.angle)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff

                if center_dist < CENTER_THRESHOLD and angle_diff < ANGLE_THRESHOLD:
                    cluster.append(other)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _create_events(self, strokes: List[Stroke]) -> List[SweepEvent]:
        """
        Create sweep line events from strokes.

        Each stroke generates two events:
        - SHIFT at its left edge (min x)
        - POP at its right edge (max x)

        Args:
            strokes: List of strokes

        Returns:
            Sorted list of events by x-coordinate
        """
        events = []

        for i, stroke in enumerate(strokes):
            # Get left and right x coordinates
            left_x = min(stroke.x1, stroke.x2)
            right_x = max(stroke.x1, stroke.x2)

            # Create SHIFT event at left edge
            events.append(SweepEvent(
                x=left_x,
                event_type=EventType.SHIFT,
                stroke=stroke,
                stroke_id=i
            ))

            # Create POP event at right edge
            events.append(SweepEvent(
                x=right_x,
                event_type=EventType.POP,
                stroke=stroke,
                stroke_id=i
            ))

        # Sort events by x coordinate
        # For ties: SHIFT events come before POP events at same x
        events.sort(key=lambda e: (e.x, 0 if e.event_type == EventType.SHIFT else 1))

        return events

    def _sweep_line_parse(self, events: List[SweepEvent]) -> List[StrokeNode]:
        """
        Run the sweep line shift-reduce parsing algorithm.

        Args:
            events: Sorted list of sweep events

        Returns:
            List of root nodes in the dependency tree
        """
        stack: List[StrokeNode] = []  # Current active strokes
        nodes: Dict[int, StrokeNode] = {}  # All nodes by stroke_id
        roots: List[StrokeNode] = []  # Root nodes (no parent)

        for event in events:
            if event.event_type == EventType.SHIFT:
                # Create a new node for this stroke
                node = StrokeNode(
                    stroke=event.stroke,
                    stroke_id=event.stroke_id
                )
                nodes[event.stroke_id] = node

                # Check for REDUCE: is this stroke contained/attached to any on stack?
                parent_node = self._find_parent(node, stack)

                if parent_node:
                    # REDUCE: attach as child
                    parent_node.children.append(node)
                    node.parent = parent_node
                else:
                    # No parent found, this is a root
                    roots.append(node)

                # SHIFT: push onto stack
                stack.append(node)

            elif event.event_type == EventType.POP:
                # POP: remove from stack
                node = nodes.get(event.stroke_id)
                if node and node in stack:
                    stack.remove(node)

        return roots

    def _find_parent(self, node: StrokeNode, stack: List[StrokeNode]) -> Optional[StrokeNode]:
        """
        Find if the new stroke should be attached to any stroke on the stack.

        REDUCE only happens when:
        - An endpoint of the new stroke touches the body of a stack stroke
        - This represents physical connection (like a diagonal attached to horizontal)

        Parallel strokes (e.g., h next to h) should NOT reduce.

        Args:
            node: The new stroke node
            stack: Current stack of active strokes

        Returns:
            Parent node if found, None otherwise
        """
        if not stack:
            return None

        stroke = node.stroke

        # Check each stroke on stack, prefer the most recent (top of stack)
        for stack_node in reversed(stack):
            stack_stroke = stack_node.stroke

            # Check if new stroke is physically attached to stack stroke
            if self._is_attached_to(stroke, stack_stroke):
                return stack_node

        return None

    def _is_attached_to(self, new_stroke: Stroke, base_stroke: Stroke) -> bool:
        """
        Check if new_stroke is physically attached to base_stroke.

        Attachment means an endpoint of new_stroke touches the BODY (not just endpoint)
        of base_stroke. This excludes parallel strokes that just happen to be close.

        Additionally, strokes of the same type that are roughly parallel should NOT
        be considered attached (they're just repeated strokes).

        Args:
            new_stroke: The stroke being added
            base_stroke: The stroke already on the stack

        Returns:
            True if new_stroke is attached to base_stroke
        """
        # Parallel strokes of the same type are never attached
        if new_stroke.stroke_type == base_stroke.stroke_type:
            if self._are_parallel(new_stroke, base_stroke):
                return False

        # Check if either endpoint of new_stroke touches base_stroke's body
        new_endpoints = [(new_stroke.x1, new_stroke.y1), (new_stroke.x2, new_stroke.y2)]

        for endpoint in new_endpoints:
            dist, t = self._point_to_segment_info(endpoint, base_stroke)

            # Endpoint must be close to base_stroke
            if dist > self.ATTACHMENT_THRESHOLD:
                continue

            # The contact point should be on the body of base_stroke (not at its endpoints)
            # t in (0.1, 0.9) means contact is on the middle portion
            if 0.05 < t < 0.95:
                return True

        return False

    def _are_parallel(self, s1: Stroke, s2: Stroke) -> bool:
        """
        Check if two strokes are roughly parallel.

        Args:
            s1: First stroke
            s2: Second stroke

        Returns:
            True if strokes are parallel (within 15 degrees)
        """
        angle_diff = abs(s1.angle - s2.angle)
        # Handle wraparound at 180 degrees
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        return angle_diff < 15

    def _point_to_segment_info(self, point: Tuple[float, float], segment: Stroke) -> Tuple[float, float]:
        """
        Calculate distance from point to segment and the parameter t.

        Args:
            point: (x, y) tuple
            segment: Stroke representing the line segment

        Returns:
            Tuple of (distance, t) where t is the parameter [0,1] along the segment
            t=0 means closest to segment start, t=1 means closest to segment end
        """
        px, py = point
        x1, y1 = segment.x1, segment.y1
        x2, y2 = segment.x2, segment.y2

        # Vector from segment start to end
        dx = x2 - x1
        dy = y2 - y1

        # Length squared of segment
        length_sq = dx*dx + dy*dy

        if length_sq == 0:
            # Segment is a point
            return math.sqrt((px - x1)**2 + (py - y1)**2), 0.5

        # Parameter t for closest point on infinite line
        t = ((px - x1)*dx + (py - y1)*dy) / length_sq
        t_clamped = max(0, min(1, t))

        # Closest point on segment
        closest_x = x1 + t_clamped * dx
        closest_y = y1 + t_clamped * dy

        dist = math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

        return dist, t_clamped

    def _generate_encoding(self, roots: List[StrokeNode]) -> str:
        """
        Generate the encoding string from the dependency tree.

        Args:
            roots: List of root nodes in the dependency tree

        Returns:
            Encoded string
        """
        if not roots:
            return ""

        # Sort roots by x position (left to right)
        roots.sort(key=lambda n: min(n.stroke.x1, n.stroke.x2))

        # Generate encoding for each root tree
        parts = []
        for root in roots:
            part = self._encode_node(root)
            parts.append(part)

        return ''.join(parts)

    def _encode_node(self, node: StrokeNode) -> str:
        """
        Recursively encode a node and its children.

        Args:
            node: The node to encode

        Returns:
            Encoded string for this subtree
        """
        stroke_char = node.stroke_type

        if not node.children:
            # Leaf node - just the stroke type
            return stroke_char

        # Sort children by x position
        node.children.sort(key=lambda n: min(n.stroke.x1, n.stroke.x2))

        # Encode children
        child_encodings = [self._encode_node(child) for child in node.children]
        children_str = ''.join(child_encodings)

        # Wrap in parentheses to show dependency
        return f"{stroke_char}({children_str})"


def encode_from_unicode(unicode_char: str) -> str:
    """
    Convenience function to encode a Unicode cuneiform character.

    Args:
        unicode_char: Unicode cuneiform character

    Returns:
        Sweep-line encoded string
    """
    try:
        from .glyph_renderer import GlyphRenderer
        from .stroke_detector import StrokeDetector
    except ImportError:
        from glyph_renderer import GlyphRenderer
        from stroke_detector import StrokeDetector

    # Render the character
    renderer = GlyphRenderer(size=128)
    binary = renderer.render(unicode_char)

    # Detect strokes
    detector = StrokeDetector()
    strokes = detector.detect_strokes(binary)

    # Encode
    encoder = SweepLineEncoder()
    return encoder.encode(strokes)


def debug_strokes(sign: str, show_raw: bool = True) -> List[Stroke]:
    """
    Debug function to print detected strokes for a sign.

    Args:
        sign: Unicode cuneiform character
        show_raw: If True, show raw strokes before deduplication

    Returns:
        List of deduplicated strokes
    """
    try:
        from .glyph_renderer import GlyphRenderer
        from .stroke_detector import StrokeDetector
    except ImportError:
        from glyph_renderer import GlyphRenderer
        from stroke_detector import StrokeDetector

    renderer = GlyphRenderer(size=128)
    detector = StrokeDetector()
    encoder = SweepLineEncoder()

    # Render
    binary = renderer.render(sign)

    # Detect strokes
    raw_strokes = detector.detect_strokes(binary)

    print(f"\nSign: {sign} (U+{ord(sign):04X})")
    print("=" * 50)

    if show_raw:
        print(f"\nRaw strokes: {len(raw_strokes)}")
        print("-" * 40)
        for i, s in enumerate(raw_strokes):
            print(f"  {i+1:2d}. type={s.stroke_type}  "
                  f"pos=({s.x1:.0f},{s.y1:.0f})->({s.x2:.0f},{s.y2:.0f})  "
                  f"center=({s.center_x:.0f},{s.center_y:.0f})  "
                  f"angle={s.angle:5.1f}°  len={s.length:.0f}")

    # Deduplicate
    deduped = encoder._deduplicate_strokes(raw_strokes)

    print(f"\nDeduplicated strokes: {len(deduped)}")
    print("-" * 40)
    for i, s in enumerate(deduped):
        print(f"  {i+1:2d}. type={s.stroke_type}  "
              f"pos=({s.x1:.0f},{s.y1:.0f})->({s.x2:.0f},{s.y2:.0f})  "
              f"center=({s.center_x:.0f},{s.center_y:.0f})  "
              f"angle={s.angle:5.1f}°  len={s.length:.0f}")

    # Show encoding
    encoding = encoder.encode(raw_strokes)
    print(f"\nEncoding: {encoding}")

    return deduped


if __name__ == "__main__":
    # Test the sweep line encoder
    print("="*70)
    print("SWEEP-LINE SHIFT-REDUCE ENCODER TEST")
    print("="*70)

    try:
        from .glyph_renderer import GlyphRenderer
        from .stroke_detector import StrokeDetector
    except ImportError:
        from glyph_renderer import GlyphRenderer
        from stroke_detector import StrokeDetector

    renderer = GlyphRenderer(size=128)
    detector = StrokeDetector()
    encoder = SweepLineEncoder()

    test_chars = [
        ('𒀸', 'ASH', 'Single horizontal (h)', 'h'),
        ('𒐀', 'TWO_ASH', 'Two horizontal (hh)', 'hh'),
        ('𒐁', 'THREE_ASH', 'Three horizontal (hhh)', 'hhh'),
        ('𒄉', 'GIR2_GUNU', 'Horizontal with attached diagonals', 'h(ud)(ud)vvvv'),
        ('𒀭', 'AN', 'Complex sign', '?'),
    ]

    for char, name, description, expected in test_chars:
        print(f"\n{'='*60}")
        print(f"{name}: {char}")
        print(f"  Description: {description}")
        print(f"  Expected: {expected}")

        # Render and detect
        binary = renderer.render(char)
        strokes = detector.detect_strokes(binary)
        print(f"  Raw strokes detected: {len(strokes)}")
        for i, s in enumerate(strokes):
            print(f"    {i+1}. {s.stroke_type} @ ({s.center_x:.0f},{s.center_y:.0f}) "
                  f"angle={s.angle:.0f}° len={s.length:.0f}")

        # Deduplicate
        deduped = encoder._deduplicate_strokes(strokes)
        print(f"  After deduplication: {len(deduped)}")
        for i, s in enumerate(deduped):
            print(f"    {i+1}. {s.stroke_type} @ ({s.center_x:.0f},{s.center_y:.0f}) "
                  f"angle={s.angle:.0f}° len={s.length:.0f}")

        # Encode
        encoding = encoder.encode(strokes)
        print(f"  Encoding: {encoding}")

        if encoding == expected:
            print(f"  ✓ MATCH")
        else:
            print(f"  ✗ MISMATCH")
