"""
Advanced visualization tools for encoding generation pipeline.

Creates detailed visual outputs showing each step of the pipeline:
- Rendered glyph
- Detected strokes with classifications
- Spatial groupings
- Generated encoding
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from .stroke_detector import Stroke
    from .spatial_analyzer import SpatialGroup
except ImportError:
    from stroke_detector import Stroke
    from spatial_analyzer import SpatialGroup


class EncodingVisualizer:
    """
    Creates detailed visualizations of the encoding generation process.
    """

    def __init__(self):
        """Initialize visualizer."""
        if not PIL_AVAILABLE or not CV2_AVAILABLE:
            raise RuntimeError("PIL and OpenCV required for visualization")

        # Colors for different stroke types (BGR format for OpenCV)
        self.stroke_colors = {
            'h': (0, 0, 255),      # Red - horizontal
            'v': (0, 255, 0),      # Green - vertical
            'u': (255, 0, 0),      # Blue - upward diagonal
            'd': (255, 255, 0),    # Cyan - downward diagonal
            'c': (255, 0, 255),    # Magenta - winkelhaken
        }

        # Colors for spatial groups (lighter shades)
        self.group_colors = {
            'horizontal': (100, 100, 255),
            'vertical': (100, 255, 100),
            'superposed': (255, 200, 100),
            'repetition': (255, 100, 200),
        }

    def create_pipeline_visualization(
        self,
        binary_image: np.ndarray,
        strokes: List[Stroke],
        spatial_groups: Dict[str, List[SpatialGroup]],
        encoding: Optional[str],
        unicode_char: str,
        output_path: str
    ):
        """
        Create a comprehensive multi-panel visualization.

        Args:
            binary_image: Original rendered glyph
            strokes: Detected strokes
            spatial_groups: Spatial analysis results
            encoding: Generated encoding
            unicode_char: Unicode character
            output_path: Where to save
        """
        # Create 4-panel layout: original, strokes, groups, final
        panel_height, panel_width = binary_image.shape

        # Create output image (2x2 grid)
        output = np.ones((panel_height * 2, panel_width * 2, 3), dtype=np.uint8) * 255

        # Panel 1: Original glyph
        original_color = cv2.cvtColor((binary_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        output[0:panel_height, 0:panel_width] = original_color

        # Panel 2: Detected strokes
        strokes_panel = self._visualize_strokes(binary_image, strokes)
        output[0:panel_height, panel_width:panel_width*2] = strokes_panel

        # Panel 3: Spatial groups
        groups_panel = self._visualize_spatial_groups(binary_image, spatial_groups)
        output[panel_height:panel_height*2, 0:panel_width] = groups_panel

        # Panel 4: Summary with encoding
        summary_panel = self._create_summary_panel(
            panel_width, panel_height,
            unicode_char, encoding, len(strokes), spatial_groups
        )
        output[panel_height:panel_height*2, panel_width:panel_width*2] = summary_panel

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, "1. Original", (10, 20), font, 0.5, (0, 0, 0), 1)
        cv2.putText(output, "2. Strokes", (panel_width + 10, 20), font, 0.5, (0, 0, 0), 1)
        cv2.putText(output, "3. Spatial Groups", (10, panel_height + 20), font, 0.5, (0, 0, 0), 1)
        cv2.putText(output, "4. Summary", (panel_width + 10, panel_height + 20), font, 0.5, (0, 0, 0), 1)

        # Save
        cv2.imwrite(output_path, output)

    def _visualize_strokes(self, binary_image: np.ndarray, strokes: List[Stroke]) -> np.ndarray:
        """Visualize detected strokes with color coding."""
        img_color = cv2.cvtColor((binary_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        for stroke in strokes:
            color = self.stroke_colors.get(stroke.stroke_type, (128, 128, 128))

            # Draw line
            cv2.line(
                img_color,
                (int(stroke.x1), int(stroke.y1)),
                (int(stroke.x2), int(stroke.y2)),
                color,
                thickness=2
            )

            # Draw stroke type label
            cv2.putText(
                img_color,
                stroke.stroke_type,
                (int(stroke.center_x), int(stroke.center_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

        return img_color

    def _visualize_spatial_groups(
        self,
        binary_image: np.ndarray,
        spatial_groups: Dict[str, List[SpatialGroup]]
    ) -> np.ndarray:
        """Visualize spatial groupings with bounding boxes."""
        img_color = cv2.cvtColor((binary_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        for group_type, groups in spatial_groups.items():
            if not groups:
                continue

            color = self.group_colors.get(group_type, (128, 128, 128))

            for group in groups:
                # Draw bounding box
                min_x, min_y, max_x, max_y = group.bounds
                cv2.rectangle(
                    img_color,
                    (int(min_x), int(min_y)),
                    (int(max_x), int(max_y)),
                    color,
                    2
                )

                # Label
                label_pos = (int(min_x), int(min_y) - 5)
                cv2.putText(
                    img_color,
                    f"{group_type[0]}{len(group.strokes)}",
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )

        return img_color

    def _create_summary_panel(
        self,
        width: int,
        height: int,
        unicode_char: str,
        encoding: Optional[str],
        stroke_count: int,
        spatial_groups: Dict[str, List[SpatialGroup]]
    ) -> np.ndarray:
        """Create summary information panel."""
        panel = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Count groups
        group_counts = {k: len(v) for k, v in spatial_groups.items() if v}

        # Create text summary
        lines = [
            f"Unicode: {unicode_char}",
            "",
            f"Strokes: {stroke_count}",
            "",
            "Groups:",
        ]

        for group_type, count in group_counts.items():
            lines.append(f"  {group_type}: {count}")

        lines.append("")
        lines.append("Encoding:")
        if encoding:
            # Wrap long encodings
            if len(encoding) > 15:
                lines.append(f"  {encoding[:15]}")
                lines.append(f"  {encoding[15:]}")
            else:
                lines.append(f"  {encoding}")
        else:
            lines.append("  (none)")

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        for line in lines:
            cv2.putText(
                panel,
                line,
                (10, y_offset),
                font,
                0.4,
                (0, 0, 0),
                1
            )
            y_offset += 18

        return panel

    def create_comparison_grid(
        self,
        results: List[Tuple[str, np.ndarray, List[Stroke], str]],
        output_path: str,
        columns: int = 3
    ):
        """
        Create a grid comparing multiple signs.

        Args:
            results: List of (name, binary_image, strokes, encoding) tuples
            output_path: Where to save
            columns: Number of columns in grid
        """
        if not results:
            return

        # Calculate grid dimensions
        rows = (len(results) + columns - 1) // columns

        # Get panel size from first image
        panel_height, panel_width = results[0][1].shape

        # Create output grid
        grid_width = panel_width * columns
        grid_height = panel_height * rows
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

        for idx, (name, binary, strokes, encoding) in enumerate(results):
            row = idx // columns
            col = idx % columns

            # Create panel with strokes
            panel = self._visualize_strokes(binary, strokes)

            # Add label
            label = f"{name}: {encoding or '?'}"
            cv2.putText(
                panel,
                label,
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )

            # Place in grid
            y_start = row * panel_height
            x_start = col * panel_width
            grid[y_start:y_start+panel_height, x_start:x_start+panel_width] = panel

        cv2.imwrite(output_path, grid)

    def create_legend(self, output_path: str, width: int = 300, height: int = 200):
        """Create a legend explaining the color coding."""
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255

        y_offset = 20

        # Stroke types
        cv2.putText(legend, "Stroke Types:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 25

        for stroke_type, color in self.stroke_colors.items():
            # Draw color box
            cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), color, -1)

            # Label
            labels = {
                'h': 'Horizontal',
                'v': 'Vertical',
                'u': 'Upward diagonal',
                'd': 'Downward diagonal',
                'c': 'Winkelhaken'
            }
            cv2.putText(
                legend,
                labels.get(stroke_type, stroke_type),
                (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1
            )
            y_offset += 20

        # Spatial groups
        y_offset += 10
        cv2.putText(legend, "Spatial Groups:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 25

        for group_type, color in self.group_colors.items():
            cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), color, -1)
            cv2.putText(
                legend,
                group_type.capitalize(),
                (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1
            )
            y_offset += 20

        cv2.imwrite(output_path, legend)


def create_debug_visualization(
    unicode_char: str,
    output_dir: Path,
    renderer,
    detector,
    analyzer,
    generator
):
    """
    Create comprehensive debug visualization for a single sign.

    Args:
        unicode_char: Unicode character to visualize
        output_dir: Output directory
        renderer: GlyphRenderer instance
        detector: StrokeDetector instance
        analyzer: SpatialAnalyzer instance
        generator: EncodingGenerator instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    binary = renderer.render(unicode_char)
    strokes = detector.detect_strokes(binary)
    spatial_groups = analyzer.analyze(strokes)
    encoding = generator.generate(strokes)

    # Create visualization
    visualizer = EncodingVisualizer()

    code_point = f"U+{ord(unicode_char):04X}"
    output_path = output_dir / f"{code_point}_debug.png"

    visualizer.create_pipeline_visualization(
        binary, strokes, spatial_groups, encoding,
        unicode_char, str(output_path)
    )

    print(f"Debug visualization saved to: {output_path}")
    print(f"Generated encoding: {encoding}")

    return encoding


if __name__ == "__main__":
    # Test visualizer
    try:
        from .glyph_renderer import GlyphRenderer
        from .stroke_detector import StrokeDetector
        from .spatial_analyzer import SpatialAnalyzer
        from .encoding_generator import EncodingGenerator
    except ImportError:
        from glyph_renderer import GlyphRenderer
        from stroke_detector import StrokeDetector
        from spatial_analyzer import SpatialAnalyzer
        from encoding_generator import EncodingGenerator

    output_dir = Path(__file__).parent.parent.parent.parent / "results/visualizations"

    renderer = GlyphRenderer(size=128)
    detector = StrokeDetector()
    analyzer = SpatialAnalyzer()
    generator = EncodingGenerator()

    # Test signs
    test_chars = [
        '𒀸',  # ASH - simple
        '𒐁',  # THREE_ASH - the problematic sign!
        '𒀭',  # AN - complex
    ]

    print("Creating debug visualizations...")
    print("="*70)

    for char in test_chars:
        code = f"U+{ord(char):04X}"
        print(f"\nProcessing {char} ({code})...")
        encoding = create_debug_visualization(
            char, output_dir,
            renderer, detector, analyzer, generator
        )

    # Create legend
    visualizer = EncodingVisualizer()
    legend_path = output_dir / "legend.png"
    visualizer.create_legend(str(legend_path))

    print(f"\n{'='*70}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"Legend saved to: {legend_path}")
