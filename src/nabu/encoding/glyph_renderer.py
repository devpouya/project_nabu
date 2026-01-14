"""
Glyph rendering utilities for converting Unicode cuneiform characters to images.

This module provides functions to render cuneiform signs using system fonts,
preparing them for stroke detection and analysis.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")


class GlyphRenderer:
    """
    Renders Unicode cuneiform characters to binary images for analysis.
    """

    def __init__(self, font_path: Optional[str] = None, size: int = 128):
        """
        Initialize the glyph renderer.

        Args:
            font_path: Path to cuneiform font file (e.g., NotoSansCuneiform.ttf)
                      If None, will try to use system fonts
            size: Size of rendered image (pixels)
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL/Pillow is required for glyph rendering")

        self.size = size
        self.font = self._load_font(font_path, size)

    def _load_font(self, font_path: Optional[str], size: int):
        """Load cuneiform font."""
        if font_path and Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

        # Try common cuneiform fonts
        font_names = [
            "NotoSansCuneiform-Regular.ttf",
            "NotoSansCuneiform.ttf",
            "Cuneiform.ttf",
            "CuneiformNA.ttf"
        ]

        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, size)
            except OSError:
                continue

        # Fallback to default font
        print("Warning: Cuneiform font not found, using default font")
        return ImageFont.load_default()

    def render(self, unicode_char: str, padding: int = 10) -> np.ndarray:
        """
        Render a Unicode character to a binary image.

        Args:
            unicode_char: Unicode cuneiform character (e.g., "𒀀")
            padding: Padding around the glyph (pixels)

        Returns:
            Binary numpy array (0 = background, 1 = stroke)
        """
        # Create image with white background
        img_size = self.size + 2 * padding
        img = Image.new('L', (img_size, img_size), 255)
        draw = ImageDraw.Draw(img)

        # Draw the character in black
        # Center it in the image
        bbox = draw.textbbox((0, 0), unicode_char, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (img_size - text_width) // 2 - bbox[0]
        y = (img_size - text_height) // 2 - bbox[1]

        draw.text((x, y), unicode_char, font=self.font, fill=0)

        # Convert to numpy array and binarize
        img_array = np.array(img)
        binary = (img_array < 128).astype(np.uint8)  # 0 = white, 1 = black

        return binary

    def render_with_bounding_box(self, unicode_char: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Render glyph and return bounding box of actual content.

        Args:
            unicode_char: Unicode cuneiform character

        Returns:
            Tuple of (binary_image, bounding_box)
            bounding_box = (min_x, min_y, max_x, max_y)
        """
        binary = self.render(unicode_char)

        # Find bounding box of non-zero pixels
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)

        if not np.any(rows) or not np.any(cols):
            # Empty image
            return binary, (0, 0, 0, 0)

        min_y, max_y = np.where(rows)[0][[0, -1]]
        min_x, max_x = np.where(cols)[0][[0, -1]]

        return binary, (min_x, min_y, max_x, max_y)

    def save_debug_image(self, binary: np.ndarray, output_path: str):
        """Save binary image for debugging."""
        img = Image.fromarray((binary * 255).astype(np.uint8), mode='L')
        img.save(output_path)
        print(f"Debug image saved to: {output_path}")


def render_unicode_glyph(unicode_char: str, size: int = 128, font_path: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to render a Unicode glyph.

    Args:
        unicode_char: Unicode cuneiform character
        size: Image size (pixels)
        font_path: Optional path to font file

    Returns:
        Binary numpy array
    """
    renderer = GlyphRenderer(font_path=font_path, size=size)
    return renderer.render(unicode_char)


if __name__ == "__main__":
    # Test rendering
    renderer = GlyphRenderer(size=128)

    test_chars = [
        ('𒀀', 'A', 'U+12000'),
        ('𒀸', 'ASH', 'U+12038'),
        ('𒐁', 'THREE_ASH', 'U+12401'),
        ('𒀭', 'AN', 'U+1202D'),
    ]

    output_dir = Path(__file__).parent.parent.parent.parent / "results/rendered_glyphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for char, name, code in test_chars:
        print(f"Rendering {name} ({code}): {char}")
        binary, bbox = renderer.render_with_bounding_box(char)

        print(f"  Image size: {binary.shape}")
        print(f"  Bounding box: {bbox}")
        print(f"  Non-zero pixels: {np.sum(binary)}")

        # Save debug image
        output_path = output_dir / f"{name}_{code.replace('+', '')}.png"
        renderer.save_debug_image(binary, str(output_path))
        print()

    print(f"Rendered images saved to: {output_dir}")
