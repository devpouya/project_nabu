"""
Stroke detection for cuneiform signs using F-Clip model.

Detects line segments in rendered glyphs using the F-Clip line detection model.

Pipeline:
1. Convert binary glyph to RGB image
2. Apply super-resolution (torchsr) to enhance details
3. Run F-Clip model for line detection
4. Return detected line segments
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import math
import sys
import os

# Add F-Clip to path
FCLIP_PATH = Path(__file__).parent.parent.parent.parent.parent / "F-Clip"
if FCLIP_PATH.exists():
    sys.path.insert(0, str(FCLIP_PATH))

# Super-resolution imports
try:
    import torch
    # Fix SSL certificate issues on macOS for downloading pretrained models
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    from torchsr.models import ninasr_b0
    from torchvision.transforms.functional import to_pil_image, to_tensor
    from PIL import Image
    TORCHSR_AVAILABLE = True
except ImportError:
    TORCHSR_AVAILABLE = False
    print("Warning: torchsr not available. Install with: pip install torchsr")


@dataclass
class Line:
    """Represents a detected line segment."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def length(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return math.sqrt(dx*dx + dy*dy)

    @property
    def angle(self) -> float:
        """Angle in degrees [0, 180)."""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        ang = math.degrees(math.atan2(-dy, dx)) % 180
        return ang

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Bounding box: (min_x, min_y, max_x, max_y)"""
        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2)
        )


# Backwards compatibility alias
Stroke = Line


class FClipDetector:
    """
    Detects lines in cuneiform glyphs using F-Clip.

    F-Clip is a fast line detection model with multiple backbone options.
    Super-resolution is applied to enhance image quality before detection.
    """

    # Class-level model cache to avoid reloading
    _detector = None
    _model_type = None
    _sr_model = None

    def __init__(self,
                 model: str = "HG2_LB",
                 checkpoint: Optional[str] = None,
                 threshold: float = 0.4,
                 use_super_resolution: bool = True,
                 sr_scale: int = 2):
        """
        Initialize F-Clip detector.

        Args:
            model: Model type - 'HR', 'HG1', 'HG2', 'HG1_D3', 'HG2_LB'
            checkpoint: Path to checkpoint file. If None, uses default.
            threshold: Confidence threshold for line detection
            use_super_resolution: Whether to apply super-resolution before detection
            sr_scale: Super-resolution scale factor (2 or 4)
        """
        if not FCLIP_PATH.exists():
            raise RuntimeError(f"F-Clip not found at: {FCLIP_PATH}")

        self.model_type = model
        self.threshold = threshold
        self.use_super_resolution = use_super_resolution and TORCHSR_AVAILABLE
        self.sr_scale = sr_scale

        # Default checkpoint path
        if checkpoint is None:
            checkpoint = str(FCLIP_PATH / "HG2_LB" / "checkpoint.pth.tar")
        self.checkpoint = checkpoint

        # Load models (cached at class level)
        self._load_model()
        if self.use_super_resolution:
            self._load_sr_model()

    def _load_model(self):
        """Load F-Clip model (cached at class level)."""
        # Check if already loaded with same model type
        if (FClipDetector._detector is not None and
            FClipDetector._model_type == self.model_type):
            self.detector = FClipDetector._detector
            return

        # Change to F-Clip directory for config loading
        original_dir = os.getcwd()
        try:
            os.chdir(str(FCLIP_PATH))

            from demo import FClipDetect
            self.detector = FClipDetect(self.model_type, self.checkpoint)

            # Cache at class level
            FClipDetector._detector = self.detector
            FClipDetector._model_type = self.model_type

            print(f"F-Clip model ({self.model_type}) loaded")
        finally:
            os.chdir(original_dir)

    def _load_sr_model(self):
        """Load super-resolution model (cached at class level)."""
        if FClipDetector._sr_model is not None:
            self.sr_model = FClipDetector._sr_model
            return

        # Load NinaSR model
        self.sr_model = ninasr_b0(scale=self.sr_scale, pretrained=True)
        self.sr_model.eval()

        # Move to GPU if available
        if torch.cuda.is_available():
            self.sr_model = self.sr_model.cuda()

        # Cache at class level
        FClipDetector._sr_model = self.sr_model

        print(f"Super-resolution model (ninasr_b0, scale={self.sr_scale}) loaded")

    def _apply_super_resolution(self, image: np.ndarray) -> np.ndarray:
        """
        Apply super-resolution to enhance image quality.

        Args:
            image: Input BGR image (H, W, 3)

        Returns:
            Super-resolved BGR image
        """
        # Convert BGR to RGB PIL Image
        rgb = image[:, :, ::-1]  # BGR to RGB
        pil_image = Image.fromarray(rgb)

        # Convert to tensor
        lr_tensor = to_tensor(pil_image).unsqueeze(0)

        # Move to GPU if available
        if torch.cuda.is_available():
            lr_tensor = lr_tensor.cuda()

        # Apply super-resolution
        with torch.no_grad():
            sr_tensor = self.sr_model(lr_tensor)

        # Convert back to PIL and then to numpy
        sr_pil = to_pil_image(sr_tensor.squeeze(0).cpu())
        sr_rgb = np.array(sr_pil)

        # Convert RGB back to BGR
        sr_bgr = sr_rgb[:, :, ::-1].copy()

        return sr_bgr

    def _apply_edge_detection(self, image: np.ndarray,
                               low_threshold: int = 50,
                               high_threshold: int = 150) -> np.ndarray:
        """
        Apply Canny edge detection to the image.

        Args:
            image: Input BGR image (H, W, 3)
            low_threshold: Lower threshold for Canny edge detection
            high_threshold: Upper threshold for Canny edge detection

        Returns:
            BGR image with edges (white edges on black background)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        # Convert back to BGR (white edges on black background)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return edges_bgr

    def detect(self, binary_image: np.ndarray) -> List[Line]:
        """
        Detect lines in a binary image using F-Clip.

        Args:
            binary_image: Binary numpy array (1 = stroke, 0 = background)

        Returns:
            List of detected Line objects
        """
        # Convert binary to BGR image
        bgr_image = self._binary_to_bgr(binary_image)
        original_h, original_w = bgr_image.shape[:2]

        # Apply super-resolution if enabled
        if self.use_super_resolution:
            bgr_image = self._apply_super_resolution(bgr_image)
            sr_h, sr_w = bgr_image.shape[:2]
            scale_x = original_w / sr_w
            scale_y = original_h / sr_h
        else:
            scale_x = scale_y = 1.0

        # Apply edge detection
        bgr_image = self._apply_edge_detection(bgr_image)

        # Run F-Clip detection
        # Output is (N, 2, 2) where each line is [[y1,x1], [y2,x2]]
        lines_array = self.detector.detect(bgr_image)

        # Convert to Line objects
        detected = []
        for line in lines_array:
            # F-Clip uses [row, col] format, convert to [x, y]
            y1, x1 = line[0]
            y2, x2 = line[1]

            # Scale back to original image coordinates
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)

            # Skip very short lines
            if length < 5:
                continue

            detected.append(Line(x1=x1, y1=y1, x2=x2, y2=y2))

        return detected

    def _binary_to_bgr(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Convert binary image to BGR image.

        Args:
            binary_image: Binary numpy array (1 = stroke, 0 = background)

        Returns:
            BGR image (black strokes on white background)
        """
        # Convert: 1 (stroke) -> 0 (black), 0 (background) -> 255 (white)
        gray = ((1 - binary_image) * 255).astype(np.uint8)

        # Convert to BGR
        bgr = np.stack([gray, gray, gray], axis=-1)

        return bgr

    # Backwards compatibility
    def detect_strokes(self, binary_image: np.ndarray) -> List[Line]:
        return self.detect(binary_image)


# Backwards compatibility aliases
StrokeDetector = FClipDetector
LINEAStrokeDetector = FClipDetector


def debug_detection(sign: str, output_dir: str = None, threshold: float = 0.4,
                    use_super_resolution: bool = True):
    """
    Debug line detection for a sign.

    Args:
        sign: Unicode cuneiform character
        output_dir: Optional directory to save debug images
        threshold: Detection confidence threshold
        use_super_resolution: Whether to apply super-resolution
    """
    try:
        from .glyph_renderer import GlyphRenderer
    except ImportError:
        from glyph_renderer import GlyphRenderer

    import cv2

    renderer = GlyphRenderer(size=128)
    detector = FClipDetector(threshold=threshold, use_super_resolution=use_super_resolution)

    print(f"\n{'='*60}")
    print(f"DEBUG: {sign} (U+{ord(sign):04X})")
    print(f"Super-resolution: {'ON' if use_super_resolution else 'OFF'}")
    print('='*60)

    # Step 1: Render
    binary = renderer.render(sign)
    print(f"\n1. Rendered image: {binary.shape}, {np.sum(binary)} foreground pixels")

    # Step 2: Detect lines with F-Clip
    lines = detector.detect(binary)
    print(f"2. F-Clip detected {len(lines)} lines:")
    for i, line in enumerate(lines):
        print(f"   {i+1}. ({line.x1:.0f},{line.y1:.0f})->({line.x2:.0f},{line.y2:.0f}) "
              f"angle={line.angle:.1f}° len={line.length:.0f}")

    # Save debug images if output_dir provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        name = f"U{ord(sign):04X}"

        # Save original
        cv2.imwrite(str(output_path / f"{name}_1_original.png"),
                   (binary * 255).astype(np.uint8))

        # Save super-resolved image if enabled
        if use_super_resolution and detector.use_super_resolution:
            bgr_image = detector._binary_to_bgr(binary)
            sr_image = detector._apply_super_resolution(bgr_image)
            cv2.imwrite(str(output_path / f"{name}_2_superres.png"), sr_image)
            print(f"   Super-resolved: {binary.shape[:2]} -> {sr_image.shape[:2]}")

            # Save edge-detected image
            edges_image = detector._apply_edge_detection(sr_image)
            cv2.imwrite(str(output_path / f"{name}_3_edges.png"), edges_image)

        # Save with detected lines
        img_color = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for line in lines:
            cv2.line(img_color, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)),
                    (0, 0, 255), 2)
        cv2.imwrite(str(output_path / f"{name}_4_lines.png"), img_color)

        print(f"\nDebug images saved to: {output_path}")

    return lines


if __name__ == "__main__":
    # Test line detection
    try:
        from glyph_renderer import GlyphRenderer
    except ImportError:
        from .glyph_renderer import GlyphRenderer

    renderer = GlyphRenderer(size=128)
    detector = FClipDetector()

    test_chars = [
        ('𒀸', 'ASH', 'U+12038'),       # Single horizontal
        ('𒐀', 'TWO_ASH', 'U+12400'),   # Two horizontal
        ('𒐁', 'THREE_ASH', 'U+12401'), # Three horizontal
        ('𒄉', 'GIR2_GUNU', 'U+12109'), # Complex with diagonals
        ('𒀭', 'AN', 'U+1202D'),        # Complex sign
    ]

    for char, name, code in test_chars:
        print(f"\n{'='*60}")
        print(f"Analyzing {name} ({code}): {char}")

        binary = renderer.render(char)
        lines = detector.detect(binary)

        print(f"Detected {len(lines)} lines:")
        for i, line in enumerate(lines, 1):
            print(f"  {i}. ({line.x1:.0f},{line.y1:.0f})->({line.x2:.0f},{line.y2:.0f}) "
                  f"angle={line.angle:.1f}° len={line.length:.0f}")

    print(f"\nDone.")
