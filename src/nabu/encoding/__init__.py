"""
Automatic cuneiform encoding generation using computer vision.

This module provides tools for automatically generating Hantatallas stroke-based
encodings from Unicode cuneiform characters.
"""

from .encoding_generator import generate_encoding_from_unicode, EncodingGenerator
from .encoding_database import EncodingDatabase, unicode_to_encoding
from .glyph_renderer import GlyphRenderer
from .stroke_detector import Line, StrokeDetector, FClipDetector, debug_detection
from .spatial_analyzer import SpatialAnalyzer
from .sweep_line_encoder import SweepLineEncoder, encode_from_unicode, debug_strokes

__all__ = [
    'generate_encoding_from_unicode',
    'EncodingGenerator',
    'EncodingDatabase',
    'unicode_to_encoding',
    'GlyphRenderer',
    'Line',
    'StrokeDetector',
    'FClipDetector',
    'debug_detection',
    'SpatialAnalyzer',
    'SweepLineEncoder',
    'encode_from_unicode',
    'debug_strokes',
]
