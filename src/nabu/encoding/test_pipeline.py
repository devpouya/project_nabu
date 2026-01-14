"""
Comprehensive test runner for the automatic encoding generation pipeline.

Tests the complete flow: Unicode → Image → Strokes → Spatial Analysis → Encoding
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass, asdict
import sys

try:
    from .glyph_renderer import GlyphRenderer
    from .stroke_detector import StrokeDetector, visualize_strokes
    from .spatial_analyzer import SpatialAnalyzer
    from .encoding_generator import EncodingGenerator, generate_encoding_from_unicode
except ImportError:
    from glyph_renderer import GlyphRenderer
    from stroke_detector import StrokeDetector, visualize_strokes
    from spatial_analyzer import SpatialAnalyzer
    from encoding_generator import EncodingGenerator, generate_encoding_from_unicode


@dataclass
class TestCase:
    """Test case for encoding generation."""
    unicode_char: str
    code_point: str
    name: str
    expected_encoding: Optional[str] = None
    description: str = ""


@dataclass
class TestResult:
    """Result of a single test."""
    test_case: TestCase
    generated_encoding: Optional[str]
    stroke_count: int
    spatial_structure: str
    success: bool
    error: Optional[str] = None
    match: bool = False


class EncodingPipelineTester:
    """
    Tests the complete encoding generation pipeline.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize pipeline tester.

        Args:
            output_dir: Directory for test outputs (visualizations, reports)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "results/encoding_tests"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.renderer = GlyphRenderer(size=128)
        self.detector = StrokeDetector()
        self.analyzer = SpatialAnalyzer()
        self.generator = EncodingGenerator()

    def run_test(self, test_case: TestCase, save_visualization: bool = True) -> TestResult:
        """
        Run a single test case through the pipeline.

        Args:
            test_case: Test case to run
            save_visualization: Whether to save visualization images

        Returns:
            TestResult with pipeline output and status
        """
        try:
            # Step 1: Render glyph
            binary = self.renderer.render(test_case.unicode_char)

            # Step 2: Detect strokes
            strokes = self.detector.detect_strokes(binary)

            # Step 3: Analyze spatial structure
            spatial_groups = self.analyzer.analyze(strokes)
            primary_structure = self.analyzer.determine_primary_structure(spatial_groups)

            # Step 4: Generate encoding
            encoding = self.generator.generate(strokes)

            # Check match if expected encoding provided
            match = False
            if test_case.expected_encoding and encoding:
                match = (encoding == test_case.expected_encoding or
                        test_case.expected_encoding in encoding or
                        encoding in test_case.expected_encoding)

            # Save visualization
            if save_visualization:
                viz_path = self.output_dir / f"{test_case.name}_{test_case.code_point.replace('+', '')}_viz.png"
                visualize_strokes(binary, strokes, str(viz_path))

            return TestResult(
                test_case=test_case,
                generated_encoding=encoding,
                stroke_count=len(strokes),
                spatial_structure=primary_structure or "simple",
                success=True,
                match=match
            )

        except Exception as e:
            return TestResult(
                test_case=test_case,
                generated_encoding=None,
                stroke_count=0,
                spatial_structure="error",
                success=False,
                error=str(e)
            )

    def run_test_suite(self, test_cases: List[TestCase], save_visualizations: bool = True) -> List[TestResult]:
        """
        Run a suite of test cases.

        Args:
            test_cases: List of test cases
            save_visualizations: Whether to save visualization images

        Returns:
            List of test results
        """
        results = []

        print("="*70)
        print("ENCODING GENERATION TEST SUITE")
        print("="*70)
        print(f"Running {len(test_cases)} test cases...")
        print()

        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: {test_case.name} ({test_case.code_point})")
            print(f"  Character: {test_case.unicode_char}")
            if test_case.expected_encoding:
                print(f"  Expected: {test_case.expected_encoding}")

            result = self.run_test(test_case, save_visualization=save_visualizations)

            if result.success:
                print(f"  ✓ Generated: {result.generated_encoding}")
                print(f"  Strokes: {result.stroke_count}, Structure: {result.spatial_structure}")
                if test_case.expected_encoding:
                    status = "✓ MATCH" if result.match else "✗ MISMATCH"
                    print(f"  {status}")
            else:
                print(f"  ✗ FAILED: {result.error}")

            print()
            results.append(result)

        return results

    def print_summary(self, results: List[TestResult]):
        """Print summary statistics."""
        print("="*70)
        print("TEST SUMMARY")
        print("="*70)

        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        # Match statistics (for tests with expected encodings)
        with_expected = [r for r in results if r.test_case.expected_encoding]
        matches = sum(1 for r in with_expected if r.match)

        print(f"Total tests: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print()

        if with_expected:
            print(f"Tests with expected encodings: {len(with_expected)}")
            print(f"Exact/partial matches: {matches} ({matches/len(with_expected)*100:.1f}%)")
            print()

        # Stroke count distribution
        stroke_counts = [r.stroke_count for r in results if r.success]
        if stroke_counts:
            print(f"Stroke count range: {min(stroke_counts)} - {max(stroke_counts)}")
            print(f"Average strokes: {sum(stroke_counts)/len(stroke_counts):.1f}")
            print()

        # Structure distribution
        structures = {}
        for r in results:
            if r.success:
                structures[r.spatial_structure] = structures.get(r.spatial_structure, 0) + 1

        if structures:
            print("Spatial structures detected:")
            for structure, count in sorted(structures.items(), key=lambda x: -x[1]):
                print(f"  {structure}: {count} ({count/successful*100:.1f}%)")

    def save_report(self, results: List[TestResult], report_path: Optional[Path] = None):
        """Save detailed test report to file."""
        if report_path is None:
            report_path = self.output_dir / "test_report.json"

        # Convert to serializable format
        report_data = {
            'total_tests': len(results),
            'successful': sum(1 for r in results if r.success),
            'failed': sum(1 for r in results if not r.success),
            'results': []
        }

        for result in results:
            result_dict = {
                'name': result.test_case.name,
                'unicode': result.test_case.unicode_char,
                'code_point': result.test_case.code_point,
                'expected_encoding': result.test_case.expected_encoding,
                'generated_encoding': result.generated_encoding,
                'stroke_count': result.stroke_count,
                'spatial_structure': result.spatial_structure,
                'success': result.success,
                'match': result.match,
                'error': result.error
            }
            report_data['results'].append(result_dict)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed report saved to: {report_path}")


def get_default_test_cases() -> List[TestCase]:
    """Get default test cases for common cuneiform signs."""
    return [
        # Simple signs
        TestCase(
            unicode_char='𒀸',
            code_point='U+12038',
            name='ASH',
            expected_encoding='h',
            description='Single horizontal stroke'
        ),
        TestCase(
            unicode_char='𒁹',
            code_point='U+12079',
            name='DIŠ',
            expected_encoding='v',
            description='Single vertical stroke'
        ),

        # Repetitions
        TestCase(
            unicode_char='𒐀',
            code_point='U+12400',
            name='TWO_ASH',
            expected_encoding='h2',
            description='Two horizontal strokes'
        ),
        TestCase(
            unicode_char='𒐁',
            code_point='U+12401',
            name='THREE_ASH',
            expected_encoding='h3',
            description='Three horizontal strokes (the sign that failed before!)'
        ),
        TestCase(
            unicode_char='𒐂',
            code_point='U+12402',
            name='FOUR_ASH',
            expected_encoding='h4',
            description='Four horizontal strokes'
        ),

        # Complex compositions
        TestCase(
            unicode_char='𒀭',
            code_point='U+1202D',
            name='AN',
            expected_encoding='(h2[00v0])',
            description='Complex superposition with nested stacking'
        ),
        TestCase(
            unicode_char='𒁀',
            code_point='U+12040',
            name='BA',
            expected_encoding=None,  # Don't know expected
            description='Complex sign for testing'
        ),
        TestCase(
            unicode_char='𒂍',
            code_point='U+1208D',
            name='E2',
            expected_encoding=None,
            description='House sign - complex structure'
        ),

        # Diagonal strokes
        TestCase(
            unicode_char='𒀀',
            code_point='U+12000',
            name='A',
            expected_encoding=None,
            description='Contains diagonal strokes'
        ),
        TestCase(
            unicode_char='𒌋',
            code_point='U+1230B',
            name='U',
            expected_encoding=None,
            description='Contains diagonal strokes'
        ),
    ]


if __name__ == "__main__":
    # Run default test suite
    tester = EncodingPipelineTester()

    test_cases = get_default_test_cases()

    # Run tests
    results = tester.run_test_suite(test_cases, save_visualizations=True)

    # Print summary
    tester.print_summary(results)

    # Save report
    tester.save_report(results)

    print(f"\nTest visualizations saved to: {tester.output_dir}")
