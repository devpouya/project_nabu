"""
Validation against HZL ground truth.

Tests the automatic encoding generation against known HZL encodings
to measure accuracy and identify areas for improvement.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import csv
from dataclasses import dataclass

try:
    from .encoding_generator import generate_encoding_from_unicode
except ImportError:
    from encoding_generator import generate_encoding_from_unicode


@dataclass
class ValidationResult:
    """Result of validating one sign."""
    unicode_char: str
    code_point: str
    hzl_id: str
    expected_encoding: str
    generated_encoding: Optional[str]
    match: bool
    similarity: float  # 0.0 to 1.0


class HZLValidator:
    """
    Validates automatic encoding generation against HZL ground truth.
    """

    def __init__(self, hantatallas_path: Optional[Path] = None):
        """
        Initialize validator.

        Args:
            hantatallas_path: Path to hantatallas repository
        """
        if hantatallas_path is None:
            hantatallas_path = Path.home() / "projects/cuneiform/hantatallas"

        self.hantatallas_path = Path(hantatallas_path)
        self.hzl_dat_path = self.hantatallas_path / "hantatallas/data/hzl.dat"
        self.unicode_csv_path = self.hantatallas_path / "hantatallas/data/unicode_cleaned.csv"

        # Load ground truth
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> List[Dict]:
        """
        Load HZL ground truth: Unicode character → expected encoding.

        Returns:
            List of dicts with 'unicode', 'code_point', 'hzl_id', 'encoding'
        """
        # Step 1: Load HZL encodings (hzl_id → encoding)
        hzl_encodings = {}

        if self.hzl_dat_path.exists():
            with open(self.hzl_dat_path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')

            current_id = None
            in_form_section = False

            for line in lines:
                tabs = len(line) - len(line.lstrip('\t'))
                content = line.strip()

                if not content:
                    continue

                if tabs == 0 and content.replace('.', '').replace('A', '').replace('B', '').isdigit():
                    current_id = content
                    in_form_section = False
                elif tabs == 1 and content == 'FORM':
                    in_form_section = True
                elif tabs == 2 and in_form_section and current_id:
                    parts = content.split()
                    if parts:
                        encoding = parts[0]
                        if encoding not in hzl_encodings.get(current_id, []):
                            if current_id not in hzl_encodings:
                                hzl_encodings[current_id] = []
                            hzl_encodings[current_id].append(encoding)

        print(f"Loaded {len(hzl_encodings)} HZL sign encodings")

        # Step 2: Load Unicode mappings (hzl_id → unicode)
        ground_truth = []

        if self.unicode_csv_path.exists():
            with open(self.unicode_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header

                for row in reader:
                    if len(row) < 7:
                        continue

                    hzl_id = row[3].strip()
                    if not hzl_id or hzl_id in [' ', '\xa0']:
                        continue

                    unicode_col = row[5].strip()

                    # Extract Unicode characters
                    unicode_chars = self._extract_unicode_chars(unicode_col)

                    if unicode_chars and hzl_id in hzl_encodings:
                        for unicode_char in unicode_chars:
                            for encoding in hzl_encodings[hzl_id]:
                                ground_truth.append({
                                    'unicode': unicode_char,
                                    'code_point': f"U+{ord(unicode_char):04X}",
                                    'hzl_id': hzl_id,
                                    'encoding': encoding
                                })

        print(f"Created {len(ground_truth)} ground truth entries")

        return ground_truth

    def _extract_unicode_chars(self, unicode_col: str) -> List[str]:
        """Extract Unicode characters from unicode column."""
        chars = []
        for char in unicode_col:
            code_point = ord(char)
            if (0x12000 <= code_point <= 0x123FF or
                0x12400 <= code_point <= 0x1247F or
                0x12480 <= code_point <= 0x1254F):
                if char not in chars:
                    chars.append(char)
        return chars

    def validate(self, limit: Optional[int] = None) -> List[ValidationResult]:
        """
        Validate encoding generation against ground truth.

        Args:
            limit: Optional limit on number of signs to test

        Returns:
            List of ValidationResult objects
        """
        results = []

        test_set = self.ground_truth[:limit] if limit else self.ground_truth

        print(f"\nValidating {len(test_set)} signs against HZL ground truth...")
        print("="*70)

        for i, entry in enumerate(test_set, 1):
            unicode_char = entry['unicode']
            expected = entry['encoding']
            hzl_id = entry['hzl_id']
            code_point = entry['code_point']

            # Generate encoding
            try:
                generated = generate_encoding_from_unicode(unicode_char)
            except Exception as e:
                print(f"Error processing {unicode_char} ({code_point}): {e}")
                generated = None

            # Check match
            match, similarity = self._compare_encodings(generated, expected)

            result = ValidationResult(
                unicode_char=unicode_char,
                code_point=code_point,
                hzl_id=hzl_id,
                expected_encoding=expected,
                generated_encoding=generated,
                match=match,
                similarity=similarity
            )

            results.append(result)

            # Print progress
            if i % 10 == 0 or i == len(test_set):
                status = "✓" if match else "✗"
                print(f"{i:3}/{len(test_set)} {status} HZL#{hzl_id:4} {unicode_char} : "
                      f"{generated or 'NONE':20} (expected: {expected:20}) "
                      f"[{similarity:.0%}]")

        return results

    def _compare_encodings(self, generated: Optional[str], expected: str) -> Tuple[bool, float]:
        """
        Compare generated encoding with expected.

        Returns:
            Tuple of (exact_match, similarity_score)
        """
        if generated is None:
            return False, 0.0

        # Exact match
        if generated == expected:
            return True, 1.0

        # Partial similarity (based on string similarity)
        similarity = self._calculate_similarity(generated, expected)

        # Consider a "match" if similarity is high (> 80%)
        match = similarity > 0.8

        return match, similarity

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate similarity between two encoding strings.

        Uses Levenshtein-based similarity.
        """
        if not s1 or not s2:
            return 0.0

        # Simple character overlap ratio
        s1_chars = set(s1)
        s2_chars = set(s2)

        overlap = len(s1_chars & s2_chars)
        total = len(s1_chars | s2_chars)

        return overlap / total if total > 0 else 0.0

    def print_summary(self, results: List[ValidationResult]):
        """Print validation summary statistics."""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)

        total = len(results)
        exact_matches = sum(1 for r in results if r.match)
        partial_matches = sum(1 for r in results if not r.match but r.similarity > 0.5)
        failures = sum(1 for r in results if r.generated_encoding is None)

        avg_similarity = sum(r.similarity for r in results) / total if total > 0 else 0

        print(f"Total signs tested:     {total}")
        print(f"Exact matches:          {exact_matches} ({exact_matches/total*100:.1f}%)")
        print(f"Partial matches:        {partial_matches} ({partial_matches/total*100:.1f}%)")
        print(f"Failures (no encoding): {failures} ({failures/total*100:.1f}%)")
        print(f"Average similarity:     {avg_similarity:.1%}")

        # Print worst cases
        print("\n" + "="*70)
        print("WORST CASES (Lowest Similarity)")
        print("="*70)

        worst = sorted(results, key=lambda r: r.similarity)[:10]
        for r in worst:
            print(f"{r.unicode_char} {r.code_point}: "
                  f"{r.generated_encoding or 'NONE':20} vs {r.expected_encoding:20} "
                  f"[{r.similarity:.0%}]")

        # Print best matches
        print("\n" + "="*70)
        print("BEST MATCHES (Exact or High Similarity)")
        print("="*70)

        best = [r for r in results if r.similarity > 0.9][:10]
        for r in best:
            print(f"✓ {r.unicode_char} {r.code_point}: {r.generated_encoding}")


if __name__ == "__main__":
    # Run validation
    print("="*70)
    print("HZL ENCODING VALIDATION")
    print("="*70)

    validator = HZLValidator()

    # Test on first 50 signs
    results = validator.validate(limit=50)

    # Print summary
    validator.print_summary(results)

    # Save results
    output_dir = Path(__file__).parent.parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "hzl_validation_results.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("HZL Validation Results\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            status = "✓" if r.match else "✗"
            f.write(f"{status} {r.unicode_char} ({r.code_point}) HZL#{r.hzl_id}\n")
            f.write(f"  Expected:  {r.expected_encoding}\n")
            f.write(f"  Generated: {r.generated_encoding or 'NONE'}\n")
            f.write(f"  Similarity: {r.similarity:.0%}\n\n")

    print(f"\nDetailed results saved to: {output_path}")
