"""Focused regression tests for the foreground-and-grid visual score."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from model.tests.evaluate_extended import (
    foreground_mask,
    rgb_arrays_for_comparison,
    rgb_uint8_arrays_for_comparison,
    visual_score_v3_components,
    visual_structural_v6_components,
)


class VisualScoreV3Test(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def image_path(self, name, rectangle=None):
        image = Image.new("RGB", (80, 80), "white")
        if rectangle is not None:
            ImageDraw.Draw(image).rectangle(rectangle, fill="black")
        path = self.directory / name
        image.save(path)
        return path

    def test_identical_images_score_one(self):
        image = self.image_path("image.png", (5, 5, 20, 20))
        result = visual_score_v3_components(image, image)
        self.assertEqual(1.0, result["visual_score_v3"])
        self.assertEqual(1.0, result["foreground_f1_v3"])
        self.assertEqual(1.0, result["foreground_grid_f1_v3"])

    def test_missing_and_extra_elements_are_penalised(self):
        target = self.image_path("target.png", (5, 5, 20, 20))
        blank = self.image_path("blank.png")
        extra = self.image_path("extra.png", (50, 50, 65, 65))
        missing_result = visual_score_v3_components(target, blank)
        extra_result = visual_score_v3_components(blank, extra)
        self.assertEqual(0.0, missing_result["foreground_recall_v3"])
        self.assertEqual(0.0, extra_result["foreground_precision_v3"])
        self.assertLess(missing_result["visual_score_v3"], 0.1)
        self.assertLess(extra_result["visual_score_v3"], 0.1)

    def test_shifted_foreground_is_penalised_symmetrically(self):
        left = self.image_path("left.png", (5, 5, 20, 20))
        right = self.image_path("right.png", (55, 55, 70, 70))
        forward = visual_score_v3_components(left, right)
        reverse = visual_score_v3_components(right, left)
        self.assertLess(forward["foreground_grid_f1_v3"], 1.0)
        self.assertLess(forward["visual_score_v3"], 1.0)
        self.assertEqual(forward["visual_score_v3"], reverse["visual_score_v3"])

    def test_v6_uses_v3_without_changing_v5(self):
        row = {
            "syntax_valid": 1,
            "render_success": 1,
            "hit_sequence_limit": 0,
            "ended_too_early": 0,
            "overlong": 0,
            "visual_score_v3": 0.20,
            "parent_edge_f1": 1.0,
            "token_f1": 1.0,
            "button_token_f1": 1.0,
            "text_token_f1": 1.0,
            "length_score": 1.0,
            "chrf": 1.0,
            "tree_similarity": 1.0,
        }
        result = visual_structural_v6_components(row)
        self.assertAlmostEqual(0.80, result["reward_base_v6"])
        self.assertAlmostEqual(0.80, result["reward_v6"])

    def test_uint8_v3_masks_match_the_original_float_thresholds(self):
        first = self.image_path("first.png", (5, 5, 20, 20))
        second = self.image_path("second.png", (10, 10, 30, 30))
        float_first, float_second = rgb_arrays_for_comparison(first, second)
        uint8_first, uint8_second = rgb_uint8_arrays_for_comparison(first, second)
        self.assertTrue((foreground_mask(float_first) == foreground_mask(uint8_first)).all())
        self.assertTrue((foreground_mask(float_second) == foreground_mask(uint8_second)).all())


if __name__ == "__main__":
    unittest.main()
