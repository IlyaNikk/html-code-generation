"""Synthetic end-to-end checks for the Step-2 experiment reporter."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from model.tests.report_step2_results import METRICS, build_report, parse_args, write_report


MODES = ("ce_control", "structural_rl_v4", "visual_structural_rl_v6")
SEEDS = (101, 202, 303)
SAMPLES = ("sample-a", "sample-b", "sample-c", "sample-d")


def evaluation_row(sample, visual_score, parent_edge):
    row = {
        "sample": sample,
        "visual_score_v3": visual_score,
        "reward_v6": visual_score,
        "parent_edge_f1": parent_edge,
        "token_f1": parent_edge,
        "text_token_f1": parent_edge,
        "button_token_f1": parent_edge,
        "syntax_valid": 1.0,
        "render_success": 1.0,
        "overlong": 0.0,
        "target_screenshot": "targets/{}.png".format(sample),
        "predicted_screenshot": "predictions/{}.png".format(sample),
    }
    return row


class Step2ReporterTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "rl_step2"
        self.write_csv(
            self.root / "evaluations" / "supervised_constrained" / "extended_metrics.csv",
            [evaluation_row(sample, 0.50, 0.50) for sample in SAMPLES],
        )
        for mode in MODES:
            for seed in SEEDS:
                run_dir = self.root / mode / "seed_{}".format(seed)
                run_dir.mkdir(parents=True)
                config = {
                    "mode": mode,
                    "steps": 2,
                    "seed": seed,
                    "source_weights_path": "bin/web/correct/new_metrics",
                    "source_checkpoint_sha256": "same-checkpoint",
                    "profile": "web_generated",
                    "input_path": "datasets/generated/web/article/training_set",
                    "min_target_length": 0,
                    "max_target_length": 100,
                    "sequence_length": 100,
                    "learning_rate": 1e-6,
                    "ce_weight": 0.05,
                }
                (run_dir / "run_config.json").write_text(json.dumps(config))
                (run_dir / "training_samples.json").write_text(json.dumps({
                    "candidate_order": list(SAMPLES),
                    "step_samples": ["sample-a", "sample-b"],
                }))
                (run_dir / "rl_metrics.jsonl").write_text(
                    "\n".join(json.dumps({"total_loss": 0.1, "gradient_norm": 0.2}) for _ in range(2)) + "\n"
                )
                (run_dir / "Main_Model.weights.h5").write_bytes(b"checkpoint")
                visual_score = 0.50
                parent_edge = 0.50
                if mode == "structural_rl_v4":
                    visual_score = 0.55
                if mode == "visual_structural_rl_v6":
                    visual_score = 0.60
                    parent_edge = 0.52
                self.write_csv(
                    self.root / "evaluations" / mode / "seed_{}".format(seed) / "extended_metrics.csv",
                    [evaluation_row(sample, visual_score, parent_edge) for sample in SAMPLES],
                )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_csv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({field for row in rows for field in row})
        with path.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_report_validates_and_aggregates_all_seeds(self):
        args = parse_args([
            "--input-root", str(self.root), "--expected-steps", "2",
            "--bootstrap-samples", "100",
        ])
        report = build_report(args)
        readiness = report["step_3_readiness"]
        self.assertEqual(3, readiness["visual_score_v3_positive_lower_ci_seeds"])
        self.assertTrue(readiness["ready_for_step_3"])
        self.assertAlmostEqual(
            0.10,
            report["seed_matched_comparisons"]["visual_structural_rl_v6"]["visual_score_v3"]
            ["delta_across_seeds"]["mean"],
        )
        output_dir = self.root / "report"
        paths = write_report(report, output_dir)
        self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_every_report_metric_is_present_in_fixture(self):
        row = evaluation_row("sample", 0.5, 0.5)
        self.assertTrue(set(METRICS).issubset(row))

    def test_replication_report_accepts_only_ce_and_visual_rl(self):
        manifest_path = self.root / "external_eval_manifest.json"
        manifest_path.write_text(json.dumps({
            "evaluation_samples": list(SAMPLES), "split_seed": 404, "distribution": 12,
        }))
        args = parse_args([
            "--input-root", str(self.root), "--expected-steps", "2",
            "--bootstrap-samples", "100", "--modes", "ce_control,visual_structural_rl_v6",
            "--expected-eval-manifest", str(manifest_path),
        ])
        report = build_report(args)
        self.assertEqual(
            ["ce_control", "visual_structural_rl_v6"], report["experiment"]["modes"]
        )
        self.assertEqual(
            {"visual_structural_rl_v6"}, set(report["seed_matched_comparisons"])
        )
        self.assertEqual(404, report["experiment"]["expected_eval_manifest"]["split_seed"])


if __name__ == "__main__":
    unittest.main()
