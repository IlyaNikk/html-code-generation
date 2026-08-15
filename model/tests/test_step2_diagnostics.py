"""Checks for the read-only Step-2 signal diagnostic."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from model.tests.evaluate_extended import parse_args as parse_evaluator_args
from model.tests.report_step2_diagnostics import build_report, parse_args


class Step2DiagnosticsTest(unittest.TestCase):
    def test_replication_generator_requires_an_explicit_seeded_count(self):
        from compiler.generate_dataset import parse_args
        args = parse_args(["1020", "--seed", "404", "--output-dir", "dataset"])
        self.assertEqual(1020, args.count)
        self.assertEqual(404, args.seed)
        self.assertEqual("dataset", args.output_dir)

    def test_evaluator_accepts_one_fixed_sample(self):
        args = parse_evaluator_args(["--only-sample", "fixed-sample"])
        self.assertEqual("fixed-sample", args.only_sample)

    def test_renderer_worker_accepts_page_timeout(self):
        from model.tests.render_html_worker import parse_args
        args = parse_args(["--input", "input.html", "--output", "output.png", "--page-timeout-seconds", "120"])
        self.assertEqual(120, args.page_timeout_seconds)

    def test_reports_advantage_and_reward_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rl_step2"
            evaluation_root = root / "evaluations"
            for mode in ("ce_control", "structural_rl_v4", "visual_structural_rl_v6"):
                for seed in (101, 202, 303):
                    run_dir = root / mode / "seed_{}".format(seed)
                    run_dir.mkdir(parents=True)
                    records = []
                    for step in range(2):
                        records.append({
                            "ce_loss": 0.1, "policy_loss": 0.2, "total_loss": 0.3,
                            "gradient_norm": 0.4, "advantage": None if mode == "ce_control" else 0.1,
                            "sample_reward_v4": 0.6, "greedy_reward_v4": 0.5,
                            "sample_reward_v6": 0.7, "greedy_reward_v6": 0.6,
                            "sample_render_success": 1, "greedy_render_success": 1,
                        })
                    (run_dir / "rl_metrics.jsonl").write_text(
                        "\n".join(json.dumps(record) for record in records) + "\n"
                    )
                    eval_dir = evaluation_root / mode / "seed_{}".format(seed)
                    eval_dir.mkdir(parents=True)
                    with (eval_dir / "extended_metrics.csv").open("w", newline="") as destination:
                        writer = csv.DictWriter(destination, fieldnames=[
                            "sample", "reward_v6", "visual_score_v3", "render_success", "render_error"
                        ])
                        writer.writeheader()
                        writer.writerows([
                            {"sample": "a", "reward_v6": 0.2, "visual_score_v3": 0.3, "render_success": 1, "render_error": ""},
                            {"sample": "b", "reward_v6": 0.8, "visual_score_v3": 0.9, "render_success": 1, "render_error": ""},
                        ])
            report = build_report(parse_args([
                "--input-root", str(root), "--evaluations-root", str(evaluation_root),
            ]))
            run = report["run_diagnostics"]["visual_structural_rl_v6"]["101"]
            self.assertEqual(2, run["advantage"]["count"])
            self.assertEqual(0.0, run["near_zero_advantage_rate"])
            self.assertAlmostEqual(
                1.0,
                report["evaluation_diagnostics"]["visual_structural_rl_v6"]["101"]
                ["reward_v6_visual_score_v3_correlation"],
            )


if __name__ == "__main__":
    unittest.main()
