"""Validate and aggregate the fixed Step-2 SFT/RL experiment.

The report deliberately refuses incomplete or incomparable experiments.  It
uses seed-matched paired evaluation rows, never a hand-picked checkpoint, and
records the evidence needed to decide whether the visual reward may proceed to
the Step-3 grounding ablation.
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np


MODES = ("ce_control", "structural_rl_v4", "visual_structural_rl_v6")
PRIMARY_MODE = "visual_structural_rl_v6"
METRICS = (
    "visual_score_v3", "reward_v6", "parent_edge_f1", "token_f1",
    "text_token_f1", "button_token_f1", "syntax_valid", "render_success",
    "overlong",
)
CONFIG_KEYS = (
    "source_weights_path", "source_checkpoint_sha256", "profile", "input_path",
    "min_target_length", "max_target_length", "sequence_length", "learning_rate",
    "ce_weight",
)


def target_token_count(gui_path):
    """Match evaluate_extended.py token counting without importing model runtime modules."""
    count = 0
    with open(gui_path) as source:
        for raw_line in source:
            token = re.sub(r"\s+", "", raw_line)
            if not token:
                continue
            if "{" in token:
                token = token.replace("{", "")
                if "," in token:
                    leaves = token.split(",")
                    count += sum(1 for leaf in leaves[:-1] if leaf)
                    token = leaves[-1]
                if token:
                    count += 1
            elif "}" not in token:
                count += sum(1 for leaf in token.split(",") if leaf)
    return count


def parse_seeds(value):
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must be non-empty and unique")
    return seeds


def parse_modes(value):
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not modes or len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("--modes must be a non-empty comma-separated list without duplicates")
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise argparse.ArgumentTypeError("unknown mode(s): {}".format(", ".join(unknown)))
    if "ce_control" not in modes or PRIMARY_MODE not in modes:
        raise argparse.ArgumentTypeError("--modes must include ce_control and {}".format(PRIMARY_MODE))
    return modes


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="bin/web/correct/new_metrics/rl_step2")
    parser.add_argument(
        "--evaluations-root", default=None,
        help="evaluation root; defaults to <input-root>/evaluations",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--expected-eval-manifest", default=None,
        help="optional split manifest whose evaluation_samples must exactly match the baseline evaluation",
    )
    parser.add_argument(
        "--expected-eval-input-path", default=None,
        help="directory containing manifest samples; enables applying the evaluation length filter to the manifest",
    )
    parser.add_argument("--expected-eval-min-target-length", type=int, default=0)
    parser.add_argument("--expected-eval-max-target-length", type=int, default=0)
    parser.add_argument(
        "--modes", type=parse_modes, default=MODES,
        help="comma-separated modes; replication may omit structural_rl_v4",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=(101, 202, 303))
    parser.add_argument("--expected-steps", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args(argv)


def read_json(path):
    with open(path) as source:
        return json.load(source)


def read_rows(path):
    with open(path, newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("{} contains no evaluation rows".format(path))
    mapped = {row["sample"]: row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError("{} contains duplicate sample identifiers".format(path))
    return mapped


def validate_expected_eval_manifest(path, samples, input_path=None, min_target_length=0, max_target_length=0):
    manifest = read_json(require_file(path, "expected eval manifest"))
    expected = manifest.get("evaluation_samples")
    if not isinstance(expected, list) or not all(isinstance(sample, str) for sample in expected):
        raise ValueError("{} must contain a string list named evaluation_samples".format(path))
    if len(expected) != len(set(expected)):
        raise ValueError("{} contains duplicate evaluation_samples".format(path))
    if min_target_length < 0 or max_target_length < 0:
        raise ValueError("expected eval target-length bounds must be non-negative")
    if input_path:
        filtered_expected = []
        for sample in expected:
            gui_path = os.path.join(input_path, "{}.gui".format(sample))
            image_path = os.path.join(input_path, "{}.png".format(sample))
            if not os.path.isfile(gui_path) or not os.path.isfile(image_path):
                raise ValueError("expected eval sample is missing its GUI or image: {}".format(sample))
            target_length = target_token_count(gui_path)
            if target_length < min_target_length:
                continue
            if max_target_length and target_length > max_target_length:
                continue
            filtered_expected.append(sample)
        expected = filtered_expected
    if set(expected) != samples:
        raise ValueError("{} does not match the immutable baseline eval sample set".format(path))
    return {
        "path": os.path.abspath(path),
        "samples": len(expected),
        "split_seed": manifest.get("split_seed"),
        "distribution": manifest.get("distribution"),
        "input_path": os.path.abspath(input_path) if input_path else None,
        "min_target_length": min_target_length if input_path else None,
        "max_target_length": max_target_length if input_path else None,
    }


def numeric_rows(rows, samples, metric):
    try:
        values = np.asarray([float(rows[sample][metric]) for sample in samples], dtype=np.float64)
    except KeyError as exc:
        raise ValueError("metric {!r} is missing from an evaluation; rerun evaluator".format(metric)) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("metric {!r} has non-finite values".format(metric))
    return values


def read_log_count(path):
    with open(path) as source:
        records = [json.loads(line) for line in source if line.strip()]
    if any(not np.isfinite(float(record["total_loss"])) for record in records):
        raise ValueError("{} contains a non-finite total loss".format(path))
    if any(not np.isfinite(float(record["gradient_norm"])) for record in records):
        raise ValueError("{} contains a non-finite gradient norm".format(path))
    return len(records)


def require_file(path, description):
    if not os.path.isfile(path):
        raise ValueError("missing {}: {}".format(description, path))
    return path


def validate_run(input_root, mode, seed, expected_steps):
    run_dir = os.path.join(input_root, mode, "seed_{}".format(seed))
    config_path = require_file(os.path.join(run_dir, "run_config.json"), "run config")
    samples_path = require_file(os.path.join(run_dir, "training_samples.json"), "training-sample manifest")
    log_path = require_file(os.path.join(run_dir, "rl_metrics.jsonl"), "step log")
    require_file(os.path.join(run_dir, "Main_Model.weights.h5"), "fine-tuned checkpoint")
    config = read_json(config_path)
    sample_manifest = read_json(samples_path)
    if config.get("mode") != mode:
        raise ValueError("{} reports mode {!r}, expected {!r}".format(run_dir, config.get("mode"), mode))
    if int(config.get("steps", -1)) != expected_steps:
        raise ValueError("{} has steps={}, expected {}".format(run_dir, config.get("steps"), expected_steps))
    if int(config.get("seed", -1)) != seed:
        raise ValueError("{} has seed={}, expected {}".format(run_dir, config.get("seed"), seed))
    if read_log_count(log_path) != expected_steps:
        raise ValueError("{} does not contain {} JSONL steps".format(log_path, expected_steps))
    candidate_order = sample_manifest.get("candidate_order")
    step_samples = sample_manifest.get("step_samples")
    if not isinstance(candidate_order, list) or not isinstance(step_samples, list):
        raise ValueError("{} has an invalid training-sample manifest".format(samples_path))
    if len(step_samples) != expected_steps:
        raise ValueError("{} does not list {} step samples".format(samples_path, expected_steps))
    if any(sample not in candidate_order for sample in step_samples):
        raise ValueError("{} contains a step sample outside candidate_order".format(samples_path))
    return config, sample_manifest


def paired_bootstrap(reference, comparator, bootstrap_indices):
    delta = reference - comparator
    boot_means = delta[bootstrap_indices].mean(axis=1)
    return {
        "reference_mean": float(reference.mean()),
        "comparator_mean": float(comparator.mean()),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "paired_bootstrap_95_ci": [
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        ],
        "reference_better_samples": int(np.sum(delta > 0.0)),
        "comparator_better_samples": int(np.sum(delta < 0.0)),
        "tied_samples": int(np.sum(delta == 0.0)),
    }


def across_seed_summary(seed_values):
    ordered = np.asarray([seed_values[str(seed)] for seed in sorted(map(int, seed_values))], dtype=np.float64)
    return {
        "seed_values": seed_values,
        "mean": float(ordered.mean()),
        "median": float(np.median(ordered)),
        "standard_deviation": float(ordered.std(ddof=0)),
    }


def require_same_values(configurations):
    reference_name = sorted(configurations)[0]
    reference = configurations[reference_name]
    for name, config in configurations.items():
        for key in CONFIG_KEYS:
            if config.get(key) != reference.get(key):
                raise ValueError("run configs differ for {!r}: {} vs {}".format(key, reference_name, name))


def report_examples(samples, reference_rows, candidate_rows, seeds):
    deltas = {}
    for sample in samples:
        deltas[sample] = float(np.mean([
            float(candidate_rows[seed][sample]["visual_score_v3"])
            - float(reference_rows[seed][sample]["visual_score_v3"])
            for seed in seeds
        ]))
    ordered = sorted(samples, key=lambda sample: (deltas[sample], sample))
    selected = [("largest_degradation", sample) for sample in ordered[:4]]
    selected.extend(("largest_improvement", sample) for sample in ordered[-4:][::-1])
    rows = []
    for category, sample in selected:
        row = {
            "category": category,
            "sample": sample,
            "mean_visual_score_v3_delta": deltas[sample],
            "target_screenshot": candidate_rows[seeds[0]][sample].get("target_screenshot", ""),
        }
        for seed in seeds:
            row["prediction_seed_{}".format(seed)] = candidate_rows[seed][sample].get("predicted_screenshot", "")
        rows.append(row)
    return rows


def build_report(args):
    if args.expected_steps <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("--expected-steps and --bootstrap-samples must be positive")
    input_root = args.input_root
    evaluations_root = args.evaluations_root or os.path.join(input_root, "evaluations")
    baseline_path = require_file(
        os.path.join(evaluations_root, "supervised_constrained", "extended_metrics.csv"),
        "supervised constrained baseline",
    )
    baseline_rows = read_rows(baseline_path)
    all_rows, all_configs, all_manifests = {}, {}, {}
    expected_samples = set(baseline_rows)
    external_eval_manifest = None
    if args.expected_eval_manifest:
        external_eval_manifest = validate_expected_eval_manifest(
            args.expected_eval_manifest,
            expected_samples,
            args.expected_eval_input_path,
            args.expected_eval_min_target_length,
            args.expected_eval_max_target_length,
        )
    comparison_modes = tuple(mode for mode in args.modes if mode != "ce_control")

    for mode in args.modes:
        all_rows[mode] = {}
        all_configs[mode] = {}
        all_manifests[mode] = {}
        for seed in args.seeds:
            config, manifest = validate_run(input_root, mode, seed, args.expected_steps)
            eval_path = require_file(
                os.path.join(evaluations_root, mode, "seed_{}".format(seed), "extended_metrics.csv"),
                "constrained evaluation",
            )
            rows = read_rows(eval_path)
            if set(rows) != expected_samples:
                raise ValueError("{} does not use the immutable baseline eval sample set".format(eval_path))
            all_rows[mode][seed] = rows
            all_configs[mode][seed] = config
            all_manifests[mode][seed] = manifest

    flat_configs = {
        "{}/{}".format(mode, seed): all_configs[mode][seed]
        for mode in args.modes for seed in args.seeds
    }
    require_same_values(flat_configs)
    for seed in args.seeds:
        reference_manifest = all_manifests["ce_control"][seed]
        for mode in comparison_modes:
            if all_manifests[mode][seed] != reference_manifest:
                raise ValueError("training-sample order differs for seed {} between {} and ce_control".format(seed, mode))

    samples = sorted(expected_samples)
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(0, len(samples), size=(args.bootstrap_samples, len(samples)))
    aggregate = {mode: {} for mode in args.modes}
    comparisons = {mode: {} for mode in comparison_modes}
    for metric in METRICS:
        for mode in args.modes:
            seed_means = {
                str(seed): float(numeric_rows(all_rows[mode][seed], samples, metric).mean())
                for seed in args.seeds
            }
            aggregate[mode][metric] = across_seed_summary(seed_means)
        for mode in comparison_modes:
            by_seed = {}
            delta_means = {}
            positive_lower_ci_count = 0
            for seed in args.seeds:
                candidate = numeric_rows(all_rows[mode][seed], samples, metric)
                control = numeric_rows(all_rows["ce_control"][seed], samples, metric)
                summary = paired_bootstrap(candidate, control, bootstrap_indices)
                by_seed[str(seed)] = summary
                delta_means[str(seed)] = summary["delta_mean"]
                positive_lower_ci_count += int(summary["paired_bootstrap_95_ci"][0] > 0.0)
            comparisons[mode][metric] = {
                "by_seed": by_seed,
                "delta_across_seeds": across_seed_summary(delta_means),
                "positive_lower_ci_seed_count": positive_lower_ci_count,
            }

    visual = comparisons[PRIMARY_MODE]
    readiness = {
        "primary_comparison": "{}_minus_ce_control".format(PRIMARY_MODE),
        "visual_score_v3_positive_lower_ci_seeds": visual["visual_score_v3"]["positive_lower_ci_seed_count"],
        "visual_score_v3_two_seed_gate": visual["visual_score_v3"]["positive_lower_ci_seed_count"] >= 2,
        "parent_edge_f1_delta": visual["parent_edge_f1"]["delta_across_seeds"]["mean"],
        "parent_edge_f1_not_worse": visual["parent_edge_f1"]["delta_across_seeds"]["mean"] >= 0.0,
        "syntax_valid_rate_delta": visual["syntax_valid"]["delta_across_seeds"]["mean"],
        "render_success_rate_delta": visual["render_success"]["delta_across_seeds"]["mean"],
        "quality_rate_floor": -0.02,
    }
    readiness["quality_rates_within_two_percentage_points"] = (
        readiness["syntax_valid_rate_delta"] >= readiness["quality_rate_floor"]
        and readiness["render_success_rate_delta"] >= readiness["quality_rate_floor"]
    )
    readiness["ready_for_step_3"] = (
        readiness["visual_score_v3_two_seed_gate"]
        and readiness["parent_edge_f1_not_worse"]
        and readiness["quality_rates_within_two_percentage_points"]
    )

    examples = report_examples(
        samples, all_rows["ce_control"], all_rows[PRIMARY_MODE], args.seeds
    )
    return {
        "experiment": {
            "input_root": input_root,
            "evaluations_root": evaluations_root,
            "modes": list(args.modes),
            "seeds": list(args.seeds),
            "expected_steps": args.expected_steps,
            "eval_samples": len(samples),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "expected_eval_manifest": external_eval_manifest,
            "shared_run_config": {key: flat_configs[sorted(flat_configs)[0]].get(key) for key in CONFIG_KEYS},
        },
        "aggregate_metrics": aggregate,
        "seed_matched_comparisons": comparisons,
        "step_3_readiness": readiness,
        "manual_review_examples": examples,
    }


def write_report(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "step2_report.json")
    csv_path = os.path.join(output_dir, "step2_summary.csv")
    examples_path = os.path.join(output_dir, "visual_score_v3_delta_examples.csv")
    with open(json_path, "w") as destination:
        json.dump(report, destination, indent=2, ensure_ascii=False)
        destination.write("\n")

    summary_rows = []
    for mode, metrics in report["aggregate_metrics"].items():
        for metric, value in metrics.items():
            summary_rows.append({"row_type": "aggregate", "mode": mode, "metric": metric, **value})
    for mode, metrics in report["seed_matched_comparisons"].items():
        for metric, value in metrics.items():
            summary_rows.append({
                "row_type": "comparison_vs_ce_control",
                "mode": mode,
                "metric": metric,
                "delta_mean": value["delta_across_seeds"]["mean"],
                "delta_median": value["delta_across_seeds"]["median"],
                "delta_standard_deviation": value["delta_across_seeds"]["standard_deviation"],
                "positive_lower_ci_seed_count": value["positive_lower_ci_seed_count"],
                "by_seed": json.dumps(value["by_seed"]),
            })
    with open(csv_path, "w", newline="") as destination:
        fieldnames = sorted({field for row in summary_rows for field in row})
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(examples_path, "w", newline="") as destination:
        fieldnames = sorted({field for row in report["manual_review_examples"] for field in row})
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["manual_review_examples"])
    return json_path, csv_path, examples_path


def main(argv):
    args = parse_args(argv)
    report = build_report(args)
    output_dir = args.output_dir or os.path.join(args.input_root, "report")
    json_path, csv_path, examples_path = write_report(report, output_dir)
    readiness = report["step_3_readiness"]
    print("[step2_report] samples={} seeds={}".format(report["experiment"]["eval_samples"], args.seeds))
    print("[step2_report] visual lower-CI-positive seeds={}".format(
        readiness["visual_score_v3_positive_lower_ci_seeds"]
    ))
    print("[step2_report] ready_for_step_3={}".format(readiness["ready_for_step_3"]))
    print("[step2_report] json={}".format(json_path))
    print("[step2_report] csv={}".format(csv_path))
    print("[step2_report] review_examples={}".format(examples_path))


if __name__ == "__main__":
    main(sys.argv[1:])
