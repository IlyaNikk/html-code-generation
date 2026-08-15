"""Summarise paired original/blank/shuffled image-ablation evaluation results."""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np


DEFAULT_INPUT_ROOT = "bin/web/correct/new_metrics/image_ablation"
METRICS = [
    "visual_score_v2", "visual_score_v3", "foreground_f1_v3", "foreground_grid_f1_v3",
    "reward_v5", "parent_edge_f1", "token_f1",
    "text_token_f1", "button_token_f1", "tree_similarity", "syntax_valid"
]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args(argv)


def read_rows(path):
    with open(path, newline="") as f:
        rows = {row["sample"]: row for row in csv.DictReader(f)}
    if not rows:
        raise ValueError("{} contains no metric rows".format(path))
    return rows


def require_same_samples(named_rows):
    names = sorted(named_rows)
    expected = set(named_rows[names[0]])
    for name in names[1:]:
        if set(named_rows[name]) != expected:
            raise ValueError("{} does not contain the same target samples as {}".format(name, names[0]))
    return sorted(expected)


def values(rows, samples, metric):
    try:
        return np.asarray([float(rows[sample][metric]) for sample in samples], dtype=np.float64)
    except KeyError:
        raise ValueError("metric {!r} is absent; rerun the evaluator with Visual Score v3".format(metric))


def paired_summary(reference, comparator, bootstrap_indices):
    delta = reference - comparator
    bootstrap_means = delta[bootstrap_indices].mean(axis=1)
    return {
        "original_mean": float(reference.mean()),
        "original_median": float(np.median(reference)),
        "comparator_mean": float(comparator.mean()),
        "comparator_median": float(np.median(comparator)),
        "delta_original_minus_comparator_mean": float(delta.mean()),
        "delta_original_minus_comparator_median": float(np.median(delta)),
        "paired_bootstrap_95_ci": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "original_better_samples": int(np.sum(delta > 0.0)),
        "comparator_better_samples": int(np.sum(delta < 0.0)),
        "tied_samples": int(np.sum(delta == 0.0)),
    }


def main(argv):
    args = parse_args(argv)
    output_dir = args.output_dir or os.path.join(args.input_root, "report_v2")
    condition_paths = {
        name: os.path.join(args.input_root, name, "extended_metrics.csv")
        for name in os.listdir(args.input_root)
        if os.path.isfile(os.path.join(args.input_root, name, "extended_metrics.csv"))
    }
    if "original" not in condition_paths or "blank" not in condition_paths:
        raise SystemExit("input root must contain original and blank evaluation directories")

    shuffle_names = sorted(
        (name for name in condition_paths if re.fullmatch(r"shuffled_seed_\d+", name)),
        key=lambda name: int(name.rsplit("_", 1)[1])
    )
    if not shuffle_names:
        raise SystemExit("input root must contain at least one shuffled_seed_<N> directory")

    rows_by_condition = {name: read_rows(path) for name, path in condition_paths.items()}
    samples = require_same_samples({
        name: rows_by_condition[name]
        for name in ["original", "blank"] + shuffle_names
    })
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, len(samples), size=(args.bootstrap_samples, len(samples))
    )

    original_rows = rows_by_condition["original"]
    blank_rows = rows_by_condition["blank"]
    summaries = {}
    report_rows = []
    for comparison_name, comparator_rows in [
        ("original_minus_blank", [blank_rows]),
        ("original_minus_mean_shuffled", [rows_by_condition[name] for name in shuffle_names]),
    ]:
        summaries[comparison_name] = {}
        for metric in METRICS:
            reference = values(original_rows, samples, metric)
            comparator = np.mean(
                [values(rows, samples, metric) for rows in comparator_rows], axis=0
            )
            summary = paired_summary(reference, comparator, bootstrap_indices)
            summaries[comparison_name][metric] = summary
            report_rows.append(dict(comparison=comparison_name, metric=metric, **summary))

    ordered_samples = sorted(samples, key=lambda sample: float(original_rows[sample]["visual_score_v3"]))
    middle = len(ordered_samples) // 2
    review_examples = []
    for category, selected_samples in [
        ("low", ordered_samples[:4]),
        ("medium", ordered_samples[middle - 2:middle + 2]),
        ("high", ordered_samples[-4:]),
    ]:
        for sample in selected_samples:
            row = original_rows[sample]
            review_examples.append({
                "category": category,
                "sample": sample,
                "visual_score_v3": float(row["visual_score_v3"]),
                "target_screenshot": row.get("target_screenshot", ""),
                "predicted_screenshot": row.get("predicted_screenshot", ""),
            })

    report = {
        "samples": len(samples),
        "conditions": ["original", "blank"] + shuffle_names,
        "shuffle_conditions": shuffle_names,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "comparisons": summaries,
        "manual_review_examples": review_examples,
    }
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "image_ablation_report_v3.json")
    csv_path = os.path.join(output_dir, "image_ablation_report_v3.csv")
    examples_path = os.path.join(output_dir, "visual_score_v3_examples.csv")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    fieldnames = [
        "comparison", "metric", "original_mean", "original_median", "comparator_mean",
        "comparator_median", "delta_original_minus_comparator_mean",
        "delta_original_minus_comparator_median", "paired_bootstrap_95_ci",
        "original_better_samples", "comparator_better_samples", "tied_samples"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report_rows:
            row = dict(row)
            row["paired_bootstrap_95_ci"] = json.dumps(row["paired_bootstrap_95_ci"])
            writer.writerow(row)
    with open(examples_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "sample", "visual_score_v3", "target_screenshot", "predicted_screenshot"
        ])
        writer.writeheader()
        writer.writerows(review_examples)

    print("[image_ablation_report] samples={}".format(len(samples)))
    for comparison in ["original_minus_blank", "original_minus_mean_shuffled"]:
        print("[image_ablation_report] {}".format(comparison))
        for metric in ["visual_score_v3", "reward_v5"]:
            metric_summary = summaries[comparison][metric]
            print(
                "  {}: delta={:.6f}, 95% CI=[{:.6f}, {:.6f}]".format(
                    metric,
                    metric_summary["delta_original_minus_comparator_mean"],
                    metric_summary["paired_bootstrap_95_ci"][0],
                    metric_summary["paired_bootstrap_95_ci"][1],
                )
            )
    print("[image_ablation_report] json={}".format(json_path))
    print("[image_ablation_report] csv={}".format(csv_path))
    print("[image_ablation_report] review_examples={}".format(examples_path))


if __name__ == "__main__":
    main(sys.argv[1:])
