"""Diagnose the fixed Step-2 training signal without changing checkpoints."""

import argparse
import csv
import json
import os
import sys

import numpy as np


MODES = ("ce_control", "structural_rl_v4", "visual_structural_rl_v6")


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
    if "ce_control" not in modes or "visual_structural_rl_v6" not in modes:
        raise argparse.ArgumentTypeError("--modes must include ce_control and visual_structural_rl_v6")
    return modes


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="bin/web/correct/new_metrics/rl_step2")
    parser.add_argument("--evaluations-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--modes", type=parse_modes, default=MODES)
    parser.add_argument("--seeds", type=parse_seeds, default=(101, 202, 303))
    parser.add_argument("--zero-advantage-tolerance", type=float, default=1e-8)
    return parser.parse_args(argv)


def read_jsonl(path):
    with open(path) as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows:
        raise ValueError("{} contains no records".format(path))
    return rows


def read_csv(path):
    with open(path, newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("{} contains no rows".format(path))
    return rows


def finite_values(rows, field):
    values = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("{} is non-finite".format(field))
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def describe(values):
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "standard_deviation": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def pearson_correlation(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def run_diagnostics(records, mode, zero_tolerance):
    result = {
        "steps": len(records),
        "ce_loss": describe(finite_values(records, "ce_loss")),
        "policy_loss": describe(finite_values(records, "policy_loss")),
        "total_loss": describe(finite_values(records, "total_loss")),
        "gradient_norm": describe(finite_values(records, "gradient_norm")),
        "training_render_failures": int(sum(
            row.get("sample_render_success") == 0 or row.get("greedy_render_success") == 0
            for row in records
        )),
    }
    if mode == "ce_control":
        return result

    advantages = finite_values(records, "advantage")
    sample_reward_field = "sample_reward_v4" if mode == "structural_rl_v4" else "sample_reward_v6"
    greedy_reward_field = "greedy_reward_v4" if mode == "structural_rl_v4" else "greedy_reward_v6"
    result.update({
        "advantage": describe(advantages),
        "near_zero_advantage_rate": float(np.mean(np.abs(advantages) <= zero_tolerance)),
        "sample_reward": describe(finite_values(records, sample_reward_field)),
        "greedy_reward": describe(finite_values(records, greedy_reward_field)),
        "advantage_policy_loss_correlation": pearson_correlation(
            advantages, finite_values(records, "policy_loss")
        ),
    })
    return result


def evaluation_diagnostics(rows):
    reward = finite_values(rows, "reward_v6")
    visual = finite_values(rows, "visual_score_v3")
    failures = [
        {"sample": row["sample"], "render_error": row.get("render_error", "")}
        for row in rows if float(row.get("render_success", 0.0)) != 1.0
    ]
    return {
        "samples": len(rows),
        "reward_v6_visual_score_v3_correlation": pearson_correlation(reward, visual),
        "render_failures": failures,
    }


def build_report(args):
    if args.zero_advantage_tolerance < 0.0:
        raise ValueError("--zero-advantage-tolerance must be non-negative")
    evaluations_root = args.evaluations_root or os.path.join(args.input_root, "evaluations")
    runs = {}
    evaluation = {}
    for mode in args.modes:
        runs[mode] = {}
        evaluation[mode] = {}
        for seed in args.seeds:
            run_path = os.path.join(args.input_root, mode, "seed_{}".format(seed), "rl_metrics.jsonl")
            eval_path = os.path.join(evaluations_root, mode, "seed_{}".format(seed), "extended_metrics.csv")
            runs[mode][str(seed)] = run_diagnostics(read_jsonl(run_path), mode, args.zero_advantage_tolerance)
            evaluation[mode][str(seed)] = evaluation_diagnostics(read_csv(eval_path))

    aggregate = {}
    for mode in args.modes:
        aggregate[mode] = {}
        for field in ("ce_loss", "policy_loss", "total_loss", "gradient_norm", "advantage"):
            means = [runs[mode][str(seed)].get(field, {}).get("mean") for seed in args.seeds]
            values = np.asarray([value for value in means if value is not None], dtype=np.float64)
            aggregate[mode][field + "_mean_across_seeds"] = describe(values)
        if mode != "ce_control":
            aggregate[mode]["near_zero_advantage_rate_across_seeds"] = describe(np.asarray([
                runs[mode][str(seed)]["near_zero_advantage_rate"] for seed in args.seeds
            ], dtype=np.float64))
        correlations = np.asarray([
            evaluation[mode][str(seed)]["reward_v6_visual_score_v3_correlation"]
            for seed in args.seeds
            if evaluation[mode][str(seed)]["reward_v6_visual_score_v3_correlation"] is not None
        ], dtype=np.float64)
        aggregate[mode]["reward_v6_visual_score_v3_correlation_across_seeds"] = describe(correlations)

    return {
        "input_root": args.input_root,
        "evaluations_root": evaluations_root,
        "modes": list(args.modes),
        "seeds": list(args.seeds),
        "zero_advantage_tolerance": args.zero_advantage_tolerance,
        "run_diagnostics": runs,
        "evaluation_diagnostics": evaluation,
        "aggregate": aggregate,
    }


def write_report(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "step2_diagnostics.json")
    csv_path = os.path.join(output_dir, "step2_diagnostics.csv")
    with open(json_path, "w") as destination:
        json.dump(report, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
    rows = []
    for mode, seeds in report["run_diagnostics"].items():
        for seed, values in seeds.items():
            row = {"mode": mode, "seed": seed, "steps": values["steps"]}
            for field in ("ce_loss", "policy_loss", "total_loss", "gradient_norm", "advantage"):
                summary = values.get(field)
                if summary:
                    for name, value in summary.items():
                        row["{}_{}".format(field, name)] = value
            row["near_zero_advantage_rate"] = values.get("near_zero_advantage_rate")
            row["training_render_failures"] = values["training_render_failures"]
            row["reward_v6_visual_score_v3_correlation"] = report["evaluation_diagnostics"][mode][seed][
                "reward_v6_visual_score_v3_correlation"
            ]
            row["evaluation_render_failures"] = len(report["evaluation_diagnostics"][mode][seed]["render_failures"])
            rows.append(row)
    with open(csv_path, "w", newline="") as destination:
        fieldnames = sorted({field for row in rows for field in row})
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main(argv):
    args = parse_args(argv)
    report = build_report(args)
    output_dir = args.output_dir or os.path.join(args.input_root, "diagnostics")
    json_path, csv_path = write_report(report, output_dir)
    print("[step2_diagnostics] json={}".format(json_path))
    print("[step2_diagnostics] csv={}".format(csv_path))


if __name__ == "__main__":
    main(sys.argv[1:])
