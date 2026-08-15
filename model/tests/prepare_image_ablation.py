"""Prepare deterministic image-ablation manifests for screenshot-to-DSL evaluation."""

import argparse
import json
import os
import random
import re
import sys


DEFAULT_INPUT_PATH = "datasets/generated/web/article/eval_set"
DEFAULT_OUTPUT_DIR = "datasets/generated/web/article/image_ablation"


def target_token_count(gui_path):
    """Match evaluate_extended.py token counting without importing model dependencies."""
    count = 0
    with open(gui_path) as gui:
        for raw_line in gui:
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


def derangement(sample_ids, seed):
    rng = random.Random(seed)
    source_ids = list(sample_ids)
    for _ in range(1000):
        rng.shuffle(source_ids)
        if all(target_id != source_id for target_id, source_id in zip(sample_ids, source_ids)):
            return source_ids
    raise RuntimeError("could not create a derangement for seed {}".format(seed))


def rows_for_mapping(sample_ids, target_lengths, mode, source_ids=None, seed=None):
    rows = []
    for index, target_id in enumerate(sample_ids):
        source_id = target_id if source_ids is None else source_ids[index]
        if mode == "blank":
            source_id = None
        rows.append({
            "target_sample": target_id,
            "target_gui": "{}.gui".format(target_id),
            "target_image": "{}.png".format(target_id),
            "source_image_sample": source_id,
            "source_image": "{}.png".format(source_id) if source_id else None,
            "target_length": target_lengths[target_id],
            "image_mode": mode,
            "seed": seed,
        })
    return rows


def write_manifest(path, metadata, rows, overwrite):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError("{} already exists; use --overwrite to replace it".format(path))
    with open(path, "w") as output:
        json.dump({"metadata": metadata, "samples": rows}, output, indent=2, sort_keys=True)
        output.write("\n")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-target-length", type=int, default=0)
    parser.add_argument("--max-target-length", type=int, default=100)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("at least one seed is required")

    sample_ids = []
    target_lengths = {}
    for file_name in sorted(os.listdir(args.input_path)):
        if not file_name.endswith(".gui"):
            continue
        sample_id = file_name[:-4]
        image_path = os.path.join(args.input_path, "{}.png".format(sample_id))
        if not os.path.isfile(image_path):
            raise FileNotFoundError("missing image for {}".format(file_name))
        length = target_token_count(os.path.join(args.input_path, file_name))
        if length < args.min_target_length:
            continue
        if args.max_target_length > 0 and length > args.max_target_length:
            continue
        sample_ids.append(sample_id)
        target_lengths[sample_id] = length

    if len(sample_ids) < 2:
        raise SystemExit("at least two filtered samples are required")

    os.makedirs(args.output_dir, exist_ok=True)
    metadata = {
        "input_path": args.input_path,
        "min_target_length": args.min_target_length,
        "max_target_length": args.max_target_length,
        "samples": len(sample_ids),
    }
    write_manifest(
        os.path.join(args.output_dir, "original.json"),
        dict(metadata, image_mode="original", seed=None),
        rows_for_mapping(sample_ids, target_lengths, "original"),
        args.overwrite,
    )
    write_manifest(
        os.path.join(args.output_dir, "blank.json"),
        dict(metadata, image_mode="blank", seed=None),
        rows_for_mapping(sample_ids, target_lengths, "blank"),
        args.overwrite,
    )
    for seed in seeds:
        write_manifest(
            os.path.join(args.output_dir, "shuffled_seed_{}.json".format(seed)),
            dict(metadata, image_mode="shuffled", seed=seed),
            rows_for_mapping(sample_ids, target_lengths, "shuffled", derangement(sample_ids, seed), seed),
            args.overwrite,
        )

    print("[prepare_image_ablation] output={}".format(args.output_dir))
    print("[prepare_image_ablation] selected_samples={}".format(len(sample_ids)))
    print("[prepare_image_ablation] manifests={}".format(2 + len(seeds)))


if __name__ == "__main__":
    main(sys.argv[1:])
