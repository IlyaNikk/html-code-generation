from __future__ import print_function
from __future__ import absolute_import

import argparse
import json
import os
import sys
import hashlib
import shutil
import numpy as np

from classes.Sampler import *

TRAINING_SET_NAME = "training_set"
EVALUATION_SET_NAME = "eval_set"

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Split screenshot/DSL pairs into deterministic train/eval sets.")
    parser.add_argument("input_path")
    parser.add_argument("distribution", nargs="?", type=int, default=12)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--inside-input", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--manifest", default=None, help="write the exact train/eval membership to this JSON file")
    args = parser.parse_args(argv)
    if args.distribution < 1:
        parser.error("distribution must be positive")
    return args


def main(argv):
    args = parse_args(argv)
    input_path = os.path.abspath(args.input_path)
    paths = []
    for filename in os.listdir(input_path):
        if filename.endswith(".gui") and os.path.isfile(os.path.join(input_path, filename)):
            file_name = filename[:-4]
            if os.path.isfile(os.path.join(input_path, "{}.png".format(file_name))):
                paths.append(file_name)

    if not paths:
        raise SystemExit("No .gui/.png pairs found in {}".format(input_path))

    evaluation_samples_number = max(1, len(paths) // (args.distribution + 1))
    training_samples_number = len(paths) - evaluation_samples_number
    print("Splitting datasets, training samples: {}, evaluation samples: {}".format(
        training_samples_number, evaluation_samples_number))
    if args.seed is None:
        np.random.shuffle(paths)
    else:
        np.random.default_rng(args.seed).shuffle(paths)

    eval_set, train_set, hashes = [], [], []
    for path in paths:
        with open(os.path.join(input_path, "{}.gui".format(path)), "r", encoding="utf-8") as source:
            content_hash = hashlib.sha256(
                source.read().replace(" ", "").replace("\n", "").encode("utf-8")
            ).hexdigest()
        if len(eval_set) >= evaluation_samples_number or content_hash in hashes:
            train_set.append(path)
        else:
            eval_set.append(path)
        hashes.append(content_hash)

    while len(eval_set) < evaluation_samples_number and train_set:
        eval_set.append(train_set.pop())
    if len(eval_set) != evaluation_samples_number or len(train_set) != training_samples_number:
        raise RuntimeError("could not create the requested train/eval split")

    output_root = input_path if args.inside_input else os.path.dirname(input_path)
    eval_output_path = os.path.join(output_root, EVALUATION_SET_NAME)
    train_output_path = os.path.join(output_root, TRAINING_SET_NAME)
    if args.clean_output:
        for output_path in (eval_output_path, train_output_path):
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
    elif os.path.exists(eval_output_path) or os.path.exists(train_output_path):
        print("Warning: output folders already exist. Use --clean-output to rebuild them without old samples.")
    os.makedirs(eval_output_path, exist_ok=True)
    os.makedirs(train_output_path, exist_ok=True)

    for path in eval_set:
        for extension in ("png", "gui"):
            shutil.copyfile(
                os.path.join(input_path, "{}.{}".format(path, extension)),
                os.path.join(eval_output_path, "{}.{}".format(path, extension)),
            )
    for path in train_set:
        for extension in ("png", "gui"):
            shutil.copyfile(
                os.path.join(input_path, "{}.{}".format(path, extension)),
                os.path.join(train_output_path, "{}.{}".format(path, extension)),
            )

    if args.manifest:
        manifest_path = os.path.abspath(args.manifest)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as destination:
            json.dump({
                "input_path": input_path,
                "split_seed": args.seed,
                "distribution": args.distribution,
                "training_samples": sorted(train_set),
                "evaluation_samples": sorted(eval_set),
            }, destination, indent=2)
            destination.write("\n")
        print("Split manifest: {}".format(manifest_path))
    print("Training dataset: {}".format(train_output_path))
    print("Evaluation dataset: {}".format(eval_output_path))


if __name__ == "__main__":
    main(sys.argv[1:])
