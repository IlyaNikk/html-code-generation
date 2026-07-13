from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import hashlib
import shutil

from classes.Sampler import *

argv = sys.argv[1:]

if len(argv) < 1:
    print("Error: not enough argument supplied:")
    print("build_datasets.py <input path> <distribution (default: 12)> [--clean-output] [--inside-input]")
    exit(0)
else:
    input_path = argv[0]

clean_output = "--clean-output" in argv
inside_input = "--inside-input" in argv
distribution_args = [
    arg for arg in argv[1:]
    if arg not in ["--clean-output", "--inside-input"]
]
distribution = 12 if len(distribution_args) == 0 else int(distribution_args[0])

TRAINING_SET_NAME = "training_set"
EVALUATION_SET_NAME = "eval_set"

paths = []
for f in os.listdir(input_path):
    if f.find(".gui") != -1 and os.path.isfile("{}/{}".format(input_path, f)):
        path_gui = "{}/{}".format(input_path, f)
        file_name = f[:f.find(".gui")]

        if os.path.isfile("{}/{}.png".format(input_path, file_name)):
            path_img = "{}/{}.png".format(input_path, file_name)
            paths.append(file_name)

if len(paths) == 0:
    print("No .gui/.png pairs found in {}".format(input_path))
    exit(1)

evaluation_samples_number = max(1, len(paths) // (distribution + 1))
training_samples_number = len(paths) - evaluation_samples_number

assert training_samples_number + evaluation_samples_number == len(paths)

print("Splitting datasets, training samples: {}, evaluation samples: {}".format(training_samples_number, evaluation_samples_number))

np.random.shuffle(paths)

eval_set = []
train_set = []

hashes = []
for path in paths:
    if sys.version_info >= (3,):
        f = open("{}/{}.gui".format(input_path, path), 'r', encoding='utf-8')
    else:
        f = open("{}/{}.gui".format(input_path, path), 'r')

    with f:
        chars = ""
        for line in f:
            chars += line
        content_hash = chars.replace(" ", "").replace("\n", "")
        content_hash = hashlib.sha256(content_hash.encode('utf-8')).hexdigest()

        if len(eval_set) >= evaluation_samples_number:
            train_set.append(path)
        else:
            is_unique = True
            for h in hashes:
                if h == content_hash:
                    is_unique = False
                    break

            if is_unique:
                eval_set.append(path)
            else:
                train_set.append(path)

        hashes.append(content_hash)

while len(eval_set) < evaluation_samples_number and len(train_set) > 0:
    eval_set.append(train_set.pop())

assert len(eval_set) == evaluation_samples_number
assert len(train_set) == training_samples_number

output_root = input_path if inside_input else os.path.dirname(input_path)
eval_output_path = "{}/{}".format(output_root, EVALUATION_SET_NAME)
train_output_path = "{}/{}".format(output_root, TRAINING_SET_NAME)

if clean_output:
    if os.path.exists(eval_output_path):
        shutil.rmtree(eval_output_path)
    if os.path.exists(train_output_path):
        shutil.rmtree(train_output_path)
elif os.path.exists(eval_output_path) or os.path.exists(train_output_path):
    print("Warning: output folders already exist. Use --clean-output to rebuild them without old samples.")

if not os.path.exists(eval_output_path):
    os.makedirs(eval_output_path)

if not os.path.exists(train_output_path):
    os.makedirs(train_output_path)

for path in eval_set:
    shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}.png".format(eval_output_path, path))
    shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}.gui".format(eval_output_path, path))

for path in train_set:
    shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}.png".format(train_output_path, path))
    shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}.gui".format(train_output_path, path))

print("Training dataset: {}".format(train_output_path))
print("Evaluation dataset: {}".format(eval_output_path))
