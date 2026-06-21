"""Dataset profiles — centralises every per-dataset knob (training/eval paths,
DSL mapping) so changing datasets is one CLI flag, not edits in four files.

Workflow:
    1. train.py picks a profile (`--profile web` / `--profile web_generated`),
       resolves paths from PROFILES, and writes the profile NAME into
       `<output_path>/training_profile.txt`.
    2. Downstream scripts (predict_one.py, functional-test.py, the in-training
       TestingCallback, …) call `profiles.load(weights_path)` to get the same
       profile dict back without being told. No more hardcoded DSL_PATH /
       eval_set constants drifting between files.

Adding a new dataset = adding one entry to PROFILES. No code changes elsewhere.
"""

import os

PROFILES = {
    "web": {                       # original Beltramelli pix2code web set
        "training_set": "datasets/web/training_set",
        "eval_set":     "datasets/web/eval_set",
        "dsl_mapping":  "compiler/assets/web-dsl-mapping.json",
    },
    "web_generated": {             # synthetic set from compiler/generate_dataset.py
        "training_set": "datasets/generated/web/training_set",
        "eval_set":     "datasets/generated/web/eval_set",
        "dsl_mapping":  "compiler/assets/web-dsl-mapping-new.json",
    },
}

_SIDECAR = "training_profile.txt"
_DEFAULT_PROFILE = "web"


def get(name):
    """Return the profile dict for `name`, or fail loudly with the list of known profiles."""
    if name not in PROFILES:
        raise SystemExit(
            "Unknown dataset profile {!r}. Known: {}".format(name, sorted(PROFILES))
        )
    return PROFILES[name]


def save(weights_path, name):
    """Record which profile this training run used, so inference scripts can pick it up later."""
    if name not in PROFILES:
        raise SystemExit("Refusing to save unknown profile {!r}".format(name))
    os.makedirs(weights_path, exist_ok=True)
    with open(os.path.join(weights_path, _SIDECAR), "w") as f:
        f.write(name)


def load(weights_path, default=_DEFAULT_PROFILE):
    """Return the profile dict that was active when `weights_path` was trained.

    Falls back to `default` (the original 'web' profile) if no sidecar file
    is present — this preserves behaviour for `bin/web/` directories trained
    before the refactor existed.
    """
    sidecar = os.path.join(weights_path, _SIDECAR)
    if os.path.isfile(sidecar):
        with open(sidecar) as f:
            name = f.read().strip()
    else:
        name = default
    return get(name)
