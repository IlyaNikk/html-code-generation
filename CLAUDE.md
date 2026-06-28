# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Generates HTML/Bootstrap web pages from screenshot images via deep learning. A mockup image is fed to the model, which predicts an intermediate DSL (`.gui` format); a separate rule-based compiler then turns that DSL into HTML. The project is a heavily-modified fork of pix2code (Tony Beltramelli) / Taneem Jan's thesis work — the original used LSTM encoders; this fork migrated the image encoder toward ResNet/autoencoder features and the text path toward a Transformer.

## Setup

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # keras 3.4.0 / tensorflow 2.17.0 / numpy 1.23.5
```

Note: `compiler/generate_dataset.py` additionally requires `playwright` (not in requirements.txt) plus `playwright install` for the browser used to screenshot generated pages.

## Common commands

All `make` targets run from the repo root. `${WEIGHTS_PATH}` is typically `bin/web`, `${WEIGHTS}` the model name (e.g. `Main_Model.weights`), `${INPUT_PATH}` a dataset dir.

```
make train_model_web              # train Main_Model on datasets/web
make train_autoencoder_web        # train autoencoder first (4th arg =1), then Main_Model
make train_model_new_web          # train on datasets/generated/web (synthetic data)

make functional_for_web           # render-diff eval: bin/web Main_Model.weights datasets/web/eval_set
make functional_for_web_diff WEIGHTS_PATH=bin/web WEIGHTS=Main_Model.weights INPUT_PATH=datasets/web/eval_set   # + saves diff PNGs
make bleu_for_web WEIGHTS_PATH=bin/web WEIGHTS=Main_Model.weights INPUT_PATH=datasets/web/eval_set              # BLEU scores

make compile_gui PATH="./tests/<uid>.gui"   # compile one .gui -> .html
make create_dataset COUNT=1000              # generate COUNT synthetic .gui/.png pairs
make autoencoder_predict                    # sanity-check autoencoder reconstruction
```

There is no lint config or unit-test runner — the "tests" are the functional/BLEU evaluation scripts above, which require trained weights to run.

Generate code for a single image directly:
```
python3 model/sample.py <weights_path> <model_name> <input_image> <output_dir> [greedy]
```

## Important execution detail: import paths

Two different working-directory conventions coexist; respect them or imports break:

- `model/train.py` and `model/sample.py` import as `from classes.model.Main_Model import *`. They work because running `python3 model/train.py` puts `model/` on `sys.path`. Run these as `python3 model/...` from the repo root (as the Makefile does).
- `model/tests/*.py`, `compiler/web-compiler.py`, and `compiler/generate_dataset.py` do `sys.path.append('./')` and import as `from model.classes... ` / `from compiler.classes...`, plus reference assets by repo-relative paths like `compiler/assets/...`. These must be run from the repo root.

## Architecture

### Two-stage pipeline
1. **Image → DSL** (`model/`): a Keras model predicts `.gui` tokens one at a time from the image + a sliding window of previously-predicted tokens.
2. **DSL → HTML** (`compiler/`): a deterministic compiler maps `.gui` tokens to HTML fragments via a JSON DSL-mapping file. No ML here.

### Model (`model/classes/`)
- `model/classes/model/Config.py` — global hyperparameters: `CONTEXT_LENGTH=48` (token window), `IMAGE_SIZE=256`, `BATCH_SIZE=64`, `EPOCHS`.
- `AModel.py` — base class; `save()`/`load()` persist architecture as `<name>.json` + weights as `<name>.h5` under `output_path`.
- `Main_Model.py` — **the active model**. It is a transformer-based variant despite the name: it loads a *pretrained autoencoder* (`autoencoder_old.weights`) at construction, freezes its conv stack up to `max_pooling2d` as the image encoder, feeds the partial token sequence through transformer-encoder blocks, concatenates the two, and softmaxes the next token. Output weights name is `Main_Model_trans.weights`.
- `Main_Model_old.py`, `Main_Model_transformer.py` — alternate/experimental architectures, not wired into `train.py`.
- `autoencoder_image_old.py` is the autoencoder `Main_Model.py` and `train.py` currently import; `autoencoder_image.py` / `new_autoencoder_image.py` are alternates.
- `dataset/Dataset.py` loads all `.gui`/`.png` (or `.npz`) pairs into memory and one-hot encodes tokens via `Vocabulary`. `dataset/Generator.py` is the streaming equivalent used for `model.fit` (yields `((images, partial_sequences), next_word)`, or images-only for autoencoder training).
- `Vocabulary.py` — builds the token↔one-hot maps; serialized to `words.vocab`. `START_TOKEN`/`END_TOKEN`/`PLACEHOLDER` frame each sequence.
- `Sampler.py` — `predict_greedy` does autoregressive decoding: maintains the `CONTEXT_LENGTH` sliding window, appends each argmax token until `<END>`.
- Training writes `meta_dataset.npy` (input_shape, output_size, size) and `words.vocab` into `output_path`; inference (`sample.py`, tests) reads them back from the weights dir.

### Compiler (`compiler/`)
- `web-compiler.py` — entrypoint; reads a `.gui`, builds a tree, renders HTML.
- `classes/Compiler.py` — `compile()` (file→file) and `compile_in_runtime()` (string→string, used by the functional tests). Parses opening-tag/closing-tag tokens into `Node` trees.
- `classes/Node.py` / `Utils.py` — tree node + rendering, including random lorem/text fill.
- `assets/*.json` — DSL→HTML mappings. **There are several and they disagree**: `web-compiler.py` uses `web-dsl-mapping.json`, while `model/tests/functional-test.py` and `compiler/generate_dataset.py` use `web-dsl-mapping-new.json`. The `-bdui` / `-bdui_v2` variants are alternate component sets. Pick the mapping that matches the DSL vocabulary your model was trained on.
- `generate_dataset.py` — synthesizes new `.gui` files from `assets/rules-for-generate.json` (allowed parent/child relations, structure, restrictions), compiles them to HTML, and screenshots via Playwright to produce training pairs.

### Data layout
- `datasets/web/{training_set,eval_set}` — image/markup pairs (gitignored; unzip from `datasets/*.zip`). Images may be raw `.png` or preprocessed `.npz` (key `"features"`).
- `bin/` — trained weights (`.h5`/`.json`), `meta_dataset.npy`, `words.vocab` (gitignored).
- `generated-output/` — model predictions. `model-architectures/` — architecture diagrams.

## Gotchas
- `Main_Model.__init__` hard-requires `autoencoder_old.weights.h5` to already exist in `output_path` — train/obtain the autoencoder before constructing `Main_Model` for training or inference.
- `Main_Model.fit_generator` currently contains a **hardcoded checkpoint path** (`main_model_trans_checkpoint_17_06_2025_06_03.weights.h5`) that it loads before training, and a `plot_model(... to_file='se2seq_resnet.png')` call. These are experiment-specific leftovers; expect to edit them when running a fresh train.
- Source files contain Russian comments (this is an active research fork); the original English author headers remain.
- `.gitignore` excludes `**/*.txt`, so `requirements.txt` and similar are force-added — be careful when adding new `.txt` files.
