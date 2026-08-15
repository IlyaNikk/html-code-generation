import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
from collections import Counter
from difflib import SequenceMatcher

sys.path.append('./')

import numpy as np
from PIL import Image

from compiler.classes.Compiler import Compiler
from compiler.classes.Utils import Utils as CompilerUtils
from model.classes.Utils import Utils
from model.classes.Sampler import Sampler
from model.classes.Vocabulary import START_TOKEN, END_TOKEN
from model.classes.dataset import profiles as dataset_profiles
from model.classes.model.Config import CONTEXT_LENGTH, IMAGE_SIZE
from model.classes.model.Main_Model import Main_Model
from model.classes.test_classes.BLEU import BLEU


DEFAULT_RULES_PATH = "compiler/assets/rules-for-generate.json"
DEFAULT_MODEL_NAME = "Main_Model.weights"
DEFAULT_VIEWPORT = {"width": 1280, "height": 2860}
STATIC_RENDERER_VERSION = "webkit_static_js_disabled_domcontentloaded_v1"
RENDER_WORKER_PATH = os.path.join(os.path.dirname(__file__), "render_html_worker.py")
REWARD_VERSION = "content_structural_v4"
VISUAL_SCORE_V2_VERSION = "visual_score_v2"
VISUAL_SCORE_V3_VERSION = "visual_score_v3_foreground_grid"
REWARD_V5_VERSION = "visual_structural_v5"
REWARD_V6_VERSION = "visual_structural_v6_v3"
FOREGROUND_LUMINANCE_THRESHOLD = 0.97
FOREGROUND_CHROMA_THRESHOLD = 0.08
FOREGROUND_GRID_SIZE = 8
BUTTON_TOKENS = {"btn-active", "btn-inactive", "btn-green", "btn-orange", "btn-red"}
TEXT_TOKENS = {"big-title", "small-title", "text"}
LAYOUT_TOKENS = {
    "body", "header", "main", "footer", "row", "single", "double", "quadruple",
    "carousel-wrapper", "carousel-indicator-wrappers", "carousel-content-wrapper"
}


class DslNode:
    def __init__(self, key):
        self.key = key
        self.children = []


def clean_prediction(text):
    return text.replace(START_TOKEN, "").replace(END_TOKEN, "")


def read_target_gui(input_path, gui_name):
    with open("{}/{}.gui".format(input_path, gui_name)) as f:
        return clean_prediction(f.read())


def parse_dsl_tree(dsl_text):
    root = DslNode("__root__")
    stack = [root]
    syntax_errors = []
    known_tokens = []
    edges = []

    for raw_line in clean_prediction(dsl_text).splitlines():
        token = re.sub(r"\s+", "", raw_line)
        if token == "":
            continue

        if "{" in token:
            token = token.replace("{", "")
            if "," in token:
                tokens = token.split(",")
                for leaf in tokens[:-1]:
                    if leaf:
                        node = DslNode(leaf)
                        stack[-1].children.append(node)
                        known_tokens.append(leaf)
                        edges.append((stack[-1].key, leaf))
                token = tokens[-1]

            if token:
                node = DslNode(token)
                stack[-1].children.append(node)
                known_tokens.append(token)
                edges.append((stack[-1].key, token))
                stack.append(node)
        elif "}" in token:
            if len(stack) == 1:
                syntax_errors.append("closing_without_parent")
            else:
                stack.pop()
        else:
            for leaf in token.split(","):
                if leaf:
                    node = DslNode(leaf)
                    stack[-1].children.append(node)
                    known_tokens.append(leaf)
                    edges.append((stack[-1].key, leaf))

    if len(stack) != 1:
        syntax_errors.append("unclosed_tags:{}".format(len(stack) - 1))

    return root, known_tokens, edges, syntax_errors


def preorder_tokens(node):
    tokens = []
    for child in node.children:
        tokens.append(child.key)
        tokens.extend(preorder_tokens(child))
    return tokens


def load_rules(path):
    with open(path) as f:
        rules = json.load(f)
    relations = dict(rules.get("relations", {}))
    relations[rules["top"]] = rules.get("structure", [])
    relations["__root__"] = [rules["top"]]
    return rules, relations


def grammar_validity(edges, relations):
    if len(edges) == 0:
        return 0.0, 0, 0

    valid = 0
    for parent, child in edges:
        if child in relations.get(parent, []):
            valid += 1

    return valid / float(len(edges)), valid, len(edges)


def tree_similarity(predicted_root, target_root):
    predicted = preorder_tokens(predicted_root)
    target = preorder_tokens(target_root)
    if len(predicted) == 0 and len(target) == 0:
        return 1.0
    return SequenceMatcher(None, predicted, target).ratio()


def counter_scores(predicted_items, target_items, allowed_items=None):
    if allowed_items is not None:
        predicted_items = [item for item in predicted_items if item in allowed_items]
        target_items = [item for item in target_items if item in allowed_items]

    predicted = Counter(predicted_items)
    target = Counter(target_items)
    predicted_total = sum(predicted.values())
    target_total = sum(target.values())

    if predicted_total == 0 and target_total == 0:
        return 1.0, 1.0, 1.0
    if predicted_total == 0 or target_total == 0:
        return 0.0, 0.0, 0.0

    overlap = sum((predicted & target).values())
    precision = overlap / float(predicted_total)
    recall = overlap / float(target_total)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def structure_content_scores(predicted_tokens, target_tokens, predicted_edges, target_edges):
    token_precision, token_recall, token_f1 = counter_scores(predicted_tokens, target_tokens)
    edge_precision, edge_recall, edge_f1 = counter_scores(predicted_edges, target_edges)
    _, _, button_token_f1 = counter_scores(predicted_tokens, target_tokens, BUTTON_TOKENS)
    _, _, text_token_f1 = counter_scores(predicted_tokens, target_tokens, TEXT_TOKENS)
    _, _, layout_token_f1 = counter_scores(predicted_tokens, target_tokens, LAYOUT_TOKENS)

    return {
        "token_precision": float(token_precision),
        "token_recall": float(token_recall),
        "token_f1": float(token_f1),
        "parent_edge_precision": float(edge_precision),
        "parent_edge_recall": float(edge_recall),
        "parent_edge_f1": float(edge_f1),
        "button_token_f1": float(button_token_f1),
        "text_token_f1": float(text_token_f1),
        "layout_token_f1": float(layout_token_f1),
    }


def render_html_to_png(html, output_path, timeout_seconds=90):
    """Use the same static renderer contract as the RL visual reward."""
    input_path = "{}.html".format(output_path)
    try:
        with open(input_path, "w") as destination:
            destination.write(html)
        completed = subprocess.run(
            [
                sys.executable, RENDER_WORKER_PATH, "--input", input_path, "--output", output_path,
                "--page-timeout-seconds", str(timeout_seconds),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            output = completed.stdout.strip().replace("\n", " ")
            raise RuntimeError("renderer worker exited {}: {}".format(completed.returncode, output[-1000:]))
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("renderer worker produced no screenshot")
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("renderer worker timed out after {}s".format(timeout_seconds)) from exc
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


def image_diff_percentage(master_path, prediction_path):
    with Image.open(master_path) as master_img:
        master = np.asarray(master_img.convert("RGB"), dtype=np.int16)

    with Image.open(prediction_path) as prediction_img:
        prediction = np.asarray(prediction_img.convert("RGB"), dtype=np.int16)

    diff = np.abs(master - prediction)
    denominator = np.sum(np.abs(prediction))
    if denominator == 0:
        return 100.0
    return float(np.sum(diff) * 100.0 / denominator)


def rgb_arrays_for_comparison(first_path, second_path):
    """Load two images at a common size chosen symmetrically from both inputs."""
    with Image.open(first_path) as first_image:
        first_rgb = first_image.convert("RGB")
    with Image.open(second_path) as second_image:
        second_rgb = second_image.convert("RGB")

    common_size = (
        max(first_rgb.width, second_rgb.width),
        max(first_rgb.height, second_rgb.height)
    )
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    if first_rgb.size != common_size:
        first_rgb = first_rgb.resize(common_size, resampling)
    if second_rgb.size != common_size:
        second_rgb = second_rgb.resize(common_size, resampling)
    return (
        np.asarray(first_rgb, dtype=np.float32) / 255.0,
        np.asarray(second_rgb, dtype=np.float32) / 255.0,
    )


def rgb_uint8_arrays_for_comparison(first_path, second_path):
    """Load comparable RGB images without expanding them to float32 arrays.

    Visual Score V3 only needs a foreground mask.  Keeping the image pixels as
    uint8 reduces the peak memory of a 1280×2860 pair by roughly 4× while
    preserving the exact luminance/chroma threshold decisions.
    """
    with Image.open(first_path) as first_image:
        first_rgb = first_image.convert("RGB")
    with Image.open(second_path) as second_image:
        second_rgb = second_image.convert("RGB")
    common_size = (
        max(first_rgb.width, second_rgb.width),
        max(first_rgb.height, second_rgb.height)
    )
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    if first_rgb.size != common_size:
        first_rgb = first_rgb.resize(common_size, resampling)
    if second_rgb.size != common_size:
        second_rgb = second_rgb.resize(common_size, resampling)
    return np.asarray(first_rgb, dtype=np.uint8), np.asarray(second_rgb, dtype=np.uint8)


def sobel_edge_magnitude(rgb):
    """Return a normalised Sobel edge map without a new OpenCV dependency."""
    grayscale = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    padded = np.pad(grayscale, 1, mode="edge")
    horizontal = (
        -padded[:-2, :-2] + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2] + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2] + padded[2:, 2:]
    )
    vertical = (
        -padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    return np.clip(np.hypot(horizontal, vertical) / (4.0 * np.sqrt(2.0)), 0.0, 1.0)


def visual_score_v2_components(first_path, second_path, grid_size=4):
    """Compute a symmetric global, edge, and worst-region visual comparison."""
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")
    first, second = rgb_arrays_for_comparison(first_path, second_path)
    rgb_error = float(np.mean(np.abs(first - second)))
    edge_error = float(np.mean(np.abs(sobel_edge_magnitude(first) - sobel_edge_magnitude(second))))

    per_pixel_error = np.mean(np.abs(first - second), axis=2)
    row_tiles = np.array_split(per_pixel_error, grid_size, axis=0)
    tile_errors = []
    for row_tile in row_tiles:
        tile_errors.extend(float(np.mean(tile)) for tile in np.array_split(row_tile, grid_size, axis=1))
    worst_count = max(1, len(tile_errors) // 4)
    worst_tile_error = float(np.mean(sorted(tile_errors, reverse=True)[:worst_count]))
    distance = 0.25 * rgb_error + 0.40 * edge_error + 0.35 * worst_tile_error
    return {
        "visual_rgb_error_v2": rgb_error,
        "visual_edge_error_v2": edge_error,
        "visual_worst_tile_error_v2": worst_tile_error,
        "visual_score_v2": float(np.clip(1.0 - distance, 0.0, 1.0)),
    }


def foreground_mask(rgb):
    """Identify UI ink, colour, and light-grey panels while excluding white page space."""
    if rgb.dtype == np.uint8:
        # This is equivalent to the float calculation below after RGB / 255:
        # luminance < 0.97 and chroma > 0.08.  Work in one uint32 plane rather
        # than materialising a second float32 image for the V3 reward.
        luminance_scaled = rgb[:, :, 0].astype(np.uint32)
        luminance_scaled *= 299
        luminance_scaled += rgb[:, :, 1].astype(np.uint32) * 587
        luminance_scaled += rgb[:, :, 2].astype(np.uint32) * 114
        chroma = np.maximum(np.maximum(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2])
        chroma = chroma - np.minimum(np.minimum(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2])
        return (luminance_scaled < 247350) | (chroma > 20)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    return (luminance < FOREGROUND_LUMINANCE_THRESHOLD) | (chroma > FOREGROUND_CHROMA_THRESHOLD)


def binary_overlap_scores(target_mask, predicted_mask):
    """Return foreground precision, recall, F1, and IoU with explicit empty-mask handling."""
    target_total = int(np.sum(target_mask))
    predicted_total = int(np.sum(predicted_mask))
    intersection = int(np.sum(target_mask & predicted_mask))
    union = int(np.sum(target_mask | predicted_mask))
    if target_total == 0 and predicted_total == 0:
        return 1.0, 1.0, 1.0, 1.0
    precision = intersection / float(predicted_total) if predicted_total else 0.0
    recall = intersection / float(target_total) if target_total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = intersection / float(union) if union else 0.0
    return precision, recall, f1, iou


def grid_foreground_f1(target_mask, predicted_mask, grid_size=FOREGROUND_GRID_SIZE):
    """Compare how much foreground occupies each spatial cell, independent of page emptiness."""
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")
    target_tiles = [
        float(np.mean(tile))
        for row in np.array_split(target_mask, grid_size, axis=0)
        for tile in np.array_split(row, grid_size, axis=1)
    ]
    predicted_tiles = [
        float(np.mean(tile))
        for row in np.array_split(predicted_mask, grid_size, axis=0)
        for tile in np.array_split(row, grid_size, axis=1)
    ]
    overlap = sum(min(target, predicted) for target, predicted in zip(target_tiles, predicted_tiles))
    total = sum(target_tiles) + sum(predicted_tiles)
    return 1.0 if total == 0.0 else 2.0 * overlap / total


def visual_score_v3_components(first_path, second_path):
    """Score UI foreground overlap and its grid location, avoiding white-background dominance."""
    target, predicted = rgb_uint8_arrays_for_comparison(first_path, second_path)
    target_mask = foreground_mask(target)
    predicted_mask = foreground_mask(predicted)
    precision, recall, f1, iou = binary_overlap_scores(target_mask, predicted_mask)
    grid_f1 = grid_foreground_f1(target_mask, predicted_mask)
    return {
        "foreground_precision_v3": float(precision),
        "foreground_recall_v3": float(recall),
        "foreground_f1_v3": float(f1),
        "foreground_iou_v3": float(iou),
        "foreground_grid_f1_v3": float(grid_f1),
        "visual_score_v3": float(0.55 * f1 + 0.45 * grid_f1),
    }


def render_and_diff(compiler, target_gui, predicted_gui, work_dir, gui_name, render_html=None,
                    calculate_image_diff=True):
    master_html = compiler.compile_in_runtime(
        target_gui,
        rendering_function=CompilerUtils.render_content_with_text
    )
    predicted_html = compiler.compile_in_runtime(
        predicted_gui,
        rendering_function=CompilerUtils.render_content_with_text
    )

    master_path = os.path.join(work_dir, "{}.target.png".format(gui_name))
    predicted_path = os.path.join(work_dir, "{}.predicted.png".format(gui_name))
    render = render_html or render_html_to_png
    render(master_html, master_path)
    render(predicted_html, predicted_path)
    image_diff = image_diff_percentage(master_path, predicted_path) if calculate_image_diff else None
    return image_diff, master_path, predicted_path


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def target_complexity_bucket(target_length):
    if target_length <= 50:
        return "short"
    if target_length <= 100:
        return "medium"
    return "long"


def load_image_ablation_manifest(path, input_path):
    """Load and validate the target-to-input-image mapping for an ablation run."""
    with open(path) as f:
        manifest = json.load(f)

    metadata = manifest.get("metadata")
    samples = manifest.get("samples")
    if not isinstance(metadata, dict) or not isinstance(samples, list):
        raise ValueError("{} must contain metadata and samples".format(path))

    image_mode = metadata.get("image_mode")
    if image_mode not in {"original", "shuffled", "blank"}:
        raise ValueError("{} has unsupported image_mode {!r}".format(path, image_mode))

    mappings = {}
    for row in samples:
        target_sample = row.get("target_sample")
        if not target_sample or target_sample in mappings:
            raise ValueError("{} has a missing or duplicate target_sample".format(path))

        target_gui = row.get("target_gui", "{}.gui".format(target_sample))
        if target_gui != "{}.gui".format(target_sample):
            raise ValueError("{} has inconsistent target_gui for {}".format(path, target_sample))
        if not os.path.isfile(os.path.join(input_path, target_gui)):
            raise FileNotFoundError("manifest target GUI does not exist: {}".format(target_gui))

        source_image = row.get("source_image")
        source_sample = row.get("source_image_sample")
        if image_mode == "blank":
            if source_image is not None or source_sample is not None:
                raise ValueError("blank manifest row {} must not name a source image".format(target_sample))
        else:
            expected_image = "{}.png".format(source_sample) if source_sample else None
            if source_image != expected_image:
                raise ValueError("{} has inconsistent source image for {}".format(path, target_sample))
            if not os.path.isfile(os.path.join(input_path, source_image)):
                raise FileNotFoundError("manifest source image does not exist: {}".format(source_image))

        mappings[target_sample] = {
            "image_mode": image_mode,
            "source_image_sample": source_sample,
            "source_image": source_image,
            "source_image_path": (
                os.path.join(input_path, source_image) if source_image is not None else None
            ),
        }

    if not mappings:
        raise ValueError("{} contains no samples".format(path))
    return metadata, mappings


def ablation_image(mapping):
    """Return the preprocessed input image specified by a manifest mapping."""
    if mapping["image_mode"] == "blank":
        # A fixed zero-valued input removes all image evidence without creating a
        # temporary PNG file. It has the same normalised shape as a real input.
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    return Utils.get_preprocessed_img(mapping["source_image_path"], IMAGE_SIZE)


def length_score(length_ratio):
    return max(0.0, 1.0 - abs(1.0 - length_ratio))


def reward_components(row):
    syntax_valid = safe_float(row["syntax_valid"])
    render_success = safe_float(row["render_success"])
    hit_sequence_limit = safe_float(row["hit_sequence_limit"])
    ended_too_early = safe_float(row["ended_too_early"])
    ratio = safe_float(row["length_ratio"])
    overlong = 1.0 if ratio > 1.25 else 0.0
    score_length = length_score(ratio)

    # Quality terms sum to 1.0. Syntax and rendering are requirements: they only
    # reduce the reward when broken, instead of hiding content mistakes.
    reward_base = (
        0.10 * safe_float(row["visual_score"])
        + 0.20 * safe_float(row["parent_edge_f1"])
        + 0.18 * safe_float(row["token_f1"])
        + 0.16 * safe_float(row["button_token_f1"])
        + 0.16 * safe_float(row["text_token_f1"])
        + 0.10 * score_length
        + 0.05 * safe_float(row["chrf"])
        + 0.05 * safe_float(row["tree_similarity"])
    )

    reward_penalty = (
        0.25 * (1.0 - syntax_valid)
        + 0.25 * (1.0 - render_success)
        + 0.20 * ended_too_early
        + 0.25 * overlong
        + 0.15 * hit_sequence_limit
    )
    reward = max(0.0, min(1.0, reward_base - reward_penalty))

    return {
        "length_score": float(score_length),
        "overlong": int(overlong),
        "reward_base": float(reward_base),
        "reward_penalty": float(reward_penalty),
        "reward": float(reward),
    }


def visual_structural_v5_components(row):
    """New reward, kept separate from the frozen v4 fields for reproducibility."""
    syntax_valid = safe_float(row["syntax_valid"])
    render_success = safe_float(row["render_success"])
    hit_sequence_limit = safe_float(row["hit_sequence_limit"])
    ended_too_early = safe_float(row["ended_too_early"])
    overlong = safe_float(row["overlong"])

    reward_base_v5 = (
        0.25 * safe_float(row["visual_score_v2"])
        + 0.20 * safe_float(row["parent_edge_f1"])
        + 0.16 * safe_float(row["token_f1"])
        + 0.15 * safe_float(row["button_token_f1"])
        + 0.15 * safe_float(row["text_token_f1"])
        + 0.05 * safe_float(row["length_score"])
        + 0.02 * safe_float(row["chrf"])
        + 0.02 * safe_float(row["tree_similarity"])
    )
    reward_penalty_v5 = (
        0.25 * (1.0 - syntax_valid)
        + 0.25 * (1.0 - render_success)
        + 0.20 * ended_too_early
        + 0.25 * overlong
        + 0.15 * hit_sequence_limit
    )
    reward_v5 = max(0.0, min(1.0, reward_base_v5 - reward_penalty_v5))
    return {
        "reward_base_v5": float(reward_base_v5),
        "reward_penalty_v5": float(reward_penalty_v5),
        "reward_v5": float(reward_v5),
    }


def visual_structural_v6_components(row):
    """Visual-structural reward using the calibrated foreground/grid score V3.

    V5 intentionally remains tied to ``visual_score_v2`` so earlier results stay
    reproducible.  V6 changes only the visual term and preserves the structural
    weights and all validity/length penalties.
    """
    syntax_valid = safe_float(row["syntax_valid"])
    render_success = safe_float(row["render_success"])
    hit_sequence_limit = safe_float(row["hit_sequence_limit"])
    ended_too_early = safe_float(row["ended_too_early"])
    overlong = safe_float(row["overlong"])

    reward_base_v6 = (
        0.25 * safe_float(row["visual_score_v3"])
        + 0.20 * safe_float(row["parent_edge_f1"])
        + 0.16 * safe_float(row["token_f1"])
        + 0.15 * safe_float(row["button_token_f1"])
        + 0.15 * safe_float(row["text_token_f1"])
        + 0.05 * safe_float(row["length_score"])
        + 0.02 * safe_float(row["chrf"])
        + 0.02 * safe_float(row["tree_similarity"])
    )
    reward_penalty_v6 = (
        0.25 * (1.0 - syntax_valid)
        + 0.25 * (1.0 - render_success)
        + 0.20 * ended_too_early
        + 0.25 * overlong
        + 0.15 * hit_sequence_limit
    )
    reward_v6 = max(0.0, min(1.0, reward_base_v6 - reward_penalty_v6))
    return {
        "reward_base_v6": float(reward_base_v6),
        "reward_penalty_v6": float(reward_penalty_v6),
        "reward_v6": float(reward_v6),
    }


def add_group_summary(summary, name, rows):
    summary[name] = {"samples": len(rows)}
    if len(rows) == 0:
        return

    fields = [
        "syntax_valid", "render_success", "tree_similarity", "bleu", "chrf",
        "visual_score", "visual_score_v2", "visual_score_v3", "foreground_f1_v3",
        "foreground_grid_f1_v3", "token_f1", "parent_edge_f1", "button_token_f1",
        "text_token_f1", "layout_token_f1", "prediction_length", "target_length",
        "length_ratio", "length_score", "ended_too_early", "overlong",
        "reward_base", "reward_penalty", "reward", "reward_base_v5",
        "reward_penalty_v5", "reward_v5", "reward_base_v6",
        "reward_penalty_v6", "reward_v6"
    ]
    for field in fields:
        values = [safe_float(row[field]) for row in rows]
        summary[name][field + "_mean"] = float(np.mean(values))
        summary[name][field + "_median"] = float(np.median(values))

    summary[name]["syntax_invalid_rate"] = 1.0 - summary[name]["syntax_valid_mean"]
    summary[name]["render_failure_rate"] = 1.0 - summary[name]["render_success_mean"]
    summary[name]["overlong_rate"] = summary[name]["overlong_mean"]


def summarize(rows):
    numeric_fields = [
        "bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "chrf",
        "image_diff", "visual_score", "visual_rgb_error_v2", "visual_edge_error_v2",
        "visual_worst_tile_error_v2", "visual_score_v2", "foreground_precision_v3",
        "foreground_recall_v3", "foreground_f1_v3", "foreground_iou_v3",
        "foreground_grid_f1_v3", "visual_score_v3", "grammar_validity", "tree_similarity",
        "token_precision", "token_recall", "token_f1",
        "parent_edge_precision", "parent_edge_recall", "parent_edge_f1",
        "button_token_f1", "text_token_f1", "layout_token_f1",
        "syntax_valid", "render_success", "prediction_length",
        "target_length", "length_ratio", "length_score", "ended_too_early",
        "overlong", "reward_base", "reward_penalty", "reward", "reward_base_v5",
        "reward_penalty_v5", "reward_v5", "reward_base_v6",
        "reward_penalty_v6", "reward_v6",
        "hit_sequence_limit"
    ]
    summary = {"samples": len(rows)}
    if len(rows) == 0:
        return summary

    for field in numeric_fields:
        values = [safe_float(row[field]) for row in rows]
        summary[field + "_mean"] = float(np.mean(values))
        summary[field + "_median"] = float(np.median(values))

    summary["render_failure_rate"] = 1.0 - summary["render_success_mean"]
    summary["syntax_invalid_rate"] = 1.0 - summary["syntax_valid_mean"]
    summary["overlong_rate"] = summary["overlong_mean"]
    for bucket in ["short", "medium", "long"]:
        add_group_summary(
            summary,
            "{}_target_summary".format(bucket),
            [row for row in rows if row["target_complexity"] == bucket]
        )
    return summary


def add_run_config(summary, args, input_path, files, image_ablation_metadata=None):
    summary["run_config"] = {
        "profile": args.profile,
        "reward_version": REWARD_VERSION,
        "visual_score_v2_version": VISUAL_SCORE_V2_VERSION,
        "visual_score_v3_version": VISUAL_SCORE_V3_VERSION,
        "reward_v5_version": REWARD_V5_VERSION,
        "reward_v6_version": REWARD_V6_VERSION,
        "mode": args.mode,
        "input_path": input_path,
        "limit": args.limit,
        "sequence_length": args.sequence_length,
        "min_target_length": args.min_target_length,
        "max_target_length": args.max_target_length,
        "only_sample": args.only_sample,
        "selected_files": len(files),
        "image_ablation_manifest": args.image_ablation_manifest,
        "image_ablation": image_ablation_metadata,
        "renderer_version": STATIC_RENDERER_VERSION,
        "render_timeout_seconds": args.render_timeout_seconds,
    }


def write_outputs(rows, summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rows_path = os.path.join(output_dir, "extended_metrics.csv")
    jsonl_path = os.path.join(output_dir, "extended_metrics.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")

    fieldnames = [
        "sample", "mode", "bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "chrf",
        "image_diff", "visual_score", "visual_score_version", "visual_rgb_error_v2",
        "visual_edge_error_v2", "visual_worst_tile_error_v2", "visual_score_v2",
        "visual_score_v3_version", "foreground_precision_v3", "foreground_recall_v3",
        "foreground_f1_v3", "foreground_iou_v3", "foreground_grid_f1_v3", "visual_score_v3",
        "render_success", "render_error",
        "syntax_valid", "syntax_errors", "grammar_validity", "grammar_valid_edges",
        "grammar_total_edges", "tree_similarity", "token_precision", "token_recall",
        "token_f1", "parent_edge_precision", "parent_edge_recall", "parent_edge_f1",
        "button_token_f1", "text_token_f1", "layout_token_f1", "prediction_length",
        "target_length", "target_complexity", "length_ratio", "ended_too_early",
        "overlong", "length_score", "reward_base", "reward_penalty", "reward",
        "reward_v5_version", "reward_base_v5", "reward_penalty_v5", "reward_v5",
        "reward_v6_version", "reward_base_v6", "reward_penalty_v6", "reward_v6",
        "hit_sequence_limit",
        "image_mode", "source_image_sample", "source_image", "target_screenshot",
        "predicted_screenshot"
    ]

    with open(rows_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[evaluate_extended] rows: {}".format(rows_path))
    print("[evaluate_extended] jsonl: {}".format(jsonl_path))
    print("[evaluate_extended] summary: {}".format(summary_path))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Extended evaluator for screenshot-to-DSL research metrics.")
    parser.add_argument("--profile", default=None, help="dataset profile name; default loads from weights sidecar")
    parser.add_argument("--weights-path", default=None, help="trained weights directory")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input-path", default=None, help="eval set path")
    parser.add_argument("--rules-path", default=DEFAULT_RULES_PATH)
    parser.add_argument("--mode", default="greedy", choices=["greedy", "constrained"])
    parser.add_argument("--sequence-length", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0, help="0 means all eval samples")
    parser.add_argument("--min-target-length", type=int, default=0, help="keep samples with at least this many target tokens")
    parser.add_argument("--max-target-length", type=int, default=0, help="0 means no upper target-token limit")
    parser.add_argument("--render-timeout-seconds", type=int, default=90)
    parser.add_argument("--only-sample", default=None, help="evaluate exactly this GUI sample id")
    parser.add_argument(
        "--image-ablation-manifest",
        default=None,
        help="JSON manifest from prepare_image_ablation.py; replaces the input image while keeping each target DSL fixed"
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    if args.render_timeout_seconds <= 0:
        raise SystemExit("--render-timeout-seconds must be positive")

    if args.weights_path:
        weights_path = args.weights_path
        profile = dataset_profiles.load(weights_path)
    else:
        profile_name = args.profile or "web_generated"
        profile = dataset_profiles.get(profile_name)
        weights_path = profile["output_dir"]

    if args.profile:
        profile = dataset_profiles.get(args.profile)

    input_path = args.input_path or profile["eval_set"]
    dsl_path = profile["dsl_mapping"]
    rules, relations = load_rules(args.rules_path)
    image_ablation_metadata = None
    image_ablation_mappings = None
    if args.image_ablation_manifest:
        image_ablation_metadata, image_ablation_mappings = load_image_ablation_manifest(
            args.image_ablation_manifest,
            input_path
        )

    if args.output_dir:
        output_dir = args.output_dir
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(weights_path, "extended_metrics_{}".format(stamp))

    print("[evaluate_extended] weights={}".format(weights_path))
    print("[evaluate_extended] input={}".format(input_path))
    print("[evaluate_extended] dsl_mapping={}".format(dsl_path))
    print("[evaluate_extended] rules={}".format(args.rules_path))
    print("[evaluate_extended] min_target_length={}".format(args.min_target_length))
    print("[evaluate_extended] max_target_length={}".format(args.max_target_length))
    if image_ablation_metadata:
        print("[evaluate_extended] image_ablation_manifest={}".format(args.image_ablation_manifest))
        print("[evaluate_extended] image_mode={}".format(image_ablation_metadata["image_mode"]))
    print("[evaluate_extended] output={}".format(output_dir))

    meta_dataset = np.load("{}/meta_dataset.npy".format(weights_path), allow_pickle=True)
    input_shape = meta_dataset[0]
    output_size = meta_dataset[1]

    model = Main_Model(input_shape, output_size, weights_path)
    model.load(args.model_name)
    sampler = Sampler(weights_path, input_shape, output_size, CONTEXT_LENGTH)
    compiler = Compiler(dsl_path)

    if image_ablation_mappings is not None:
        # The manifest freezes the sample set. Do not apply new length filters here:
        # all conditions must evaluate exactly the same targets for a paired test.
        files = ["{}.gui".format(sample) for sample in sorted(image_ablation_mappings)]
    else:
        files = sorted(f for f in os.listdir(input_path) if re.search(r"\.gui$", f))
        filtered_files = []
        for file_name in files:
            gui_name = file_name.replace(".gui", "")
            target_gui = read_target_gui(input_path, gui_name)
            _, target_tokens, _, _ = parse_dsl_tree(target_gui)
            target_length = len(target_tokens)
            if target_length < args.min_target_length:
                continue
            if args.max_target_length > 0 and target_length > args.max_target_length:
                continue
            filtered_files.append(file_name)
        files = filtered_files
    if args.only_sample:
        requested_file = "{}.gui".format(args.only_sample)
        if requested_file not in files:
            raise SystemExit("--only-sample is absent after filtering: {}".format(args.only_sample))
        files = [requested_file]
    elif args.limit > 0:
        files = files[:args.limit]
    print("[evaluate_extended] selected_files={}".format(len(files)))

    rows = []
    screenshot_dir = os.path.join(output_dir, "screenshots")
    prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(screenshot_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    for index, file_name in enumerate(files, start=1):
        gui_name = file_name.replace(".gui", "")
        print("[evaluate_extended] [{}/{}] {}".format(index, len(files), gui_name), flush=True)

        target_gui = read_target_gui(input_path, gui_name)
        if image_ablation_mappings is None:
            image_mapping = {
                "image_mode": "original",
                "source_image_sample": gui_name,
                "source_image": "{}.png".format(gui_name),
                "source_image_path": os.path.join(input_path, "{}.png".format(gui_name)),
            }
        else:
            image_mapping = image_ablation_mappings[gui_name]
        evaluation_img = ablation_image(image_mapping)

        if args.mode == "constrained":
            raw_prediction, _ = sampler.predict_constrained(
                model,
                np.array([evaluation_img]),
                rules_path=args.rules_path,
                sequence_length=args.sequence_length,
                while_testing=True
            )
        else:
            raw_prediction, _ = sampler.predict_greedy(
                model,
                np.array([evaluation_img]),
                sequence_length=args.sequence_length,
                while_testing=True
            )
        predicted_gui = clean_prediction(raw_prediction)

        with open(os.path.join(prediction_dir, "{}.predicted.gui".format(gui_name)), "w") as f:
            f.write(predicted_gui)
        with open(os.path.join(prediction_dir, "{}.target.gui".format(gui_name)), "w") as f:
            f.write(target_gui)

        predicted_root, predicted_tokens, predicted_edges, syntax_errors = parse_dsl_tree(predicted_gui)
        target_root, target_tokens, target_edges, _ = parse_dsl_tree(target_gui)
        syntax_valid = 1 if len(syntax_errors) == 0 and len(predicted_tokens) > 0 else 0
        grammar_score, grammar_valid_edges, grammar_total_edges = grammar_validity(predicted_edges, relations)
        tree_score = tree_similarity(predicted_root, target_root)
        content_scores = structure_content_scores(
            predicted_tokens,
            target_tokens,
            predicted_edges,
            target_edges
        )

        try:
            bleu = BLEU.get_bleu_score(predicted_gui, gui_name, input_path)
        except Exception:
            bleu = [0, 0, 0, 0, 0]

        try:
            chrf = BLEU.get_chrf_score(predicted_gui, gui_name, input_path)
        except Exception:
            chrf = 0

        render_success = 0
        render_error = ""
        image_diff = 100.0
        visual_v2 = {
            "visual_rgb_error_v2": 1.0,
            "visual_edge_error_v2": 1.0,
            "visual_worst_tile_error_v2": 1.0,
            "visual_score_v2": 0.0,
        }
        visual_v3 = {
            "foreground_precision_v3": 0.0,
            "foreground_recall_v3": 0.0,
            "foreground_f1_v3": 0.0,
            "foreground_iou_v3": 0.0,
            "foreground_grid_f1_v3": 0.0,
            "visual_score_v3": 0.0,
        }
        target_screenshot = os.path.join(screenshot_dir, "{}.target.png".format(gui_name))
        predicted_screenshot = os.path.join(screenshot_dir, "{}.predicted.png".format(gui_name))
        try:
            image_diff, target_image_path, predicted_image_path = render_and_diff(
                compiler, target_gui, predicted_gui, screenshot_dir, gui_name,
                render_html=lambda html, path: render_html_to_png(
                    html, path, timeout_seconds=args.render_timeout_seconds
                ),
            )
            visual_v2 = visual_score_v2_components(target_image_path, predicted_image_path)
            visual_v3 = visual_score_v3_components(target_image_path, predicted_image_path)
            render_success = 1
        except Exception as e:
            render_error = "{}: {}".format(type(e).__name__, e)

        visual_score = max(0.0, 1.0 - (image_diff / 100.0))

        row = {
            "sample": gui_name,
            "mode": args.mode,
            "image_mode": image_mapping["image_mode"],
            "source_image_sample": image_mapping["source_image_sample"],
            "source_image": image_mapping["source_image"],
            "target_screenshot": target_screenshot,
            "predicted_screenshot": predicted_screenshot,
            "bleu": float(bleu[0]),
            "bleu_1": float(bleu[1]),
            "bleu_2": float(bleu[2]),
            "bleu_3": float(bleu[3]),
            "bleu_4": float(bleu[4]),
            "chrf": float(chrf),
            "image_diff": float(image_diff),
            "visual_score": float(visual_score),
            "visual_score_version": VISUAL_SCORE_V2_VERSION,
            "visual_score_v3_version": VISUAL_SCORE_V3_VERSION,
            "reward_v5_version": REWARD_V5_VERSION,
            "reward_v6_version": REWARD_V6_VERSION,
            "render_success": render_success,
            "render_error": render_error,
            "syntax_valid": syntax_valid,
            "syntax_errors": ";".join(syntax_errors),
            "grammar_validity": float(grammar_score),
            "grammar_valid_edges": grammar_valid_edges,
            "grammar_total_edges": grammar_total_edges,
            "tree_similarity": float(tree_score),
            "token_precision": content_scores["token_precision"],
            "token_recall": content_scores["token_recall"],
            "token_f1": content_scores["token_f1"],
            "parent_edge_precision": content_scores["parent_edge_precision"],
            "parent_edge_recall": content_scores["parent_edge_recall"],
            "parent_edge_f1": content_scores["parent_edge_f1"],
            "button_token_f1": content_scores["button_token_f1"],
            "text_token_f1": content_scores["text_token_f1"],
            "layout_token_f1": content_scores["layout_token_f1"],
            "prediction_length": len(predicted_tokens),
            "target_length": len(target_tokens),
            "target_complexity": target_complexity_bucket(len(target_tokens)),
            "length_ratio": (
                len(predicted_tokens) / float(len(target_tokens))
                if len(target_tokens) > 0 else 0.0
            ),
            "ended_too_early": (
                1 if len(target_tokens) > 50
                and len(predicted_tokens) < 0.75 * len(target_tokens)
                else 0
            ),
            "hit_sequence_limit": 1 if len(predicted_tokens) >= args.sequence_length else 0,
        }
        row.update(visual_v2)
        row.update(visual_v3)
        row.update(reward_components(row))
        row.update(visual_structural_v5_components(row))
        row.update(visual_structural_v6_components(row))
        rows.append(row)

    summary = summarize(rows)
    add_run_config(summary, args, input_path, files, image_ablation_metadata)
    write_outputs(rows, summary, output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
