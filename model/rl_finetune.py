"""Reproducible continued-SFT and self-critical RL for screenshot-to-DSL.

Every run starts from a supervised checkpoint and uses one deterministic order
of training pages.  ``structural_rl_v4`` keeps the historical reward frozen;
``visual_structural_rl_v6`` replaces only its visual component with the
calibrated foreground-and-grid Visual Score V3.  The policy gradient is
computed from the same grammar-masked distribution used to sample a rollout.
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.append("./")

import numpy as np
import tensorflow as tf

from compiler.classes.Compiler import Compiler
from model.classes.GrammarConstrainedDecoder import GrammarConstrainedDecoder
from model.classes.Sampler import Sampler
from model.classes.Utils import Utils
from model.classes.Vocabulary import END_TOKEN, PLACEHOLDER, START_TOKEN
from model.classes.dataset.Dataset import Dataset
from model.classes.dataset import profiles as dataset_profiles
from model.classes.model.Config import CONTEXT_LENGTH, IMAGE_SIZE
from model.classes.model.Main_Model import Main_Model
from model.tests import evaluate_extended as metrics


DEFAULT_RULES_PATH = "compiler/assets/rules-for-generate.json"
RENDER_WORKER_PATH = os.path.join(os.path.dirname(__file__), "tests", "render_html_worker.py")
MODES = ("ce_control", "structural_rl_v4", "visual_structural_rl_v6")


def reward_version_for_mode(mode):
    if mode == "ce_control":
        return "none"
    if mode == "structural_rl_v4":
        return metrics.REWARD_VERSION
    if mode == "visual_structural_rl_v6":
        return metrics.REWARD_V6_VERSION
    raise ValueError("unsupported training mode {!r}".format(mode))


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "{}h {:02d}m {:02d}s".format(hours, minutes, seconds)
    if minutes:
        return "{}m {:02d}s".format(minutes, seconds)
    return "{}s".format(seconds)


def append_token(context, token_id, output_size):
    next_context = list(context[1:])
    encoded = np.zeros(output_size, dtype=np.float32)
    encoded[token_id] = 1.0
    next_context.append(encoded)
    return next_context


def valid_token_mask(output_size, valid_ids):
    """Build a boolean grammar mask; an empty constraint permits all tokens."""
    mask = np.zeros(output_size, dtype=bool)
    if not valid_ids:
        mask[:] = True
        return mask
    for token_id in valid_ids:
        if 0 <= token_id < output_size:
            mask[token_id] = True
    if not np.any(mask):
        raise ValueError("grammar decoder did not yield a vocabulary token")
    return mask


def normalize_masked_probabilities(probabilities, mask):
    """Return the exact categorical distribution used by constrained sampling."""
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size != len(mask):
        raise ValueError("probability and grammar-mask sizes differ")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("model returned invalid probabilities")
    masked = values * np.asarray(mask, dtype=np.float64)
    total = float(masked.sum())
    if total <= 0.0:
        raise ValueError("grammar-masked probability mass is zero")
    return masked / total


def mask_probabilities(probabilities, valid_ids):
    """Backward-compatible constrained distribution helper used in tests."""
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    return normalize_masked_probabilities(values, valid_token_mask(len(values), valid_ids))


def masked_action_probabilities(probabilities, action_masks):
    """Renormalise model probabilities under the rollout's grammar masks.

    ``action_masks`` is captured before every sampled action.  Reusing it here
    prevents the common REINFORCE bug of scoring an action under the unmasked
    model distribution instead of the distribution that selected it.
    """
    masks = tf.cast(action_masks, probabilities.dtype)
    masked = probabilities * masks
    normalizer = tf.reduce_sum(masked, axis=1, keepdims=True)
    if tf.executing_eagerly() and bool(tf.reduce_any(normalizer <= 0.0).numpy()):
        raise ValueError("grammar-masked probability mass is zero during replay")
    return masked / tf.maximum(normalizer, tf.keras.backend.epsilon())


def isolated_render_html(html, output_path, timeout_seconds):
    """Render outside the TensorFlow process so a WebKit abort is recoverable."""
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


def rollout(network, sampler, image, sequence_length, rules_path, sampled, rng):
    """Generate a grammar-valid rollout and retain data needed by REINFORCE."""
    placeholder = sampler.voc.vocabulary[PLACEHOLDER]
    start = sampler.voc.vocabulary[START_TOKEN]
    context = [np.eye(sampler.output_size, dtype=np.float32)[placeholder]] * (sampler.context_length - 1)
    context.append(np.eye(sampler.output_size, dtype=np.float32)[start])
    decoder = GrammarConstrainedDecoder(rules_path, sampler.voc)
    action_contexts, action_ids, action_masks = [], [], []
    tokens = [START_TOKEN]
    reached_limit = True

    for _ in range(sequence_length):
        probabilities = network(
            [np.asarray([image]), np.asarray([context])], training=False
        ).numpy()[0]
        mask = valid_token_mask(len(probabilities), decoder.valid_token_ids())
        distribution = normalize_masked_probabilities(probabilities, mask)
        token_id = int(rng.choice(len(distribution), p=distribution)) if sampled else int(np.argmax(distribution))
        action_contexts.append(np.asarray(context, dtype=np.float32))
        action_ids.append(token_id)
        action_masks.append(mask)
        token = sampler.voc.token_lookup[token_id]
        tokens.append(token)
        context = append_token(context, token_id, sampler.output_size)
        decoder.update(token)
        if token == END_TOKEN:
            reached_limit = False
            break

    # Forced closing is part of the constrained output, but it was not sampled
    # and must not contribute to the policy-gradient likelihood.
    if tokens[-1] != END_TOKEN:
        for token in decoder.force_close_tokens():
            tokens.append(token)
            if token == END_TOKEN:
                break

    return (
        "".join(tokens),
        np.asarray(action_contexts),
        np.asarray(action_ids),
        np.asarray(action_masks, dtype=np.float32),
        reached_limit,
    )


def score_rollout(compiler, input_path, gui_name, target_gui, predicted, sequence_length,
                  reached_limit, render_reward, work_dir, label=None, render_html=None,
                  include_legacy_reward=True):
    """Calculate frozen V4 plus V5/V6 fields for a sampled or greedy output."""
    predicted_gui = metrics.clean_prediction(predicted)
    predicted_root, predicted_tokens, predicted_edges, syntax_errors = metrics.parse_dsl_tree(predicted_gui)
    target_root, target_tokens, target_edges, _ = metrics.parse_dsl_tree(target_gui)
    content = metrics.structure_content_scores(predicted_tokens, target_tokens, predicted_edges, target_edges)
    render_success, image_diff, render_error = 0, 100.0, ""
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
    if render_reward:
        if label:
            print("[rl]   rendering and scoring {} rollout...".format(label), flush=True)
        try:
            image_diff, target_image_path, predicted_image_path = metrics.render_and_diff(
                compiler, target_gui, predicted_gui, work_dir, gui_name, render_html=render_html,
                calculate_image_diff=include_legacy_reward
            )
            if include_legacy_reward:
                visual_v2 = metrics.visual_score_v2_components(target_image_path, predicted_image_path)
            visual_v3 = metrics.visual_score_v3_components(target_image_path, predicted_image_path)
            render_success = 1
        except Exception as exc:
            render_error = "{}: {}".format(type(exc).__name__, exc)

    try:
        chrf = float(metrics.BLEU.get_chrf_score(predicted_gui, gui_name, input_path))
    except Exception:
        chrf = 0.0
    target_length = len(target_tokens)
    row = {
        "syntax_valid": int(not syntax_errors and bool(predicted_tokens)),
        "render_success": render_success,
        "hit_sequence_limit": int(reached_limit),
        "ended_too_early": int(target_length > 50 and len(predicted_tokens) < 0.75 * target_length),
        "length_ratio": len(predicted_tokens) / float(target_length) if target_length else 0.0,
        "visual_score": max(0.0, 1.0 - image_diff / 100.0) if image_diff is not None else 0.0,
        "parent_edge_f1": content["parent_edge_f1"],
        "token_f1": content["token_f1"],
        "button_token_f1": content["button_token_f1"],
        "text_token_f1": content["text_token_f1"],
        "chrf": chrf,
        "tree_similarity": metrics.tree_similarity(predicted_root, target_root),
        "render_error": render_error,
    }
    row.update(visual_v2)
    row.update(visual_v3)
    row.update(metrics.reward_components(row))
    row.update(metrics.visual_structural_v5_components(row))
    row.update(metrics.visual_structural_v6_components(row))
    return row


def reward_for_mode(score, mode):
    if mode == "structural_rl_v4":
        return float(score["reward"])
    if mode == "visual_structural_rl_v6":
        return float(score["reward_v6"])
    raise ValueError("{} has no RL reward".format(mode))


def teacher_forced_batch(sampler, target_gui):
    tokens = list(Dataset.tokenize_gui(target_gui.splitlines(True)))
    token_ids = [sampler.voc.vocabulary[token] for token in tokens if token in sampler.voc.vocabulary]
    context = [np.eye(sampler.output_size, dtype=np.float32)[sampler.voc.vocabulary[PLACEHOLDER]]] * (sampler.context_length - 1)
    context.append(np.eye(sampler.output_size, dtype=np.float32)[sampler.voc.vocabulary[START_TOKEN]])
    contexts, labels = [], []
    for token_id in token_ids[1:]:
        contexts.append(np.asarray(context, dtype=np.float32))
        labels.append(token_id)
        context = append_token(context, token_id, sampler.output_size)
    if not labels:
        raise ValueError("target GUI contains no teacher-forced prediction tokens")
    return np.asarray(contexts), np.asarray(labels, dtype=np.int32)


def copy_evaluation_sidecars(source_dir, output_dir):
    for name in ["Main_Model.weights.json", "meta_dataset.npy", "words.vocab", "training_profile.txt",
                 "resnet50.weights.h5", "resnet50.weights.json"]:
        source = os.path.join(source_dir, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(output_dir, name))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True
        ).stdout.strip().splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}


def write_json(path, value):
    with open(path, "w") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False)
        destination.write("\n")


def write_json_atomic(path, value):
    temporary_path = "{}.tmp".format(path)
    write_json(temporary_path, value)
    os.replace(temporary_path, path)


def save_weights_atomically(model, path):
    temporary_path = "{}.tmp.weights.h5".format(path)
    model.model.save_weights(temporary_path)
    os.replace(temporary_path, path)


def save_training_state(model, checkpoint_manager, output_dir, completed_steps, rng):
    """Persist weights, optimizer state, and sampling state after a safe step."""
    checkpoint_manager.save(checkpoint_number=completed_steps)
    save_weights_atomically(model, os.path.join(output_dir, "Main_Model.weights.h5"))
    write_json_atomic(os.path.join(output_dir, "progress.json"), {
        "completed_steps": completed_steps,
        "checkpoint": checkpoint_manager.latest_checkpoint,
        "rng_state": rng.bit_generator.state,
        "saved_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    })


def read_jsonl(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def recover_log_for_resume(path, completed_steps):
    """Discard records newer than the last durable checkpoint after a hard kill."""
    records = read_jsonl(path)
    if len(records) < completed_steps:
        raise SystemExit("step log has fewer records than progress.json")
    if any(record.get("step") != index + 1 for index, record in enumerate(records[:completed_steps])):
        raise SystemExit("step log is not sequential up to the durable checkpoint")
    if len(records) == completed_steps:
        return
    with open(path, "w") as destination:
        for record in records[:completed_steps]:
            destination.write(json.dumps(record) + "\n")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="web_generated")
    parser.add_argument("--weights-path", default=None, help="supervised checkpoint directory")
    parser.add_argument("--input-path", default=None, help="defaults to selected profile training_set")
    parser.add_argument("--rules-path", default=DEFAULT_RULES_PATH)
    parser.add_argument("--mode", default="structural_rl_v4", choices=MODES)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--min-target-length", type=int, default=0)
    parser.add_argument("--max-target-length", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--ce-weight", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--render-timeout-seconds", type=int, default=90)
    parser.add_argument("--gradient-microbatch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="continue from the latest durable checkpoint in --output-dir")
    parser.add_argument("--no-render-reward", action="store_true", help="only allowed for the CE control")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def checked_float(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise FloatingPointError("{} is not finite".format(name))
    return value


def add_gradients(accumulated, gradients, variables):
    """Accumulate gradients without retaining a tape for all rollout actions."""
    result = []
    for current, gradient, variable in zip(accumulated, gradients, variables):
        if gradient is None:
            result.append(current)
        else:
            # Token embeddings produce IndexedSlices.  TensorFlow intentionally
            # does not define ``IndexedSlices + IndexedSlices``; densifying this
            # small vocabulary gradient makes microbatch accumulation uniform.
            dense_gradient = tf.convert_to_tensor(gradient)
            result.append(dense_gradient if current is None else current + dense_gradient)
    return result


def apply_gradients(optimizer, variables, gradients):
    pairs = [(gradient, variable) for gradient, variable in zip(gradients, variables) if gradient is not None]
    if not pairs:
        raise RuntimeError("loss produced no gradients")
    clipped_gradients, gradient_norm = tf.clip_by_global_norm([pair[0] for pair in pairs], 1.0)
    checked_float(gradient_norm.numpy(), "gradient_norm")
    optimizer.apply_gradients(zip(clipped_gradients, [pair[1] for pair in pairs]))
    return checked_float(gradient_norm.numpy(), "gradient_norm")


def combine_gradients(primary, secondary, secondary_weight):
    """Form ``primary + secondary_weight * secondary`` without dropping None grads."""
    combined = []
    for first, second in zip(primary, secondary):
        if first is None:
            combined.append(None if second is None else secondary_weight * second)
        elif second is None:
            combined.append(first)
        else:
            combined.append(first + secondary_weight * second)
    return combined


def supervised_loss_and_gradients(network, image, contexts, actions, microbatch_size):
    """Compute CE gradients in bounded batches while preserving mean loss."""
    total_actions = len(actions)
    accumulated = [None] * len(network.trainable_variables)
    negative_log_probability_sum = 0.0
    for start in range(0, total_actions, microbatch_size):
        end = min(total_actions, start + microbatch_size)
        batch_contexts = tf.convert_to_tensor(contexts[start:end], dtype=tf.float32)
        batch_actions = tf.convert_to_tensor(actions[start:end], dtype=tf.int32)
        batch_images = tf.repeat(tf.convert_to_tensor(image[None, ...], dtype=tf.float32), end - start, axis=0)
        with tf.GradientTape() as tape:
            probabilities = network([batch_images, batch_contexts], training=True)
            action_probabilities = tf.gather(probabilities, batch_actions, batch_dims=1)
            loss = -tf.reduce_sum(tf.math.log(tf.clip_by_value(action_probabilities, 1e-8, 1.0))) / total_actions
        accumulated = add_gradients(
            accumulated, tape.gradient(loss, network.trainable_variables), network.trainable_variables
        )
        negative_log_probability_sum += checked_float(loss.numpy(), "ce_loss_chunk") * total_actions
    return negative_log_probability_sum / total_actions, accumulated


def policy_loss_and_gradients(network, image, contexts, actions, masks, advantage, microbatch_size):
    """Replay a grammar-masked policy gradient without a rollout-sized tape."""
    total_actions = len(actions)
    accumulated = [None] * len(network.trainable_variables)
    policy_loss = 0.0
    advantage = float(advantage)
    for start in range(0, total_actions, microbatch_size):
        end = min(total_actions, start + microbatch_size)
        batch_contexts = tf.convert_to_tensor(contexts[start:end], dtype=tf.float32)
        batch_actions = tf.convert_to_tensor(actions[start:end], dtype=tf.int32)
        batch_masks = tf.convert_to_tensor(masks[start:end], dtype=tf.float32)
        batch_images = tf.repeat(tf.convert_to_tensor(image[None, ...], dtype=tf.float32), end - start, axis=0)
        with tf.GradientTape() as tape:
            probabilities = network([batch_images, batch_contexts], training=False)
            constrained = masked_action_probabilities(probabilities, batch_masks)
            action_probabilities = tf.gather(constrained, batch_actions, batch_dims=1)
            loss = -tf.stop_gradient(tf.cast(advantage, probabilities.dtype)) * tf.reduce_sum(
                tf.math.log(tf.clip_by_value(action_probabilities, 1e-8, 1.0))
            ) / total_actions
        accumulated = add_gradients(
            accumulated, tape.gradient(loss, network.trainable_variables), network.trainable_variables
        )
        policy_loss += checked_float(loss.numpy(), "policy_loss_chunk")
    return policy_loss, accumulated


def main(argv):
    started_at = time.monotonic()
    args = parse_args(argv)
    if (args.steps <= 0 or args.sequence_length <= 0 or args.checkpoint_every <= 0
            or args.render_timeout_seconds <= 0 or args.gradient_microbatch_size <= 0):
        raise SystemExit(
            "--steps, --sequence-length, --checkpoint-every, --render-timeout-seconds, "
            "and --gradient-microbatch-size must be positive"
        )
    if args.learning_rate <= 0.0 or args.ce_weight < 0.0:
        raise SystemExit("--learning-rate must be positive and --ce-weight must be non-negative")
    if args.no_render_reward and args.mode != "ce_control":
        raise SystemExit("--no-render-reward is only valid for ce_control; RL rewards require rendering")

    profile = dataset_profiles.get(args.profile)
    weights_path = args.weights_path or profile["output_dir"]
    input_path = args.input_path or profile["training_set"]
    source_weight_path = os.path.join(weights_path, "Main_Model.weights.h5")
    if not os.path.isfile(source_weight_path):
        raise SystemExit("supervised weights not found: {}".format(source_weight_path))
    output_dir = args.output_dir or os.path.join(
        weights_path, "rl_step2_{}_{}".format(args.mode, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    if args.resume:
        if not os.path.isdir(output_dir):
            raise SystemExit("--resume requires an existing output directory")
    else:
        os.makedirs(output_dir, exist_ok=False)
        copy_evaluation_sidecars(weights_path, output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    candidates = []
    for filename in sorted(os.listdir(input_path)):
        if not filename.endswith(".gui"):
            continue
        gui_name = filename[:-4]
        target_gui = metrics.read_target_gui(input_path, gui_name)
        _, target_tokens, _, _ = metrics.parse_dsl_tree(target_gui)
        if len(target_tokens) < args.min_target_length:
            continue
        if args.max_target_length and len(target_tokens) > args.max_target_length:
            continue
        candidates.append((gui_name, target_gui))
    if not candidates:
        raise SystemExit("No training samples match the target-length filters")
    np.random.default_rng(args.seed).shuffle(candidates)
    step_samples = [candidates[step % len(candidates)][0] for step in range(args.steps)]

    reward_version = reward_version_for_mode(args.mode)
    config = vars(args).copy()
    config.update({
        "mode": args.mode,
        "reward_version": reward_version,
        "source_weights_path": weights_path,
        "source_checkpoint_path": source_weight_path,
        "source_checkpoint_sha256": sha256_file(source_weight_path),
        "input_path": input_path,
        "selected_samples": len(candidates),
        "git": git_metadata(),
    })
    sample_manifest = {
        "candidate_order": [sample for sample, _ in candidates],
        "step_samples": step_samples,
    }
    config_path = os.path.join(output_dir, "run_config.json")
    samples_path = os.path.join(output_dir, "training_samples.json")
    progress_path = os.path.join(output_dir, "progress.json")
    log_path = os.path.join(output_dir, "rl_metrics.jsonl")
    completed_steps = 0
    if args.resume:
        try:
            with open(config_path) as source:
                previous_config = json.load(source)
            with open(samples_path) as source:
                previous_manifest = json.load(source)
            with open(progress_path) as source:
                progress = json.load(source)
        except (OSError, ValueError) as exc:
            raise SystemExit("--resume needs valid config, sample manifest, and progress files: {}".format(exc))
        for key in ["mode", "steps", "sequence_length", "min_target_length", "max_target_length",
                    "learning_rate", "ce_weight", "seed", "source_checkpoint_sha256"]:
            if previous_config.get(key) != config.get(key):
                raise SystemExit("--resume argument differs from original run for {!r}".format(key))
        if previous_manifest != sample_manifest:
            raise SystemExit("--resume training-sample order differs from the original run")
        completed_steps = int(progress.get("completed_steps", -1))
        if completed_steps < 0 or completed_steps > args.steps:
            raise SystemExit("progress.json has an invalid completed_steps value")
        rng_state = progress.get("rng_state")
        if not isinstance(rng_state, dict):
            raise SystemExit("progress.json has no recoverable NumPy RNG state")
        rng.bit_generator.state = rng_state
        recover_log_for_resume(log_path, completed_steps)
    else:
        write_json(config_path, config)
        write_json(samples_path, sample_manifest)

    print("[rl] mode={} reward={} output={}".format(args.mode, reward_version, output_dir), flush=True)
    print("[rl] source weights={} sha256={}".format(weights_path, config["source_checkpoint_sha256"]), flush=True)
    print("[rl] selected {} training samples; steps={}, completed={}".format(
        len(candidates), args.steps, completed_steps
    ), flush=True)

    meta = np.load(os.path.join(weights_path, "meta_dataset.npy"), allow_pickle=True)
    input_shape, output_size = meta[0], int(meta[1])
    model = Main_Model(input_shape, output_size, weights_path)
    model.load("Main_Model.weights")
    sampler = Sampler(weights_path, input_shape, output_size, CONTEXT_LENGTH)
    compiler = Compiler(profile["dsl_mapping"]) if args.mode != "ce_control" else None
    include_legacy_reward = args.mode == "structural_rl_v4"
    render_html = (
        (lambda html, path: isolated_render_html(html, path, args.render_timeout_seconds))
        if args.mode != "ce_control" else None
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    if hasattr(optimizer, "build"):
        optimizer.build(model.model.trainable_variables)
    training_checkpoint = tf.train.Checkpoint(model=model.model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        training_checkpoint, os.path.join(output_dir, "training_state"), max_to_keep=2
    )
    if args.resume:
        latest_checkpoint = checkpoint_manager.latest_checkpoint
        if not latest_checkpoint:
            raise SystemExit("--resume found no TensorFlow training-state checkpoint")
        training_checkpoint.restore(latest_checkpoint).expect_partial()
        print("[rl] resumed from {} at step {}".format(latest_checkpoint, completed_steps), flush=True)

    log_mode = "a" if args.resume else "w"
    with open(log_path, log_mode) as log_file:
        for step in range(completed_steps, args.steps):
            gui_name, target_gui = candidates[step % len(candidates)]
            step_started_at = time.monotonic()
            image = Utils.get_preprocessed_img(os.path.join(input_path, gui_name + ".png"), IMAGE_SIZE)
            ce_contexts, ce_actions = teacher_forced_batch(sampler, target_gui)
            variables = model.model.trainable_variables

            record = {
                "step": step + 1,
                "sample": gui_name,
                "mode": args.mode,
                "reward_version": reward_version,
                "sampled_actions": 0,
                "greedy_actions": 0,
                "sample_reward_v4": None,
                "greedy_reward_v4": None,
                "sample_reward_v6": None,
                "greedy_reward_v6": None,
                "advantage": None,
                "sample_render_success": None,
                "greedy_render_success": None,
                "sample_render_error": "",
                "greedy_render_error": "",
            }

            if args.mode == "ce_control":
                ce_loss, ce_gradients = supervised_loss_and_gradients(
                    model.model, image, ce_contexts, ce_actions, args.gradient_microbatch_size
                )
                policy_loss = 0.0
                total_loss = ce_loss
                gradient_norm = apply_gradients(optimizer, variables, ce_gradients)
            else:
                print("[rl] step {}/{}: {} — sampling constrained rollouts...".format(
                    step + 1, args.steps, gui_name), flush=True)
                sampled, sample_contexts, sample_ids, sample_masks, sample_limit = rollout(
                    model.model, sampler, image, args.sequence_length, args.rules_path, True, rng
                )
                greedy, _, greedy_ids, _, greedy_limit = rollout(
                    model.model, sampler, image, args.sequence_length, args.rules_path, False, rng
                )
                # Keep every screenshot in an isolated WebKit process.  The
                # carousel-heavy step-7 page crashes WebKit when target and
                # prediction share a browser, while this original path handles
                # it correctly.  Durable checkpoints make the extra churn safe.
                with tempfile.TemporaryDirectory(dir=output_dir) as work_dir:
                    sample_score = score_rollout(
                        compiler, input_path, gui_name, target_gui, sampled, args.sequence_length,
                        sample_limit, True, work_dir, label="sampled", render_html=render_html,
                        include_legacy_reward=include_legacy_reward
                    )
                    greedy_score = score_rollout(
                        compiler, input_path, gui_name, target_gui, greedy, args.sequence_length,
                        greedy_limit, True, work_dir, label="greedy", render_html=render_html,
                        include_legacy_reward=include_legacy_reward
                    )
                advantage = reward_for_mode(sample_score, args.mode) - reward_for_mode(greedy_score, args.mode)
                # A 100-token rollout used to be kept in one GradientTape.  At
                # 256x256 this retains a full model's activations for every
                # action and can trigger macOS's OOM killer.  Accumulate the
                # same mean-loss gradients in small independent tapes instead.
                policy_loss, policy_gradients = policy_loss_and_gradients(
                    model.model, image, sample_contexts, sample_ids, sample_masks,
                    advantage, args.gradient_microbatch_size
                )
                ce_loss, ce_gradients = supervised_loss_and_gradients(
                    model.model, image, ce_contexts, ce_actions, args.gradient_microbatch_size
                )
                total_loss = policy_loss + args.ce_weight * ce_loss
                gradients = combine_gradients(policy_gradients, ce_gradients, args.ce_weight)
                gradient_norm = apply_gradients(optimizer, variables, gradients)
                record.update({
                    "sampled_actions": int(len(sample_ids)),
                    "greedy_actions": int(len(greedy_ids)),
                    "sample_reward_v4": float(sample_score["reward"]) if include_legacy_reward else None,
                    "greedy_reward_v4": float(greedy_score["reward"]) if include_legacy_reward else None,
                    "sample_reward_v6": float(sample_score["reward_v6"]),
                    "greedy_reward_v6": float(greedy_score["reward_v6"]),
                    "advantage": float(advantage),
                    "sample_render_success": int(sample_score["render_success"]),
                    "greedy_render_success": int(greedy_score["render_success"]),
                    "sample_render_error": sample_score["render_error"],
                    "greedy_render_error": greedy_score["render_error"],
                })

            record.update({
                "policy_loss": checked_float(policy_loss, "policy_loss"),
                "ce_loss": checked_float(ce_loss, "ce_loss"),
                "total_loss": checked_float(total_loss, "total_loss"),
                "gradient_norm": gradient_norm,
            })
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            completed_steps = step + 1
            if completed_steps % args.checkpoint_every == 0 or completed_steps == args.steps:
                save_training_state(model, checkpoint_manager, output_dir, completed_steps, rng)
                print("[rl] durable checkpoint saved at step {}".format(completed_steps), flush=True)
            average_step = (time.monotonic() - started_at) / float(step + 1)
            eta = average_step * (args.steps - step - 1)
            print("[rl] step {}/{} complete: loss={:.4f}, grad_norm={:.3f}, elapsed={}, eta={}".format(
                step + 1, args.steps, record["total_loss"], gradient_norm,
                format_duration(time.monotonic() - step_started_at), format_duration(eta)
            ), flush=True)

    checkpoint_path = os.path.join(output_dir, "Main_Model.weights.h5")
    print("[rl] saved {} after {}".format(checkpoint_path, format_duration(time.monotonic() - started_at)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
