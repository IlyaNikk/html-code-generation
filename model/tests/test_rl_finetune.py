"""Unit tests for the constrained-distribution part of Step-2 RL."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from model.rl_finetune import (
    add_gradients,
    mask_probabilities,
    masked_action_probabilities,
    isolated_render_html,
    policy_loss_and_gradients,
    recover_log_for_resume,
    reward_version_for_mode,
    supervised_loss_and_gradients,
    valid_token_mask,
)


class GrammarMaskedPolicyTest(unittest.TestCase):
    def test_sampling_distribution_excludes_invalid_tokens(self):
        distribution = mask_probabilities(np.asarray([0.1, 0.2, 0.3, 0.4]), [1, 3])
        np.testing.assert_allclose(distribution, [0.0, 1.0 / 3.0, 0.0, 2.0 / 3.0])
        self.assertEqual(1.0, float(distribution.sum()))

    def test_empty_constraint_allows_every_token(self):
        mask = valid_token_mask(3, [])
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_replayed_policy_probability_matches_sampling_distribution(self):
        probabilities = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)
        masks = tf.constant([[0.0, 1.0, 0.0, 1.0]], dtype=tf.float32)
        replayed = masked_action_probabilities(probabilities, masks).numpy()[0]
        expected = mask_probabilities(np.asarray([0.1, 0.2, 0.3, 0.4]), [1, 3])
        np.testing.assert_allclose(replayed, expected, rtol=1e-6, atol=1e-6)

    def test_mode_reward_versions_are_explicit(self):
        self.assertEqual("none", reward_version_for_mode("ce_control"))
        self.assertEqual("content_structural_v4", reward_version_for_mode("structural_rl_v4"))
        self.assertEqual("visual_structural_v6_v3", reward_version_for_mode("visual_structural_rl_v6"))

    def test_resume_discards_records_after_last_durable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl_metrics.jsonl"
            path.write_text("\n".join(
                json.dumps({"step": step, "total_loss": 0.1, "gradient_norm": 0.2})
                for step in [1, 2, 3]
            ) + "\n")
            recover_log_for_resume(path, completed_steps=2)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([1, 2], [record["step"] for record in records])

    def test_renderer_worker_failure_is_reported_without_killing_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "render.png"
            with patch("model.rl_finetune.subprocess.run") as run:
                run.return_value = SimpleNamespace(returncode=-9, stdout="worker killed")
                with self.assertRaisesRegex(RuntimeError, "renderer worker exited -9"):
                    isolated_render_html("<html></html>", str(output), timeout_seconds=5)
            self.assertIn("--page-timeout-seconds", run.call_args.args[0])
            self.assertFalse(Path(str(output) + ".html").exists())

    def test_microbatch_gradients_match_full_mean_objective(self):
        image_input = tf.keras.Input(shape=(1,))
        context_input = tf.keras.Input(shape=(2,))
        features = tf.keras.layers.Concatenate()([image_input, context_input])
        probabilities = tf.keras.layers.Dense(3, activation="softmax", use_bias=False)(features)
        network = tf.keras.Model([image_input, context_input], probabilities)
        network.layers[-1].set_weights([np.asarray([
            [0.1, -0.2, 0.3],
            [0.4, 0.2, -0.1],
            [-0.3, 0.5, 0.2],
        ], dtype=np.float32)])
        image = np.asarray([0.25], dtype=np.float32)
        contexts = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        actions = np.asarray([0, 1, 2, 1, 0], dtype=np.int32)
        masks = np.ones((len(actions), 3), dtype=np.float32)
        images = tf.repeat(tf.convert_to_tensor(image[None, ...]), len(actions), axis=0)
        context_tensor = tf.convert_to_tensor(contexts)
        action_tensor = tf.convert_to_tensor(actions)

        with tf.GradientTape() as tape:
            full_probabilities = network([images, context_tensor], training=True)
            selected = tf.gather(full_probabilities, action_tensor, batch_dims=1)
            full_ce_loss = -tf.reduce_mean(tf.math.log(selected))
        full_ce_gradients = tape.gradient(full_ce_loss, network.trainable_variables)
        micro_ce_loss, micro_ce_gradients = supervised_loss_and_gradients(
            network, image, contexts, actions, microbatch_size=2
        )
        self.assertAlmostEqual(float(full_ce_loss.numpy()), micro_ce_loss, places=6)
        for expected, actual in zip(full_ce_gradients, micro_ce_gradients):
            np.testing.assert_allclose(expected.numpy(), actual.numpy(), rtol=1e-6, atol=1e-6)

        advantage = 0.75
        with tf.GradientTape() as tape:
            full_probabilities = network([images, context_tensor], training=False)
            selected = tf.gather(full_probabilities, action_tensor, batch_dims=1)
            full_policy_loss = -advantage * tf.reduce_mean(tf.math.log(selected))
        full_policy_gradients = tape.gradient(full_policy_loss, network.trainable_variables)
        micro_policy_loss, micro_policy_gradients = policy_loss_and_gradients(
            network, image, contexts, actions, masks, advantage, microbatch_size=2
        )
        self.assertAlmostEqual(float(full_policy_loss.numpy()), micro_policy_loss, places=6)
        for expected, actual in zip(full_policy_gradients, micro_policy_gradients):
            np.testing.assert_allclose(expected.numpy(), actual.numpy(), rtol=1e-6, atol=1e-6)

    def test_gradient_accumulation_densifies_sparse_embedding_gradient(self):
        variable = tf.Variable(tf.zeros((4, 2), dtype=tf.float32))
        first = tf.IndexedSlices(
            values=tf.constant([[1.0, 2.0]], dtype=tf.float32),
            indices=tf.constant([1]),
            dense_shape=tf.constant([4, 2]),
        )
        second = tf.IndexedSlices(
            values=tf.constant([[3.0, 4.0]], dtype=tf.float32),
            indices=tf.constant([1]),
            dense_shape=tf.constant([4, 2]),
        )
        accumulated = add_gradients([None], [first], [variable])
        accumulated = add_gradients(accumulated, [second], [variable])
        np.testing.assert_array_equal(
            accumulated[0].numpy(),
            np.asarray([[0.0, 0.0], [4.0, 6.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
