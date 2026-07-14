from __future__ import print_function
from __future__ import absolute_import

__author__ = 'Taneem Jan, taneemishere.github.io'

from .Vocabulary import *
from .Utils import *
from .GrammarConstrainedDecoder import GrammarConstrainedDecoder
import tensorflow as tf
import numpy as np

class Sampler:
    def __init__(self, voc_path, input_shape, output_size, context_length):
        self.voc = Vocabulary()
        self.voc.retrieve(voc_path)

        self.input_shape = input_shape
        self.output_size = output_size

        print("Vocabulary size: {}".format(self.voc.size))
        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

        self.context_length = context_length

    def predict_greedy(self, model, input_img, require_sparse_label=True, sequence_length=150, verbose=False, while_testing=False):
        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Utils.sparsify(current_context, self.output_size)

        predictions = START_TOKEN
        out_probas = []

        for i in range(0, sequence_length):
            if verbose:
                print("predicting {}/{}...".format(i, sequence_length))

            if while_testing:
                probas = model.predict(input_img, np.array([current_context]))
            else:
                probas = model.predict([input_img, np.array([current_context])], batch_size=1, steps=None, verbose=0)
            prediction = np.argmax(tf.nn.softmax(probas))

            out_probas.append(probas)

            new_context = []
            for j in range(1, self.context_length):
                new_context.append(current_context[j])

            if require_sparse_label:
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)
            else:
                new_context.append(prediction)

            current_context = new_context

            predictions += self.voc.token_lookup[prediction]

            if self.voc.token_lookup[prediction] == END_TOKEN:
                break

        return predictions, out_probas

    def predict_constrained(self, model, input_img, rules_path, require_sparse_label=True,
                            sequence_length=150, verbose=False, while_testing=False):
        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Utils.sparsify(current_context, self.output_size)

        predictions = START_TOKEN
        out_probas = []
        decoder = GrammarConstrainedDecoder(rules_path, self.voc)

        for i in range(0, sequence_length):
            if verbose:
                print("predicting {}/{}...".format(i, sequence_length))

            if while_testing:
                probas = model.predict(input_img, np.array([current_context]))
            else:
                probas = model.predict([input_img, np.array([current_context])], batch_size=1, steps=None, verbose=0)

            prediction = self._argmax_with_valid_mask(probas, decoder.valid_token_ids())
            out_probas.append(probas)

            current_context = self._append_prediction_to_context(
                current_context, prediction, require_sparse_label
            )

            token = self.voc.token_lookup[prediction]
            predictions += token
            decoder.update(token)

            if token == END_TOKEN:
                break

        if END_TOKEN not in predictions:
            for token in decoder.force_close_tokens():
                if token not in self.voc.vocabulary:
                    continue
                prediction = self.voc.vocabulary[token]
                current_context = self._append_prediction_to_context(
                    current_context, prediction, require_sparse_label
                )
                predictions += token
                if token == END_TOKEN:
                    break

        return predictions, out_probas

    def _argmax_with_valid_mask(self, probas, valid_ids):
        scores = np.asarray(tf.nn.softmax(probas)).reshape(-1)
        if len(valid_ids) == 0:
            return int(np.argmax(scores))

        masked = np.full(scores.shape, -np.inf)
        for valid_id in valid_ids:
            if valid_id < len(scores):
                masked[valid_id] = scores[valid_id]

        if np.all(np.isneginf(masked)):
            return int(np.argmax(scores))
        return int(np.argmax(masked))

    def _append_prediction_to_context(self, current_context, prediction, require_sparse_label):
        new_context = []
        for j in range(1, self.context_length):
            new_context.append(current_context[j])

        if require_sparse_label:
            sparse_label = np.zeros(self.output_size)
            sparse_label[prediction] = 1
            new_context.append(sparse_label)
        else:
            new_context.append(prediction)

        return new_context
