# coding=utf-8
# Copyright 2025 TF.Text Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Microbenchmarks for tokenizers on IMDB dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np

import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow_text.python import ops as text_ops
from tensorflow_text.python.benchmarks import benchmark_utils


FLAGS = flags.FLAGS
flags.DEFINE_integer("run_iters", 1000, "Number of iterations to run")
flags.DEFINE_integer("burn_iters", 10, "Number of warmup runs")
flags.DEFINE_integer("batch_size", 32, "The size of a batch")
flags.DEFINE_boolean("run_eagerly", True, "Run in eager mode")
flags.DEFINE_boolean(
    "use_tf_function", True,
    "Wraps the op in a tf.function. Only works when eager mode is enabled")
flags.DEFINE_boolean("xprof_tracing", False, "Enables xprof tracing")
flags.DEFINE_boolean("with_offsets", False,
                     "Runs the op with offsets additionally")
flags.DEFINE_boolean(
    "ragged_vs_dense", False,
    "Run the tokenizers using ragged inputs and its dense counterpart")


class OpsBenchmark(benchmark_utils.OpsBaseBenchmark):
  """Benchmarks for various ops in TF Text."""

  def __init__(self):
    if not FLAGS.run_eagerly:
      ops.disable_eager_execution()

    self.use_tf_function = FLAGS.use_tf_function
    self.load_input_data(FLAGS.batch_size)

  def _run(self, op, kwargs=None):
    if FLAGS.ragged_vs_dense:
      self.run_and_report_ragged_vs_dense(
          op,
          FLAGS.run_iters,
          FLAGS.burn_iters,
          xprof_enabled=FLAGS.xprof_tracing,
          **(kwargs or {}))
      return

    self.run_and_report(
        op,
        FLAGS.run_iters,
        FLAGS.burn_iters,
        xprof_enabled=FLAGS.xprof_tracing,
        **(kwargs or {}))

  def benchmark_ngrams(self):
    self.input_data = text_ops.WhitespaceTokenizer().tokenize(self.input_data)

    self._run(
        text_ops.ngrams, {
            "width": 2,
            "axis": -1,
            "reduction_type": text_ops.Reduction.STRING_JOIN,
            "string_separator": "|"
        })

  def benchmark_sliding_window(self):
    self.input_data = text_ops.WhitespaceTokenizer().tokenize(self.input_data)

    self._run(text_ops.sliding_window, {"width": 3, "axis": -1})

  def benchmark_case_fold_utf8(self):
    self._run(text_ops.case_fold_utf8)

  def benchmark_normalize_utf8(self):
    self._run(text_ops.normalize_utf8, {"normalization_form": "NFKC"})

  def benchmark_normalize_utf8_with_offsets(self):
    if FLAGS.with_offsets:
      self._run(text_ops.normalize_utf8_with_offsets_map,
                {"normalization_form": "NFKC"})

  def benchmark_coerce_to_structurally_valid_utf8(self):
    if FLAGS.ragged_vs_dense:
      return

    # The input here is a valid UTF-8 input
    self._run(text_ops.coerce_to_structurally_valid_utf8)

  def benchmark_pad_along_dimension(self):
    self.input_data = text_ops.WhitespaceTokenizer().tokenize(self.input_data)

    self._run(text_ops.pad_along_dimension, {
        "axis": -1,
        "right_pad": ["RP"],
        "left_pad": ["LP"]
    })

  def benchmark_state_based_sentence_breaking(self):
    if FLAGS.ragged_vs_dense:
      return

    # TODO(b/167267653): Remove custom input(line below) when the bug is fixed
    self.input_data = constant_op.constant(["Hello (who are you)? Foo bar!"])

    sentence_breaker = text_ops.StateBasedSentenceBreaker()
    self._run(sentence_breaker.break_sentences)

  def benchmark_create_feature_bitmask(self):
    if FLAGS.ragged_vs_dense:
      return

    self.input_data = array_ops.placeholder_with_default(
        constant_op.constant([[[True, True, False], [True, False, False]],
                              [[False, False, True], [True, False, True]]]),
        shape=None)

    self._run(text_ops.create_feature_bitmask)


class ConstrainedSequenceOpsBenchmark(benchmark_utils.OpsBaseBenchmark):
  """Benchmarks for constrained sequence ops in TF Text."""

  def __init__(self):
    if not FLAGS.run_eagerly:
      ops.disable_eager_execution()

    self.use_tf_function = FLAGS.use_tf_function
    self.load_input_data(FLAGS.batch_size)

  def load_input_data(self, batch_size):
    scores = [[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]]

    self.input_data = constant_op.constant([scores, scores, scores],
                                           dtype=np.float32)
    self.transition_weights = constant_op.constant(
        [[.1, .2, .3, .4, .1], [.5, .6, .7, .8, .1], [.9, 1, .15, 1, .1],
         [.25, .35, .45, .55, .5], [.1, .5, .1, .1, 1]],
        dtype=np.float32)

    self.allowed_transitions = constant_op.constant(
        [[True, True, True, True, True], [True, True, True, True, True],
         [True, False, True, False, True], [True, True, True, True, True],
         [True, True, True, True, False]])

  def _run(self, op, kwargs=None):
    self.run_and_report(
        op,
        FLAGS.run_iters,
        FLAGS.burn_iters,
        xprof_enabled=FLAGS.xprof_tracing,
        **(kwargs or {}))

  def benchmark_greedy_constrained_sequence(self):
    if FLAGS.ragged_vs_dense:
      return

    self._run(
        text_ops.greedy_constrained_sequence, {
            "transition_weights": self.transition_weights,
            "allowed_transitions": self.allowed_transitions,
            "use_log_space": True,
            "use_start_and_end_states": True
        })

  def benchmark_viterb_constrained_sequence(self):
    if FLAGS.ragged_vs_dense:
      return

    self._run(
        text_ops.greedy_constrained_sequence, {
            "transition_weights": self.transition_weights,
            "allowed_transitions": self.allowed_transitions,
            "use_log_space": True,
            "use_start_and_end_states": True
        })


if __name__ == "__main__":
  app.run(tf.test.main())
