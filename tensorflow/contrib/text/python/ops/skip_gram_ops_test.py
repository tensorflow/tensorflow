# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Skip-gram sampling ops tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

from tensorflow.contrib import lookup
from tensorflow.contrib import text
from tensorflow.contrib.text.python.ops import skip_gram_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl


class SkipGramOpsTest(test.TestCase):

  def _split_tokens_labels(self, output):
    tokens = [x[0] for x in output]
    labels = [x[1] for x in output]
    return tokens, labels

  def test_skip_gram_sample_skips_2(self):
    """Tests skip-gram with min_skips = max_skips = 2."""
    input_tensor = constant_op.constant(
        [b"the", b"quick", b"brown", b"fox", b"jumps"])
    tokens, labels = text.skip_gram_sample(
        input_tensor, min_skips=2, max_skips=2)
    expected_tokens, expected_labels = self._split_tokens_labels([
        (b"the", b"quick"),
        (b"the", b"brown"),
        (b"quick", b"the"),
        (b"quick", b"brown"),
        (b"quick", b"fox"),
        (b"brown", b"the"),
        (b"brown", b"quick"),
        (b"brown", b"fox"),
        (b"brown", b"jumps"),
        (b"fox", b"quick"),
        (b"fox", b"brown"),
        (b"fox", b"jumps"),
        (b"jumps", b"brown"),
        (b"jumps", b"fox"),
    ])
    with self.cached_session():
      self.assertAllEqual(expected_tokens, tokens.eval())
      self.assertAllEqual(expected_labels, labels.eval())

  def test_skip_gram_sample_emit_self(self):
    """Tests skip-gram with emit_self_as_target = True."""
    input_tensor = constant_op.constant(
        [b"the", b"quick", b"brown", b"fox", b"jumps"])
    tokens, labels = text.skip_gram_sample(
        input_tensor, min_skips=2, max_skips=2, emit_self_as_target=True)
    expected_tokens, expected_labels = self._split_tokens_labels([
        (b"the", b"the"),
        (b"the", b"quick"),
        (b"the", b"brown"),
        (b"quick", b"the"),
        (b"quick", b"quick"),
        (b"quick", b"brown"),
        (b"quick", b"fox"),
        (b"brown", b"the"),
        (b"brown", b"quick"),
        (b"brown", b"brown"),
        (b"brown", b"fox"),
        (b"brown", b"jumps"),
        (b"fox", b"quick"),
        (b"fox", b"brown"),
        (b"fox", b"fox"),
        (b"fox", b"jumps"),
        (b"jumps", b"brown"),
        (b"jumps", b"fox"),
        (b"jumps", b"jumps"),
    ])
    with self.cached_session():
      self.assertAllEqual(expected_tokens, tokens.eval())
      self.assertAllEqual(expected_labels, labels.eval())

  def test_skip_gram_sample_skips_0(self):
    """Tests skip-gram with min_skips = max_skips = 0."""
    input_tensor = constant_op.constant([b"the", b"quick", b"brown"])

    # If emit_self_as_target is False (default), output will be empty.
    tokens, labels = text.skip_gram_sample(
        input_tensor, min_skips=0, max_skips=0, emit_self_as_target=False)
    with self.cached_session():
      self.assertEqual(0, tokens.eval().size)
      self.assertEqual(0, labels.eval().size)

    # If emit_self_as_target is True, each token will be its own label.
    tokens, labels = text.skip_gram_sample(
        input_tensor, min_skips=0, max_skips=0, emit_self_as_target=True)
    expected_tokens, expected_labels = self._split_tokens_labels([
        (b"the", b"the"),
        (b"quick", b"quick"),
        (b"brown", b"brown"),
    ])
    with self.cached_session():
      self.assertAllEqual(expected_tokens, tokens.eval())
      self.assertAllEqual(expected_labels, labels.eval())

  def test_skip_gram_sample_skips_exceed_length(self):
    """Tests skip-gram when min/max_skips exceed length of input."""
    input_tensor = constant_op.constant([b"the", b"quick", b"brown"])
    tokens, labels = text.skip_gram_sample(
        input_tensor, min_skips=100, max_skips=100)
    expected_tokens, expected_labels = self._split_tokens_labels([
        (b"the", b"quick"),
        (b"the", b"brown"),
        (b"quick", b"the"),
        (b"quick", b"brown"),
        (b"brown", b"the"),
        (b"brown", b"quick"),
    ])
    with self.cached_session():
      self.assertAllEqual(expected_tokens, tokens.eval())
      self.assertAllEqual(expected_labels, labels.eval())

  def test_skip_gram_sample_start_limit(self):
    """Tests skip-gram over a limited portion of the input."""
    input_tensor = constant_op.constant(
        [b"foo", b"the", b"quick", b"brown", b"bar"])
    tokens, labels = text.skip_gram_sample(
        input_tensor, min_skips=1, max_skips=1, start=1, limit=3)
    expected_tokens, expected_labels = self._split_tokens_labels([
        (b"the", b"quick"),
        (b"quick", b"the"),
        (b"quick", b"brown"),
        (b"brown", b"quick"),
    ])
    with self.cached_session():
      self.assertAllEqual(expected_tokens, tokens.eval())
      self.assertAllEqual(expected_labels, labels.eval())

  def test_skip_gram_sample_limit_exceeds(self):
    """Tests skip-gram when limit exceeds the length of the input."""
    input_tensor = constant_op.constant([b"foo", b"the", b"quick", b"brown"])
    tokens, labels = text.skip_gram_sample(
        input_tensor, min_skips=1, max_skips=1, start=1, limit=100)
    expected_tokens, expected_labels = self._split_tokens_labels([
        (b"the", b"quick"),
        (b"quick", b"the"),
        (b"quick", b"brown"),
        (b"brown", b"quick"),
    ])
    with self.cached_session():
      self.assertAllEqual(expected_tokens, tokens.eval())
      self.assertAllEqual(expected_labels, labels.eval())

  def test_skip_gram_sample_random_skips(self):
    """Tests skip-gram with min_skips != max_skips, with random output."""
    # The number of outputs is non-deterministic in this case, so set random
    # seed to help ensure the outputs remain constant for this test case.
    random_seed.set_random_seed(42)

    input_tensor = constant_op.constant(
        [b"the", b"quick", b"brown", b"fox", b"jumps", b"over"])
    tokens, labels = text.skip_gram_sample(
        input_tensor, min_skips=1, max_skips=2, seed=9)
    expected_tokens, expected_labels = self._split_tokens_labels([
        (b"the", b"quick"),
        (b"the", b"brown"),
        (b"quick", b"the"),
        (b"quick", b"brown"),
        (b"quick", b"fox"),
        (b"brown", b"the"),
        (b"brown", b"quick"),
        (b"brown", b"fox"),
        (b"brown", b"jumps"),
        (b"fox", b"brown"),
        (b"fox", b"jumps"),
        (b"jumps", b"fox"),
        (b"jumps", b"over"),
        (b"over", b"fox"),
        (b"over", b"jumps"),
    ])
    with self.cached_session() as sess:
      tokens_eval, labels_eval = sess.run([tokens, labels])
      self.assertAllEqual(expected_tokens, tokens_eval)
      self.assertAllEqual(expected_labels, labels_eval)

  def test_skip_gram_sample_random_skips_default_seed(self):
    """Tests outputs are still random when no op-level seed is specified."""
    # This is needed since tests set a graph-level seed by default. We want to
    # explicitly avoid setting both graph-level seed and op-level seed, to
    # simulate behavior under non-test settings when the user doesn't provide a
    # seed to us. This results in random_seed.get_seed() returning None for both
    # seeds, forcing the C++ kernel to execute its default seed logic.
    random_seed.set_random_seed(None)

    # Uses an input tensor with 10 words, with possible skip ranges in [1,
    # 5]. Thus, the probability that two random samplings would result in the
    # same outputs is 1/5^10 ~ 1e-7 (aka the probability of this test being
    # flaky).
    input_tensor = constant_op.constant([str(x) for x in range(10)])

    # Do not provide an op-level seed here!
    tokens_1, labels_1 = text.skip_gram_sample(
        input_tensor, min_skips=1, max_skips=5)
    tokens_2, labels_2 = text.skip_gram_sample(
        input_tensor, min_skips=1, max_skips=5)

    with self.cached_session() as sess:
      tokens_1_eval, labels_1_eval, tokens_2_eval, labels_2_eval = sess.run(
          [tokens_1, labels_1, tokens_2, labels_2])

    if len(tokens_1_eval) == len(tokens_2_eval):
      self.assertNotEqual(tokens_1_eval.tolist(), tokens_2_eval.tolist())
    if len(labels_1_eval) == len(labels_2_eval):
      self.assertNotEqual(labels_1_eval.tolist(), labels_2_eval.tolist())

  def test_skip_gram_sample_batch(self):
    """Tests skip-gram with batching."""
    input_tensor = constant_op.constant([b"the", b"quick", b"brown", b"fox"])
    tokens, labels = text.skip_gram_sample(
        input_tensor, min_skips=1, max_skips=1, batch_size=3)
    expected_tokens, expected_labels = self._split_tokens_labels([
        (b"the", b"quick"),
        (b"quick", b"the"),
        (b"quick", b"brown"),
        (b"brown", b"quick"),
        (b"brown", b"fox"),
        (b"fox", b"brown"),
    ])
    with self.cached_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)

      tokens_eval, labels_eval = sess.run([tokens, labels])
      self.assertAllEqual(expected_tokens[:3], tokens_eval)
      self.assertAllEqual(expected_labels[:3], labels_eval)
      tokens_eval, labels_eval = sess.run([tokens, labels])
      self.assertAllEqual(expected_tokens[3:6], tokens_eval)
      self.assertAllEqual(expected_labels[3:6], labels_eval)

      coord.request_stop()
      coord.join(threads)

  def test_skip_gram_sample_non_string_input(self):
    """Tests skip-gram with non-string input."""
    input_tensor = constant_op.constant([1, 2, 3], dtype=dtypes.int16)
    tokens, labels = text.skip_gram_sample(
        input_tensor, min_skips=1, max_skips=1)
    expected_tokens, expected_labels = self._split_tokens_labels([
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
    ])
    with self.cached_session():
      self.assertAllEqual(expected_tokens, tokens.eval())
      self.assertAllEqual(expected_labels, labels.eval())

  def test_skip_gram_sample_errors(self):
    """Tests various errors raised by skip_gram_sample()."""
    input_tensor = constant_op.constant([b"the", b"quick", b"brown"])

    invalid_skips = (
        # min_skips and max_skips must be >= 0.
        (-1, 2),
        (1, -2),
        # min_skips must be <= max_skips.
        (2, 1))
    for min_skips, max_skips in invalid_skips:
      tokens, labels = text.skip_gram_sample(
          input_tensor, min_skips=min_skips, max_skips=max_skips)
      with self.cached_session() as sess, self.assertRaises(
          errors.InvalidArgumentError):
        sess.run([tokens, labels])

    # input_tensor must be of rank 1.
    with self.assertRaises(ValueError):
      invalid_tensor = constant_op.constant([[b"the"], [b"quick"], [b"brown"]])
      text.skip_gram_sample(invalid_tensor)

    # vocab_freq_table must be provided if vocab_min_count, vocab_subsampling,
    # or corpus_size is specified.
    dummy_input = constant_op.constant([""])
    with self.assertRaises(ValueError):
      text.skip_gram_sample(
          dummy_input, vocab_freq_table=None, vocab_min_count=1)
    with self.assertRaises(ValueError):
      text.skip_gram_sample(
          dummy_input, vocab_freq_table=None, vocab_subsampling=1e-5)
    with self.assertRaises(ValueError):
      text.skip_gram_sample(dummy_input, vocab_freq_table=None, corpus_size=100)
    with self.assertRaises(ValueError):
      text.skip_gram_sample(
          dummy_input,
          vocab_freq_table=None,
          vocab_subsampling=1e-5,
          corpus_size=100)

    # vocab_subsampling and corpus_size must both be present or absent.
    dummy_table = lookup.HashTable(
        lookup.KeyValueTensorInitializer([b"foo"], [10]), -1)
    with self.assertRaises(ValueError):
      text.skip_gram_sample(
          dummy_input,
          vocab_freq_table=dummy_table,
          vocab_subsampling=None,
          corpus_size=100)
    with self.assertRaises(ValueError):
      text.skip_gram_sample(
          dummy_input,
          vocab_freq_table=dummy_table,
          vocab_subsampling=1e-5,
          corpus_size=None)

  def test_filter_input_filter_vocab(self):
    """Tests input filtering based on vocab frequency table and thresholds."""
    input_tensor = constant_op.constant(
        [b"the", b"answer", b"to", b"life", b"and", b"universe"])
    keys = constant_op.constant([b"and", b"life", b"the", b"to", b"universe"])
    values = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)
    vocab_freq_table = lookup.HashTable(
        lookup.KeyValueTensorInitializer(keys, values), -1)

    with self.cached_session():
      vocab_freq_table.init.run()

      # No vocab_freq_table specified - output should be the same as input.
      no_table_output = skip_gram_ops._filter_input(
          input_tensor=input_tensor,
          vocab_freq_table=None,
          vocab_min_count=None,
          vocab_subsampling=None,
          corpus_size=None,
          seed=None)
      self.assertAllEqual(input_tensor.eval(), no_table_output.eval())

      # vocab_freq_table specified, but no vocab_min_count - output should have
      # filtered out tokens not in the table (b"answer").
      table_output = skip_gram_ops._filter_input(
          input_tensor=input_tensor,
          vocab_freq_table=vocab_freq_table,
          vocab_min_count=None,
          vocab_subsampling=None,
          corpus_size=None,
          seed=None)
      self.assertAllEqual([b"the", b"to", b"life", b"and", b"universe"],
                          table_output.eval())

      # vocab_freq_table and vocab_min_count specified - output should have
      # filtered out tokens whose frequencies are below the threshold
      # (b"and": 0, b"life": 1).
      threshold_output = skip_gram_ops._filter_input(
          input_tensor=input_tensor,
          vocab_freq_table=vocab_freq_table,
          vocab_min_count=2,
          vocab_subsampling=None,
          corpus_size=None,
          seed=None)
      self.assertAllEqual([b"the", b"to", b"universe"], threshold_output.eval())

  def test_filter_input_subsample_vocab(self):
    """Tests input filtering based on vocab subsampling."""
    # The outputs are non-deterministic, so set random seed to help ensure that
    # the outputs remain constant for testing.
    random_seed.set_random_seed(42)

    input_tensor = constant_op.constant([
        # keep_prob = (sqrt(30/(0.05*100)) + 1) * (0.05*100/30) = 0.57.
        b"the",
        b"answer",  # Not in vocab. (Always discarded)
        b"to",  # keep_prob = 0.75.
        b"life",  # keep_prob > 1. (Always kept)
        b"and",  # keep_prob = 0.48.
        b"universe"  # Below vocab threshold of 3. (Always discarded)
    ])
    keys = constant_op.constant([b"and", b"life", b"the", b"to", b"universe"])
    values = constant_op.constant([40, 8, 30, 20, 2], dtypes.int64)
    vocab_freq_table = lookup.HashTable(
        lookup.KeyValueTensorInitializer(keys, values), -1)

    with self.cached_session():
      vocab_freq_table.init.run()
      output = skip_gram_ops._filter_input(
          input_tensor=input_tensor,
          vocab_freq_table=vocab_freq_table,
          vocab_min_count=3,
          vocab_subsampling=0.05,
          corpus_size=math_ops.reduce_sum(values),
          seed=9)
      self.assertAllEqual([b"the", b"to", b"life", b"and"], output.eval())

  def _make_text_vocab_freq_file(self):
    filepath = os.path.join(test.get_temp_dir(), "vocab_freq.txt")
    with open(filepath, "w") as f:
      writer = csv.writer(f)
      writer.writerows([
          ["and", 40],
          ["life", 8],
          ["the", 30],
          ["to", 20],
          ["universe", 2],
      ])
    return filepath

  def _make_text_vocab_float_file(self):
    filepath = os.path.join(test.get_temp_dir(), "vocab_freq_float.txt")
    with open(filepath, "w") as f:
      writer = csv.writer(f)
      writer.writerows([
          ["and", 0.4],
          ["life", 0.08],
          ["the", 0.3],
          ["to", 0.2],
          ["universe", 0.02],
      ])
    return filepath

  def test_skip_gram_sample_with_text_vocab_filter_vocab(self):
    """Tests skip-gram sampling with text vocab and freq threshold filtering."""
    input_tensor = constant_op.constant([
        b"the",
        b"answer",  # Will be filtered before candidate generation.
        b"to",
        b"life",
        b"and",
        b"universe"  # Will be filtered before candidate generation.
    ])

    # b"answer" is not in vocab file, and b"universe"'s frequency is below
    # threshold of 3.
    vocab_freq_file = self._make_text_vocab_freq_file()

    tokens, labels = text.skip_gram_sample_with_text_vocab(
        input_tensor=input_tensor,
        vocab_freq_file=vocab_freq_file,
        vocab_token_index=0,
        vocab_freq_index=1,
        vocab_min_count=3,
        min_skips=1,
        max_skips=1)

    expected_tokens, expected_labels = self._split_tokens_labels([
        (b"the", b"to"),
        (b"to", b"the"),
        (b"to", b"life"),
        (b"life", b"to"),
        (b"life", b"and"),
        (b"and", b"life"),
    ])
    with self.cached_session():
      lookup_ops.tables_initializer().run()
      self.assertAllEqual(expected_tokens, tokens.eval())
      self.assertAllEqual(expected_labels, labels.eval())

  def _text_vocab_subsample_vocab_helper(self, vocab_freq_file, vocab_min_count,
                                         vocab_freq_dtype, corpus_size=None):
    # The outputs are non-deterministic, so set random seed to help ensure that
    # the outputs remain constant for testing.
    random_seed.set_random_seed(42)

    input_tensor = constant_op.constant([
        # keep_prob = (sqrt(30/(0.05*100)) + 1) * (0.05*100/30) = 0.57.
        b"the",
        b"answer",  # Not in vocab. (Always discarded)
        b"to",  # keep_prob = 0.75.
        b"life",  # keep_prob > 1. (Always kept)
        b"and",  # keep_prob = 0.48.
        b"universe"  # Below vocab threshold of 3. (Always discarded)
    ])
    # keep_prob calculated from vocab file with relative frequencies of:
    # and: 40
    # life: 8
    # the: 30
    # to: 20
    # universe: 2

    tokens, labels = text.skip_gram_sample_with_text_vocab(
        input_tensor=input_tensor,
        vocab_freq_file=vocab_freq_file,
        vocab_token_index=0,
        vocab_freq_index=1,
        vocab_freq_dtype=vocab_freq_dtype,
        vocab_min_count=vocab_min_count,
        vocab_subsampling=0.05,
        corpus_size=corpus_size,
        min_skips=1,
        max_skips=1,
        seed=123)

    expected_tokens, expected_labels = self._split_tokens_labels([
        (b"the", b"to"),
        (b"to", b"the"),
        (b"to", b"life"),
        (b"life", b"to"),
    ])
    with self.cached_session() as sess:
      lookup_ops.tables_initializer().run()
      tokens_eval, labels_eval = sess.run([tokens, labels])
      self.assertAllEqual(expected_tokens, tokens_eval)
      self.assertAllEqual(expected_labels, labels_eval)

  def test_skip_gram_sample_with_text_vocab_subsample_vocab(self):
    """Tests skip-gram sampling with text vocab and vocab subsampling."""
    # Vocab file frequencies
    # and: 40
    # life: 8
    # the: 30
    # to: 20
    # universe: 2
    #
    # corpus_size for the above vocab is 40+8+30+20+2 = 100.
    text_vocab_freq_file = self._make_text_vocab_freq_file()
    self._text_vocab_subsample_vocab_helper(
        vocab_freq_file=text_vocab_freq_file,
        vocab_min_count=3,
        vocab_freq_dtype=dtypes.int64)
    self._text_vocab_subsample_vocab_helper(
        vocab_freq_file=text_vocab_freq_file,
        vocab_min_count=3,
        vocab_freq_dtype=dtypes.int64,
        corpus_size=100)

    # The user-supplied corpus_size should not be less than the sum of all
    # the frequency counts of vocab_freq_file, which is 100.
    with self.assertRaises(ValueError):
      self._text_vocab_subsample_vocab_helper(
          vocab_freq_file=text_vocab_freq_file,
          vocab_min_count=3,
          vocab_freq_dtype=dtypes.int64,
          corpus_size=99)

  def test_skip_gram_sample_with_text_vocab_subsample_vocab_float(self):
    """Tests skip-gram sampling with text vocab and subsampling with floats."""
    # Vocab file frequencies
    # and: 0.4
    # life: 0.08
    # the: 0.3
    # to: 0.2
    # universe: 0.02
    #
    # corpus_size for the above vocab is 0.4+0.08+0.3+0.2+0.02 = 1.
    text_vocab_float_file = self._make_text_vocab_float_file()
    self._text_vocab_subsample_vocab_helper(
        vocab_freq_file=text_vocab_float_file,
        vocab_min_count=0.03,
        vocab_freq_dtype=dtypes.float32)
    self._text_vocab_subsample_vocab_helper(
        vocab_freq_file=text_vocab_float_file,
        vocab_min_count=0.03,
        vocab_freq_dtype=dtypes.float32,
        corpus_size=1.0)

    # The user-supplied corpus_size should not be less than the sum of all
    # the frequency counts of vocab_freq_file, which is 1.
    with self.assertRaises(ValueError):
      self._text_vocab_subsample_vocab_helper(
          vocab_freq_file=text_vocab_float_file,
          vocab_min_count=0.03,
          vocab_freq_dtype=dtypes.float32,
          corpus_size=0.99)

  def test_skip_gram_sample_with_text_vocab_errors(self):
    """Tests various errors raised by skip_gram_sample_with_text_vocab()."""
    dummy_input = constant_op.constant([""])
    vocab_freq_file = self._make_text_vocab_freq_file()

    invalid_indices = (
        # vocab_token_index can't be negative.
        (-1, 0),
        # vocab_freq_index can't be negative.
        (0, -1),
        # vocab_token_index can't be equal to vocab_freq_index.
        (0, 0),
        (1, 1),
        # vocab_freq_file only has two columns.
        (0, 2),
        (2, 0))

    for vocab_token_index, vocab_freq_index in invalid_indices:
      with self.assertRaises(ValueError):
        text.skip_gram_sample_with_text_vocab(
            input_tensor=dummy_input,
            vocab_freq_file=vocab_freq_file,
            vocab_token_index=vocab_token_index,
            vocab_freq_index=vocab_freq_index)


if __name__ == "__main__":
  test.main()
