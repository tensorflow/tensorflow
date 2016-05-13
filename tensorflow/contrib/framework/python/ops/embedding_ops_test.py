# Copyright 2016 Google Inc. All Rights Reserved.
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
"""embedding_ops tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

import numpy as np
import tensorflow as tf


class SafeEmbeddingLookupSparseTest(tf.test.TestCase):

  def _random_ids_and_weights(self, vocab_size=4, embed_dim=4, num_shards=1):
    assert vocab_size > 0
    assert embed_dim > 0
    assert num_shards > 0
    assert num_shards <= vocab_size

    embedding_weights = tf.create_partitioned_variables(
        shape=[vocab_size, embed_dim],
        slicing=[num_shards, 1],
        initializer=tf.truncated_normal_initializer(
            mean=0.0,
            stddev=1.0 / math.sqrt(vocab_size),
            dtype=tf.float32))
    for w in embedding_weights:
      w.initializer.run()
    embedding_weights = [w.eval() for w in embedding_weights]

    # Each row demonstrates a test case:
    #   Row 0: multiple valid ids, 1 invalid id, weighted mean
    #   Row 1: all ids are invalid (leaving no valid ids after pruning)
    #   Row 2: no ids to begin with
    #   Row 3: single id
    #   Row 4: all ids have <=0 weight
    indices = [[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [4, 0], [4, 1]]
    ids = [0, 1, -1, -1, 2, 0, 1]
    weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
    shape = [5, 4]

    sparse_ids = tf.SparseTensor(
        tf.constant(indices, tf.int64),
        tf.constant(ids, tf.int64),
        tf.constant(shape, tf.int64))

    sparse_weights = tf.SparseTensor(
        tf.constant(indices, tf.int64),
        tf.constant(weights, tf.float32),
        tf.constant(shape, tf.int64))

    return embedding_weights, sparse_ids, sparse_weights

  def test_safe_embedding_lookup_sparse_return_zero_vector(self):
    with self.test_session():
      embedding_weights, sparse_ids, sparse_weights = (
          self._random_ids_and_weights())

      embedding_lookup_result = (
          tf.contrib.framework.safe_embedding_lookup_sparse(
              embedding_weights, sparse_ids, sparse_weights).eval())

      self.assertAllClose(embedding_lookup_result, [
          (1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) / 3.0,
          [0] * 4, [0] * 4, embedding_weights[0][2], [0] * 4
      ])

  def test_safe_embedding_lookup_sparse_return_special_vector(self):
    with self.test_session():
      embedding_weights, sparse_ids, sparse_weights = (
          self._random_ids_and_weights())

      embedding_lookup_result = (
          tf.contrib.framework.safe_embedding_lookup_sparse(
              embedding_weights, sparse_ids,
              sparse_weights, default_id=3).eval())

      self.assertAllClose(embedding_lookup_result, [
          (1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) / 3.0,
          embedding_weights[0][3], embedding_weights[0][3],
          embedding_weights[0][2], embedding_weights[0][3]])

  def test_safe_embedding_lookup_sparse_no_weights(self):
    with self.test_session():
      embedding_weights, sparse_ids, _ = self._random_ids_and_weights()

      embedding_lookup_result = (
          tf.contrib.framework.safe_embedding_lookup_sparse(
              embedding_weights, sparse_ids, None).eval())

      self.assertAllClose(embedding_lookup_result, [
          (embedding_weights[0][0] + embedding_weights[0][1]) / 2.0, [0] * 4,
          [0] * 4, embedding_weights[0][2],
          (embedding_weights[0][0] + embedding_weights[0][1]) / 2.0
      ])

  def test_safe_embedding_lookup_sparse_partitioned(self):
    with self.test_session():
      embedding_weights, sparse_ids, _ = self._random_ids_and_weights(
          vocab_size=4, num_shards=3)

      embedding_lookup_result = (
          tf.contrib.framework.safe_embedding_lookup_sparse(
              embedding_weights, sparse_ids, None).eval())

      embedding_weights = list(itertools.chain(*embedding_weights))
      self.assertAllClose(embedding_lookup_result, [
          (embedding_weights[0] + embedding_weights[1]) / 2.0, [0] * 4, [0] * 4,
          embedding_weights[2],
          (embedding_weights[0] + embedding_weights[1]) / 2.0
      ])

  def test_safe_embedding_lookup_sparse_partitioned_inconsistent_weights(self):
    with self.test_session():
      embedding_weights, sparse_ids, sparse_weights = (
          self._random_ids_and_weights(vocab_size=4, num_shards=3))

      embedding_weights[1] = embedding_weights[1].astype(np.float64)
      self.assertRaises(ValueError,
                        tf.contrib.framework.safe_embedding_lookup_sparse,
                        embedding_weights, sparse_ids)
      embedding_weights = [
          tf.constant(w, dtype=tf.float64) for w in embedding_weights]
      self.assertRaises(
          ValueError, tf.contrib.framework.safe_embedding_lookup_sparse,
          embedding_weights, sparse_ids, sparse_weights)


if __name__ == "__main__":
  tf.test.main()
