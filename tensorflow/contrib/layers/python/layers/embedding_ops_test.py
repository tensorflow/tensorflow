# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

  def _random_weights(self, vocab_size=4, embed_dim=4, num_shards=1):
    assert vocab_size > 0
    assert embed_dim > 0
    assert num_shards > 0
    assert num_shards <= vocab_size

    embedding_weights = tf.create_partitioned_variables(
        shape=[vocab_size, embed_dim],
        slicing=[num_shards, 1],
        initializer=tf.truncated_normal_initializer(mean=0.0,
                                                    stddev=1.0 /
                                                    math.sqrt(vocab_size),
                                                    dtype=tf.float32))
    for w in embedding_weights:
      w.initializer.run()
    embedding_weights = [w.eval() for w in embedding_weights]
    return embedding_weights

  def _ids_and_weights_2d(self):
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
        tf.constant(indices, tf.int64), tf.constant(ids, tf.int64),
        tf.constant(shape, tf.int64))

    sparse_weights = tf.SparseTensor(
        tf.constant(indices, tf.int64), tf.constant(weights, tf.float32),
        tf.constant(shape, tf.int64))

    return sparse_ids, sparse_weights

  def _ids_and_weights_3d(self):
    # Each (2-D) index demonstrates a test case:
    #   Index 0, 0: multiple valid ids, 1 invalid id, weighted mean
    #   Index 0, 1: all ids are invalid (leaving no valid ids after pruning)
    #   Index 0, 2: no ids to begin with
    #   Index 1, 0: single id
    #   Index 1, 1: all ids have <=0 weight
    #   Index 1, 2: no ids to begin with
    indices = [
        [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [1, 0, 0], [1, 1, 0],
        [1, 1, 1]
    ]
    ids = [0, 1, -1, -1, 2, 0, 1]
    weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
    shape = [2, 3, 4]

    sparse_ids = tf.SparseTensor(
        tf.constant(indices, tf.int64), tf.constant(ids, tf.int64),
        tf.constant(shape, tf.int64))

    sparse_weights = tf.SparseTensor(
        tf.constant(indices, tf.int64), tf.constant(weights, tf.float32),
        tf.constant(shape, tf.int64))

    return sparse_ids, sparse_weights

  def test_safe_embedding_lookup_sparse_return_zero_vector(self):
    with self.test_session():
      embedding_weights = self._random_weights()
      sparse_ids, sparse_weights = self._ids_and_weights_2d()

      embedding_lookup_result = (
          tf.contrib.layers.safe_embedding_lookup_sparse(
              embedding_weights, sparse_ids, sparse_weights).eval())

      self.assertAllClose(embedding_lookup_result, [
          (1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) / 3.0,
          [0] * 4, [0] * 4, embedding_weights[0][2], [0] * 4
      ])

  def test_safe_embedding_lookup_sparse_return_special_vector(self):
    with self.test_session():
      embedding_weights = self._random_weights()
      sparse_ids, sparse_weights = self._ids_and_weights_2d()

      embedding_lookup_result = (
          tf.contrib.layers.safe_embedding_lookup_sparse(
              embedding_weights,
              sparse_ids,
              sparse_weights,
              default_id=3).eval())

      self.assertAllClose(embedding_lookup_result, [
          (1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) / 3.0,
          embedding_weights[0][3], embedding_weights[0][3],
          embedding_weights[0][2], embedding_weights[0][3]
      ])

  def test_safe_embedding_lookup_sparse_no_weights(self):
    with self.test_session():
      embedding_weights = self._random_weights()
      sparse_ids, _ = self._ids_and_weights_2d()

      embedding_lookup_result = (
          tf.contrib.layers.safe_embedding_lookup_sparse(embedding_weights,
                                                         sparse_ids,
                                                         None).eval())

      self.assertAllClose(embedding_lookup_result, [
          (embedding_weights[0][0] + embedding_weights[0][1]) / 2.0, [0] * 4,
          [0] * 4, embedding_weights[0][2],
          (embedding_weights[0][0] + embedding_weights[0][1]) / 2.0
      ])

  def test_safe_embedding_lookup_sparse_partitioned(self):
    with self.test_session():
      embedding_weights = self._random_weights(num_shards=3)
      sparse_ids, _ = self._ids_and_weights_2d()

      embedding_lookup_result = (
          tf.contrib.layers.safe_embedding_lookup_sparse(embedding_weights,
                                                         sparse_ids,
                                                         None).eval())

      embedding_weights = list(itertools.chain(*embedding_weights))
      self.assertAllClose(embedding_lookup_result, [
          (embedding_weights[0] + embedding_weights[1]) / 2.0, [0] * 4, [0] * 4,
          embedding_weights[2],
          (embedding_weights[0] + embedding_weights[1]) / 2.0
      ])

  def test_safe_embedding_lookup_sparse_partitioned_inconsistent_weights(self):
    with self.test_session():
      embedding_weights = self._random_weights(num_shards=3)
      sparse_ids, sparse_weights = self._ids_and_weights_2d()

      embedding_weights[1] = embedding_weights[1].astype(np.float64)
      self.assertRaises(ValueError,
                        tf.contrib.layers.safe_embedding_lookup_sparse,
                        embedding_weights, sparse_ids)
      embedding_weights = [
          tf.constant(w, dtype=tf.float64) for w in embedding_weights
      ]
      self.assertRaises(ValueError,
                        tf.contrib.layers.safe_embedding_lookup_sparse,
                        embedding_weights, sparse_ids, sparse_weights)

  def test_safe_embedding_lookup_sparse_3d_return_zero_vector(self):
    with self.test_session():
      embedding_weights = self._random_weights()
      sparse_ids, sparse_weights = self._ids_and_weights_3d()

      embedding_lookup_result = (
          tf.contrib.layers.safe_embedding_lookup_sparse(
              embedding_weights, sparse_ids, sparse_weights).eval())

      self.assertAllClose(embedding_lookup_result, [
          [(1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) /
           3.0, [0] * 4, [0] * 4], [embedding_weights[0][2], [0] * 4, [0] * 4]
      ])

  def test_safe_embedding_lookup_sparse_3d_return_special_vector(self):
    with self.test_session():
      embedding_weights = self._random_weights()
      sparse_ids, sparse_weights = self._ids_and_weights_3d()

      embedding_lookup_result = (
          tf.contrib.layers.safe_embedding_lookup_sparse(
              embedding_weights,
              sparse_ids,
              sparse_weights,
              default_id=3).eval())

      self.assertAllClose(embedding_lookup_result, [
          [(1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) /
           3.0, embedding_weights[0][3], embedding_weights[0][3]],
          [embedding_weights[0][2], embedding_weights[0][3],
           embedding_weights[0][3]]
      ])

  def test_safe_embedding_lookup_sparse_3d_no_weights(self):
    with self.test_session():
      embedding_weights = self._random_weights()
      sparse_ids, _ = self._ids_and_weights_3d()

      embedding_lookup_result = (
          tf.contrib.layers.safe_embedding_lookup_sparse(embedding_weights,
                                                         sparse_ids,
                                                         None).eval())

      self.assertAllClose(embedding_lookup_result, [
          [(embedding_weights[0][0] + embedding_weights[0][1]) / 2.0, [0] * 4,
           [0] * 4], [embedding_weights[0][2],
                      (embedding_weights[0][0] + embedding_weights[0][1]) / 2.0,
                      [0] * 4]
      ])

  def test_safe_embedding_lookup_sparse_3d_partitioned(self):
    with self.test_session():
      embedding_weights = self._random_weights(num_shards=3)
      sparse_ids, _ = self._ids_and_weights_3d()

      embedding_lookup_result = (
          tf.contrib.layers.safe_embedding_lookup_sparse(embedding_weights,
                                                         sparse_ids,
                                                         None).eval())

      embedding_weights = list(itertools.chain(*embedding_weights))
      self.assertAllClose(embedding_lookup_result, [
          [(embedding_weights[0] + embedding_weights[1]) / 2.0, [0] * 4,
           [0] * 4], [embedding_weights[2],
                      (embedding_weights[0] + embedding_weights[1]) / 2.0,
                      [0] * 4]
      ])

  def test_safe_embedding_lookup_sparse_3d_partitioned_inconsistent_weights(
      self):
    with self.test_session():
      embedding_weights = self._random_weights(num_shards=3)
      sparse_ids, sparse_weights = self._ids_and_weights_3d()

      embedding_weights[1] = embedding_weights[1].astype(np.float64)
      self.assertRaises(ValueError,
                        tf.contrib.layers.safe_embedding_lookup_sparse,
                        embedding_weights, sparse_ids)
      embedding_weights = [
          tf.constant(w, dtype=tf.float64) for w in embedding_weights
      ]
      self.assertRaises(ValueError,
                        tf.contrib.layers.safe_embedding_lookup_sparse,
                        embedding_weights, sparse_ids, sparse_weights)


class HashedEmbeddingLookupTest(tf.test.TestCase):

  def setUp(self):
    tf.set_random_seed(1)

  def _random_weights(self, size=50, num_shards=1):
    assert size > 0
    assert num_shards > 0
    assert num_shards <= size

    embedding_weights = tf.create_partitioned_variables(
        shape=[size],
        slicing=[num_shards],
        initializer=tf.truncated_normal_initializer(mean=0.0,
                                                    stddev=1.0,
                                                    dtype=tf.float32))
    for w in embedding_weights:
      w.initializer.run()
    return embedding_weights

  def test_hashed_embedding_consistency(self):
    with self.test_session():
      embedding_weights = self._random_weights()
      values = tf.constant(["foo", "foo"])

      embedding_lookup_result = tf.contrib.layers.hashed_embedding_lookup(
          embedding_weights, values, dimension=10).eval()

      self.assertAllEqual(embedding_lookup_result.shape, [2, 10])
      self.assertAllEqual(embedding_lookup_result[0],
                          embedding_lookup_result[1])

  def test_hashed_embedding_multiple_partition(self):
    with self.test_session():
      embedding_weights = self._random_weights(num_shards=7)
      values = tf.constant([4, 4, 5])

      embedding_lookup_result = tf.contrib.layers.hashed_embedding_lookup(
          embedding_weights, values, dimension=5).eval()

      self.assertAllEqual(embedding_lookup_result.shape, [3, 5])
      self.assertAllEqual(embedding_lookup_result[0],
                          embedding_lookup_result[1])
      # Different embedding expected for different value.
      embedding_diff = np.min((embedding_lookup_result[2] -
                               embedding_lookup_result[0]) ** 2)
      self.assertGreater(embedding_diff, 0)

  def test_hashed_embedding_coverage(self):
    with self.test_session():
      size = 8
      embedding_weights = self._random_weights(size=size, num_shards=3)
      values = tf.constant(["foo"])

      # Large embedding dimension to cover the full range of weights.
      embedding_lookup_result = tf.contrib.layers.hashed_embedding_lookup(
          embedding_weights, values, dimension=100).eval()

      self.assertEqual(len(np.unique(embedding_lookup_result[0])), size)

  def test_hashed_embedding_multi_dimension(self):
    with self.test_session():
      embedding_weights = self._random_weights()
      values = tf.constant([["foo", "bar", "bar"], ["bar", "bar", "foo"]])

      embedding_lookup_result = tf.contrib.layers.hashed_embedding_lookup(
          embedding_weights, values, dimension=10).eval()

      self.assertAllEqual(embedding_lookup_result.shape, [2, 3, 10])
      self.assertAllEqual(embedding_lookup_result[0][0],
                          embedding_lookup_result[1][2])

  def test_hashed_embedding_lookup_sparse(self):
    with self.test_session():
      embedding_weights = self._random_weights(num_shards=3)
      sparse_tensor = tf.SparseTensor(values=["foo", "bar", "foo", "bar"],
                                      indices=[[0, 0], [1, 0], [1, 1], [3, 0]],
                                      shape=[5, 2])

      embedding_lookup_result = (
          tf.contrib.layers.hashed_embedding_lookup_sparse(
              embedding_weights, sparse_tensor, dimension=5, combiner="mean")
          .eval())

      self.assertAllEqual(embedding_lookup_result.shape, [5, 5])
      # Same non-zero embedding for the empty rows filled with a default value.
      self.assertAllEqual(embedding_lookup_result[2],
                          embedding_lookup_result[4])
      embedding_norm = np.sum(embedding_lookup_result[2] ** 2)
      self.assertGreater(embedding_norm, 0)

      self.assertAllEqual(embedding_lookup_result[1],
                          0.5 * (embedding_lookup_result[0] +
                                 embedding_lookup_result[3]))

  def test_embedding_lookup_unique(self):
    d_embed = 5
    n_embed = 10
    idx_shape = (2, 3, 4)
    embeds = np.random.randn(n_embed, d_embed)
    idx = np.random.randint(0, n_embed, idx_shape)

    with self.test_session():
      embedded_np = embeds[idx]
      embedded_tf = tf.contrib.layers.embedding_lookup_unique(
          embeds, idx).eval()

    self.assertEqual(embedded_np.shape, embedded_tf.shape)
    np.testing.assert_almost_equal(embedded_np, embedded_tf)

  def test_embedding_lookup_unique_param3d(self):
    embeds = np.random.randn(5, 3, 3)
    idx = np.random.randint(0, 5, 10)
    idx2d = np.random.randint(0, 5, (10, 2))

    with self.test_session():
      embedded_np = embeds[idx]
      embedded_np2d = embeds[idx2d]
      embedded_tf = tf.contrib.layers.embedding_lookup_unique(
          embeds, idx).eval()
      embedded_tf_lst = tf.contrib.layers.embedding_lookup_unique(
          [embeds], idx).eval()
      embedded_tf2d = tf.contrib.layers.embedding_lookup_unique(
          embeds, idx2d).eval()

    self.assertEqual(embedded_np.shape, embedded_tf.shape)
    np.testing.assert_almost_equal(embedded_np, embedded_tf)
    self.assertEqual(embedded_np.shape, embedded_tf_lst.shape)
    np.testing.assert_almost_equal(embedded_np, embedded_tf_lst)
    self.assertEqual(embedded_np2d.shape, embedded_tf2d.shape)
    np.testing.assert_almost_equal(embedded_np2d, embedded_tf2d)


if __name__ == "__main__":
  tf.test.main()
