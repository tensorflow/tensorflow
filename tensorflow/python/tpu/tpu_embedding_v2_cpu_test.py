# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPU Embeddings mid level API on CPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_test_lib
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.util import nest


class CPUEmbeddingTest(tpu_embedding_v2_test_lib.EmbeddingTestBase):

  def setUp(self):
    super(CPUEmbeddingTest, self).setUp()
    self._create_initial_data()

  def _create_mid_level(self):
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    return tpu_embedding_v2.TPUEmbedding(
        feature_config=self.feature_config,
        batch_size=self.batch_size,
        optimizer=optimizer)

  def _get_dense_tensors(self, dtype=dtypes.int32):
    feature0 = constant_op.constant(self.feature_watched_values, dtype=dtype)
    feature1 = constant_op.constant(self.feature_favorited_values, dtype=dtype)
    feature2 = constant_op.constant(self.feature_friends_values, dtype=dtype)
    return (feature0, feature1, feature2)

  def test_cpu_dense_lookup(self):
    mid_level = self._create_mid_level()
    features = self._get_dense_tensors()
    results = tpu_embedding_v2.cpu_embedding_lookup(
        features,
        weights=None,
        tables=mid_level.embedding_tables,
        feature_config=self.feature_config)
    all_lookups = []
    for feature, config in zip(nest.flatten(features), self.feature_config):
      table = mid_level.embedding_tables[config.table].numpy()
      all_lookups.append(table[feature.numpy()])
    self.assertAllClose(results, nest.pack_sequence_as(results, all_lookups))

  def test_cpu_dense_lookup_with_weights(self):
    mid_level = self._create_mid_level()
    features = self._get_dense_tensors()
    weights = self._get_dense_tensors(dtype=dtypes.float32)

    with self.assertRaisesRegex(
        ValueError, 'Weight specified for .*, but input is dense.'):
      tpu_embedding_v2.cpu_embedding_lookup(
          features,
          weights=weights,
          tables=mid_level.embedding_tables,
          feature_config=self.feature_config)

  def _get_sparse_tensors(self, dtype=dtypes.int32):
    feature0 = sparse_tensor.SparseTensor(
        indices=self.feature_watched_indices,
        values=constant_op.constant(self.feature_watched_values, dtype=dtype),
        dense_shape=[self.data_batch_size, 2])
    feature1 = sparse_tensor.SparseTensor(
        indices=self.feature_favorited_indices,
        values=constant_op.constant(self.feature_favorited_values, dtype=dtype),
        dense_shape=[self.data_batch_size, 2])
    feature2 = sparse_tensor.SparseTensor(
        indices=self.feature_friends_indices,
        values=constant_op.constant(self.feature_friends_values, dtype=dtype),
        dense_shape=[self.data_batch_size, 3])
    return (feature0, feature1, feature2)

  def test_cpu_sparse_lookup(self):
    mid_level = self._create_mid_level()
    features = self._get_sparse_tensors()
    results = tpu_embedding_v2.cpu_embedding_lookup(
        features,
        weights=None,
        tables=mid_level.embedding_tables,
        feature_config=self.feature_config)
    reduced = []
    for feature, config in zip(nest.flatten(features), self.feature_config):
      table = mid_level.embedding_tables[config.table].numpy()
      all_lookups = table[feature.values.numpy()]
      # With row starts we can use reduceat in numpy. Get row starts from the
      # ragged tensor API.
      ragged = ragged_tensor.RaggedTensor.from_sparse(feature)
      row_starts = ragged.row_starts().numpy()
      reduced.append(np.add.reduceat(all_lookups, row_starts))
      if config.table.combiner == 'mean':
        # for mean, divide by the row lengths.
        reduced[-1] /= np.expand_dims(ragged.row_lengths().numpy(), axis=1)
    self.assertAllClose(results, nest.pack_sequence_as(results, reduced))

  def test_cpu_sparse_lookup_with_weights(self):
    mid_level = self._create_mid_level()
    features = self._get_sparse_tensors()
    weights = self._get_sparse_tensors(dtype=dtypes.float32)
    results = tpu_embedding_v2.cpu_embedding_lookup(
        features,
        weights=weights,
        tables=mid_level.embedding_tables,
        feature_config=self.feature_config)
    weighted_sum = []
    for feature, weight, config in zip(nest.flatten(features),
                                       nest.flatten(weights),
                                       self.feature_config):
      table = mid_level.embedding_tables[config.table].numpy()
      # Expand dims here needed to broadcast this multiplication properly.
      weight = np.expand_dims(weight.values.numpy(), axis=1)
      all_lookups = table[feature.values.numpy()] * weight
      # With row starts we can use reduceat in numpy. Get row starts from the
      # ragged tensor API.
      row_starts = ragged_tensor.RaggedTensor.from_sparse(feature).row_starts()
      row_starts = row_starts.numpy()
      weighted_sum.append(np.add.reduceat(all_lookups, row_starts))
      if config.table.combiner == 'mean':
        weighted_sum[-1] /= np.add.reduceat(weight, row_starts)
    self.assertAllClose(results, nest.pack_sequence_as(results,
                                                       weighted_sum))

  def test_cpu_sparse_lookup_with_non_sparse_weights(self):
    mid_level = self._create_mid_level()
    features = self._get_sparse_tensors()
    weights = self._get_dense_tensors(dtype=dtypes.float32)
    with self.assertRaisesRegex(
        ValueError, 'but it does not match type of the input which is'):
      tpu_embedding_v2.cpu_embedding_lookup(
          features,
          weights=weights,
          tables=mid_level.embedding_tables,
          feature_config=self.feature_config)

  def _get_ragged_tensors(self, dtype=dtypes.int32):
    feature0 = ragged_tensor.RaggedTensor.from_row_lengths(
        values=constant_op.constant(self.feature_watched_values, dtype=dtype),
        row_lengths=self.feature_watched_row_lengths)
    feature1 = ragged_tensor.RaggedTensor.from_row_lengths(
        values=constant_op.constant(self.feature_favorited_values, dtype=dtype),
        row_lengths=self.feature_favorited_row_lengths)
    feature2 = ragged_tensor.RaggedTensor.from_row_lengths(
        values=constant_op.constant(self.feature_friends_values, dtype=dtype),
        row_lengths=self.feature_friends_row_lengths)
    return (feature0, feature1, feature2)

  def test_cpu_ragged_lookup_with_weights(self):
    mid_level = self._create_mid_level()
    features = self._get_ragged_tensors()
    weights = self._get_ragged_tensors(dtype=dtypes.float32)
    results = tpu_embedding_v2.cpu_embedding_lookup(
        features,
        weights=weights,
        tables=mid_level.embedding_tables,
        feature_config=self.feature_config)
    weighted_sum = []
    for feature, weight, config in zip(nest.flatten(features),
                                       nest.flatten(weights),
                                       self.feature_config):
      table = mid_level.embedding_tables[config.table].numpy()
      # Expand dims here needed to broadcast this multiplication properly.
      weight = np.expand_dims(weight.values.numpy(), axis=1)
      all_lookups = table[feature.values.numpy()] * weight
      row_starts = feature.row_starts().numpy()
      weighted_sum.append(np.add.reduceat(all_lookups, row_starts))
      if config.table.combiner == 'mean':
        weighted_sum[-1] /= np.add.reduceat(weight, row_starts)
    self.assertAllClose(results, nest.pack_sequence_as(results,
                                                       weighted_sum))

  def test_cpu_invalid_structure_for_features(self):
    mid_level = self._create_mid_level()
    # Remove one element of the tuple, self.feature_config has 3 so we need to
    # pass 3.
    features = tuple(self._get_sparse_tensors()[:2])
    with self.assertRaises(ValueError):
      tpu_embedding_v2.cpu_embedding_lookup(
          features,
          weights=None,
          tables=mid_level.embedding_tables,
          feature_config=self.feature_config)

  def test_cpu_invalid_structure_for_weights(self):
    mid_level = self._create_mid_level()
    features = self._get_sparse_tensors()
    # Remove one element of the tuple, self.feature_config has 3 so we need to
    # pass 3 (or None).
    weights = tuple(self._get_dense_tensors(dtype=dtypes.float32)[:2])
    with self.assertRaises(ValueError):
      tpu_embedding_v2.cpu_embedding_lookup(
          features,
          weights=weights,
          tables=mid_level.embedding_tables,
          feature_config=self.feature_config)

  def test_cpu_sequence_lookup(self):
    feature_config = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched', max_sequence_length=2),)
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    mid_level = tpu_embedding_v2.TPUEmbedding(
        feature_config=feature_config,
        batch_size=self.batch_size,
        optimizer=optimizer)
    features = tuple(self._get_sparse_tensors()[:1])
    with self.assertRaisesRegex(
        ValueError, 'Sequence features unsupported at this time.'):
      tpu_embedding_v2.cpu_embedding_lookup(
          features,
          weights=None,
          tables=mid_level.embedding_tables,
          feature_config=feature_config)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
