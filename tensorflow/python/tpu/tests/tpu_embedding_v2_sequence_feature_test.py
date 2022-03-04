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
"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

  @parameterized.parameters([True, False])
  def test_sequence_feature(self, is_sparse):
    seq_length = 3
    # Set the max_seq_length in feature config
    for feature in self.feature_config:
      feature.max_sequence_length = seq_length
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    if is_sparse:
      dataset = self._create_sparse_dataset(strategy)
    else:
      dataset = self._create_ragged_dataset(strategy)
    feature_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_fn():

      def step():
        return mid_level_api.dequeue()

      mid_level_api.enqueue(next(feature_iter), training=False)
      return strategy.run(step)

    output = test_fn()
    self.assertEqual(
        self._get_replica_numpy(output[0], strategy, 0).shape, (2, 3, 4))
    self.assertEqual(
        self._get_replica_numpy(output[1], strategy, 0).shape, (2, 3, 4))
    self.assertEqual(
        self._get_replica_numpy(output[2], strategy, 0).shape, (2, 3, 2))

  @parameterized.parameters([True, False])
  def test_sequence_feature_with_build(self, is_updated_shape):
    seq_length = 3
    # Set the max_seq_length in feature config
    for feature in self.feature_config:
      feature.max_sequence_length = seq_length
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    dataset = self._create_sparse_dataset(strategy)
    feature_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))
    if is_updated_shape:
      mid_level_api.build([
          TensorShape([self.batch_size, seq_length, 2]),
          TensorShape([self.batch_size, seq_length, 2]),
          TensorShape([self.batch_size, seq_length, 3])
      ])
    else:
      mid_level_api.build([
          TensorShape([self.batch_size, 2]),
          TensorShape([self.batch_size, 2]),
          TensorShape([self.batch_size, 3])
      ])

    @def_function.function
    def test_fn():

      def step():
        return mid_level_api.dequeue()

      mid_level_api.enqueue(next(feature_iter), training=False)
      return strategy.run(step)

    output = test_fn()
    self.assertEqual(
        self._get_replica_numpy(output[0], strategy, 0).shape, (2, 3, 4))
    self.assertEqual(
        self._get_replica_numpy(output[1], strategy, 0).shape, (2, 3, 4))
    self.assertEqual(
        self._get_replica_numpy(output[2], strategy, 0).shape, (2, 3, 2))

if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
