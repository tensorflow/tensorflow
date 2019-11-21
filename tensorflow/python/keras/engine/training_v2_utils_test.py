# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.keras.engine.training_v2_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import mock
import numpy as np


from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import training_v2_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


class AggregatePredictResultsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(AggregatePredictResultsTest, self).setUp()
    strategy_combinations.set_virtual_cpus_to_at_least(3)
    self.num_replica = 3
    self.batch_size = 16
    self.dense_shape = (2, 3)
    self.total_sample = 2 * self.batch_size

    mock_model = collections.namedtuple('Model', ['outputs'])
    self.mock_model = mock_model([1])

    strategy = mirrored_strategy.MirroredStrategy(
        ['/cpu:0', '/cpu:1', '/cpu:2'])

    execution_function = lambda *inp: inp
    @def_function.function
    def predict_loop(batch):
      batch_result = strategy.experimental_run_v2(execution_function, batch)
      batch_result = dist_utils.unwrap_output_dict(
          strategy, batch_result, ModeKeys.PREDICT)
      # swap the order of replica 1 and 2, to mimic random order.
      batch_result[2], batch_result[1] = batch_result[1], batch_result[2]
      batch_result[5], batch_result[4] = batch_result[4], batch_result[5]
      return batch_result

    self.strategy = strategy
    self.predict_loop = predict_loop

  @combinations.generate(combinations.combine(tf_api_version=[1, 2],
                                              mode='eager'))
  def test_aggregate_predict_results_dense(self):
    dataset = dataset_ops.Dataset.range(self.total_sample)
    def dense_map_fn(i):
      # Mimic what we do for adding batch index
      return i, array_ops.fill(self.dense_shape, i)
    dense_dataset = dataset.map(dense_map_fn).batch(self.batch_size)
    distributed_data = self.strategy.experimental_distribute_dataset(
        dense_dataset)

    start = 0
    for batch in distributed_data:
      with mock.patch.object(training_v2_utils,
                             '_should_add_batch_index_to_element',
                             fake_should_add_batch_index_to_element):
        batch_result = self.predict_loop(batch)
        final_result = training_v2_utils._aggregate_predict_results(
            self.strategy, batch_result, self.mock_model)

        # Make sure the dense result is in a sorted order.
        expected_result = np.arange(
            start=start, stop=start+self.batch_size).reshape((-1, 1))
        expected_result = np.tile(expected_result, 6).reshape(
            (-1,) + self.dense_shape)
        self.assertAllClose(final_result[0], expected_result)
        start += self.batch_size

  @combinations.generate(combinations.combine(tf_api_version=[1, 2],
                                              mode='eager'))
  def test_aggregate_predict_results_sparse(self):
    dataset = dataset_ops.Dataset.range(self.total_sample)
    def sparse_map_fn(i):
      return i, sparse_tensor.SparseTensor(
          indices=[(0, 0)],
          values=[i],
          dense_shape=self.dense_shape)
    sparse_dataset = dataset.map(sparse_map_fn).batch(self.batch_size)
    distributed_data = self.strategy.experimental_distribute_dataset(
        sparse_dataset)

    start = 0
    for batch in distributed_data:
      with mock.patch.object(training_v2_utils,
                             '_should_add_batch_index_to_element',
                             fake_should_add_batch_index_to_element):
        batch_result = self.predict_loop(batch)
        final_result = training_v2_utils._aggregate_predict_results(
            self.strategy, batch_result, self.mock_model)

        # Make sure the dense result is in a sorted order.
        expected_values = np.arange(start=start, stop=start+self.batch_size)
        self.assertAllClose(final_result[0].values, expected_values)
        start += self.batch_size

  @combinations.generate(combinations.combine(tf_api_version=[1, 2],
                                              mode='eager'))
  def test_aggregate_predict_results_ragged(self):
    dataset = dataset_ops.Dataset.range(self.total_sample)
    def ragged_map_fn(i):
      return i, ragged_factory_ops.constant([[0], [], []], dtype=np.int64) + i
    ragged_dataset = dataset.map(ragged_map_fn).batch(self.batch_size)
    distributed_data = self.strategy.experimental_distribute_dataset(
        ragged_dataset)

    start = 0
    for batch in distributed_data:
      with mock.patch.object(training_v2_utils,
                             '_should_add_batch_index_to_element',
                             fake_should_add_batch_index_to_element):
        batch_result = self.predict_loop(batch)
        final_result = training_v2_utils._aggregate_predict_results(
            self.strategy, batch_result, self.mock_model)

        # Make sure the dense result is in a sorted order.
        expected_values = np.arange(start=start, stop=start+self.batch_size)
        self.assertAllClose(final_result[0].flat_values, expected_values)
        start += self.batch_size


def fake_should_add_batch_index_to_element(strategy, mode):
  # Ignore the strategy instance check since we were using the MirroredStrategy
  # for testing.
  del strategy
  return mode == ModeKeys.PREDICT


if __name__ == '__main__':
  test.main()
