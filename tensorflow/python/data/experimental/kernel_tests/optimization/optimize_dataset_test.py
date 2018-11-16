# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the private `_OptimizeDataset` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental.ops import threadpool
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class OptimizeDatasetTest(test_base.DatasetTestBase):

  def testOptimizationStatefulFunction(self):
    dataset = dataset_ops.Dataset.range(10).map(
        lambda _: random_ops.random_uniform([])).batch(10)
    dataset = dataset_ops._OptimizeDataset(dataset, [])
    get_next = self.getNext(dataset)
    self.evaluate(get_next())

  # TODO(b/117581999): Add eager coverage for the following tests.
  def testSkipEagerOptimizationLargeInputFromTensor(self):
    input_t = array_ops.placeholder(dtypes.int32, (None, None, None))
    dataset = dataset_ops.Dataset.from_tensors(input_t)
    dataset = dataset_ops._OptimizeDataset(dataset, [])
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op, {input_t: np.ones([512, 1024, 1025], np.int32)})
      sess.run(get_next)

  # TODO(b/117581999): Add eager coverage for the following tests.
  def testSkipEagerOptimizationLargeInputFromTensorSlices(self):
    input_t = array_ops.placeholder(dtypes.int32, (None, None, None, None))
    dataset = dataset_ops.Dataset.from_tensor_slices(input_t)
    dataset = dataset_ops._OptimizeDataset(dataset, [])
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op, {input_t: np.ones([1, 512, 1024, 1025], np.int32)})
      sess.run(get_next)

  def testOptimizationNestedDataset(self):

    def flat_map_fn(_):
      dataset = dataset_ops.Dataset.from_tensors(0)
      dataset = dataset.apply(optimization.assert_next(["MemoryCacheImpl"]))
      dataset = dataset.skip(0)  # Should be removed by noop elimination
      dataset = dataset.cache()
      return dataset

    dataset = dataset_ops.Dataset.range(1)
    dataset = dataset.flat_map(flat_map_fn)
    dataset = dataset_ops._OptimizeDataset(dataset, ["noop_elimination"])
    self.assertDatasetProduces(dataset, expected_output=[0])

  def testOptimizationThreadPoolDataset(self):
    dataset = dataset_ops.Dataset.range(10).batch(10)

    dataset = threadpool.override_threadpool(
        dataset,
        threadpool.PrivateThreadPool(
            2, display_name="private_thread_pool_%d" % 2))

    dataset = dataset_ops._OptimizeDataset(dataset, [])
    self.assertDatasetProduces(
        dataset,
        expected_output=[list(range(10))],
        requires_initialization=True)

  def testOptimizationNonSerializable(self):
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.apply(optimization.assert_next(["FiniteSkip"]))
    dataset = dataset.skip(0)  # Should not be removed by noop elimination
    dataset = dataset.apply(optimization.non_serializable())
    dataset = dataset.apply(optimization.assert_next(["MemoryCacheImpl"]))
    dataset = dataset.skip(0)  # Should be removed by noop elimination
    dataset = dataset.cache()
    dataset = dataset_ops._OptimizeDataset(dataset, ["noop_elimination"])
    self.assertDatasetProduces(dataset, expected_output=[0])

  def testOptimizationNonSerializableAsDirectInput(self):
    """Tests that non-serializable dataset can be OptimizeDataset's input.
    """
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.apply(optimization.non_serializable())
    dataset = dataset_ops._OptimizeDataset(dataset, ["noop_elimination"])
    self.assertDatasetProduces(dataset, expected_output=[0])


if __name__ == "__main__":
  test.main()
