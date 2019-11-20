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
"""Tests for the ParallelInterleaveDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class ParallelInterleaveDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def setUp(self):
    self.input_values = np.array([4, 5, 6], dtype=np.int64)
    self.num_repeats = 2
    self.num_outputs = np.sum(self.input_values) * 2

  def _build_ds(self, cycle_length, block_length, sloppy=False):
    return (dataset_ops.Dataset.from_tensor_slices(
        self.input_values).repeat(self.num_repeats).apply(
            interleave_ops.parallel_interleave(
                lambda x: dataset_ops.Dataset.range(10 * x, 11 * x),
                cycle_length, block_length, sloppy)))

  @combinations.generate(test_base.default_test_combinations())
  def testSerializationCore(self):
    # cycle_length > 1, block_length > 1
    cycle_length = 2
    block_length = 3
    self.run_core_tests(lambda: self._build_ds(cycle_length, block_length),
                        self.num_outputs)
    # cycle_length = 1
    cycle_length = 1
    block_length = 3
    self.run_core_tests(lambda: self._build_ds(cycle_length, block_length),
                        self.num_outputs)
    # block_length = 1
    cycle_length = 2
    block_length = 1
    self.run_core_tests(lambda: self._build_ds(cycle_length, block_length),
                        self.num_outputs)

  @combinations.generate(test_base.default_test_combinations())
  def testSerializationWithSloppy(self):
    break_points = self.gen_break_points(self.num_outputs, 10)
    expected_outputs = np.repeat(
        np.concatenate([np.arange(10 * x, 11 * x) for x in self.input_values]),
        self.num_repeats).tolist()

    def run_test(cycle_length, block_length):
      actual = self.gen_outputs(
          lambda: self._build_ds(cycle_length, block_length, True),
          break_points, self.num_outputs)
      self.assertSequenceEqual(sorted(actual), expected_outputs)

    # cycle_length > 1, block_length > 1
    run_test(2, 3)
    # cycle_length = 1
    run_test(1, 3)
    # block_length = 1
    run_test(2, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testSparseCore(self):

    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _interleave_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    def _build_dataset():
      return dataset_ops.Dataset.range(10).map(_map_fn).apply(
          interleave_ops.parallel_interleave(_interleave_fn, 1))

    self.run_core_tests(_build_dataset, 20)


if __name__ == '__main__':
  test.main()
