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
"""Tests for the ChooseFastestBranchDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ChooseFastestBranchDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def testCore(self):

    def build_ds(size):
      dataset = dataset_ops.Dataset.range(size)

      def branch_0(dataset):
        return dataset.map(lambda x: x).batch(10)

      def branch_1(dataset):
        return dataset.batch(10).map(lambda x: x)

      return optimization._ChooseFastestBranchDataset(  # pylint: disable=protected-access
          dataset, [branch_0, branch_1],
          ratio_numerator=10)

    for size in [100, 1000]:
      self.run_core_tests(lambda: build_ds(size), None, size // 10)  # pylint: disable=cell-var-from-loop

  def testWithCapture(self):

    def build_ds():
      dataset = dataset_ops.Dataset.range(10)
      const_64 = constant_op.constant(1, dtypes.int64)
      const_32 = constant_op.constant(1, dtypes.int32)

      def branch_0(dataset):
        return dataset.map(lambda x: x + const_64)

      def branch_1(dataset):
        return dataset.map(lambda x: x + math_ops.cast(const_32, dtypes.int64))

      return optimization._ChooseFastestBranchDataset(
          dataset, [branch_0, branch_1], num_elements_per_branch=3)

    self.run_core_tests(build_ds, None, 10)

  def testWithPrefetch(self):

    def build_ds():
      dataset = dataset_ops.Dataset.range(10)
      const_64 = constant_op.constant(1, dtypes.int64)
      const_32 = constant_op.constant(1, dtypes.int32)

      def branch_0(dataset):
        return dataset.map(lambda x: x + const_64)

      def branch_1(dataset):
        return dataset.map(lambda x: x + math_ops.cast(const_32, dtypes.int64))

      return optimization._ChooseFastestBranchDataset(
          dataset, [branch_0, branch_1], num_elements_per_branch=3)

    self.run_core_tests(build_ds, None, 10)

  def testWithMoreOutputThanInput(self):

    def build_ds():
      dataset = dataset_ops.Dataset.from_tensors(0).repeat(1000).batch(100)

      def branch(dataset):
        return dataset.apply(batching.unbatch())

      return optimization._ChooseFastestBranchDataset(
          dataset, [branch, branch],
          ratio_denominator=10,
          num_elements_per_branch=100)

    self.run_core_tests(build_ds, None, 1000)


if __name__ == "__main__":
  test.main()
