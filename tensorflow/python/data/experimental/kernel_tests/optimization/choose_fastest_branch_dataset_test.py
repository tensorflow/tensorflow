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
"""Tests for `tf.data.experimental._ChooseFastestBranchDataset`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ChooseFastestBranchDatasetTest(test_base.DatasetTestBase,
                                     parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testSimple(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2, 3, 4])

    def branch(dataset):
      return dataset.map(lambda x: x)

    choose_fastest = optimization._ChooseFastestBranchDataset(
        dataset, [branch, branch])

    self.assertDatasetProduces(
        choose_fastest,
        expected_output=[0, 1, 2, 3, 4],
        expected_shapes=dataset_ops.get_legacy_output_shapes(dataset))

  @combinations.generate(test_base.default_test_combinations())
  def testCaptureSimple(self):
    dataset = dataset_ops.Dataset.range(10)

    const_64 = constant_op.constant(1, dtypes.int64)
    const_32 = constant_op.constant(1, dtypes.int32)

    def branch_0(dataset):
      return dataset.map(lambda x: x + const_64)

    def branch_1(dataset):
      return dataset.map(lambda x: x + math_ops.cast(const_32, dtypes.int64))

    choose_fastest = optimization._ChooseFastestBranchDataset(
        dataset, [branch_0, branch_1])

    self.assertDatasetProduces(
        choose_fastest, expected_output=list(range(1, 11)))

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentFunctions(self):
    dataset = dataset_ops.Dataset.range(100)

    def branch_0(dataset):
      return dataset.map(lambda x: x).batch(10)

    def branch_1(dataset):
      return dataset.batch(10).map(lambda x: x)

    choose_fastest = optimization._ChooseFastestBranchDataset(
        dataset, [branch_0, branch_1], ratio_numerator=10)

    self.assertDatasetProduces(
        choose_fastest,
        expected_output=[list(range(10 * x, 10 * x + 10)) for x in range(10)])

  @combinations.generate(test_base.default_test_combinations())
  def testWithRepeatBeforeAndAfter(self):
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)

    def branch_0(dataset):
      return dataset.map(lambda x: x).batch(10)

    def branch_1(dataset):
      return dataset.batch(10).map(lambda x: x)

    choose_fastest = optimization._ChooseFastestBranchDataset(
        dataset, [branch_0, branch_1], ratio_numerator=10)
    choose_fastest = choose_fastest.repeat(10)

    self.assertDatasetProduces(
        choose_fastest, expected_output=[[0] * 10 for _ in range(10)])

  @combinations.generate(test_base.default_test_combinations())
  def testWithPrefetch(self):
    """Should maintain ordering even if the branches do prefetching."""
    dataset = dataset_ops.Dataset.range(100)

    def branch_0(dataset):
      return dataset.prefetch(1)

    def branch_1(dataset):
      return dataset.prefetch(2)

    choose_fastest = optimization._ChooseFastestBranchDataset(
        dataset, [branch_0, branch_1])

    self.assertDatasetProduces(choose_fastest, expected_output=list(range(100)))

  @combinations.generate(test_base.default_test_combinations())
  def testWithMoreOutputThanInput(self):

    dataset = dataset_ops.Dataset.from_tensors(0).repeat(1000).batch(100)

    def branch(dataset):
      return dataset.unbatch()

    choose_fastest = optimization._ChooseFastestBranchDataset(
        dataset, [branch, branch],
        ratio_denominator=100,
        num_elements_per_branch=100)

    self.assertDatasetProduces(choose_fastest, expected_output=[0] * 1000)

  @combinations.generate(test_base.default_test_combinations())
  def testWithBadNumElements(self):

    dataset = dataset_ops.Dataset.from_tensors(0).repeat(1000).batch(100)

    def branch(dataset):
      return dataset.unbatch()

    def make_dataset():
      return optimization._ChooseFastestBranchDataset(
          dataset, [branch, branch],
          ratio_denominator=100,
          num_elements_per_branch=10)

    expected_error_msg = ("`num_elements_per_branch` must be divisible by "
                          "`ratio_denominator`")
    if context.executing_eagerly():
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   expected_error_msg):
        make_dataset()
    else:
      choose_fastest = make_dataset()
      self.assertDatasetProduces(
          choose_fastest,
          expected_error=(errors.InvalidArgumentError, expected_error_msg))

  @combinations.generate(test_base.default_test_combinations())
  def testErrorWithRepeat(self):
    dataset = dataset_ops.Dataset.from_tensors(0)

    def branch(dataset):
      return dataset.repeat(10)

    choose_fastest = optimization._ChooseFastestBranchDataset(
        dataset, [branch, branch],
        ratio_denominator=10,
        num_elements_per_branch=10)
    self.assertDatasetProduces(
        choose_fastest,
        expected_error=(
            errors.InvalidArgumentError,
            "Cannot create more than one WrapperIterator per WrapperDataset."),
        expected_error_iter=2)


if __name__ == "__main__":
  test.main()
