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
"""Tests for the sequence datasets serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class SkipDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_skip_dataset(self, count):
    components = (np.arange(10),)
    return dataset_ops.Dataset.from_tensor_slices(components).skip(count)

  @combinations.generate(test_base.default_test_combinations())
  def testSkipFewerThanInputs(self):
    count = 4
    num_outputs = 10 - count
    self.run_core_tests(lambda: self._build_skip_dataset(count), num_outputs)

  @combinations.generate(test_base.default_test_combinations())
  def testSkipVarious(self):
    # Skip more than inputs
    self.run_core_tests(lambda: self._build_skip_dataset(20), 0)
    # Skip exactly the input size
    self.run_core_tests(lambda: self._build_skip_dataset(10), 0)
    self.run_core_tests(lambda: self._build_skip_dataset(-1), 0)
    # Skip nothing
    self.run_core_tests(lambda: self._build_skip_dataset(0), 10)

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidSkip(self):
    with self.assertRaisesRegex(ValueError,
                                'Shape must be rank 0 but is rank 1'):
      self.run_core_tests(lambda: self._build_skip_dataset([1, 2]), 0)


class TakeDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_take_dataset(self, count):
    components = (np.arange(10),)
    return dataset_ops.Dataset.from_tensor_slices(components).take(count)

  @combinations.generate(test_base.default_test_combinations())
  def testTakeFewerThanInputs(self):
    count = 4
    self.run_core_tests(lambda: self._build_take_dataset(count), count)

  @combinations.generate(test_base.default_test_combinations())
  def testTakeVarious(self):
    # Take more than inputs
    self.run_core_tests(lambda: self._build_take_dataset(20), 10)
    # Take exactly the input size
    self.run_core_tests(lambda: self._build_take_dataset(10), 10)
    # Take all
    self.run_core_tests(lambda: self._build_take_dataset(-1), 10)
    # Take nothing
    self.run_core_tests(lambda: self._build_take_dataset(0), 0)

  def testInvalidTake(self):
    with self.assertRaisesRegex(ValueError,
                                'Shape must be rank 0 but is rank 1'):
      self.run_core_tests(lambda: self._build_take_dataset([1, 2]), 0)


class RepeatDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_repeat_dataset(self, count, take_count=3):
    components = (np.arange(10),)
    return dataset_ops.Dataset.from_tensor_slices(components).take(
        take_count).repeat(count)

  @combinations.generate(test_base.default_test_combinations())
  def testFiniteRepeat(self):
    count = 10
    self.run_core_tests(lambda: self._build_repeat_dataset(count), 3 * count)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyRepeat(self):
    self.run_core_tests(lambda: self._build_repeat_dataset(0), 0)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteRepeat(self):
    self.verify_unused_iterator(
        lambda: self._build_repeat_dataset(-1), 10, verify_exhausted=False)
    self.verify_multiple_breaks(
        lambda: self._build_repeat_dataset(-1), 20, verify_exhausted=False)
    self.verify_reset_restored_iterator(
        lambda: self._build_repeat_dataset(-1), 20, verify_exhausted=False)

    # Test repeat empty dataset
    self.run_core_tests(lambda: self._build_repeat_dataset(-1, 0), 0)

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidRepeat(self):
    with self.assertRaisesRegex(ValueError,
                                'Shape must be rank 0 but is rank 1'):
      self.run_core_tests(lambda: self._build_repeat_dataset([1, 2], 0), 0)


if __name__ == '__main__':
  test.main()
