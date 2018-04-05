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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


class SequenceDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_skip_dataset(self, count):
    components = (np.arange(10),)
    return dataset_ops.Dataset.from_tensor_slices(components).skip(count)

  def testSkipFewerThanInputs(self):
    count = 4
    num_outputs = 10 - count
    self.run_core_tests(lambda: self._build_skip_dataset(count),
                        lambda: self._build_skip_dataset(count + 2),
                        num_outputs)

  def testSkipVarious(self):
    # Skip more than inputs
    self.run_core_tests(lambda: self._build_skip_dataset(20), None, 0)
    # Skip exactly the input size
    self.run_core_tests(lambda: self._build_skip_dataset(10), None, 0)
    self.run_core_tests(lambda: self._build_skip_dataset(-1), None, 0)
    # Skip nothing
    self.run_core_tests(lambda: self._build_skip_dataset(0), None, 10)

  def testInvalidSkip(self):
    with self.assertRaisesRegexp(
        ValueError, 'Shape must be rank 0 but is rank 1'):
      self.run_core_tests(lambda: self._build_skip_dataset([1, 2]), None, 0)

  def _build_take_dataset(self, count):
    components = (np.arange(10),)
    return dataset_ops.Dataset.from_tensor_slices(components).take(count)

  def testTakeFewerThanInputs(self):
    count = 4
    self.run_core_tests(
        lambda: self._build_take_dataset(count),
        lambda: self._build_take_dataset(count + 2),
        count,
    )

  def testTakeVarious(self):
    # Take more than inputs
    self.run_core_tests(lambda: self._build_take_dataset(20), None, 10)
    # Take exactly the input size
    self.run_core_tests(lambda: self._build_take_dataset(10), None, 10)
    # Take all
    self.run_core_tests(lambda: self._build_take_dataset(-1), None, 10)
    # Take nothing
    self.run_core_tests(lambda: self._build_take_dataset(0), None, 0)

  def testInvalidTake(self):
    with self.assertRaisesRegexp(
        ValueError, 'Shape must be rank 0 but is rank 1'):
      self.run_core_tests(lambda: self._build_take_dataset([1, 2]), None, 0)

  def _build_repeat_dataset(self, count, take_count=3):
    components = (np.arange(10),)
    return dataset_ops.Dataset.from_tensor_slices(components).take(
        take_count).repeat(count)

  def testFiniteRepeat(self):
    count = 10
    self.run_core_tests(lambda: self._build_repeat_dataset(count),
                        lambda: self._build_repeat_dataset(count + 2),
                        3 * count)

  def testEmptyRepeat(self):
    self.run_core_tests(lambda: self._build_repeat_dataset(0), None, 0)

  def testInfiniteRepeat(self):
    self.verify_unused_iterator(
        lambda: self._build_repeat_dataset(-1), 10, verify_exhausted=False)
    self.verify_init_before_restore(
        lambda: self._build_repeat_dataset(-1), 10, verify_exhausted=False)
    self.verify_multiple_breaks(
        lambda: self._build_repeat_dataset(-1), 20, verify_exhausted=False)
    self.verify_reset_restored_iterator(
        lambda: self._build_repeat_dataset(-1), 20, verify_exhausted=False)
    self.verify_restore_in_modified_graph(
        lambda: self._build_repeat_dataset(-1),
        lambda: self._build_repeat_dataset(2),
        20,
        verify_exhausted=False)
    # Test repeat empty dataset
    self.run_core_tests(lambda: self._build_repeat_dataset(-1, 0), None, 0)

  def testInvalidRepeat(self):
    with self.assertRaisesRegexp(
        ValueError, 'Shape must be rank 0 but is rank 1'):
      self.run_core_tests(lambda: self._build_repeat_dataset([1, 2], 0),
                          None, 0)


if __name__ == "__main__":
  test.main()
