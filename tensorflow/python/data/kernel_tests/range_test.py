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
"""Tests for `tf.data.Dataset.range()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class RangeTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testStop(self):
    dataset = dataset_ops.Dataset.range(5)
    self.assertDatasetProduces(dataset, expected_output=range(5))

  @combinations.generate(test_base.default_test_combinations())
  def testStartStop(self):
    start, stop = 2, 5
    dataset = dataset_ops.Dataset.range(start, stop)
    self.assertDatasetProduces(dataset, expected_output=range(2, 5))

  @combinations.generate(test_base.default_test_combinations())
  def testStartStopStep(self):
    start, stop, step = 2, 10, 2
    dataset = dataset_ops.Dataset.range(start, stop, step)
    self.assertDatasetProduces(dataset, expected_output=range(2, 10, 2))

  @combinations.generate(test_base.default_test_combinations())
  def testZeroStep(self):
    start, stop, step = 2, 10, 0
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(start, stop, step)
      self.evaluate(dataset._variant_tensor)

  @combinations.generate(test_base.default_test_combinations())
  def testNegativeStep(self):
    start, stop, step = 2, 10, -1
    dataset = dataset_ops.Dataset.range(start, stop, step)
    self.assertDatasetProduces(dataset, expected_output=range(2, 10, -1))

  @combinations.generate(test_base.default_test_combinations())
  def testStopLessThanStart(self):
    start, stop = 10, 2
    dataset = dataset_ops.Dataset.range(start, stop)
    self.assertDatasetProduces(dataset, expected_output=range(10, 2))

  @combinations.generate(test_base.default_test_combinations())
  def testStopLessThanStartWithPositiveStep(self):
    start, stop, step = 10, 2, 2
    dataset = dataset_ops.Dataset.range(start, stop, step)
    self.assertDatasetProduces(dataset, expected_output=range(10, 2, 2))

  @combinations.generate(test_base.default_test_combinations())
  def testStopLessThanStartWithNegativeStep(self):
    start, stop, step = 10, 2, -1
    dataset = dataset_ops.Dataset.range(start, stop, step)
    self.assertDatasetProduces(dataset, expected_output=range(10, 2, -1))


if __name__ == "__main__":
  test.main()
