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
"""Tests for `tf.data.Dataset.repeat()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class RepeatTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(count=[0, 3, 7])))
  def testFiniteRepeat(self, count):
    """Test a dataset that repeats its input multiple times."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    dataset = dataset_ops.Dataset.from_tensors(components).repeat(count)
    self.assertEqual(
        [c.shape for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    self.assertDatasetProduces(dataset, [components] * count)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteRepeat(self):
    # NOTE(mrry): There's not a good way to test that the sequence is infinite.
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    dataset = dataset_ops.Dataset.from_tensors(components).repeat(-1)
    self.assertEqual(
        [c.shape for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    get_next = self.getNext(dataset)
    for _ in range(17):
      results = self.evaluate(get_next())
      for component, result_component in zip(components, results):
        self.assertAllEqual(component, result_component)

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatRepeat(self):
    """Test the composition of repeat datasets."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    inner_count, outer_count = 7, 14

    dataset = dataset_ops.Dataset.from_tensors(components).repeat(
        inner_count).repeat(outer_count)
    self.assertEqual(
        [c.shape for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    self.assertDatasetProduces(dataset,
                               [components] * (inner_count * outer_count))


if __name__ == "__main__":
  test.main()
