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
"""Tests for the `MapFusion` optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class MapFusionTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _testMapFusion(self, functions):
    dataset = dataset_ops.Dataset.range(5).apply(
        testing.assert_next(["Map", "MemoryCacheImpl"]))
    for function in functions:
      dataset = dataset.map(function)

    dataset = dataset.cache()
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_fusion = True
    dataset = dataset.with_options(options)
    expected_output = []
    for x in range(5):
      r = x
      for function in functions:
        if isinstance(r, tuple):
          r = function(*r)  # Pass tuple as multiple arguments.
        else:
          r = function(r)
      expected_output.append(r)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMapFusionScalar(self):
    identity = lambda x: x
    increment = lambda x: x + 1

    def increment_and_square(x):
      y = x + 1
      return y * y

    functions = [identity, increment, increment_and_square]

    for x in functions:
      for y in functions:
        self._testMapFusion([x, y])
        for z in functions:
          self._testMapFusion([x, y, z])

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndFilterFusionTuple(self):
    with_42 = lambda x: (x, 42)
    swap = lambda x, y: (y, x)

    self._testMapFusion([with_42, swap])
    self._testMapFusion([with_42, swap, swap])


if __name__ == "__main__":
  test.main()
