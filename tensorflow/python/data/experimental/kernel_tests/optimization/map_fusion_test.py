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

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def _map_fusion_test_cases():
  """Generates test cases for the MapFusion optimization."""

  identity = lambda x: x
  increment = lambda x: x + 1

  def increment_and_square(x):
    y = x + 1
    return y * y

  functions = [identity, increment, increment_and_square]
  tests = []
  for i, fun1 in enumerate(functions):
    for j, fun2 in enumerate(functions):
      tests.append((
          "Test{}{}".format(i, j),
          [fun1, fun2],
      ))
      for k, fun3 in enumerate(functions):
        tests.append((
            "Test{}{}{}".format(i, j, k),
            [fun1, fun2, fun3],
        ))

  swap = lambda x, n: (n, x)
  tests.append((
      "Swap1",
      [lambda x: (x, 42), swap],
  ))
  tests.append((
      "Swap2",
      [lambda x: (x, 42), swap, swap],
  ))
  return tuple(tests)


@test_util.run_all_in_graph_and_eager_modes
class MapFusionTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(*_map_fusion_test_cases())
  def testMapFusion(self, functions):
    dataset = dataset_ops.Dataset.range(5).apply(
        optimization.assert_next(["Map", "MemoryCacheImpl"]))
    for function in functions:
      dataset = dataset.map(function)

    dataset = dataset.cache()
    options = dataset_ops.Options()
    options.experimental_map_fusion = True
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


if __name__ == "__main__":
  test.main()
