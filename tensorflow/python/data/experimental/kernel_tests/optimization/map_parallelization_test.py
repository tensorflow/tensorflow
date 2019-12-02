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
"""Tests for the `MapParallelization` optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class MapParallelizationTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _testMapParallelization(self, function, should_optimize):
    next_nodes = ["ParallelMap"] if should_optimize else ["Map"]
    dataset = dataset_ops.Dataset.range(5).apply(
        testing.assert_next(next_nodes)).map(function)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset, expected_output=[function(x) for x in range(5)])

  @combinations.generate(test_base.default_test_combinations())
  def testIdentity(self):
    self._testMapParallelization(lambda x: x, should_optimize=True)

  @combinations.generate(test_base.default_test_combinations())
  def testIncrement(self):
    self._testMapParallelization(lambda x: x + 1, should_optimize=True)

  @combinations.generate(test_base.default_test_combinations())
  def testAssert(self):

    def assert_greater(x):
      assert_op = control_flow_ops.Assert(math_ops.greater(x, -1), [x])
      with ops.control_dependencies([assert_op]):
        return x

    self._testMapParallelization(assert_greater, should_optimize=True)

  @combinations.generate(test_base.default_test_combinations())
  def testCapturedConstant(self):
    captured_t = constant_op.constant(42, dtype=dtypes.int64)
    def fn(x):
      return x + captured_t
    dataset = dataset_ops.Dataset.range(5).apply(
        testing.assert_next(["ParallelMap"])).map(fn)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset, expected_output=[x + 42 for x in range(5)])

  @combinations.generate(test_base.default_test_combinations())
  def testCapturedVariable(self):
    captured_t = variables.Variable(42, dtype=dtypes.int64)
    def fn(x):
      return x + captured_t
    dataset = dataset_ops.Dataset.range(5).apply(
        testing.assert_next(["Map"])).map(fn)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    self.evaluate(variables.global_variables_initializer())
    self.assertDatasetProduces(
        dataset,
        expected_output=[x + 42 for x in range(5)],
        requires_initialization=True)


if __name__ == "__main__":
  test.main()
