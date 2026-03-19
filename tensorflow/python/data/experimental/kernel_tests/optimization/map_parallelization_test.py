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
import functools

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _test_combinations():
  def assert_greater(x):
    assert_op = control_flow_assert.Assert(math_ops.greater(x, -1), [x])
    with ops.control_dependencies([assert_op]):
      return x

  cases = [
      ("Identity", lambda x: x, True),
      ("Increment", lambda x: x + 1, True),
      ("AssertGreater", assert_greater, True),
  ]

  def reduce_fn(x, y):
    name, function, should_optimize = y
    return x + combinations.combine(
        function=combinations.NamedObject(name, function),
        should_optimize=should_optimize)

  return functools.reduce(reduce_fn, cases, [])


class MapParallelizationTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_combinations()))
  def testMapParallelization(self, function, should_optimize):
    next_nodes = ["ParallelMap"] if should_optimize else ["Map"]
    dataset = dataset_ops.Dataset.range(5).apply(
        testing.assert_next(next_nodes)).map(function)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset, expected_output=[function(x) for x in range(5)])

  @combinations.generate(test_base.default_test_combinations())
  def testNoMapParallelizationWhenSynchronous(self):
    dataset = (
        dataset_ops.Dataset.range(5)
        .apply(testing.assert_next(["Map"]))
        .map(lambda x: x + 1, synchronous=True)
    )
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset, expected_output=[x + 1 for x in range(5)]
    )

  @combinations.generate(test_base.default_test_combinations())
  def testCapturedConstant(self):
    captured_t = constant_op.constant(42, dtype=dtypes.int64)
    def fn(x):
      return x + captured_t
    dataset = dataset_ops.Dataset.range(5).apply(
        testing.assert_next(["ParallelMap"])).map(fn)
    options = options_lib.Options()
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
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    self.evaluate(variables.global_variables_initializer())
    self.assertDatasetProduces(
        dataset,
        expected_output=[x + 42 for x in range(5)],
        requires_initialization=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(apply_autotune=[None, True, False])))
  def testAutotuneOption(self, apply_autotune):
    next_nodes = ["ParallelMap"] if (apply_autotune is not False) else ["Map"]  # pylint: disable=g-bool-id-comparison
    dataset = dataset_ops.Dataset.range(4).apply(
        testing.assert_next(next_nodes)).map(lambda x: x + 2)

    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_parallelization = True
    if apply_autotune is not None:
      options.autotune.enabled = apply_autotune
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[2, 3, 4, 5])

  @combinations.generate(test_base.default_test_combinations())
  def testNoParallelizationInsideInterleave(self):

    def func(i):
      ds = dataset_ops.Dataset.range(i).apply(testing.assert_next(
          ["Map"])).map(lambda x: x + 1)
      return ds

    dataset = dataset_ops.Dataset.range(1, 4).interleave(
        map_func=func, cycle_length=2, block_length=2)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)

    self.assertDatasetProduces(dataset, expected_output=[1, 1, 2, 1, 2, 3])

  @combinations.generate(test_base.default_test_combinations())
  def testNoParallelizationInsideFlatMap(self):

    def func(i):
      ds = dataset_ops.Dataset.range(i).apply(testing.assert_next(
          ["Map"])).map(lambda x: x + 1)
      return ds

    dataset = dataset_ops.Dataset.range(1, 4).flat_map(map_func=func)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)

    self.assertDatasetProduces(dataset, expected_output=[1, 1, 2, 1, 2, 3])


if __name__ == "__main__":
  test.main()
