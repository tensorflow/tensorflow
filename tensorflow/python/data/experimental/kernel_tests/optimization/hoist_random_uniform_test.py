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
"""Tests for the `HoistRandomUniform` optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


def _hoist_random_uniform_test_cases():
  """Generates test cases for the HoistRandomUniform optimization."""

  plus_one = lambda x: x + 1

  def random(_):
    return random_ops.random_uniform([],
                                     minval=1,
                                     maxval=10,
                                     dtype=dtypes.float32,
                                     seed=42)

  def random_with_assert(x):
    y = random(x)
    assert_op = control_flow_ops.Assert(math_ops.greater_equal(y, 1), [y])
    with ops.control_dependencies([assert_op]):
      return y

  twice_random = lambda x: (random(x) + random(x)) / 2.

  tests = [("PlusOne", plus_one, False), ("RandomUniform", random, True),
           ("RandomWithAssert", random_with_assert, True),
           ("TwiceRandom", twice_random, False)]
  return tuple(tests)


@test_util.run_all_in_graph_and_eager_modes
class HoistRandomUniformTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _testDataset(self, dataset):
    previous_result = 0
    get_next = self.getNext(dataset)
    for _ in range(5):
      result = self.evaluate(get_next())
      self.assertLessEqual(1, result)
      self.assertLessEqual(result, 10)
      # This checks if the result is somehow random by checking if we are not
      # generating the same values.
      self.assertNotEqual(previous_result, result)
      previous_result = result
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @parameterized.named_parameters(*_hoist_random_uniform_test_cases())
  def testHoisting(self, function, will_optimize):
    dataset = dataset_ops.Dataset.range(5).apply(
        testing.assert_next(
            ["Zip[0]", "Map"] if will_optimize else ["Map"])).map(function)

    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.hoist_random_uniform = True
    dataset = dataset.with_options(options)
    self._testDataset(dataset)

  def testCapturedInputs(self):
    a = constant_op.constant(1, dtype=dtypes.float32)
    b = constant_op.constant(0, dtype=dtypes.float32)
    some_tensor = math_ops.mul(a, b)

    def random_with_capture(_):
      return some_tensor + random_ops.random_uniform(
          [], minval=1, maxval=10, dtype=dtypes.float32, seed=42)

    dataset = dataset_ops.Dataset.range(5).apply(
        testing.assert_next(["Zip[0]", "Map"])).map(random_with_capture)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.hoist_random_uniform = True
    dataset = dataset.with_options(options)
    self._testDataset(dataset)


if __name__ == "__main__":
  test.main()
