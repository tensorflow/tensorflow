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
"""Tests for HostState optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.contrib.data.python.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class HoistRandomUniformTest(test_base.DatasetTestBase, parameterized.TestCase):

  @staticmethod
  def map_functions():
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

  @parameterized.named_parameters(*map_functions.__func__())
  def testHoisting(self, function, will_optimize):
    dataset = dataset_ops.Dataset.range(5).apply(
        optimization.assert_next(
            ["Zip[0]", "Map"] if will_optimize else ["Map"])).map(function)

    dataset = dataset.apply(optimization.optimize(["hoist_random_uniform"]))
    self._testDataset(dataset)

  def testAdditionalInputs(self):
    a = constant_op.constant(1, dtype=dtypes.float32)
    b = constant_op.constant(0, dtype=dtypes.float32)
    some_tensor = math_ops.mul(a, b)

    def random_with_capture(_):
      return some_tensor + random_ops.random_uniform(
          [], minval=1, maxval=10, dtype=dtypes.float32, seed=42)

    dataset = dataset_ops.Dataset.range(5).apply(
        optimization.assert_next(
            ["Zip[0]", "Map"])).map(random_with_capture).apply(
                optimization.optimize(["hoist_random_uniform"]))
    self._testDataset(dataset)

  def _testDataset(self, dataset):
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()
    previous_result = 0
    with self.cached_session() as sess:
      for _ in range(5):
        result = sess.run(get_next)
        self.assertLessEqual(1, result)
        self.assertLessEqual(result, 10)
        # This checks if the result is somehow random by checking if we are not
        # generating the same values.
        self.assertNotEqual(previous_result, result)
        previous_result = result
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
