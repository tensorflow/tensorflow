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
"""Tests for the MapParallelization optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.contrib.data.python.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class MapParallelizationTest(test_base.DatasetTestBase, parameterized.TestCase):

  @staticmethod
  def map_functions():
    identity = lambda x: x
    increment = lambda x: x + 1

    def assert_greater(x):
      assert_op = control_flow_ops.Assert(math_ops.greater(x, -1), [x])
      with ops.control_dependencies([assert_op]):
        return x

    def random(_):
      return random_ops.random_uniform([],
                                       minval=0,
                                       maxval=10,
                                       dtype=dtypes.int64,
                                       seed=42)

    def assert_with_random(x):
      x = assert_greater(x)
      return random(x)

    return (("Identity", identity, True), ("Increment", increment, True),
            ("AssertGreater", assert_greater, True), ("Random", random, False),
            ("AssertWithRandom", assert_with_random, False))

  @parameterized.named_parameters(*map_functions.__func__())
  def testMapParallelization(self, function, should_optimize):
    next_nodes = ["ParallelMap"] if should_optimize else ["Map"]
    dataset = dataset_ops.Dataset.range(5).apply(
        optimization.assert_next(next_nodes)).map(function).apply(
            optimization.optimize(["map_parallelization"]))
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      for x in range(5):
        result = sess.run(get_next)
        # No need to run the pipeline if it was not optimized.  Also the results
        # might be hard to check because of random.
        if not should_optimize:
          return
        r = function(x)
        self.assertAllEqual(r, result)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
