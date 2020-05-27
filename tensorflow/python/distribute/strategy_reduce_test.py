# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `strategy.reduce`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op


class StrategyReduceTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def test_reduce_with_axis(self, distribution):

    @def_function.function
    def fn():
      return constant_op.constant([1., 2.])
    x = distribution.run(fn)

    x_m = distribution.reduce(reduce_util.ReduceOp.MEAN, x, axis=0)
    self.assertEqual(1.5, self.evaluate(x_m))
    x_s = distribution.reduce(reduce_util.ReduceOp.SUM, x, axis=0)
    self.assertEqual(3 * distribution.num_replicas_in_sync, self.evaluate(x_s))


if __name__ == "__main__":
  test.main()
