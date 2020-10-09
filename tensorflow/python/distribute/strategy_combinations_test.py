# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for a little bit of strategy_combinations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class StrategyCombinationsTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          strategy=strategy_combinations.two_replica_strategies,
          mode=["graph", "eager"]))
  def testTwoReplicaStrategy(self, strategy):
    with strategy.scope():

      @def_function.function
      def one():
        return array_ops.identity(1.)

      one_per_replica = strategy.run(one)
      num_replicas = strategy.reduce(
          reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
      self.assertEqual(self.evaluate(num_replicas), 2.)

  @combinations.generate(
      combinations.combine(
          strategy=strategy_combinations.four_replica_strategies,
          mode=["graph", "eager"]))
  def testFourReplicaStrategy(self, strategy):
    with strategy.scope():

      @def_function.function
      def one():
        return array_ops.identity(1.)

      one_per_replica = strategy.run(one)
      num_replicas = strategy.reduce(
          reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
      self.assertEqual(self.evaluate(num_replicas), 4.)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2
          ],
          mode=["graph", "eager"]))
  def testMirrored2CPUs(self, distribution):
    with distribution.scope():
      one_per_replica = distribution.run(lambda: constant_op.constant(1))
      num_replicas = distribution.reduce(
          reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
      self.assertEqual(2, self.evaluate(num_replicas))


if __name__ == "__main__":
  test_util.main()
