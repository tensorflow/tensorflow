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
"""Tests for MirroredFunctionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_function_strategy
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util


class MirroredFunctionStrategyTest(test.TestCase):

  def setUp(self):
    super(MirroredFunctionStrategyTest, self).setUp()
    strategy_combinations.set_virtual_cpus_to_at_least(3)
    self._strategy = mirrored_function_strategy.MirroredFunctionStrategy(
        devices=("/cpu:1", "/cpu:2"))

  def testReplicaId(self):
    f_traces = []

    @def_function.function
    def f(x):
      f_traces.append(None)  # Only happens on trace.
      replica_context = distribution_strategy_context.get_replica_context()
      # This is a non-constant tensor.
      replica_id = replica_context.replica_id_in_sync_group
      self.assertIsInstance(replica_id, ops.Tensor)
      self.assertIsNone(tensor_util.constant_value(replica_id))
      return x + replica_id

    one = constant_op.constant(1)
    self.assertLen(f_traces, 0)
    result1 = self._strategy.run(f, args=(one,))
    self.assertLen(f_traces, 1)  # Function traced once, not for each replica.
    # Returns a per-replica value.
    self.assertIsInstance(result1, values.PerReplica)
    self.assertAllEqual([1, 2],
                        self._strategy.experimental_local_results(result1))

    # Try passing a per-replica value as an argument.
    result2 = self._strategy.run(f, args=(result1,))
    self.assertLen(f_traces, 1)
    self.assertIsInstance(result2, values.PerReplica)
    self.assertAllEqual([1, 3],
                        self._strategy.experimental_local_results(result2))

  def testMergeCall(self):
    f_traces = []
    g_traces = []

    def g(strategy, z):
      g_traces.append(None)  # Only happens on trace.
      self.assertIs(strategy, self._strategy)
      self.assertTrue(distribution_strategy_context.in_cross_replica_context())
      self.assertIsInstance(z, mirrored_function_strategy.FnMergedValue)
      return z

    @def_function.function
    def f(x):
      f_traces.append(None)  # Only happens on trace.
      replica_context = distribution_strategy_context.get_replica_context()
      y = replica_context.merge_call(g, args=(x,))
      self.assertIsInstance(y, ops.Tensor)
      return y

    one = constant_op.constant(1)
    self.assertLen(f_traces, 0)
    self.assertLen(g_traces, 0)
    result = self._strategy.run(f, args=(one,))
    # Functions traced once, not for each replica.
    self.assertLen(f_traces, 1)
    self.assertLen(g_traces, 1)
    # Returns a per-replica value.
    self.assertIsInstance(result, values.PerReplica)
    self.assertAllEqual([1, 1],
                        self._strategy.experimental_local_results(result))


if __name__ == "__main__":
  test.main()
