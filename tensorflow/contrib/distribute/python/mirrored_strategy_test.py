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
"""Tests for class MirroredStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import strategy_test_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import distribute as distribute_lib


class MirroredOneCPUDistributionTest(strategy_test_lib.DistributionTestBase):

  def _get_distribution_strategy(self):
    return mirrored_strategy.MirroredStrategy(["/device:CPU:0"])

  def testMinimizeLossEager(self):
    self._test_minimize_loss_eager(self._get_distribution_strategy())

  def testMinimizeLossGraph(self):
    self._test_minimize_loss_graph(self._get_distribution_strategy())

  def testMapReduce(self):
    self._test_map_reduce(self._get_distribution_strategy())

  def testDeviceIndex(self):
    self._test_device_index(self._get_distribution_strategy())

  def testTowerId(self):
    self._test_tower_id(self._get_distribution_strategy())

  @test_util.run_in_graph_and_eager_modes
  def testCallAndMergeExceptions(self):
    self._test_call_and_merge_exceptions(self._get_distribution_strategy())


class VariableCreatorStackTest(test.TestCase):

  def testCreatorStacksAreThreadLocal(self):
    devices = ["/device:CPU:0", "/device:GPU:0"]
    dist = mirrored_strategy.MirroredStrategy(devices)

    def model_fn(device_id):
      assert isinstance(device_id, int)
      def thread_creator_fn(next_creator, *args, **kwargs):
        return next_creator(*args, **kwargs) + ":thread_" + str(device_id)

      with variable_scope.variable_creator_scope(thread_creator_fn):
        # Create a variable in this scope.
        v = variable_scope.variable(1.0)

        # This will pause the current thread, and execute the other thread.
        distribute_lib.get_tower_context().merge_call(lambda _: _)
      return v

    def main_thread_creator(next_creator, *args, **kwargs):
      # We are not using the underlying next_creator for test purposes.
      del next_creator, args, kwargs
      return "main_thread"

    with context.graph_mode(), \
        dist.scope(), \
        variable_scope.variable_creator_scope(main_thread_creator):
      result = dist.call_for_each_tower(model_fn, dist.worker_device_index)
      result = dist.unwrap(result)
      expected = ["main_thread:thread_0", "main_thread:thread_1"]
      self.assertEquals(expected, result)


if __name__ == "__main__":
  test.main()
