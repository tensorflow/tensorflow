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
"""Tests for training.moving_averages when using a DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.contrib.distribute.python import combinations
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import moving_averages


all_combinations = combinations.combine(
    distribution=[combinations.default_strategy,
                  combinations.one_device_strategy,
                  combinations.mirrored_strategy_with_gpu_and_cpu,
                  combinations.core_mirrored_strategy_with_gpu_and_cpu],
    mode=["graph"])


class AssignMovingAveragesTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(all_combinations)
  def testReplicaModeWithoutZeroDebias(self, distribution):
    replica_id = [0]

    def replica_fn():
      var = variables.Variable([10.0, 11.0])
      val = constant_op.constant([1.0 + replica_id[0], 2.0 - replica_id[0]])
      replica_id[0] += 1
      decay = 0.25
      assign = moving_averages.assign_moving_average(
          var, val, decay, zero_debias=False)
      return var, assign

    with distribution.scope(), self.cached_session() as sess:
      var, assign = distribution.extended.call_for_each_replica(replica_fn)
      variables.global_variables_initializer().run()
      self.assertAllClose([10.0, 11.0], var.eval())
      sess.run(distribution.unwrap(assign))
      # Mean of val across calls to replica_fn().
      average_val = [1.0 + 0.5 * (replica_id[0] - 1),
                     2.0 - 0.5 * (replica_id[0] - 1)]
      val_weight = 1.0 - 0.25
      self.assertAllClose(
          [10.0 * 0.25 + average_val[0] * val_weight,
           11.0 * 0.25 + average_val[1] * val_weight],
          var.eval())

  @combinations.generate(all_combinations)
  def testReplicaMode(self, distribution):
    replica_id = [0]

    def replica_fn():
      var = variables.Variable([0.0, 0.0])
      val = constant_op.constant([1.0 + replica_id[0], 2.0 - replica_id[0]])
      replica_id[0] += 1
      decay = 0.25
      assign = moving_averages.assign_moving_average(var, val, decay)
      return var, assign.op

    with distribution.scope(), self.cached_session() as sess:
      var, assign_op = distribution.extended.call_for_each_replica(replica_fn)
      variables.global_variables_initializer().run()
      self.assertAllClose([0.0, 0.0], var.eval())
      sess.run(distribution.unwrap(assign_op))
      # Mean of val across calls to replica_fn().
      average_val = [1.0 + 0.5 * (replica_id[0] - 1),
                     2.0 - 0.5 * (replica_id[0] - 1)]
      self.assertAllClose(average_val, var.eval())

  @combinations.generate(all_combinations)
  def testCrossDeviceWithoutZeroDebias(self, distribution):
    with distribution.scope(), self.cached_session() as sess:
      var = variables.Variable([10.0, 11.0])
      val = constant_op.constant([1.0, 2.0])
      decay = 0.25
      # NOTE(josh11b): We currently generate an error if val is a PerReplica
      # value.
      assign = moving_averages.assign_moving_average(
          var, val, decay, zero_debias=False)

      variables.global_variables_initializer().run()
      self.assertAllClose([10.0, 11.0], var.eval())
      sess.run(assign)
      average_val = [1.0, 2.0]
      val_weight = 1.0 - 0.25
      self.assertAllClose(
          [10.0 * 0.25 + average_val[0] * val_weight,
           11.0 * 0.25 + average_val[1] * val_weight],
          var.eval())
      # Also try assign.op.
      sess.run(assign.op)
      orig_weight = 0.25 * 0.25
      val_weight = 1.0 - orig_weight
      self.assertAllClose(
          [10.0 * orig_weight + average_val[0] * val_weight,
           11.0 * orig_weight + average_val[1] * val_weight],
          var.eval())

  @combinations.generate(all_combinations)
  def testCrossDevice(self, distribution):
    with distribution.scope(), self.cached_session() as sess:
      var = variables.Variable([0.0, 0.0])
      val = array_ops.placeholder(dtypes.float32)
      decay = 0.25
      # NOTE(josh11b): We currently generate an error if val is a PerReplica
      # value.
      assign = moving_averages.assign_moving_average(var, val, decay)

      variables.global_variables_initializer().run()
      self.assertAllClose([0.0, 0.0], var.eval())
      sess.run(assign, feed_dict={val: [1.0, 2.0]})
      self.assertAllClose([1.0, 2.0], var.eval())

      # Also try assign.op.
      sess.run(assign.op, feed_dict={val: [10.0, 0.0]})
      self.assertAllClose(
          [(1.0 * 0.25 + 10.0) / (1.0 * 0.25 + 1.0),
           (2.0 * 0.25 + 0.0) / (1.0 * 0.25 + 1.0)],
          var.eval())

  @combinations.generate(all_combinations)
  def testAssignVariable(self, distribution):

    def replica_fn():
      var = variables.Variable([10.0, 11.0])
      # Here we expect to check the case when input value are variable.
      val = variables.Variable([1., 2.])
      decay = 0.25
      assign = moving_averages.assign_moving_average(
          var, val, decay, zero_debias=False)
      return var, assign

    with distribution.scope(), self.cached_session() as sess:
      var, assign = distribution.extended.call_for_each_replica(replica_fn)
      variables.global_variables_initializer().run()
      self.assertAllClose([10.0, 11.0], var.eval())
      sess.run(distribution.unwrap(assign))
      self.assertAllClose(
          [10 * 0.25 + 1. * (1 - 0.25), 11 * 0.25 + 2. * (1 - 0.25)],
          var.eval())


if __name__ == "__main__":
  test.main()
