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
"""Tests for CollectiveAllReduceStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.distribute.python import collective_all_reduce_strategy
from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import cross_tower_utils
from tensorflow.contrib.distribute.python import multi_worker_test_base
from tensorflow.contrib.distribute.python import strategy_test_lib
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class DistributedCollectiveAllReduceStrategyTest(
    multi_worker_test_base.MultiWorkerTestBase, parameterized.TestCase):

  collective_key_base = 0

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 2 workers."""
    cls._workers, cls._ps = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=0)
    cls._cluster_spec = {
        run_config.TaskType.WORKER: [
            'fake_worker_0', 'fake_worker_1', 'fake_worker_2'
        ]
    }

  def setUp(self):
    self._run_options = config_pb2.RunOptions()
    self._run_options.experimental.collective_graph_key = 6

    self._sess_config = config_pb2.ConfigProto()
    self._sess_config.experimental.collective_group_leader = (
        '/job:worker/replica:0/task:0')

    # We use a different key_base for each test so that collective keys won't be
    # reused.
    # TODO(yuefengz, tucker): enable it to reuse collective keys in different
    # tests.
    DistributedCollectiveAllReduceStrategyTest.collective_key_base += 100000
    super(DistributedCollectiveAllReduceStrategyTest, self).setUp()

  def _get_test_object(self, task_type, task_id, num_gpus=0):
    distribution = collective_all_reduce_strategy.CollectiveAllReduceStrategy(
        num_gpus_per_worker=num_gpus,
        cluster_spec=self._cluster_spec,
        task_type=task_type,
        task_id=task_id)
    collective_keys = cross_tower_utils.CollectiveKeys(
        group_key_start=10 * num_gpus +
        DistributedCollectiveAllReduceStrategyTest.collective_key_base,
        instance_key_start=num_gpus * 100 +
        DistributedCollectiveAllReduceStrategyTest.collective_key_base,
        instance_key_with_id_start=num_gpus * 10000 +
        DistributedCollectiveAllReduceStrategyTest.collective_key_base)
    distribution._collective_keys = collective_keys
    distribution._cross_tower_ops._collective_keys = collective_keys
    return distribution, self._workers[task_id].target

  def _test_minimize_loss_graph(self, task_type, task_id, num_gpus):
    d, master_target = self._get_test_object(task_type, task_id, num_gpus)
    with ops.Graph().as_default(), \
         self.test_session(config=self._sess_config,
                           target=master_target) as sess, \
         d.scope():
      l = core.Dense(1, use_bias=False, name='gpu_%d' % d._num_gpus_per_worker)

      def loss_fn(x):
        y = array_ops.reshape(l(x), []) - constant_op.constant(1.)
        return y * y

      # TODO(yuefengz, apassos): eager.backprop.implicit_grad is not safe for
      # multiple graphs (b/111216820).
      def grad_fn(x):
        loss = loss_fn(x)
        var_list = (
            variables.trainable_variables() + ops.get_collection(
                ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        grads = gradients.gradients(loss, var_list)
        ret = list(zip(grads, var_list))
        return ret

      def update(v, g):
        return v.assign_sub(0.05 * g, use_locking=True)

      one = d.broadcast(constant_op.constant([[1.]]))

      def step():
        """Perform one optimization step."""
        # Run forward & backward to get gradients, variables list.
        g_v = d.call_for_each_tower(grad_fn, one)
        # Update the variables using the gradients and the update() function.
        before_list = []
        after_list = []
        for g, v in g_v:
          fetched = d.read_var(v)
          before_list.append(fetched)
          with ops.control_dependencies([fetched]):
            # TODO(yuefengz): support non-Mirrored variable as destinations.
            g = d.reduce(
                variable_scope.VariableAggregation.SUM, g, destinations=v)
            with ops.control_dependencies(d.unwrap(d.update(v, update, g))):
              after_list.append(d.read_var(v))
        return before_list, after_list

      before_out, after_out = step()

      if context.num_gpus() < d._num_gpus_per_worker:
        return True

      sess.run(
          variables.global_variables_initializer(), options=self._run_options)

      for i in range(10):
        b, a = sess.run((before_out, after_out), options=self._run_options)
        if i == 0:
          before, = b
        after, = a

      error_before = abs(before - 1)
      error_after = abs(after - 1)
      # Error should go down
      self.assertLess(error_after, error_before)
      return error_after < error_before

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2]))
  def testMinimizeLossGraph(self, num_gpus):
    self._run_between_graph_clients(self._test_minimize_loss_graph,
                                    self._cluster_spec, num_gpus)

  def _test_variable_initialization(self, task_type, task_id, num_gpus):
    distribution, master_target = self._get_test_object(task_type, task_id,
                                                        num_gpus)
    with ops.Graph().as_default(), \
         self.test_session(config=self._sess_config,
                           target=master_target) as sess, \
         distribution.scope():

      def model_fn():
        x = variable_scope.get_variable(
            'x',
            shape=(2, 3),
            initializer=init_ops.random_uniform_initializer(
                1.0, 10.0, dtype=dtypes.float32))
        return array_ops.identity(x)

      x = distribution.call_for_each_tower(model_fn)
      reduced_x = distribution.unwrap(
          distribution.reduce(
              variable_scope.VariableAggregation.MEAN, x,
              destinations='/cpu:0'))[0]

      sess.run(
          variables.global_variables_initializer(), options=self._run_options)
      x_value, reduced_x_value = sess.run(
          [x, reduced_x], options=self._run_options)
      self.assertTrue(np.array_equal(x_value, reduced_x_value))
    return np.array_equal(x_value, reduced_x_value)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2]))
  def testVariableInitialization(self, num_gpus):
    if context.num_gpus() < num_gpus:
      return
    self._run_between_graph_clients(
        self._test_variable_initialization,
        self._cluster_spec,
        num_gpus=num_gpus)


class LocalCollectiveAllReduceStrategy(strategy_test_lib.DistributionTestBase,
                                       parameterized.TestCase):

  def testMinimizeLossGraph(self, num_gpus=2):
    # Collective ops doesn't support strategy with one device.
    if context.num_gpus() < num_gpus:
      return
    distribution = collective_all_reduce_strategy.CollectiveAllReduceStrategy(
        num_gpus_per_worker=num_gpus)
    self._test_minimize_loss_graph(distribution)


if __name__ == '__main__':
  test.main()
