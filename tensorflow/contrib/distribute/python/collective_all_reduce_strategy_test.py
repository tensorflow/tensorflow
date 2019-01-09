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
from tensorflow.contrib.distribute.python import multi_worker_test_base
from tensorflow.contrib.distribute.python import strategy_test_lib
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import training_util


class CollectiveAllReduceStrategyTestBase(
    multi_worker_test_base.MultiWorkerTestBase):

  collective_key_base = 0

  def setUp(self):
    # We use a different key_base for each test so that collective keys won't be
    # reused.
    # TODO(yuefengz, tucker): enable it to reuse collective keys in different
    # tests.
    CollectiveAllReduceStrategyTestBase.collective_key_base += 100000
    super(CollectiveAllReduceStrategyTestBase, self).setUp()

  def _get_test_object(self, task_type, task_id, num_gpus=0):
    distribution = collective_all_reduce_strategy.CollectiveAllReduceStrategy(
        num_gpus_per_worker=num_gpus)
    session_config = config_pb2.ConfigProto()
    if task_type and task_id is not None:
      distribution.configure(
          session_config=session_config,
          cluster_spec=self._cluster_spec,
          task_type=task_type,
          task_id=task_id)
    collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=10 * num_gpus +
        CollectiveAllReduceStrategyTestBase.collective_key_base,
        instance_key_start=num_gpus * 100 +
        CollectiveAllReduceStrategyTestBase.collective_key_base,
        instance_key_with_id_start=num_gpus * 10000 +
        CollectiveAllReduceStrategyTestBase.collective_key_base)
    distribution.extended._collective_keys = collective_keys
    distribution.extended._cross_device_ops._collective_keys = (
        collective_keys)
    if task_type and task_id is not None:
      return distribution, 'grpc://' + self._cluster_spec[task_type][
          task_id], session_config
    else:
      return distribution, '', session_config

  def _test_minimize_loss_graph(self, task_type, task_id, num_gpus):
    d, master_target, config = self._get_test_object(task_type, task_id,
                                                     num_gpus)
    with ops.Graph().as_default(), \
         self.cached_session(config=config,
                             target=master_target) as sess, \
         d.scope():
      l = core.Dense(1, use_bias=False,
                     name='gpu_%d' % d.extended._num_gpus_per_worker)

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
        g_v = d.call_for_each_replica(grad_fn, args=[one])
        # Update the variables using the gradients and the update() function.
        before_list = []
        after_list = []
        for g, v in g_v:
          fetched = d.extended.read_var(v)
          before_list.append(fetched)
          with ops.control_dependencies([fetched]):
            # TODO(yuefengz): support non-Mirrored variable as destinations.
            g = d.extended.reduce_to(
                reduce_util.ReduceOp.SUM, g, destinations=v)
            with ops.control_dependencies(
                d.update(v, update, g, grouped=False)):
              after_list.append(d.extended.read_var(v))
        return before_list, after_list

      before_out, after_out = step()

      if context.num_gpus() < d.extended._num_gpus_per_worker:
        return True

      sess.run(variables.global_variables_initializer())

      for i in range(10):
        b, a = sess.run((before_out, after_out))
        if i == 0:
          before, = b
        after, = a

      error_before = abs(before - 1)
      error_after = abs(after - 1)
      # Error should go down
      self.assertLess(error_after, error_before)
      return error_after < error_before

  def _test_complex_model(self, task_type, task_id, num_gpus):
    d, master_target, config = self._get_test_object(task_type, task_id,
                                                     num_gpus)

    def model_fn():
      """Mnist model with synthetic input."""
      data_format = 'channels_last'
      input_shape = [28, 28, 1]
      l = keras.layers
      max_pool = l.MaxPooling2D((2, 2), (2, 2),
                                padding='same',
                                data_format=data_format)
      model = keras.Sequential([
          l.Reshape(target_shape=input_shape, input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=nn.relu), max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=nn.relu), max_pool,
          l.Flatten(),
          l.Dense(1024, activation=nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])
      image = random_ops.random_uniform([2, 28, 28])
      label = random_ops.random_uniform([2, 1], maxval=10, dtype=dtypes.int32)
      logits = model(image, training=True)
      loss = losses.sparse_softmax_cross_entropy(labels=label, logits=logits)
      optimizer = adam.AdamOptimizer(learning_rate=1e-4)
      train_op = optimizer.minimize(loss,
                                    training_util.get_or_create_global_step())
      return train_op

    with ops.Graph().as_default(), \
         self.cached_session(config=config,
                             target=master_target) as sess:
      with d.scope():
        train_op = d.call_for_each_replica(model_fn)
        train_op = d.group(d.unwrap(train_op))

      sess.run(variables.global_variables_initializer())
      sess.run(train_op)
      return True

  def _test_variable_initialization(self, task_type, task_id, num_gpus):
    distribution, master_target, config = self._get_test_object(
        task_type, task_id, num_gpus)
    with ops.Graph().as_default(), \
         self.cached_session(config=config,
                             target=master_target) as sess, \
         distribution.scope():

      def model_fn():
        x = variable_scope.get_variable(
            'x',
            shape=(2, 3),
            initializer=init_ops.random_uniform_initializer(
                1.0, 10.0, dtype=dtypes.float32))
        return array_ops.identity(x)

      x = distribution.call_for_each_replica(model_fn)
      reduced_x = distribution.reduce(reduce_util.ReduceOp.MEAN, x)
      x = distribution.unwrap(x)[0]

      sess.run(variables.global_variables_initializer())

      x_value, reduced_x_value = sess.run([x, reduced_x])
      self.assertTrue(
          np.allclose(x_value, reduced_x_value, atol=1e-5),
          msg=('x_value = %r, reduced_x_value = %r' % (x_value,
                                                       reduced_x_value)))
    return np.allclose(x_value, reduced_x_value, atol=1e-5)

  def _test_input_fn_iterator(self, task_type, task_id, num_gpus, input_fn,
                              expected_values):
    distribution, master_target, config = self._get_test_object(
        task_type, task_id, num_gpus)
    devices = distribution.extended.worker_devices

    with ops.Graph().as_default(), \
         self.cached_session(config=config,
                             target=master_target) as sess:
      iterator = distribution.make_input_fn_iterator(input_fn)
      sess.run(iterator.initialize())

      for expected_value in expected_values:
        next_element = iterator.get_next()
        computed_value = sess.run([values.select_replica(r, next_element)
                                   for r in range(len(devices))])
        self.assertEqual(expected_value, computed_value)

      with self.assertRaises(errors.OutOfRangeError):
        next_element = iterator.get_next()
        sess.run([values.select_replica(r, next_element)
                  for r in range(len(devices))])

      # After re-initializing the iterator, should be able to iterate again.
      sess.run(iterator.initialize())

      for expected_value in expected_values:
        next_element = iterator.get_next()
        computed_value = sess.run([values.select_replica(r, next_element)
                                   for r in range(len(devices))])
        self.assertEqual(expected_value, computed_value)


class DistributedCollectiveAllReduceStrategyTest(
    CollectiveAllReduceStrategyTestBase,
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 3 workers."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=0)

  def test_num_replicas_in_sync(self):
    distribution = collective_all_reduce_strategy.CollectiveAllReduceStrategy(
        num_gpus_per_worker=2)
    distribution.configure(cluster_spec=self._cluster_spec, task_type='worker',
                           task_id=0)
    num_workers = len(self._cluster_spec.get('chief', []) +
                      self._cluster_spec.get('worker', []))
    self.assertEqual(2 * num_workers,
                     distribution.num_replicas_in_sync)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2], required_gpus=1))
  def testMinimizeLossGraph(self, num_gpus):
    self._run_between_graph_clients(self._test_minimize_loss_graph,
                                    self._cluster_spec, num_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2], required_gpus=1))
  def testVariableInitialization(self, num_gpus):
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    self._run_between_graph_clients(
        self._test_variable_initialization,
        self._cluster_spec,
        num_gpus=num_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2], required_gpus=1))
  def testComplexModel(self, num_gpus):
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    self._run_between_graph_clients(
        self._test_complex_model, self._cluster_spec, num_gpus=num_gpus)

  # TODO(yuefengz): Update how we use num_gpus and required_gpus
  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2], required_gpus=1))
  def testMakeInputFnIterator(self, num_gpus):
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    dataset_fn = lambda: dataset_ops.Dataset.range(100)
    # We use CPU as the device when num_gpus = 0
    devices_per_worker = max(1, num_gpus)
    expected_values = [[i+j for j in range(devices_per_worker)]
                       for i in range(0, 100, devices_per_worker)]

    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=3*devices_per_worker,
        expected_num_input_pipelines=3,
        expected_input_pipeline_id=1)  # because task_id = 1
    self._test_input_fn_iterator('worker', 1, num_gpus,
                                 input_fn, expected_values)

  def testUpdateConfigProto(self):
    distribution = collective_all_reduce_strategy.CollectiveAllReduceStrategy(
        num_gpus_per_worker=2)
    distribution.configure(
        cluster_spec=self._cluster_spec, task_type='worker', task_id=1)

    config_proto = config_pb2.ConfigProto(device_filters=['to_be_overridden'])
    rewrite_options = config_proto.graph_options.rewrite_options
    rewrite_options.scoped_allocator_opts.enable_op.append('to_be_removed')

    new_config = distribution.update_config_proto(config_proto)

    # Verify group leader
    self.assertEqual('/job:worker/replica:0/task:0',
                     new_config.experimental.collective_group_leader)

    # Verify device filters.
    self.assertEqual(['/job:worker/task:1'], new_config.device_filters)

    # Verify rewrite options.
    new_rewrite_options = new_config.graph_options.rewrite_options
    self.assertEqual(rewriter_config_pb2.RewriterConfig.ON,
                     new_rewrite_options.scoped_allocator_optimization)
    self.assertEqual(['CollectiveReduce'],
                     new_rewrite_options.scoped_allocator_opts.enable_op)


class DistributedCollectiveAllReduceStrategyTestWithChief(
    CollectiveAllReduceStrategyTestBase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 3 workers and 1 chief."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=0, has_chief=True)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2], required_gpus=1))
  def testMinimizeLossGraph(self, num_gpus):
    self._run_between_graph_clients(self._test_minimize_loss_graph,
                                    self._cluster_spec, num_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2], required_gpus=1))
  def testVariableInitialization(self, num_gpus):
    if context.num_gpus() < num_gpus:
      return
    self._run_between_graph_clients(
        self._test_variable_initialization,
        self._cluster_spec,
        num_gpus=num_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2], required_gpus=1))
  def testComplexModel(self, num_gpus):
    if context.num_gpus() < num_gpus:
      return
    self._run_between_graph_clients(
        self._test_complex_model, self._cluster_spec, num_gpus=num_gpus)


class LocalCollectiveAllReduceStrategy(
    CollectiveAllReduceStrategyTestBase,
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.TwoDeviceDistributionTestBase,
    parameterized.TestCase):

  def testMinimizeLossGraph(self, num_gpus=2):
    # Collective ops doesn't support strategy with one device.
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    self._test_minimize_loss_graph(None, None, num_gpus)

  def testComplexModel(self, num_gpus=2):
    # Collective ops doesn't support strategy with one device.
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    self._test_complex_model(None, None, num_gpus)

  def testMakeInputFnIterator(self, num_gpus=2):
    # Collective ops doesn't support strategy with one device.
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    dataset_fn = lambda: dataset_ops.Dataset.range(10)
    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=num_gpus,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    self._test_input_fn_iterator(None, None, num_gpus,
                                 input_fn, expected_values)

  def testAllReduceSum(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_sum(distribution)

  def testAllReduceSumGradients(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_sum_gradients(distribution)

  def testAllReduceSumGradientTape(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_sum_gradient_tape(distribution)

  def testAllReduceMean(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_mean(distribution)

  def testAllReduceMeanGradients(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_mean_gradients(distribution)

  def testAllReduceMeanGradientTape(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_mean_gradient_tape(distribution)

  def testNumpyIterator(self):
    num_gpus = 2
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    strategy, _, _ = self._get_test_object(None, None, num_gpus)
    self._test_numpy_iterator(strategy)


if __name__ == '__main__':
  test.main()
