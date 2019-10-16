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

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
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
from tensorflow.python.training.server_lib import ClusterSpec


def create_test_objects(cluster_spec=None,
                        task_type=None,
                        task_id=None,
                        num_gpus=None):
  sess_config = config_pb2.ConfigProto()
  if num_gpus is None:
    num_gpus = context.num_gpus()

  if cluster_spec and task_type and task_id is not None:
    cluster_resolver = SimpleClusterResolver(
        cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
        task_type=task_type,
        task_id=task_id,
        num_accelerators={'GPU': num_gpus})
    target = 'grpc://' + cluster_spec[task_type][task_id]
  else:
    cluster_resolver = SimpleClusterResolver(
        ClusterSpec({}), num_accelerators={'GPU': num_gpus})
    target = ''

  strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy(
      cluster_resolver=cluster_resolver)
  sess_config = strategy.update_config_proto(sess_config)

  return strategy, target, sess_config


class CollectiveAllReduceStrategyTestBase(
    multi_worker_test_base.MultiWorkerTestBase):

  collective_key_base = 0

  def setUp(self):
    # We use a different key_base for each test so that collective keys won't be
    # reused.
    # TODO(yuefengz, ayushd): enable it to reuse collective keys in different
    # tests.
    CollectiveAllReduceStrategyTestBase.collective_key_base += 100000
    super(CollectiveAllReduceStrategyTestBase, self).setUp()

  def _get_test_object(self, task_type, task_id, num_gpus=0):
    strategy, target, session_config = create_test_objects(
        cluster_spec=self._cluster_spec,
        task_type=task_type,
        task_id=task_id,
        num_gpus=num_gpus)

    collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=10 +
        CollectiveAllReduceStrategyTestBase.collective_key_base,
        op_instance_key_start=100 +
        CollectiveAllReduceStrategyTestBase.collective_key_base,
        variable_instance_key_start=10000 +
        CollectiveAllReduceStrategyTestBase.collective_key_base)
    strategy.extended._collective_keys = collective_keys
    strategy.extended._cross_device_ops._collective_keys = (collective_keys)

    return strategy, target, session_config

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

      one = constant_op.constant([[1.]])

      def step():
        """Perform one optimization step."""
        # Run forward & backward to get gradients, variables list.
        g_v = d.extended.call_for_each_replica(grad_fn, args=[one])
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
                d.extended.update(v, update, args=(g,), group=False)):
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
      # TODO(yuefengz): make loss a callable for eager mode.
      loss = losses.sparse_softmax_cross_entropy(labels=label, logits=logits)
      optimizer = adam.AdamOptimizer(learning_rate=1e-4)
      train_op = optimizer.minimize(loss,
                                    training_util.get_or_create_global_step())
      return train_op

    with ops.Graph().as_default(), \
         self.cached_session(config=config,
                             target=master_target) as sess:
      with d.scope():
        train_op = d.extended.call_for_each_replica(model_fn)
        train_op = d.group(d.experimental_local_results(train_op))

      sess.run(variables.global_variables_initializer())
      sess.run(train_op)

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

      x = distribution.extended.call_for_each_replica(model_fn)
      reduced_x = distribution.reduce(reduce_util.ReduceOp.MEAN, x, axis=None)
      x = distribution.experimental_local_results(x)[0]

      sess.run(variables.global_variables_initializer())

      x_value, reduced_x_value = sess.run([x, reduced_x])
      self.assertTrue(
          np.allclose(x_value, reduced_x_value, atol=1e-5),
          msg=('x_value = %r, reduced_x_value = %r' % (x_value,
                                                       reduced_x_value)))

  def _test_input_fn_iterator(self,
                              task_type,
                              task_id,
                              num_gpus,
                              input_fn,
                              expected_values,
                              test_reinitialize=True,
                              ignore_order=False):
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
        if ignore_order:
          self.assertCountEqual(expected_value, computed_value)
        else:
          self.assertEqual(expected_value, computed_value)

      with self.assertRaises(errors.OutOfRangeError):
        next_element = iterator.get_next()
        sess.run([values.select_replica(r, next_element)
                  for r in range(len(devices))])

      # After re-initializing the iterator, should be able to iterate again.
      if test_reinitialize:
        sess.run(iterator.initialize())

        for expected_value in expected_values:
          next_element = iterator.get_next()
          computed_value = sess.run([values.select_replica(r, next_element)
                                     for r in range(len(devices))])
          if ignore_order:
            self.assertCountEqual(expected_value, computed_value)
          else:
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

  @combinations.generate(combinations.combine(mode=['graph']))
  def test_num_replicas_in_sync(self):
    distribution, _, _ = create_test_objects(
        cluster_spec=self._cluster_spec,
        task_type='worker',
        task_id=0,
        num_gpus=2)
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
      combinations.combine(
          mode=['graph'],
          num_gpus=[0, 1, 2],
          required_gpus=1,
          use_dataset=[True, False]))
  def testMakeInputFnIterator(self, num_gpus, use_dataset):
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    if use_dataset:
      fn = lambda: dataset_ops.Dataset.range(100)
    else:
      def fn():
        dataset = dataset_ops.Dataset.range(100)
        it = dataset_ops.make_one_shot_iterator(dataset)
        return it.get_next
    # We use CPU as the device when num_gpus = 0
    devices_per_worker = max(1, num_gpus)
    expected_values = [[i+j for j in range(devices_per_worker)]
                       for i in range(0, 100, devices_per_worker)]

    input_fn = self._input_fn_to_test_input_context(
        fn,
        expected_num_replicas_in_sync=3*devices_per_worker,
        expected_num_input_pipelines=3,
        expected_input_pipeline_id=1)  # because task_id = 1
    self._test_input_fn_iterator(
        'worker',
        1,
        num_gpus,
        input_fn,
        expected_values,
        test_reinitialize=use_dataset,
        ignore_order=not use_dataset)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testUpdateConfigProto(self):
    strategy, _, _ = self._get_test_object(
        task_type='worker', task_id=1, num_gpus=2)

    config_proto = config_pb2.ConfigProto(device_filters=['to_be_overridden'])
    rewrite_options = config_proto.graph_options.rewrite_options
    rewrite_options.scoped_allocator_opts.enable_op.append('to_be_removed')

    new_config = strategy.update_config_proto(config_proto)

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

  @combinations.generate(combinations.combine(mode=['eager']))
  def testEnableCollectiveOps(self):
    mock_called = [False]

    # pylint: disable=dangerous-default-value
    def mock_enable_collective_ops(server_def, mock_called=mock_called):
      self.assertEqual('worker', server_def.job_name)
      self.assertEqual(1, server_def.task_index)
      self.assertEqual('grpc', server_def.protocol)
      mock_called[0] = True

    def mock_configure_collective_ops(*args, **kwargs):
      del args, kwargs

    with test.mock.patch.object(context.context(), 'enable_collective_ops',
                                mock_enable_collective_ops), \
         test.mock.patch.object(context.context(), 'configure_collective_ops',
                                mock_configure_collective_ops):
      strategy, _, _ = self._get_test_object(
          task_type='worker', task_id=1, num_gpus=2)
    self.assertTrue(strategy.extended._std_server_started)
    self.assertTrue(mock_called[0])


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

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'], num_gpus=[2, 4], required_gpus=2))
  def testMinimizeLoss(self, num_gpus):
    # Collective ops doesn't support strategy with one device.
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    if context.executing_eagerly():
      strategy, _, _ = self._get_test_object(None, None, num_gpus)
      self._test_minimize_loss_eager(strategy)
    else:
      self._test_minimize_loss_graph(None, None, num_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[2, 4], required_gpus=2))
  def testComplexModel(self, num_gpus):
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    self._test_complex_model(None, None, num_gpus)

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'], required_gpus=2, use_dataset=[True, False]))
  def testMakeInputFnIterator(self, use_dataset):
    num_gpus = 2
    if use_dataset:
      fn = lambda: dataset_ops.Dataset.range(5 * num_gpus)
    else:
      def fn():
        dataset = dataset_ops.Dataset.range(5 * num_gpus)
        it = dataset.make_one_shot_iterator()
        return it.get_next
    expected_values = [range(i, i + num_gpus) for i in range(0, 10, num_gpus)]

    input_fn = self._input_fn_to_test_input_context(
        fn,
        expected_num_replicas_in_sync=num_gpus,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    self._test_input_fn_iterator(
        None,
        None,
        num_gpus,
        input_fn,
        expected_values,
        test_reinitialize=use_dataset,
        ignore_order=not use_dataset)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testAllReduceSum(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_sum(distribution)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testAllReduceSumGradients(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_sum_gradients(distribution)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testAllReduceSumGradientTape(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_sum_gradient_tape(distribution)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testAllReduceMean(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_mean(distribution)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testAllReduceMeanGradients(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_mean_gradients(distribution)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testAllReduceMeanGradientTape(self):
    if context.num_gpus() < 2: self.skipTest('Not enough GPUs')
    distribution, target, config = self._get_test_object(None, None, num_gpus=2)
    with self.cached_session(config=config, target=target):
      self._test_all_reduce_mean_gradient_tape(distribution)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testNumpyDataset(self):
    num_gpus = 2
    if context.num_gpus() < num_gpus:
      self.skipTest('Not enough GPUs')
    strategy, target, config = self._get_test_object(
        None, None, num_gpus=num_gpus)
    self._test_numpy_dataset(
        strategy, session=self.cached_session(config=config, target=target))


if __name__ == '__main__':
  test.main()
