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

import copy
import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


CollectiveAllReduceStrategy = (
    collective_all_reduce_strategy.CollectiveAllReduceStrategy)
CollectiveAllReduceExtended = (
    collective_all_reduce_strategy.CollectiveAllReduceExtended)
_CollectiveAllReduceStrategyExperimental = (
    collective_all_reduce_strategy._CollectiveAllReduceStrategyExperimental)


# TODO(b/231630416): Create more tests to cover the case that strategy uses
# different number of GPUs than the number of physical devices.
def create_test_objects(cluster_spec=None,
                        task_type=None,
                        task_id=None,
                        num_gpus=None,
                        num_tpus=None):
  if num_gpus is None:
    num_gpus = context.num_gpus()
  if num_tpus is None:
    num_tpus = context.context().list_physical_devices('TPU')
  if num_tpus:
    tpu_cluster_resolver.initialize_tpu_system()

  if cluster_spec and task_type and task_id is not None:
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
        task_type=task_type,
        task_id=task_id,
        num_accelerators={'GPU': num_gpus, 'TPU': num_tpus})
    target = 'grpc://' + cluster_spec[task_type][task_id]
  else:
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        server_lib.ClusterSpec({}),
        num_accelerators={'GPU': num_gpus, 'TPU': num_tpus},
    )
    target = ''

  strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy(
      cluster_resolver=cluster_resolver)

  return strategy, target


class CollectiveAllReduceStrategyTestBase(
    multi_worker_test_base.MultiWorkerTestBase):

  def setUp(self):
    # We use a different key_base for each test so that collective keys won't be
    # reused.
    CollectiveAllReduceStrategy._collective_key_base += 100000
    super(CollectiveAllReduceStrategyTestBase, self).setUp()

  def _get_test_object(self,
                       task_type,
                       task_id,
                       num_gpus=0,
                       num_tpus=0,
                       use_devices_arg=False):
    strategy, target = create_test_objects(
        cluster_spec=self._cluster_spec,
        task_type=task_type,
        task_id=task_id,
        num_gpus=num_gpus,
        num_tpus=num_tpus)

    if use_devices_arg and num_gpus > 0:
      devices = ['GPU:%d' % i for i in range(num_gpus)]
      # Temporary workaround to manually set the `_extended` field before device
      # initialization is exposed as a public interface.
      strategy._extended = CollectiveAllReduceExtended(
          container_strategy=strategy,
          cluster_resolver=None,
          communication_options=collective_util.Options(),
          devices=devices)
      # Manually set the field since the workaround bypasses the base
      # constructor, resulting in the absence of this field.
      strategy._extended._retrace_functions_for_each_device = (num_gpus > 1)

    return strategy, target

  def _test_minimize_loss_graph(self, task_type, task_id, num_gpus):
    distribution, master_target = self._get_test_object(task_type, task_id,
                                                        num_gpus)
    with ops.Graph().as_default(), \
         self.cached_session(target=master_target) as sess, \
         distribution.scope():
      initializer = functools.partial(
          init_ops_v2.GlorotUniform(), (1, 1), dtype=dtypes.float32)
      kernel = variables.Variable(
          initial_value=initializer,
          name='gpu_%d/kernel' % distribution.extended._num_devices_per_worker,
          trainable=True)

      def loss_fn(x):
        y = array_ops.reshape(
            gen_math_ops.mat_mul(x, kernel), []) - constant_op.constant(1.)
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
        g_v = distribution.extended.call_for_each_replica(grad_fn, args=[one])
        # Update the variables using the gradients and the update() function.
        before_list = []
        after_list = []
        for g, v in g_v:
          fetched = distribution.extended.read_var(v)
          before_list.append(fetched)
          with ops.control_dependencies([fetched]):
            # TODO(yuefengz): support non-Mirrored variable as destinations.
            g = distribution.extended.reduce_to(
                reduce_util.ReduceOp.SUM, g, destinations=v)
            with ops.control_dependencies(
                distribution.extended.update(v, update, args=(g,),
                                             group=False)):
              after_list.append(distribution.extended.read_var(v))
        return before_list, after_list

      before_out, after_out = step()

      if (distribution.extended._local_device_type == 'GPU' and
          context.num_gpus() < distribution.extended._num_devices_per_worker):
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

  def _test_variable_initialization(self, task_type, task_id, num_gpus):
    distribution, master_target = self._get_test_object(task_type, task_id,
                                                        num_gpus)
    with ops.Graph().as_default(), \
         self.cached_session(target=master_target) as sess, \
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
                              ignore_order=False,
                              use_devices_arg=False):
    distribution, master_target = self._get_test_object(
        task_type, task_id, num_gpus, use_devices_arg=use_devices_arg)
    devices = distribution.extended.worker_devices

    with ops.Graph().as_default(), \
         self.cached_session(target=master_target) as sess:
      iterator = distribution.make_input_fn_iterator(input_fn)
      sess.run(iterator.initializer)

      for expected_value in expected_values:
        next_element = iterator.get_next()
        computed_value = sess.run([distribute_utils.select_replica(
            r, next_element) for r in range(len(devices))])
        if ignore_order:
          self.assertCountEqual(list(expected_value), list(computed_value))
        else:
          self.assertEqual(list(expected_value), list(computed_value))

      with self.assertRaises(errors.OutOfRangeError):
        next_element = iterator.get_next()
        sess.run([distribute_utils.select_replica(r, next_element)
                  for r in range(len(devices))])

      # After re-initializing the iterator, should be able to iterate again.
      if test_reinitialize:
        sess.run(iterator.initializer)

        for expected_value in expected_values:
          next_element = iterator.get_next()
          computed_value = sess.run([
              distribute_utils.select_replica(r, next_element)
              for r in range(len(devices))])
          if ignore_order:
            self.assertCountEqual(list(expected_value), list(computed_value))
          else:
            self.assertEqual(list(expected_value), list(computed_value))


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
    distribution, _ = create_test_objects(
        cluster_spec=self._cluster_spec,
        task_type='worker',
        task_id=0,
        num_gpus=2)
    num_workers = len(self._cluster_spec.get('chief', []) +
                      self._cluster_spec.get('worker', []))
    self.assertEqual(2 * num_workers,
                     distribution.num_replicas_in_sync)

  @combinations.generate(combinations.combine(
      mode=['graph'],
      prefetch_to_device=[None, True]))
  def test_prefetch_to_device_dataset(self, prefetch_to_device):
    distribution, _ = self._get_test_object(
        task_type='worker', task_id=0, num_gpus=2)
    if prefetch_to_device is None:
      input_options = None
    else:
      input_options = distribute_lib.InputOptions(
          experimental_fetch_to_device=prefetch_to_device)
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.batch(distribution.num_replicas_in_sync)
    dataset = distribution.experimental_distribute_dataset(
        dataset, options=input_options)
    if isinstance(dataset, input_lib_v1.DistributedDatasetV1):
      item = dataset.make_initializable_iterator().get_next()
    else:
      self.skipTest('unsupported test combination')
    device_types = {
        tf_device.DeviceSpec.from_string(tensor.device).device_type for
        tensor in item.values}
    self.assertAllEqual(list(device_types), ['GPU'])

  @combinations.generate(combinations.combine(mode=['graph']))
  def test_prefetch_to_host_dataset(self):
    distribution, _ = self._get_test_object(
        task_type='worker', task_id=0, num_gpus=2)
    input_options = distribute_lib.InputOptions(
        experimental_fetch_to_device=False)
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.batch(distribution.num_replicas_in_sync)
    dataset = distribution.experimental_distribute_dataset(
        dataset, options=input_options)
    if isinstance(dataset, input_lib_v1.DistributedDatasetV1):
      item = dataset.make_initializable_iterator().get_next()
    else:
      self.skipTest('unsupported test combination')
    device_types = {
        tf_device.DeviceSpec.from_string(tensor.device).device_type for
        tensor in item.values}
    self.assertAllEqual(list(device_types), ['CPU'])

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testMinimizeLossGraph(self, required_gpus):
    self._run_between_graph_clients(self._test_minimize_loss_graph,
                                    self._cluster_spec, required_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testVariableInitialization(self, required_gpus):
    self._run_between_graph_clients(
        self._test_variable_initialization,
        self._cluster_spec,
        num_gpus=required_gpus)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=[0, 1, 2], use_dataset=[True, False]))
  def testMakeInputFnIterator(self, required_gpus, use_dataset):
    def _worker_fn(task_type, task_id, required_gpus):
      if use_dataset:
        fn = lambda: dataset_ops.Dataset.range(20)
      else:
        def fn():
          dataset = dataset_ops.Dataset.range(20)
          it = dataset_ops.make_one_shot_iterator(dataset)
          return it.get_next
      # We use CPU as the device when required_gpus = 0
      devices_per_worker = max(1, required_gpus)
      expected_values = [[i+j for j in range(devices_per_worker)]
                         for i in range(0, 20, devices_per_worker)]

      input_fn = self._input_fn_to_test_input_context(
          fn,
          expected_num_replicas_in_sync=3*devices_per_worker,
          expected_num_input_pipelines=3,
          expected_input_pipeline_id=task_id)
      self._test_input_fn_iterator(
          task_type,
          task_id,
          required_gpus,
          input_fn,
          expected_values,
          test_reinitialize=use_dataset,
          ignore_order=not use_dataset)

    self._run_between_graph_clients(_worker_fn, self._cluster_spec,
                                    required_gpus)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testUpdateConfigProto(self):
    strategy, _ = self._get_test_object(
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


class DistributedCollectiveAllReduceStrategyTestWithChief(
    CollectiveAllReduceStrategyTestBase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 3 workers and 1 chief."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=0, has_chief=True)

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testMinimizeLossGraph(self, required_gpus):
    self._run_between_graph_clients(self._test_minimize_loss_graph,
                                    self._cluster_spec, required_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testVariableInitialization(self, required_gpus):
    self._run_between_graph_clients(
        self._test_variable_initialization,
        self._cluster_spec,
        num_gpus=required_gpus)


class SingleWorkerCollectiveAllReduceStrategy(
    CollectiveAllReduceStrategyTestBase, strategy_test_lib.DistributionTestBase,
    strategy_test_lib.TwoDeviceDistributionTestBase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['eager']))
  def testStrategyInitializationError(self):
    with self.assertRaisesRegex(
        ValueError,
        'cluster_resolver and devices cannot be set at the same time'):
      _ = collective_all_reduce_strategy.CollectiveAllReduceExtended(
          container_strategy=None,
          cluster_resolver=multi_worker_test_base.create_in_process_cluster(
              num_workers=3, num_ps=0),
          communication_options=collective_util.Options(),
          devices=['GPU:0', 'GPU:1'])

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          required_gpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testMinimizeLoss(self, required_gpus, use_devices_arg):
    # Collective ops doesn't support strategy with one device.
    if context.executing_eagerly():
      strategy, _ = self._get_test_object(
          None, None, required_gpus, use_devices_arg=use_devices_arg)
      self._test_minimize_loss_eager(strategy)
    else:
      self._test_minimize_loss_graph(None, None, required_gpus)

  @combinations.generate(
      combinations.combine(
          mode=['eager'], required_gpus=[1, 2], use_devices_arg=[True, False]))
  def testNumReplicasInSync(self, required_gpus, use_devices_arg):
    strategy, _ = self._get_test_object(
        None, None, required_gpus, use_devices_arg=use_devices_arg)
    self.assertEqual(required_gpus, strategy.num_replicas_in_sync)

  @combinations.generate(
      combinations.combine(
          mode=['eager'],
          required_tpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testMinimizeLossTPU(self, required_tpus, use_devices_arg):
    strategy, _ = self._get_test_object(
        None, None, num_tpus=required_tpus, use_devices_arg=use_devices_arg)
    self._test_minimize_loss_eager(strategy)

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          required_gpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testCallAndMergeExceptions(self, required_gpus, use_devices_arg):
    strategy, _ = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    self._test_call_and_merge_exceptions(strategy)

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          required_gpus=2,
          use_dataset=[True, False],
          use_devices_arg=[True, False]))
  def testMakeInputFnIterator(self, required_gpus, use_dataset,
                              use_devices_arg):
    if use_dataset:
      fn = lambda: dataset_ops.Dataset.range(5 * required_gpus)
    else:
      def fn():
        dataset = dataset_ops.Dataset.range(5 * required_gpus)
        it = dataset_ops.make_one_shot_iterator(dataset)
        return it.get_next

    expected_values = [
        range(i, i + required_gpus) for i in range(0, 10, required_gpus)
    ]

    input_fn = self._input_fn_to_test_input_context(
        fn,
        expected_num_replicas_in_sync=required_gpus,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    self._test_input_fn_iterator(
        None,
        None,
        required_gpus,
        input_fn,
        expected_values,
        test_reinitialize=use_dataset,
        ignore_order=not use_dataset)

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          required_gpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testReduceToCpu(self, required_gpus, use_devices_arg):
    strategy, _ = self._get_test_object(
        None, None, required_gpus, use_devices_arg=use_devices_arg)
    with strategy.scope():
      result = strategy.extended.call_for_each_replica(_replica_id_f32)
      reduced = strategy.reduce(reduce_util.ReduceOp.SUM, result, axis=None)
      expected = sum(range(strategy.num_replicas_in_sync))
      self.assertEqual(expected, self.evaluate(reduced))

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceSum(self, required_gpus, use_devices_arg):
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_sum(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceSumGradients(self, required_gpus, use_devices_arg):
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_sum_gradients(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceSumGradientTape(self, required_gpus, use_devices_arg):
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_sum_gradient_tape(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceMean(self, required_gpus, use_devices_arg):
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_mean(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceMeanGradients(self, required_gpus, use_devices_arg):
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_mean_gradients(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceMeanGradientTape(self, required_gpus, use_devices_arg):
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_mean_gradient_tape(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testNumpyDataset(self, required_gpus, use_devices_arg):
    strategy, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    self._test_numpy_dataset(
        strategy, session=self.cached_session(target=target))

  @combinations.generate(
      combinations.combine(
          mode=['eager'], required_gpus=2, use_devices_arg=[True, False]))
  def testReplicateDataset(self, required_gpus, use_devices_arg):
    strategy, _ = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    dataset_fn = lambda: dataset_ops.Dataset.range(10)
    expected_values = [[i, i + 1] for i in range(0, 10, 2)]
    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=required_gpus,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    self._test_input_fn_iterable(strategy, input_fn, expected_values)

  @combinations.generate(
      combinations.combine(mode=['graph'], use_devices_arg=[True, False]))
  def testDeepCopy(self, use_devices_arg):
    distribution, _ = self._get_test_object(
        None, None, use_devices_arg=use_devices_arg)
    copy.deepcopy(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          required_gpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testSummaryForReplicaZeroOnly(self, required_gpus, use_devices_arg):
    strategy, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_summary_for_replica_zero_only(strategy)

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          required_gpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testTrainableVariables(self, required_gpus, use_devices_arg):
    strategy, _ = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    self._test_trainable_variable(strategy)


class LogicalDeviceTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['eager'], required_gpus=1))
  def testKeepLogicalDevice(self):
    gpus = tf_config.list_physical_devices('GPU')
    if len(gpus) > 1:
      self.skipTest('Skip logical device test on multi GPUs, since partial GPU '
                    'virtualization is not permitted.')
    # Cannot change logical device after the context initialization.
    context._reset_context()  # pylint: disable=protected-access
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=False, num_workers=1)
    resolver = cluster_resolver_lib.SimpleClusterResolver(
        cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
        task_type='worker',
        task_id=0)

    logical_gpus = len(gpus) * 2
    for i, device in enumerate(gpus):
      n = (i + 1) * logical_gpus // len(gpus) - i * logical_gpus // len(gpus)
      assert n > 0  # guaranteed if count >= len(devices)
      configs = []
      for ordinal in range(n):
        config = context.LogicalDeviceConfiguration(
            memory_limit=64,
            experimental_device_ordinal=ordinal)
        configs.append(config)

      tf_config.set_logical_device_configuration(device, configs)

    collective_all_reduce_strategy.CollectiveAllReduceStrategy(
        cluster_resolver=resolver)
    # Since we create two logical GPUs out of the last GPU, there should be one
    # more logical GPUs than physical GPUs.
    self.assertLen(tf_config.list_logical_devices('GPU'), logical_gpus)
    context._reset_context()  # pylint: disable=protected-access


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
        ],
        mode=['eager']))
class CollectiveAllReduceStrategyV2Test(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if context.context().list_physical_devices('TPU'):
      self.skipTest('Test not supported on TPUs')

  def test_replica_id_in_sync_group(self, strategy):

    def replica_fn():
      replica_ctx = distribute_lib.get_replica_context()
      return replica_ctx.replica_id_in_sync_group, replica_ctx._replica_id

    results = test_util.gather(strategy, strategy.run(replica_fn))
    self.assertAllEqual(list(range(strategy.extended._num_replicas_in_sync)),
                        results[0].numpy())
    self.assertAllEqual(
        list(range(len(strategy.extended.worker_devices))) *
        strategy.extended._num_workers, results[1].numpy())

  def test_deep_copy_not_allowed(self, strategy):
    # Check health is disabled in tests by default. We need to enable it for
    # this test to simulate the real world.
    strategy.extended._start_check_health_thread()
    try:
      with self.assertRaisesRegex(ValueError, 'cannot be deep copied'):
        copy.deepcopy(strategy)
      with self.assertRaisesRegex(ValueError, 'cannot be deep copied'):
        with ops.Graph().as_default():
          copy.deepcopy(strategy)
    finally:
      strategy.extended._stop_check_health_thread()


class ExperimentalCompatibilityTest(test.TestCase):

  def testIsInstance(self):
    # It's not uncommon for people to special case MultiWorkerMirroredStrategy,
    # so we need to make sure isinstance check works for combinations between
    # the experimental and non-experimental endpoints.
    strategy = CollectiveAllReduceStrategy()
    experimental_strategy = _CollectiveAllReduceStrategyExperimental()
    self.assertIsInstance(strategy, CollectiveAllReduceStrategy)
    self.assertIsInstance(strategy, _CollectiveAllReduceStrategyExperimental)
    self.assertIsInstance(experimental_strategy, CollectiveAllReduceStrategy)
    self.assertIsInstance(experimental_strategy,
                          _CollectiveAllReduceStrategyExperimental)

  def testName(self):
    self.assertEqual(CollectiveAllReduceStrategy.__name__,
                     'CollectiveAllReduceStrategy')
    self.assertEqual(_CollectiveAllReduceStrategyExperimental.__name__,
                     'CollectiveAllReduceStrategy')


def _replica_id_f32():
  return math_ops.cast(
      distribute_lib.get_replica_context()
      .replica_id_in_sync_group, dtypes.float32)


if __name__ == '__main__':
  # TODO(b/172304955): enable logical devices.
  test_util.main(config_logical_devices=False)
