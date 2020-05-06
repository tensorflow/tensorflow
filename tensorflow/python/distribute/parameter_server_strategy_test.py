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
"""Tests for ParameterServerStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import threading

from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import training_util

CHIEF = run_config.TaskType.CHIEF
WORKER = run_config.TaskType.WORKER
PS = run_config.TaskType.PS


def _get_replica_id_integer():
  replica_id = ds_context.get_replica_context().replica_id_in_sync_group
  if isinstance(replica_id, ops.Tensor):
    replica_id = tensor_util.constant_value(replica_id)
  return replica_id


def create_test_objects(cluster_spec=None,
                        task_type=None,
                        task_id=None,
                        num_gpus=None,
                        sess_config=None):
  sess_config = sess_config or config_pb2.ConfigProto()
  if num_gpus is None:
    num_gpus = context.num_gpus()
  if cluster_spec and task_type and task_id is not None:
    cluster_resolver = SimpleClusterResolver(
        cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
        task_type=task_type,
        task_id=task_id,
        num_accelerators={'GPU': num_gpus})
    distribution = parameter_server_strategy.ParameterServerStrategy(
        cluster_resolver)
    target = 'grpc://' + cluster_spec[WORKER][task_id]
  else:
    distribution = (
        central_storage_strategy.CentralStorageStrategy._from_num_gpus(num_gpus)
    )
    target = ''

  sess_config = copy.deepcopy(sess_config)
  sess_config = distribution.update_config_proto(sess_config)

  return distribution, target, sess_config


class ParameterServerStrategyTestBase(
    multi_worker_test_base.MultiWorkerTestBase):

  def setUp(self):
    self._result = 0
    self._lock = threading.Lock()
    self._init_condition = threading.Condition()
    self._init_reached = 0
    self._finish_condition = threading.Condition()
    self._finish_reached = 0
    self._sess_config = config_pb2.ConfigProto(allow_soft_placement=True)
    super(ParameterServerStrategyTestBase, self).setUp()

  def _get_test_objects(self, task_type, task_id, num_gpus):
    return create_test_objects(
        cluster_spec=self._cluster_spec,
        task_type=task_type,
        task_id=task_id,
        num_gpus=num_gpus,
        sess_config=self._sess_config)

  def _test_device_assignment_distributed(self, task_type, task_id, num_gpus):
    worker_device = '/job:%s/replica:0/task:%d' % (task_type, task_id)
    d, _, sess_config = self._get_test_objects(task_type, task_id, num_gpus)
    with ops.Graph().as_default(), \
         self.cached_session(target=self._default_target,
                             config=sess_config) as sess, \
         d.scope():

      # Define a variable outside the call_for_each_replica scope.
      n = variable_scope.get_variable('n', initializer=10.0)
      self.assertEqual(n.device, '/job:ps/task:0')

      def model_fn():
        if num_gpus == 0:
          last_part_device = 'device:CPU:0'
        else:
          replica_id = _get_replica_id_integer()
          last_part_device = ('device:GPU:%d' % replica_id)

        a = constant_op.constant(1.0)
        b = constant_op.constant(2.0)
        c = a + b
        self.assertEqual(a.device, worker_device + '/' + last_part_device)
        self.assertEqual(b.device, worker_device + '/' + last_part_device)
        self.assertEqual(c.device, worker_device + '/' + last_part_device)

        # The device scope is ignored for variables but not for normal ops.
        with ops.device('/job:worker/task:0'):
          x = variable_scope.get_variable(
              'x', initializer=10.0,
              aggregation=variable_scope.VariableAggregation.SUM)
          x_add = x.assign_add(c)
          e = a + c
        # The variable x is on the task 1 since the device_function has been
        # called once before the model_fn.
        self.assertEqual(x.device, '/job:ps/task:1')
        self.assertEqual(x_add.device, x.device)
        self.assertEqual(e.device,
                         '/job:worker/replica:0/task:0/%s' % last_part_device)

        # The colocate_vars_with can override the distribution's device.
        with d.extended.colocate_vars_with(x):
          y = variable_scope.get_variable(
              'y', initializer=20.0,
              aggregation=variable_scope.VariableAggregation.SUM)
        # We add an identity here to avoid complaints about summing
        # non-distributed values.
        y_add = y.assign_add(array_ops.identity(x_add))
        self.assertEqual(y.device, '/job:ps/task:1')
        self.assertEqual(y_add.device, y.device)
        self.assertEqual(y.device, x.device)

        z = variable_scope.get_variable(
            'z', initializer=10.0,
            aggregation=variable_scope.VariableAggregation.SUM)
        self.assertEqual(z.device, '/job:ps/task:0')
        self.assertNotEqual(z.device, x.device)

        with ops.control_dependencies([y_add]):
          # We add an identity here to avoid complaints about summing
          # non-distributed values.
          z_add = z.assign_add(array_ops.identity(y))
        with ops.control_dependencies([z_add]):
          f = z + c
        self.assertEqual(f.device, worker_device + '/' + last_part_device)

        # The device scope would merge with the default worker device.
        with ops.device('/CPU:1'):
          g = e + 1.0
        self.assertEqual(g.device, worker_device + '/device:CPU:1')

        # Ths ops.colocate_with will be ignored when defining a variable but not
        # for a normal tensor.
        with ops.colocate_with(x):
          u = variable_scope.get_variable('u', initializer=30.0)
          v = variable_scope.get_variable('v', initializer=30.0)
          h = f + 1.0
        self.assertIn('/job:ps/', u.device)
        self.assertIn('/job:ps/', v.device)
        # u and v are on different parameter servers.
        self.assertTrue(u.device != x.device or v.device != x.device)
        self.assertTrue(u.device == x.device or v.device == x.device)
        # Here h is not on one worker. Note h.device is canonical while x.device
        # is not but.
        self.assertIn('/job:ps/', h.device)
        return y_add, z_add, f

      y, z, f = d.extended.call_for_each_replica(model_fn)
      self.assertNotEqual(y, None)
      self.assertNotEqual(z, None)
      self.assertNotEqual(f, None)

      if context.num_gpus() >= 1 and num_gpus <= 1:
        variables.global_variables_initializer().run()
        y_val, z_val, f_val = sess.run([y, z, f])
        self.assertEqual(y_val, 33.0)
        self.assertEqual(z_val, 43.0)
        self.assertEqual(f_val, 46.0)

  def _test_device_assignment_distributed_enable_partitioner(
      self, task_type, task_id, num_gpus):
    d, _, sess_config = self._get_test_objects(task_type, task_id, num_gpus)
    num_shards = len(d.extended.parameter_devices)
    partitioner = partitioned_variables.fixed_size_partitioner(num_shards)
    with ops.Graph().as_default(), \
         self.cached_session(target=self._default_target,
                             config=sess_config) as sess, \
         d.scope():

      n = variable_scope.get_variable(
          'n',
          initializer=constant_op.constant([10.0, 20.0]),
          aggregation=variable_scope.VariableAggregation.SUM,
          partitioner=partitioner)

      for part_id, var in enumerate(n):
        self.assertEqual(var.device, '/job:ps/task:%d' % part_id)

      def model_fn():
        a = constant_op.constant([3.0, 5.0])
        # The device scope is ignored for variables but not for normal ops.
        with ops.device('/job:worker/task:0'):
          x = variable_scope.get_variable(
              'x',
              initializer=constant_op.constant([10.0, 20.0]),
              aggregation=variable_scope.VariableAggregation.SUM,
              partitioner=partitioner)
          x_add = x.assign_add(a, name='x_add')
        # The variable x is on the task 1 since the device_function has been
        # called once before the model_fn.
        for part_id, var in enumerate(x):
          self.assertEqual(var.device, '/job:ps/task:%d' % part_id)
          self.assertEqual(var.device, x_add[part_id].device)

        return x_add

      x = d.extended.call_for_each_replica(model_fn)

      if context.num_gpus() >= 1:
        variables.global_variables_initializer().run()
        x_val = sess.run(x)
        if num_gpus < 1:
          self.assertEqual(x_val, [13.0, 25.0])
        else:
          x_expect = [10.0 + 3 * num_gpus, 20.0 + 5 * num_gpus]
          self.assertEqual(x_val, x_expect)

  def _test_device_assignment_local(self,
                                    d,
                                    compute_device='CPU',
                                    variable_device='CPU',
                                    num_gpus=0):
    with ops.Graph().as_default(), \
         self.cached_session(target=self._default_target,
                             config=self._sess_config) as sess, \
         d.scope():

      def model_fn():
        if 'CPU' in compute_device:
          replica_compute_device = '/device:CPU:0'
        else:
          replica_id = _get_replica_id_integer()
          replica_compute_device = ('/device:GPU:%d' % replica_id)
        replica_compute_device = device_util.canonicalize(
            replica_compute_device)

        if 'CPU' in variable_device:
          replica_variable_device = '/device:CPU:0'
        else:
          replica_id = _get_replica_id_integer()
          replica_variable_device = ('/device:GPU:%d' % replica_id)
        replica_variable_device = device_util.canonicalize(
            replica_variable_device)

        a = constant_op.constant(1.0)
        b = constant_op.constant(2.0)
        c = a + b
        self.assertEqual(a.device, replica_compute_device)
        self.assertEqual(b.device, replica_compute_device)
        self.assertEqual(c.device, replica_compute_device)

        # The device scope is ignored for variables but not for normal ops.
        with ops.device('/device:GPU:2'):
          x = variable_scope.get_variable(
              'x', initializer=10.0,
              aggregation=variable_scope.VariableAggregation.SUM)
          x_add = x.assign_add(c)
          e = a + c
        self.assertEqual(
            device_util.canonicalize(x.device), replica_variable_device)
        self.assertEqual(x_add.device, x.device)
        self.assertEqual(e.device, device_util.canonicalize('/device:GPU:2'))

        # The colocate_vars_with can override the distribution's device.
        with d.extended.colocate_vars_with(x):
          y = variable_scope.get_variable(
              'y', initializer=20.0,
              aggregation=variable_scope.VariableAggregation.SUM)
        # We add an identity here to avoid complaints about summing
        # non-distributed values.
        y_add = y.assign_add(array_ops.identity(x_add))
        self.assertEqual(
            device_util.canonicalize(y.device), replica_variable_device)
        self.assertEqual(y_add.device, y.device)
        self.assertEqual(y.device, x.device)

        z = variable_scope.get_variable(
            'z', initializer=10.0,
            aggregation=variable_scope.VariableAggregation.SUM)
        self.assertEqual(
            device_util.canonicalize(z.device), replica_variable_device)

        with ops.control_dependencies([y_add]):
          # We add an identity here to avoid complaints about summing
          # non-distributed values.
          z_add = z.assign_add(array_ops.identity(y))
        with ops.control_dependencies([z_add]):
          f = z + c
        self.assertEqual(f.device, replica_compute_device)

        # The device scope would merge with the default worker device.
        with ops.device('/CPU:1'):
          g = e + 1.0
        self.assertEqual(g.device, device_util.canonicalize('/device:CPU:1'))

        # Ths ops.colocate_with will be ignored when defining a variable but not
        # for a normal tensor.
        with ops.colocate_with(x):
          u = variable_scope.get_variable('u', initializer=30.0)
          h = f + 1.0
        self.assertEqual(
            device_util.canonicalize(u.device), replica_variable_device)
        self.assertEqual(
            device_util.canonicalize(x.device),
            device_util.canonicalize(h.device))
        return y_add, z_add, f

      y, z, f = d.extended.call_for_each_replica(model_fn)
      self.assertNotEqual(y, None)
      self.assertNotEqual(z, None)
      self.assertNotEqual(f, None)

      if context.num_gpus() >= 1 and num_gpus <= 1:
        variables.global_variables_initializer().run()
        y_val, z_val, f_val = sess.run([y, z, f])
        self.assertEqual(y_val, 33.0)
        self.assertEqual(z_val, 43.0)
        self.assertEqual(f_val, 46.0)

  def _test_simple_increment(self, task_type, task_id, num_gpus):
    d, master_target, sess_config = self._get_test_objects(
        task_type, task_id, num_gpus)
    if d.extended._cluster_spec:
      num_workers = len(d.extended._cluster_spec.as_dict().get(WORKER))
      if 'chief' in d.extended._cluster_spec.as_dict():
        num_workers += 1
    else:
      num_workers = 1
    with ops.Graph().as_default(), \
         self.cached_session(target=master_target,
                             config=sess_config) as sess, \
         d.scope():

      def model_fn():
        x = variable_scope.get_variable(
            'x', initializer=10.0,
            aggregation=variable_scope.VariableAggregation.SUM)
        y = variable_scope.get_variable(
            'y', initializer=20.0,
            aggregation=variable_scope.VariableAggregation.SUM)
        z = variable_scope.get_variable(
            'z', initializer=30.0,
            aggregation=variable_scope.VariableAggregation.ONLY_FIRST_REPLICA)

        # We explicitly make a constant tensor here to avoid complaints about
        # summing non-distributed values.
        one = constant_op.constant(1.0)
        x_add = x.assign_add(one, use_locking=True)
        y_add = y.assign_add(one, use_locking=True)
        z_add = z.assign_add(one, use_locking=True)

        train_op = control_flow_ops.group(x_add, y_add, z_add)
        return x, y, z, train_op

      x, y, z, train_op = d.extended.call_for_each_replica(model_fn)
      train_op = d.group(train_op)

      if task_id == 0:
        variables.global_variables_initializer().run()

      # Workers waiting for chief worker's initializing variables.
      self._init_condition.acquire()
      self._init_reached += 1
      while self._init_reached != num_workers:
        self._init_condition.wait()
      self._init_condition.notify_all()
      self._init_condition.release()

      sess.run(train_op)

      # Wait for other workers to finish training.
      self._finish_condition.acquire()
      self._finish_reached += 1
      while self._finish_reached != num_workers:
        self._finish_condition.wait()
      self._finish_condition.notify_all()
      self._finish_condition.release()

      x_val, y_val, z_val = sess.run([x, y, z])
      self.assertEqual(x_val, 10.0 + 1.0 * num_workers * d.num_replicas_in_sync)
      self.assertEqual(y_val, 20.0 + 1.0 * num_workers * d.num_replicas_in_sync)
      self.assertEqual(z_val, 30.0 + 1.0 * num_workers)

  def _test_minimize_loss_graph(self, task_type, task_id, num_gpus):
    d, master_target, sess_config = self._get_test_objects(
        task_type, task_id, num_gpus)
    if task_type:
      # Multi-worker
      assert hasattr(d.extended, '_cluster_spec') and d.extended._cluster_spec
      num_workers = len(d.extended._cluster_spec.as_dict().get(WORKER))
      if CHIEF in d.extended._cluster_spec.as_dict():
        num_workers += 1
    else:
      # local
      num_workers = 1

    with ops.Graph().as_default(), \
         self.cached_session(target=master_target,
                             config=sess_config) as sess, \
         d.scope():
      l = core.Dense(1, use_bias=False)

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
        g_v = d.extended.call_for_each_replica(grad_fn, args=(one,))
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

      if (not task_type or
          multi_worker_util.is_chief(
              d.extended._cluster_spec, task_type, task_id)):
        variables.global_variables_initializer().run()

      # Workers waiting for chief worker's initializing variables.
      self._init_condition.acquire()
      self._init_reached += 1
      while self._init_reached != num_workers:
        self._init_condition.wait()
      self._init_condition.notify_all()
      self._init_condition.release()

      for i in range(10):
        b, a = sess.run((before_out, after_out))
        if i == 0:
          before, = b
        after, = a

      error_before = abs(before - 1)
      error_after = abs(after - 1)
      # Error should go down
      self.assertLess(error_after, error_before)

  def _test_input_fn_iterator(self,
                              task_type,
                              task_id,
                              num_gpus,
                              input_fn,
                              expected_values,
                              test_reinitialize=True,
                              ignore_order=False):
    distribution, master_target, config = self._get_test_objects(
        task_type, task_id, num_gpus)
    devices = distribution.extended.worker_devices

    with ops.Graph().as_default(), \
         self.cached_session(config=config,
                             target=master_target) as sess:
      iterator = distribution.make_input_fn_iterator(input_fn)
      sess.run(iterator.initializer)

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
        sess.run(iterator.initializer)

        for expected_value in expected_values:
          next_element = iterator.get_next()
          computed_value = sess.run([values.select_replica(r, next_element)
                                     for r in range(len(devices))])
          if ignore_order:
            self.assertCountEqual(expected_value, computed_value)
          else:
            self.assertEqual(expected_value, computed_value)


class ParameterServerStrategyTest(
    ParameterServerStrategyTestBase,
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.TwoDeviceDistributionTestBase,
    parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=2)
    cls._default_target = 'grpc://' + cls._cluster_spec[WORKER][0]

  @combinations.generate(combinations.combine(mode=['graph']))
  def test_num_replicas_in_sync(self):
    strategy, _, _ = create_test_objects(num_gpus=2)
    # All the devices on a given worker are in sync which in this case is the
    # number of gpus on each worker.
    self.assertEqual(2, strategy.num_replicas_in_sync)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testDeviceAssignmentLocalCPU(self):
    strategy, _, _ = create_test_objects(num_gpus=0)
    self._test_device_assignment_local(
        strategy, compute_device='CPU', variable_device='CPU', num_gpus=0)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testDeviceAssignmentLocalOneGPU(self):
    strategy, _, _ = create_test_objects(num_gpus=1)
    self._test_device_assignment_local(
        strategy, compute_device='GPU', variable_device='GPU', num_gpus=1)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testDeviceAssignmentLocalTwoGPUs(self):
    strategy, _, _ = create_test_objects(num_gpus=2)
    self._test_device_assignment_local(
        strategy, compute_device='GPU', variable_device='CPU', num_gpus=2)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2]))
  def testDeviceAssignmentDistributed(self, num_gpus):
    self._test_device_assignment_distributed('worker', 1, num_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2]))
  def testDeviceAssignmentDistributedEnablePartitioner(self, num_gpus):
    self._test_device_assignment_distributed_enable_partitioner(
        'worker', 1, num_gpus)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testSimpleBetweenGraph(self):
    self._run_between_graph_clients(self._test_simple_increment,
                                    self._cluster_spec, context.num_gpus())

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testLocalSimpleIncrement(self, required_gpus):
    self._test_simple_increment(None, 0, required_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testMinimizeLossGraphDistributed(self, required_gpus):
    self._run_between_graph_clients(self._test_minimize_loss_graph,
                                    self._cluster_spec, required_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testMinimizeLossGraphLocal(self, required_gpus):
    self._test_minimize_loss_graph(None, None, required_gpus)

  # TODO(priyag): Refactor this and other multi worker tests.
  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=[1, 2], use_dataset=[True, False]))
  def testMakeInputFnIteratorDistributed(self, required_gpus, use_dataset):
    if use_dataset:
      fn = lambda: dataset_ops.Dataset.range(100)
    else:
      def fn():
        dataset = dataset_ops.Dataset.range(100)
        it = dataset_ops.make_one_shot_iterator(dataset)
        return it.get_next

    expected_values = [[i + j
                        for j in range(required_gpus)]
                       for i in range(0, 100, required_gpus)]

    input_fn = self._input_fn_to_test_input_context(
        fn,
        expected_num_replicas_in_sync=required_gpus,
        expected_num_input_pipelines=3,
        expected_input_pipeline_id=1)  # because task_id = 1
    self._test_input_fn_iterator(
        'worker',
        1,
        required_gpus,
        input_fn,
        expected_values,
        test_reinitialize=use_dataset,
        ignore_order=not use_dataset)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=[1, 2], use_dataset=[True, False]))
  def testMakeInputFnIteratorLocal(self, required_gpus, use_dataset):
    if use_dataset:
      fn = lambda: dataset_ops.Dataset.range(100)
    else:

      def fn():
        dataset = dataset_ops.Dataset.range(100)
        it = dataset_ops.make_one_shot_iterator(dataset)
        return it.get_next

    expected_values = [[i + j
                        for j in range(required_gpus)]
                       for i in range(0, 100, required_gpus)]

    input_fn = self._input_fn_to_test_input_context(
        fn,
        expected_num_replicas_in_sync=required_gpus,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)  # only one worker and pipeline for local.
    self._test_input_fn_iterator(
        None,
        None,
        required_gpus,
        input_fn,
        expected_values,
        test_reinitialize=use_dataset,
        ignore_order=not use_dataset)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testGlobalStepUpdate(self):
    strategy, _, _ = create_test_objects()
    self._test_global_step_update(strategy)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testUpdateConfigProtoMultiWorker(self):
    strategy, _, _ = create_test_objects(
        cluster_spec=self._cluster_spec,
        task_type='worker',
        task_id=1,
        num_gpus=2)

    config_proto = config_pb2.ConfigProto(device_filters=['to_be_overridden'])

    new_config = strategy.update_config_proto(config_proto)

    # Verify device filters.
    self.assertEqual(['/job:worker/task:1', '/job:ps'],
                     new_config.device_filters)

    # Verify isolate_session_state
    self.assertFalse(new_config.isolate_session_state)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testUpdateConfigProtoLocal(self):
    strategy, _, _ = create_test_objects(num_gpus=2)

    config_proto = config_pb2.ConfigProto()
    new_config = strategy.update_config_proto(config_proto)

    # Verify isolate_session_state
    self.assertTrue(new_config.isolate_session_state)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInMultiWorkerMode(self):
    strategy, _, _ = create_test_objects(
        cluster_spec=self._cluster_spec,
        task_type='worker',
        task_id=1,
        num_gpus=0)
    self.assertTrue(strategy.extended._in_multi_worker_mode())

  @combinations.generate(combinations.combine(mode=['eager']))
  def testEagerCustomTrainingUnimplementedError(self):
    cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=2)
    cluster_resolver = SimpleClusterResolver(
        cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
        task_type='worker',
        task_id=1,
        num_accelerators={'GPU': 0})
    strategy = parameter_server_strategy.ParameterServerStrategy(
        cluster_resolver)
    dataset = dataset_ops.DatasetV2.from_tensor_slices([5., 6., 7., 8.])

    def train_step(data):
      return math_ops.square(data)

    self.assertRaisesRegex(NotImplementedError, 'ParameterServerStrategy*',
                           strategy.experimental_distribute_dataset,
                           dataset.batch(2))

    self.assertRaisesRegex(
        NotImplementedError, 'ParameterServerStrategy*',
        strategy.experimental_distribute_datasets_from_function,
        lambda _: dataset)

    self.assertRaisesRegex(NotImplementedError, 'ParameterServerStrategy*',
                           strategy.scope)

    self.assertRaisesRegex(NotImplementedError, 'ParameterServerStrategy*',
                           strategy.run, train_step)


class ParameterServerStrategyWithChiefTest(ParameterServerStrategyTestBase,
                                           parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=2, has_chief=True)
    cls._default_target = 'grpc://' + cls._cluster_spec[CHIEF][0]

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testSimpleBetweenGraph(self, required_gpus):
    self._run_between_graph_clients(self._test_simple_increment,
                                    self._cluster_spec, required_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], num_gpus=[0, 1, 2]))
  def testMinimizeLossGraph(self, num_gpus):
    self._run_between_graph_clients(self._test_minimize_loss_graph,
                                    self._cluster_spec, num_gpus)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testGlobalStepIsWrappedOnTwoGPUs(self):
    strategy, _, _ = create_test_objects(num_gpus=2)
    with ops.Graph().as_default(), strategy.scope():
      created_step = training_util.create_global_step()
      get_step = training_util.get_global_step()
      self.assertEqual(created_step, get_step,
                       msg=('created_step %s type %s vs. get_step %s type %s' %
                            (id(created_step), created_step.__class__.__name__,
                             id(get_step), get_step.__class__.__name__)))
      self.assertIs(values.AggregatingVariable, type(created_step))
      self.assertIs(values.AggregatingVariable, type(get_step))
      self.assertIs(strategy, created_step.distribute_strategy)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testGlobalStepIsNotWrappedOnOneGPU(self):
    strategy, _, _ = create_test_objects(num_gpus=1)
    with ops.Graph().as_default(), strategy.scope():
      created_step = training_util.create_global_step()
      get_step = training_util.get_global_step()
      self.assertEqual(created_step, get_step,
                       msg=('created_step %s type %s vs. get_step %s type %s' %
                            (id(created_step), created_step.__class__.__name__,
                             id(get_step), get_step.__class__.__name__)))
      self.assertIs(resource_variable_ops.ResourceVariable, type(created_step))
      self.assertIs(resource_variable_ops.ResourceVariable, type(get_step))
      # All variables have an _distribute_strategy parameter. Only variable
      # subclasses in distribution strategy expose it publicly.
      self.assertFalse(hasattr(strategy, 'distribute_strategy'))
      self.assertIs(strategy, created_step._distribute_strategy)

  @combinations.generate(combinations.combine(mode=['graph'], required_gpus=2))
  def testValueContainer(self):
    strategy, _, _ = create_test_objects(num_gpus=2)
    with ops.Graph().as_default(), strategy.scope():

      def f():
        with backprop.GradientTape() as tape:
          v = variable_scope.get_variable('v', initializer=10.0)
          _ = v * v
        v, = tape.watched_variables()
        w = strategy.extended.value_container(v)
        self.assertIs(values.AggregatingVariable, type(w))

      strategy.extended.call_for_each_replica(f)


class CentralStorageStrategyTest(strategy_test_lib.DistributionTestBase,
                                 parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager'],
                                              required_gpus=2))
  def testNumpyDataset(self):
    strategy, _, _ = create_test_objects(num_gpus=2)
    self._test_numpy_dataset(strategy)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInMultiWorkerMode(self):
    strategy, _, _ = create_test_objects(num_gpus=0)
    self.assertFalse(strategy.extended._in_multi_worker_mode())


if __name__ == '__main__':
  test.main()
