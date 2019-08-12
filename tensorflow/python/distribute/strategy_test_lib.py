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
"""Library for testing DistributionStrategy descendants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import core
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_util
from tensorflow.python.util import nest


class _TestException(Exception):
  pass


# May be the argument to either distribution.extended.call_for_each_replica() or
# get_replica_context().merge_call()
def _raise_exception_fn(_=None):
  raise _TestException()


# Must be the argument to a distribution.extended.call_for_each_replica() call,
# calls a get_replica_context().merge_call() that raises an exception.
def _merge_raises_fn():
  ds_context.get_replica_context().merge_call(_raise_exception_fn)


# Must be the argument to a get_replica_context().merge_call() call, calls
# dist.extended.call_for_each_replica() with a function that raises an
# exception.
def _call_raises_fn(dist):
  dist.extended.call_for_each_replica(_raise_exception_fn)


# Must be the argument to a distribution.extended.call_for_each_replica() call,
# calls a get_replica_context().merge_call() that calls a
# call_for_each_replica() that raises an exception.
def _merge_call_raises_fn():
  ds_context.get_replica_context().merge_call(_call_raises_fn)


# Must be the argument to a get_replica_context().merge_call() call, calls
# dist.extended.call_for_each_replica() with a function that calls a
# get_replica_context().merge_call() that raises an exception.
def _call_merge_raises_fn(dist):
  dist.extended.call_for_each_replica(_merge_raises_fn)


# Must be the argument to a distribution.extended.call_for_each_replica() call,
# calls a get_replica_context().merge_call() that calls a
# call_for_each_replica() that calls a get_replica_context().merge_call() that
# raises an exception.
def _merge_call_merge_raises_fn():
  ds_context.get_replica_context().merge_call(_call_merge_raises_fn)


def _events_from_logdir(test_case, logdir):
  """Reads summary events from log directory."""
  test_case.assertTrue(gfile.Exists(logdir))
  files = gfile.ListDirectory(logdir)
  test_case.assertLen(files, 1)
  records = list(tf_record.tf_record_iterator(os.path.join(logdir, files[0])))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


class DistributionTestBase(test.TestCase):
  """Some tests that should work with any DistributionStrategy."""

  def _test_minimize_loss_eager(self, d):
    with d.scope():
      l = core.Dense(1, use_bias=False)

      def loss(x):
        y = array_ops.reshape(l(x), []) - constant_op.constant(1.)
        return y * y
      # TODO(isaprykin): Extract implicit_grad+get_filtered_grad_fn into a
      # common `implicit_grad` function and put it in DistributionStrategy.
      grad_fn = backprop.implicit_grad(loss)
      grad_fn = optimizer.get_filtered_grad_fn(grad_fn)

      def update(v, g):
        return v.assign_sub(0.2 * g)

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
          # control_dependencies irrelevant but harmless in eager execution
          with ops.control_dependencies([fetched]):
            g = d.extended.reduce_to(
                reduce_util.ReduceOp.SUM, g, destinations=v)
            with ops.control_dependencies(
                d.extended.update(v, update, args=(g,), group=False)):
              after_list.append(d.extended.read_var(v))
        return before_list, after_list

      for i in range(10):
        b, a = step()
        if i == 0:
          before, = b  # pylint: disable=unbalanced-tuple-unpacking
        after, = a  # pylint: disable=unbalanced-tuple-unpacking

      error_before = abs(before.numpy() - 1)
      error_after = abs(after.numpy() - 1)
      # Error should go down
      self.assertLess(error_after, error_before)

  def _test_minimize_loss_graph(self,
                                d,
                                soft_placement=False,
                                learning_rate=0.2):
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = soft_placement
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with context.graph_mode(), \
         ops.Graph().as_default(), \
         self.cached_session(config=config) as sess, \
         d.scope():
      l = core.Dense(1, use_bias=False)

      def loss(x):
        y = array_ops.reshape(l(x), []) - constant_op.constant(1.)
        return y * y

      grad_fn = backprop.implicit_grad(loss)

      def update(v, g):
        return v.assign_sub(learning_rate * g)

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
            g = d.extended.reduce_to(
                reduce_util.ReduceOp.SUM, g, destinations=v)
            with ops.control_dependencies(
                d.extended.update(v, update, args=(g,), group=False)):
              after_list.append(d.extended.read_var(v))
        return before_list, after_list

      before_out, after_out = step()
      variables.global_variables_initializer().run()
      for i in range(10):
        b, a = sess.run((before_out, after_out))
        if i == 0:
          before, = b
        after, = a

      error_before = abs(before - 1)
      error_after = abs(after - 1)
      # Error should go down
      self.assertLess(error_after, error_before)

  def _test_summary_for_replica_zero_only(self, d):
    logdir = tempfile.mkdtemp()

    def run_fn():
      """Function executed for each replica."""
      with summary_writer.as_default():
        replica_id = ds_context.get_replica_context().replica_id_in_sync_group
        return summary_ops.write("a", replica_id)

    with self.cached_session() as sess, d.scope(), \
        summary_ops.always_record_summaries():
      # We need global_step because summary writing op *always* has global_step
      # as input, even when we always record summary or never record summary.
      global_step = training_util.get_or_create_global_step()
      if not context.executing_eagerly():
        # When executing eagerly, variables are initialized immediately after
        # creation, and its initializer will be None.
        global_step.initializer.run()
      summary_ops.set_step(0)
      summary_writer = summary_ops.create_file_writer(logdir)
      output = d.extended.call_for_each_replica(run_fn)
      unwrapped = d.unwrap(output)
      if not context.executing_eagerly():
        sess.run(summary_writer.init())
        sess.run(unwrapped)
        sess.run(summary_writer.close())

      events = _events_from_logdir(self, logdir)
      # There will be 2 entries: 1 summary file header entry, and 1 entry
      # written by replica 0.
      self.assertLen(events, 2)
      self.assertEqual(events[1].summary.value[0].tag, "a")
      self.assertEqual(events[1].summary.value[0].simple_value, 0.0)

  def _test_replica_id(self, d):
    with d.scope():
      expected_devices = [False] * len(d.extended.worker_devices)

      def mark_devices_fn():
        replica_id = self.evaluate(
            ds_context.get_replica_context().replica_id_in_sync_group)
        self.assertLess(replica_id, len(d.extended.worker_devices))
        self.assertFalse(expected_devices[replica_id])
        expected_devices[replica_id] = True

      d.extended.call_for_each_replica(mark_devices_fn)
      self.assertAllEqual(expected_devices,
                          [True] * len(d.extended.worker_devices))

  def _test_call_and_merge_exceptions(self, dist):
    with dist.scope():
      with self.assertRaises(_TestException):
        dist.extended.call_for_each_replica(_raise_exception_fn)
      with self.assertRaises(_TestException):
        dist.extended.call_for_each_replica(_merge_raises_fn)
      with self.assertRaises(_TestException):
        dist.extended.call_for_each_replica(_merge_call_raises_fn)
      with self.assertRaises(_TestException):
        dist.extended.call_for_each_replica(_merge_call_merge_raises_fn)

  def _input_fn_to_test_input_context(self, dataset_or_callable_fn,
                                      expected_num_replicas_in_sync,
                                      expected_num_input_pipelines,
                                      expected_input_pipeline_id):
    # Use a list of one element as counter so that it can be captured by the
    # `_input_fn`. This counter is incremented by 1 each time an input_fn is
    # called. We use this counter to check whether the `input_pipeline_id`
    # matches the counter in the in-graph replication.
    worker_id_counter = [0]

    def _input_fn(input_context):
      """Input fn for testing."""
      self.assertIsNotNone(input_context)
      self.assertEqual(expected_num_replicas_in_sync,
                       input_context.num_replicas_in_sync)
      self.assertEqual(expected_num_input_pipelines,
                       input_context.num_input_pipelines)
      if expected_input_pipeline_id is not None:
        self.assertEqual(expected_input_pipeline_id,
                         input_context.input_pipeline_id)
      else:
        self.assertEqual(worker_id_counter[0], input_context.input_pipeline_id)
        worker_id_counter[0] += 1

      return dataset_or_callable_fn()

    return _input_fn

  def _test_input_fn_iterable(
      self, strategy, input_fn, expected_values, ignore_order=False):
    assert_same = self.assertCountEqual if ignore_order else self.assertEqual

    iterable = strategy.experimental_distribute_datasets_from_function(input_fn)
    if context.executing_eagerly():
      iterator = iter(iterable)

      for expected_value in expected_values:
        computed_value = self.evaluate(
            list(strategy.experimental_local_results(next(iterator))))
        assert_same(expected_value, computed_value)

      with self.assertRaises(StopIteration):
        self.evaluate(strategy.experimental_local_results(next(iterator)))

      # After re-initializing the iterator, should be able to iterate again.
      iterator = iter(iterable)

      for expected_value in expected_values:
        computed_value = self.evaluate(
            list(strategy.experimental_local_results(next(iterator))))
        assert_same(expected_value, computed_value)
    else:
      iterator = dataset_ops.make_initializable_iterator(iterable)
      self._test_input_fn_iterator(iterator, strategy.extended.worker_devices,
                                   expected_values, test_reinitialize=True,
                                   ignore_order=ignore_order)

  def _test_input_fn_iterator(self,
                              iterator,
                              devices,
                              expected_values,
                              sess=None,
                              test_reinitialize=True,
                              ignore_order=False):
    evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)
    evaluate(iterator.initialize())

    for expected_value in expected_values:
      next_element = iterator.get_next()
      computed_value = evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])
      if ignore_order:
        self.assertCountEqual(expected_value, computed_value)
      else:
        self.assertEqual(expected_value, computed_value)

    with self.assertRaises(errors.OutOfRangeError):
      next_element = iterator.get_next()
      evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])

    # After re-initializing the iterator, should be able to iterate again.
    if test_reinitialize:
      evaluate(iterator.initialize())

      for expected_value in expected_values:
        next_element = iterator.get_next()
        computed_value = evaluate([
            values.select_replica(r, next_element) for r in range(len(devices))
        ])
        if ignore_order:
          self.assertCountEqual(expected_value, computed_value)
        else:
          self.assertEqual(expected_value, computed_value)

  def _test_global_step_update(self, strategy):
    with strategy.scope():
      global_step = variable_scope.get_variable(
          "global_step",
          shape=[],
          dtype=dtypes.int64,
          initializer=init_ops.zeros_initializer(),
          trainable=False,
          aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)
      self.evaluate(variables.global_variables_initializer())

      def model_fn():
        train_op = global_step.assign_add(1)
        value = global_step.read_value()
        return train_op, value

      train_ops, value = strategy.extended.call_for_each_replica(model_fn)
      self.evaluate(strategy.group(train_ops))
      global_step_tensors = strategy.experimental_local_results(value)
      global_step_values = self.evaluate(global_step_tensors)
      self.assertEqual((1,) * len(global_step_tensors), global_step_values)

  def _test_numpy_dataset(self, strategy):
    with strategy.scope(), self.cached_session() as sess:
      x = np.asarray([[1, 2], [6, 12], [2, 4], [5, 10], [3, 6], [4, 8]])
      y = np.asarray([5, 4, 3, 2, 1, 0])
      batch_size = 6
      if not strategy.extended._global_batch_size:  # pylint: disable=protected-access
        batch_size = batch_size // strategy.num_replicas_in_sync

      ds = strategy.extended.experimental_make_numpy_dataset((x, y),
                                                             session=sess)
      ds = ds.repeat(2)  # 2 epochs
      # We need to use the drop_remainder argument to get a known static
      # input shape which is required for TPUs.
      drop_remainder = strategy.extended.experimental_require_static_shapes
      ds = ds.batch(batch_size, drop_remainder=drop_remainder)
      i = strategy.make_dataset_iterator(ds)

      self.evaluate(i.initialize())

      def run_and_concatenate(strategy, i):
        x, y = strategy.experimental_run(lambda z: z, i)
        x, y = self.evaluate((strategy.experimental_local_results(x),
                              strategy.experimental_local_results(y)))
        return np.concatenate(x), np.concatenate(y)

      x_1, y_1 = run_and_concatenate(strategy, i)
      self.assertAllEqual(x, x_1)
      self.assertAllEqual(y, y_1)
      x_2, y_2 = run_and_concatenate(strategy, i)
      self.assertAllEqual(x, x_2)
      self.assertAllEqual(y, y_2)
      with self.assertRaises(errors.OutOfRangeError):
        run_and_concatenate(strategy, i)

  def _test_trainable_variable(self, strategy):
    for cls in [variables.VariableV1, variables.Variable]:
      with strategy.scope():
        v1 = cls(1.0)
        self.assertEqual(True, v1.trainable)

        v2 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ)
        self.assertEqual(False, v2.trainable)

        v3 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ,
                 trainable=True)
        self.assertEqual(True, v3.trainable)

        v4 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ,
                 trainable=False)
        self.assertEqual(False, v4.trainable)


class OneDeviceDistributionTestBase(test.TestCase):
  """Some tests that should work with any one-device DistributionStrategy."""

  def _test_run(self, strategy):
    out1 = strategy.experimental_run_v2(lambda: constant_op.constant(4.))
    self.assertAllEqual([4.], self.evaluate(strategy.unwrap(out1)))

    out2 = strategy.experimental_run_v2(
        lambda x: {"a": x * 2, "b": x * x}, args=(out1,))
    out2_vals = self.evaluate(nest.map_structure(strategy.unwrap, out2))
    self.assertAllEqual([8.], out2_vals["a"])
    self.assertAllEqual([16.], out2_vals["b"])

    out3 = strategy.experimental_run_v2(lambda b, a: a + 2 * b + 2, kwargs=out2)
    self.assertAllEqual([42.], self.evaluate(strategy.unwrap(out3)))

  def _test_all_reduce_sum(self, strategy):
    self._test_collective_comms(
        strategy, _all_sum, inputs=(4., [42., 43.]), expected=(4., [42., 43.]))

  def _test_all_reduce_sum_gradients(self, strategy):
    self._test_collective_comms_gradients(
        strategy, _all_sum, inputs=[4.], expected_grads=[4.])

  def _test_all_reduce_sum_gradient_tape(self, strategy):
    self._test_collective_comms_gradient_tape(
        strategy, _all_sum, inputs=[4.], expected_grads=[4.])

  def _test_all_reduce_mean(self, strategy):
    self._test_collective_comms(
        strategy, _all_mean, inputs=(2., [21., 22.]), expected=(2., [21., 22.]))

  def _test_all_reduce_mean_gradients(self, strategy):
    self._test_collective_comms_gradients(
        strategy, _all_mean, inputs=[5.], expected_grads=[5.])

  def _test_all_reduce_mean_gradient_tape(self, strategy):
    self._test_collective_comms_gradient_tape(
        strategy, _all_mean, inputs=[5.], expected_grads=[5.])

  def _test_collective_comms(self, strategy, comm_fn, inputs, expected):
    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensors(inputs))

    self.evaluate(inputs.initialize())
    outputs = self.evaluate(
        list(
            map(strategy.experimental_local_results,
                strategy.experimental_run(comm_fn, inputs))))
    self.assertAllEqual([expected[0]], outputs[0])
    self.assertAllEqual([expected[1]], outputs[1])

  def _test_collective_comms_gradients(self, strategy, comm_fn, inputs,
                                       expected_grads):
    if context.executing_eagerly():
      self.skipTest("`tf.gradients` is not supported with eager execution.")

    def step(c):
      x = constant_op.constant(42.)
      y = comm_fn(x) * c
      return gradients_impl.gradients(y, [x])[0]

    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensors(inputs))

    self.evaluate(inputs.initialize())
    self.assertAllEqual(
        expected_grads,
        self.evaluate(
            strategy.experimental_local_results(
                strategy.experimental_run(step, inputs))))

  def _test_collective_comms_gradient_tape(self, strategy, comm_fn, inputs,
                                           expected_grads):

    def step(c):
      x = constant_op.constant(42.)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = comm_fn(x) * c
      return tape.gradient(y, x)

    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensors(inputs))

    self.evaluate(inputs.initialize())
    self.assertAllEqual(
        expected_grads,
        self.evaluate(
            strategy.experimental_local_results(
                strategy.experimental_run(step, inputs))))

  def _test_device_and_input_device_are_colocated(self, strategy):
    if context.executing_eagerly():
      self.skipTest(
          "cross-device tests are not supported with eager execution.")
    workers, _ = test_util.create_local_cluster(2, 0)
    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.range(5))
    comm_fn = lambda x: x + 1
    run_op = strategy.experimental_run(comm_fn, inputs)
    with session_lib.Session(target=workers[1].target) as sess:
      sess.run(inputs.initialize())
      sess.run(run_op)

  def _test_device_and_input_device_are_colocated_with_function(self, strategy):
    if context.executing_eagerly():
      self.skipTest(
          "cross-device tests are not supported with eager execution.")
    workers, _ = test_util.create_local_cluster(2, 0)
    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.range(5))
    comm_fn = lambda x: x + 1
    experimental_run = def_function.function()(strategy.experimental_run)
    with ops.device("/job:worker/replica:0/task:1/device:CPU:0"):
      # The tf.function must be defined on the right device as well.
      run_op = experimental_run(comm_fn, inputs)
    with session_lib.Session(target=workers[1].target) as sess:
      sess.run(inputs.initialize())
      sess.run(run_op)


class TwoDeviceDistributionTestBase(test.TestCase):
  """Some tests that should work with any two-device DistributionStrategy."""

  def _test_run(self, strategy):
    out1 = strategy.experimental_run_v2(
        lambda: ds_context.get_replica_context().replica_id_in_sync_group + 1)
    self.assertAllEqual([1, 2], self.evaluate(strategy.unwrap(out1)))

    out2 = strategy.experimental_run_v2(
        lambda x: {"a": x * 2, "b": x * x}, args=(out1,))
    out2_vals = self.evaluate(nest.map_structure(strategy.unwrap, out2))
    self.assertAllEqual([2, 4], out2_vals["a"])
    self.assertAllEqual([1, 4], out2_vals["b"])

    out3 = strategy.experimental_run_v2(lambda b, a: a + 2 * b + 2, kwargs=out2)
    self.assertAllEqual([6, 14], self.evaluate(strategy.unwrap(out3)))

  def _test_all_reduce_sum(self, strategy):
    self._test_collective_comms(
        strategy,
        _all_sum,
        inputs=([1., 3.], [[39., 2.], [3., 41.]]),
        expected=(4., [42., 43.]))

  def _test_all_reduce_sum_gradients(self, strategy):
    self._test_collective_comms_gradients(
        strategy, _all_sum, inputs=[1., 3.], expected_grads=[4., 4.])

  def _test_all_reduce_sum_gradient_tape(self, strategy):
    self._test_collective_comms_gradient_tape(
        strategy, _all_sum, inputs=[1., 3.], expected_grads=[4., 4.])

  def _test_all_reduce_mean(self, strategy):
    self._test_collective_comms(
        strategy,
        _all_mean,
        inputs=([1., 3.], [[39., 2.], [3., 41.]]),
        expected=(2., [21., 21.5]))

  def _test_all_reduce_mean_gradients(self, strategy):
    self._test_collective_comms_gradients(
        strategy, _all_mean, inputs=[1., 3.], expected_grads=[2., 2.])

  def _test_all_reduce_mean_gradient_tape(self, strategy):
    self._test_collective_comms_gradient_tape(
        strategy, _all_mean, inputs=[1., 3.], expected_grads=[2., 2.])

  def _test_collective_comms(self, strategy, comm_fn, inputs, expected):
    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))

    self.evaluate(inputs.initialize())
    outputs = self.evaluate(
        list(
            map(strategy.experimental_local_results,
                strategy.experimental_run(comm_fn, inputs))))
    self.assertAllEqual([expected[0], expected[0]], outputs[0])
    self.assertAllEqual([expected[1], expected[1]], outputs[1])

  def _test_collective_comms_gradients(self, strategy, comm_fn, inputs,
                                       expected_grads):
    if context.executing_eagerly():
      self.skipTest("`tf.gradients` is not supported with eager execution.")

    def step(c):
      x = constant_op.constant(42.)
      y = comm_fn(x) * c
      return gradients_impl.gradients(y, [x])[0]

    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))

    self.evaluate(inputs.initialize())
    self.assertAllEqual(
        expected_grads,
        self.evaluate(
            strategy.experimental_local_results(
                strategy.experimental_run(step, inputs))))

  def _test_collective_comms_gradient_tape(self, strategy, comm_fn, inputs,
                                           expected_grads):

    def step(c):
      x = constant_op.constant(42.)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = comm_fn(x) * c
      return tape.gradient(y, x)

    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))

    self.evaluate(inputs.initialize())
    self.assertAllEqual(
        expected_grads,
        self.evaluate(
            strategy.experimental_local_results(
                strategy.experimental_run(step, inputs))))


def _all_sum(value):
  ctx = ds_context.get_replica_context()
  return ctx.all_reduce(reduce_util.ReduceOp.SUM, value)


def _all_mean(value):
  ctx = ds_context.get_replica_context()
  return ctx.all_reduce(reduce_util.ReduceOp.MEAN, value)
