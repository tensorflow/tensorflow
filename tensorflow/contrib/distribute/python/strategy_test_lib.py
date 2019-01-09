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

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer


class _TestException(Exception):
  pass


# May be the argument to either distribution.call_for_each_replica() or
# get_replica_context().merge_call()
def _raise_exception_fn(_=None):
  raise _TestException()


# Must be the argument to a distribution.call_for_each_replica() call, calls a
# get_replica_context().merge_call() that raises an exception.
def _merge_raises_fn():
  ds_context.get_replica_context().merge_call(_raise_exception_fn)


# Must be the argument to a get_replica_context().merge_call() call, calls
# dist.call_for_each_replica() with a function that raises an exception.
def _call_raises_fn(dist):
  dist.call_for_each_replica(_raise_exception_fn)


# Must be the argument to a distribution.call_for_each_replica() call,
# calls a get_replica_context().merge_call() that calls a
# call_for_each_replica() that raises an exception.
def _merge_call_raises_fn():
  ds_context.get_replica_context().merge_call(_call_raises_fn)


# Must be the argument to a get_replica_context().merge_call() call, calls
# dist.call_for_each_replica() with a function that calls a
# get_replica_context().merge_call() that raises an exception.
def _call_merge_raises_fn(dist):
  dist.call_for_each_replica(_merge_raises_fn)


# Must be the argument to a distribution.call_for_each_replica() call, calls a
# get_replica_context().merge_call() that calls a call_for_each_replica() that
# calls a get_replica_context().merge_call() that raises an exception.
def _merge_call_merge_raises_fn():
  ds_context.get_replica_context().merge_call(_call_merge_raises_fn)


class DistributionTestBase(test.TestCase):
  """Some tests that should work with any DistributionStrategy."""

  def _test_minimize_loss_eager(self, d):
    with d.scope():
      l = core.Dense(1, use_bias=False)

      def loss(x):
        # TODO(josh11b): What if this constant was instead a captured
        # value?  Would it need to be a value that has been passed
        # through d.broadcast()?
        y = array_ops.reshape(l(x), []) - constant_op.constant(1.)
        return y * y
      # TODO(isaprykin): Extract implicit_grad+get_filtered_grad_fn into a
      # common `implicit_grad` function and put it in DistributionStrategy.
      grad_fn = backprop.implicit_grad(loss)
      grad_fn = optimizer.get_filtered_grad_fn(grad_fn)

      def update(v, g):
        return v.assign_sub(0.2 * g)

      one = d.broadcast(constant_op.constant([[1.]]))

      def step():
        """Perform one optimization step."""
        # Run forward & backward to get gradients, variables list.
        g_v = d.call_for_each_replica(grad_fn, args=(one,))

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
            with ops.control_dependencies(d.update(
                v, update, g, grouped=False)):
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

  def _test_minimize_loss_graph(self, d, soft_placement=False,
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
        # TODO(josh11b): What if this constant was instead a captured
        # value?  Would it need to be a value that has been passed
        # through d.broadcast()?
        y = array_ops.reshape(l(x), []) - constant_op.constant(1.)
        return y * y

      grad_fn = backprop.implicit_grad(loss)

      def update(v, g):
        return v.assign_sub(learning_rate * g)

      one = d.broadcast(constant_op.constant([[1.]]))

      def step():
        """Perform one optimization step."""
        # Run forward & backward to get gradients, variables list.
        g_v = d.call_for_each_replica(grad_fn, args=(one,))

        # Update the variables using the gradients and the update() function.
        before_list = []
        after_list = []
        for g, v in g_v:
          fetched = d.extended.read_var(v)
          before_list.append(fetched)
          with ops.control_dependencies([fetched]):
            g = d.extended.reduce_to(
                reduce_util.ReduceOp.SUM, g, destinations=v)
            with ops.control_dependencies(d.update(
                v, update, g, grouped=False)):
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

  def _test_replica_id(self, d):
    with d.scope():
      expected_devices = [False] * len(d.extended.worker_devices)

      def mark_devices_fn():
        replica_id = self.evaluate(
            ds_context.get_replica_context().replica_id_in_sync_group)
        self.assertLess(replica_id, len(d.extended.worker_devices))
        self.assertFalse(expected_devices[replica_id])
        expected_devices[replica_id] = True

      d.call_for_each_replica(mark_devices_fn)
      self.assertAllEqual(expected_devices,
                          [True] * len(d.extended.worker_devices))

  def _test_call_and_merge_exceptions(self, dist):
    with dist.scope():
      with self.assertRaises(_TestException):
        dist.call_for_each_replica(_raise_exception_fn)
      with self.assertRaises(_TestException):
        dist.call_for_each_replica(_merge_raises_fn)
      with self.assertRaises(_TestException):
        dist.call_for_each_replica(_merge_call_raises_fn)
      with self.assertRaises(_TestException):
        dist.call_for_each_replica(_merge_call_merge_raises_fn)

  def _input_fn_to_test_input_context(self,
                                      dataset_fn,
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

      return dataset_fn()

    return _input_fn

  def _test_input_fn_iterator(self, iterator, devices, expected_values,
                              sess=None):
    evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)
    evaluate(iterator.initialize())

    for expected_value in expected_values:
      next_element = iterator.get_next()
      computed_value = evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])
      self.assertEqual(expected_value, computed_value)

    with self.assertRaises(errors.OutOfRangeError):
      next_element = iterator.get_next()
      evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])

    # After re-initializing the iterator, should be able to iterate again.
    evaluate(iterator.initialize())

    for expected_value in expected_values:
      next_element = iterator.get_next()
      computed_value = evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])
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

      train_ops, value = strategy.call_for_each_replica(model_fn)
      self.evaluate(strategy.group(train_ops))
      global_step_tensors = strategy.unwrap(value)
      global_step_values = self.evaluate(global_step_tensors)
      self.assertEqual((1,) * len(global_step_tensors), global_step_values)


class OneDeviceDistributionTestBase(test.TestCase):
  """Some tests that should work with any one-device DistributionStrategy."""

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
        list(map(strategy.unwrap, strategy.experimental_run(comm_fn, inputs))))
    self.assertAllEqual([expected[0]], outputs[0])
    self.assertAllEqual([expected[1]], outputs[1])

  def _test_collective_comms_gradients(
      self, strategy, comm_fn, inputs, expected_grads):
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
        self.evaluate(strategy.unwrap(strategy.experimental_run(step, inputs))))

  def _test_collective_comms_gradient_tape(
      self, strategy, comm_fn, inputs, expected_grads):
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
        self.evaluate(strategy.unwrap(strategy.experimental_run(step, inputs))))


class TwoDeviceDistributionTestBase(test.TestCase):
  """Some tests that should work with any two-device DistributionStrategy."""

  def _test_all_reduce_sum(self, strategy):
    self._test_collective_comms(
        strategy, _all_sum,
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
        strategy, _all_mean,
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
        list(map(strategy.unwrap, strategy.experimental_run(comm_fn, inputs))))
    self.assertAllEqual([expected[0], expected[0]], outputs[0])
    self.assertAllEqual([expected[1], expected[1]], outputs[1])

  def _test_collective_comms_gradients(
      self, strategy, comm_fn, inputs, expected_grads):
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
        self.evaluate(strategy.unwrap(strategy.experimental_run(step, inputs))))

  def _test_collective_comms_gradient_tape(
      self, strategy, comm_fn, inputs, expected_grads):
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
        self.evaluate(strategy.unwrap(strategy.experimental_run(step, inputs))))


def _all_sum(value):
  ctx = ds_context.get_replica_context()
  return ctx.all_reduce(reduce_util.ReduceOp.SUM, value)


def _all_mean(value):
  ctx = ds_context.get_replica_context()
  return ctx.all_reduce(reduce_util.ReduceOp.MEAN, value)
