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
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import optimizer


class _TestException(Exception):
  pass


# May be the argument to either distribution.call_for_each_tower() or
# get_tower_context().merge_call()
def _raise_exception_fn(_=None):
  raise _TestException()


# Must be the argument to a distribution.call_for_each_tower() call, calls a
# get_tower_context().merge_call() that raises an exception.
def _merge_raises_fn():
  distribute_lib.get_tower_context().merge_call(_raise_exception_fn)


# Must be the argument to a get_tower_context().merge_call() call, calls
# dist.call_for_each_tower() with a function that raises an exception.
def _call_raises_fn(dist):
  dist.call_for_each_tower(_raise_exception_fn)


# Must be the argument to a distribution.call_for_each_tower() call,
# calls a get_tower_context().merge_call() that calls a
# call_for_each_tower() that raises an exception.
def _merge_call_raises_fn():
  distribute_lib.get_tower_context().merge_call(_call_raises_fn)


# Must be the argument to a get_tower_context().merge_call() call, calls
# dist.call_for_each_tower() with a function that calls a
# get_tower_context().merge_call() that raises an exception.
def _call_merge_raises_fn(dist):
  dist.call_for_each_tower(_merge_raises_fn)


# Must be the argument to a distribution.call_for_each_tower() call, calls a
# get_tower_context().merge_call() that calls a call_for_each_tower() that
# calls a get_tower_context().merge_call() that raises an exception.
def _merge_call_merge_raises_fn():
  distribute_lib.get_tower_context().merge_call(_call_merge_raises_fn)


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
        g_v = d.call_for_each_tower(grad_fn, one, run_concurrently=l.built)

        # Update the variables using the gradients and the update() function.
        before_list = []
        after_list = []
        for g, v in g_v:
          fetched = d.read_var(v)
          before_list.append(fetched)
          # control_dependencies irrelevant but harmless in eager execution
          with ops.control_dependencies([fetched]):
            g = d.reduce("sum", g, destinations=v)
            with ops.control_dependencies(d.unwrap(d.update(v, update, g))):
              after_list.append(d.read_var(v))
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

  def _test_minimize_loss_graph(self, d, soft_placement=False):
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = soft_placement
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with context.graph_mode(), \
         ops.Graph().as_default(), \
         self.test_session(config=config) as sess, \
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
        return v.assign_sub(0.2 * g)

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
            g = d.reduce("sum", g, destinations=v)
            with ops.control_dependencies(d.unwrap(d.update(v, update, g))):
              after_list.append(d.read_var(v))
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

  def _test_map_reduce(self, d, in_graph=None):
    with d.scope():
      map_in = [constant_op.constant(i) for i in range(10)]
      map_out = d.map(map_in, lambda x, y: x * y, 2)
      observed = d.reduce("sum", map_out)
      expected = 90  # 2 * (0 + 1 + ... + 9)
      self.assertEqual(expected, observed.numpy())

  def _test_device_index(self, d):
    with d.scope():
      expected_devices = [False] * len(d.worker_devices)

      def mark_devices_fn(device_id):
        self.assertLess(device_id, len(d.worker_devices))
        self.assertFalse(expected_devices[device_id])
        expected_devices[device_id] = True

      d.call_for_each_tower(mark_devices_fn, d.worker_device_index)
      self.assertAllEqual(expected_devices, [True] * len(d.worker_devices))

  def _test_tower_id(self, d):
    with d.scope():
      expected_devices = [False] * len(d.worker_devices)

      def mark_devices_fn():
        tower_id = distribute_lib.get_tower_context().tower_id
        self.assertLess(tower_id, len(d.worker_devices))
        self.assertFalse(expected_devices[tower_id])
        expected_devices[tower_id] = True

      d.call_for_each_tower(mark_devices_fn)
      self.assertAllEqual(expected_devices, [True] * len(d.worker_devices))

  def _test_call_and_merge_exceptions(self, dist):
    with dist.scope():
      with self.assertRaises(_TestException):
        dist.call_for_each_tower(_raise_exception_fn)
      with self.assertRaises(_TestException):
        dist.call_for_each_tower(_merge_raises_fn)
      with self.assertRaises(_TestException):
        dist.call_for_each_tower(_merge_call_raises_fn)
      with self.assertRaises(_TestException):
        dist.call_for_each_tower(_merge_call_merge_raises_fn)
