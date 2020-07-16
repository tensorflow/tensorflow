# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for remote execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from absl.testing import parameterized
import numpy as np
import six

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import server_lib
from tensorflow.python.training.server_lib import ClusterSpec


class SingleWorkerTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SingleWorkerTest, self).setUp()

    workers, _ = test_util.create_local_cluster(1, 0)
    remote.connect_to_remote_host(workers[0].target)

  def tearDown(self):
    super(SingleWorkerTest, self).tearDown()

    # Clear the current device scope to avoid polluting other test cases.
    ops.device(None).__enter__()
    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  @test_util.eager_lazy_remote_copy_on_and_off
  def testMultiDeviceFunctionBasic(self):

    @def_function.function
    def basic(i):
      with ops.device('/job:localhost/replica:0/task:0/cpu:0'):
        a = constant_op.constant([2]) + i
      with ops.device('/job:worker/replica:0/task:0/cpu:0'):
        b = constant_op.constant([1])

      return a + b

    self.assertAllEqual(basic(constant_op.constant([2])).numpy(), [5])
    self.assertAllEqual(basic(constant_op.constant([1])).numpy(), [4])

  @test_util.eager_lazy_remote_copy_on_and_off
  def testMultiDeviceFunctionVariable(self):
    with ops.device('/job:worker/replica:0/task:0/cpu:0'):
      variable_b = variables.Variable(1)

    @def_function.function
    def with_variable(i):
      return i + variable_b

    self.assertAllEqual(with_variable(constant_op.constant([2])).numpy(), [3])

  @test_util.eager_lazy_remote_copy_on_and_off
  def testMultiDeviceFunctionRemoteOutput(self):
    with ops.device('/job:worker/replica:0/task:0/cpu:0'):
      variable_b = variables.Variable(1)

    @def_function.function
    def remote_output(i):
      with ops.device('/job:worker/replica:0/task:0/cpu:0'):
        c = variable_b + 1
      return c, i + variable_b

    self.assertAllEqual(
        remote_output(constant_op.constant([1]))[0].numpy(), 2)

  def testMultiDeviceFunctionAmbiguousDevice(self):

    @def_function.function
    def ambiguous_device(i):
      with ops.device('cpu:0'):
        return i + constant_op.constant([2])

    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device('/job:worker/replica:0/task:0/cpu:0'):
        ambiguous_device(constant_op.constant([2])).numpy()

    self.assertIn('the output node must match exactly one device',
                  cm.exception.message)

  def testStreaming(self):
    """A mini stress test for streaming - issuing many RPCs back to back."""
    with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
      x = array_ops.ones([2, 2])
      y = array_ops.zeros([2, 2])
      num_iters = 200
      for _ in range(num_iters):
        y = x + y
        # Ask for y's shape after every 10 additions on average.
        # This exercises waiting for remote shape logic in TensorHandle.
        if random.randint(1, 10) == 1:
          _ = y.shape
    np.testing.assert_array_equal(
        [[num_iters, num_iters], [num_iters, num_iters]], y.numpy())

  def testShapeError_OpByOp(self):
    with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
      x = array_ops.ones([2, 3])
      y = array_ops.zeros([2, 2])
      with self.assertRaises(errors.InvalidArgumentError) as cm:
        math_ops.matmul(x, y)

    self.assertIn('Dimensions must be equal', cm.exception.message)

  @test_util.eager_lazy_remote_copy_on_and_off
  def testShapeError_Function(self):

    @def_function.function
    def matmul_func(x, y):
      return math_ops.matmul(x, y)

    x = array_ops.ones([2, 3])
    y = array_ops.zeros([2, 2])

    with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
      with self.assertRaises(ValueError) as cm:
        matmul_func(x, y)

    if six.PY2:
      self.assertIn('Dimensions must be equal', cm.exception.message)
    else:
      self.assertIn('Dimensions must be equal', cm.exception.args[0])

  def testClientVarible(self):
    var = variables.Variable(initial_value=0)

    @def_function.function
    def func():
      with ops.device('/job:localhost/task:0'):
        read = var.read_value()
      return read + 1

    with ops.device('/job:worker/task:0'):
      self.assertAllEqual(func(), 1)

  @test_util.eager_lazy_remote_copy_on_and_off
  def testRemoteCall(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def _remote_fn(x):
      return constant_op.constant(1) + x

    remote_fn = _remote_fn.get_concrete_function()

    @def_function.function
    def func(x):
      return functional_ops.remote_call(
          args=[x],
          Tout=[dtypes.int32],
          f=remote_fn,
          target='/job:worker/task:0')

    with ops.device('/job:localhost/task:0'):
      self.assertAllEqual(func(constant_op.constant(1)), [2])


class RemoteAsyncTest(test.TestCase):

  def setUp(self):
    super(RemoteAsyncTest, self).setUp()

    workers, _ = test_util.create_local_cluster(1, 0)
    remote.connect_to_remote_host(workers[0].target)

  def tearDown(self):
    super(RemoteAsyncTest, self).tearDown()

    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  def test_out_of_range_with_while_loop(self):

    with ops.device('/job:worker/task:0'):
      dataset = dataset_ops.Dataset.from_tensor_slices([1.0, 2.0])
      dataset = dataset.batch(1, drop_remainder=False)
      iterator = iter(dataset)
      v = variables.Variable(1.0)

    @def_function.function
    def train_step(iterator):
      i = next(iterator)
      v.assign_add(math_ops.reduce_mean(i))

    while True:
      try:
        with ops.device('/job:worker/task:0'):
          train_step(iterator)
      except (errors.OutOfRangeError, errors.InternalError):
        context.async_clear_error()
        break

    self.assertAllEqual(v.numpy(), 4.0)

  def test_out_of_range_with_for_loop(self):

    with ops.device('/job:worker/task:0'):
      dataset = dataset_ops.Dataset.from_tensor_slices([1.0, 2.0])
      dataset = dataset.batch(1, drop_remainder=False)
      iterator = iter(dataset)
      v = variables.Variable(1.0)

    @def_function.function
    def train_step(iterator):
      i = next(iterator)
      v.assign_add(math_ops.reduce_mean(i))

    num_steps = 3
    for i in range(num_steps):
      try:
        with ops.device('/job:worker/task:0'):
          train_step(iterator)
        if i == num_steps - 1:
          context.async_wait()
      except errors.OutOfRangeError:
        context.async_clear_error()
        break

    self.assertAllEqual(v.numpy(), 4.0)

  def test_out_of_range_with_async_scope(self):

    with ops.device('/job:worker/task:0'):
      dataset = dataset_ops.Dataset.from_tensor_slices([1.0, 2.0])
      dataset = dataset.batch(1, drop_remainder=False)
      iterator = iter(dataset)
      v = variables.Variable(1.0)

    @def_function.function
    def train_step(iterator):
      i = next(iterator)
      v.assign_add(math_ops.reduce_mean(i))

    num_steps = 3
    try:
      with context.async_scope():
        for _ in range(num_steps):
          with ops.device('/job:worker/task:0'):
            train_step(iterator)
    except errors.OutOfRangeError:
      context.async_clear_error()

    self.assertAllEqual(v.numpy(), 4.0)


class MultiWorkersTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(MultiWorkersTest, self).setUp()

    workers, _ = test_util.create_local_cluster(3, 0)
    remote.connect_to_remote_host(
        [workers[0].target, workers[1].target, workers[2].target])

  def tearDown(self):
    super(MultiWorkersTest, self).tearDown()

    # Clear the current device scope to avoid polluting other test cases.
    ops.device(None).__enter__()
    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  @test_util.eager_lazy_remote_copy_on_and_off
  def testReturnRemoteArgument(self):

    @def_function.function
    def local_func(i):
      return i

    with ops.device('/job:worker/replica:0/task:0'):
      x = constant_op.constant([2, 1])

    with ops.device('/job:worker/replica:0/task:1'):
      self.assertAllEqual(local_func(x), [2, 1])

  @test_util.eager_lazy_remote_copy_on_and_off
  def testMultiDeviceFunctionOnLocalDevice(self):
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable(1.0)

    @def_function.function
    def remote_function(i):
      with ops.device('/job:worker/replica:0/task:0'):
        a = i + variable_b
      c = a + 1.0
      return c

    self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

  def testMultiDeviceFunctionWithPackedVariable(self):
    with ops.device('/job:worker/replica:0/task:0/device:CPU:0'):
      var0 = resource_variable_ops.ResourceVariable(1.0)
    with ops.device('/job:worker/replica:0/task:1/device:CPU:0'):
      var1 = resource_variable_ops.ResourceVariable(2.0)

    packed_var = ops.pack_eager_tensors([var0.handle, var1.handle])
    self.assertEqual(packed_var.device,
                     '/job:localhost/replica:0/task:0/device:COMPOSITE:0')
    self.assertEqual(packed_var.backing_device,
                     '/job:localhost/replica:0/task:0/device:COMPOSITE:0')

    @def_function.function
    def add_variables():
      with ops.device('/job:worker/replica:0/task:0/device:CPU:0'):
        read0 = resource_variable_ops.read_variable_op(
            packed_var, dtype=dtypes.float32)
      with ops.device('/job:worker/replica:0/task:1/device:CPU:0'):
        read1 = resource_variable_ops.read_variable_op(
            packed_var, dtype=dtypes.float32)

      return read0 + read1

    # Run the function on a remote device
    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(add_variables().numpy(), 3.0)

    # Run the function on a local worker
    self.assertAllEqual(add_variables().numpy(), 3.0)

  @test_util.eager_lazy_remote_copy_on_and_off
  def testMultiDeviceFunctionOnRemoteDeviceWithWait(self):
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable([1.0])

    @def_function.function
    def remote_function(i):
      x = array_ops.ones([1000, 1000])
      for _ in range(1, 1000):
        x = x * x
      variable_b.assign_add(i)
      a = 1.0 + variable_b
      return a

    @def_function.function
    def remote_function2(i):
      variable_b.assign_add(i)
      a = 1.0 + variable_b
      return a

    # Runs first function:
    # - on remote device
    # - needs remote input
    # - is side impacting
    # - runs much slower
    with ops.device('/job:worker/replica:0/task:0'):
      remote_function(constant_op.constant([2.0]))

    # Runs second function:
    # - on remote device
    # - is side impacting
    # There should be a sync point here and the next function will be executed
    # only after the first function has completed.
    with ops.device('/job:worker/replica:0/task:2'):
      self.assertAllEqual(remote_function2(constant_op.constant([3.0])), [7.0])

  @test_util.eager_lazy_remote_copy_on_and_off
  def testMultiDeviceFunctionOnRemoteDevice(self):
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable(1.0)

    @def_function.function
    def remote_function(i):
      with ops.device('/job:worker/replica:0/task:0'):
        a = i + variable_b
      c = a + 1.0
      return c

    context.context().mirroring_policy = context.MIRRORING_NONE

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    if test_util.is_gpu_available():
      with ops.device('/job:worker/replica:0/task:0/device:GPU:0'):
        self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    context.context().mirroring_policy = context.MIRRORING_ALL

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    if test_util.is_gpu_available():
      with ops.device('/job:worker/replica:0/task:0/device:GPU:0'):
        self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

  @test_util.eager_lazy_remote_copy_on_and_off
  def testMultiDeviceWhileLoopOnRemoteDevice(self):
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable(1.0)

    @def_function.function
    def remote_function(i):

      def body(i, _):
        with ops.device('/job:worker/replica:0/task:0'):
          a = i + variable_b
        return a + 1.0, 1

      return control_flow_ops.while_loop_v2(lambda _, d: d < 1, body, [i, 0])[0]

    context.context().mirroring_policy = context.MIRRORING_NONE

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    if test_util.is_gpu_available():
      with ops.device('/job:worker/replica:0/task:0/device:GPU:0'):
        self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    context.context().mirroring_policy = context.MIRRORING_ALL

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    if test_util.is_gpu_available():
      with ops.device('/job:worker/replica:0/task:0/device:GPU:0'):
        self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

  @test_util.eager_lazy_remote_copy_on_and_off
  def testSimpleParameterServer(self):

    with ops.device('/job:worker/task:2/device:CPU:0'):
      v1 = variables.Variable(initial_value=0)
      v2 = variables.Variable(initial_value=10)

    @def_function.function
    def worker_fn():
      v1.assign_add(1)
      v2.assign_sub(2)
      return v1.read_value() + v2.read_value()

    with ops.device('/job:worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 9)

    with ops.device('/job:worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 8)


_GRPC_PREFIX = 'grpc://'


class MultiJobsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(MultiJobsTest, self).setUp()

    workers, ps = test_util.create_local_cluster(num_workers=2, num_ps=2)
    cluster = {
        'my_worker': [_strip_prefix(t.target, _GRPC_PREFIX) for t in workers],
        'my_ps': [_strip_prefix(t.target, _GRPC_PREFIX) for t in ps],
    }
    self._cluster = server_lib.ClusterSpec(cluster)
    self._cluster_resolver = SimpleClusterResolver(
        cluster_spec=self._cluster, master=ps[0].target)

  def tearDown(self):
    super(MultiJobsTest, self).tearDown()

    # Clear the current device scope to avoid polluting other test cases.
    ops.device(None).__enter__()
    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  @test_util.eager_lazy_remote_copy_on_and_off
  def testSimpleParameterServer(self):
    remote.connect_to_cluster(self._cluster)

    with ops.device('/job:my_ps/task:0/device:CPU:0'):
      v1 = variables.Variable(initial_value=0)
      v2 = variables.Variable(initial_value=10)

    @def_function.function
    def worker_fn():
      v1.assign_add(1)
      v2.assign_sub(2)
      return v1.read_value() + v2.read_value()

    with ops.device('/job:my_worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 9)

    with ops.device('/job:my_worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 8)

  # TODO(b/152224115): Re-enable this test.
  @test_util.eager_lazy_remote_copy_on_and_off
  def DISABLED_testSimpleParameterServerWithDeviceFilters(self):
    cluster_device_filters = server_lib.ClusterDeviceFilters()
    for i in range(2):
      cluster_device_filters.set_device_filters('my_worker', i, ['/job:my_ps'])
      cluster_device_filters.set_device_filters('my_ps', i, ['/job:my_worker'])
    remote.connect_to_cluster(
        self._cluster, cluster_device_filters=cluster_device_filters)

    with ops.device('/job:my_ps/task:0/device:CPU:0'):
      v1 = variables.Variable(initial_value=0)
    with ops.device('/job:my_ps/task:1/device:CPU:0'):
      v2 = variables.Variable(initial_value=10)

    @def_function.function
    def worker_fn():
      v1.assign_add(1)
      v2.assign_sub(2)
      return v1.read_value() + v2.read_value()

    with ops.device('/job:my_worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 9)
    with ops.device('/job:my_worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 8)

    # The following remote call would fail because the ps nodes cannot see each
    # other due to the device filters.
    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device('/job:my_ps/task:0/device:CPU:0'):
        worker_fn().numpy()
    self.assertIn('/job:my_ps/replica:0/task:1/device:CPU:0 unknown device',
                  cm.exception.message)

    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device('/job:my_ps/task:1/device:CPU:0'):
        worker_fn().numpy()
    self.assertIn('/job:my_ps/replica:0/task:0/device:CPU:0 unknown device',
                  cm.exception.message)

    with ops.device('/job:my_worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 7)
    with ops.device('/job:my_worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 6)
    # Explicitly delete variables to avoid triggering errors when being GC'ed in
    # subsequent tests.
    del v1, v2

  @test_util.eager_lazy_remote_copy_on_and_off
  def testConnectWithClusterResolver(self):
    remote.connect_to_cluster(self._cluster_resolver)

    v1 = variables.Variable(initial_value=0)
    v2 = variables.Variable(initial_value=10)

    @def_function.function
    def worker_fn():
      v1.assign_add(1)
      v2.assign_sub(2)
      return v1.read_value() + v2.read_value()

    with ops.device('/job:my_worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 9)

    with ops.device('/job:my_worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 8)

  @test_util.eager_lazy_remote_copy_on_and_off
  def testConnectToClusterTwiceOk(self):
    remote.connect_to_cluster(self._cluster_resolver)
    remote.connect_to_cluster(self._cluster_resolver)

  @test_util.eager_lazy_remote_copy_on_and_off
  def testConnectToClusterOnMismatchedDevice(self):
    remote.connect_to_cluster(self._cluster_resolver)

    # enter into another device scope.
    ops.device('/job:my_worker/task:0/device:CPU:0').__enter__()

    with self.assertRaises(ValueError):
      remote.connect_to_cluster(self._cluster_resolver)

  @test_util.eager_lazy_remote_copy_on_and_off
  def testConnectToClusterWithLocalMaster(self):
    local_resolver = SimpleClusterResolver(ClusterSpec({}), master='local')
    remote.connect_to_cluster(local_resolver)

  @test_util.eager_lazy_remote_copy_on_and_off
  def testConnectToClusterInGraphModeWillFail(self):
    ops.disable_eager_execution()
    with self.assertRaises(ValueError):
      remote.connect_to_cluster(self._cluster_resolver)
    ops.enable_eager_execution()


def _strip_prefix(s, prefix):
  return s[len(prefix):] if s.startswith(prefix) else s


if __name__ == '__main__':
  test.main()
