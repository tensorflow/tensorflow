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
"""Test cases for eager execution using XLA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.layers import pooling
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import adam


class EagerTest(xla_test.XLATestCase):

  def testBasic(self):
    with self.test_scope():
      three = constant_op.constant(3)
      five = constant_op.constant(5)
      product = three * five
      self.assertAllEqual(15, product)

  def testGradientTape(self):
    with self.test_scope():

      x = constant_op.constant(1.0)
      y = constant_op.constant(10.0)
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        a = x + y + x * y
      da_dx = tape.gradient(a, x)
      da_dy = tape.gradient(a, y)

    self.assertEqual(11.0, da_dx.numpy())
    self.assertEqual(2.0, da_dy.numpy())

  def testExecuteListOutputLen0(self):
    with self.test_scope():
      empty = constant_op.constant([], dtype=dtypes.float32)
      result = array_ops.unstack(empty, 0)
      self.assertTrue(isinstance(result, list))
      self.assertEqual(0, len(result))

  def testExecuteListOutputLen1(self):
    with self.test_scope():
      split_dim = constant_op.constant(1)
      value = constant_op.constant([[0., 1., 2.], [3., 4., 5.]])
      result = array_ops.split(value, 1, axis=split_dim)
      self.assertTrue(isinstance(result, list))
      self.assertEqual(1, len(result))
      self.assertAllEqual([[0, 1, 2], [3, 4, 5]], result[0])

  def testExecuteListOutputLen3(self):
    with self.test_scope():
      split_dim = constant_op.constant(1)
      value = constant_op.constant([[0., 1., 2.], [3., 4., 5.]])
      result = array_ops.split(value, 3, axis=split_dim)
      self.assertTrue(isinstance(result, list))
      self.assertEqual(3, len(result))
      self.assertAllEqual([[0], [3]], result[0])
      self.assertAllEqual([[1], [4]], result[1])
      self.assertAllEqual([[2], [5]], result[2])

  def testBasicGraph(self):
    # Run some ops eagerly
    with self.test_scope():
      three = constant_op.constant(3)
      five = constant_op.constant(5)
      product = three * five
      self.assertAllEqual(15, product)

    # Run some ops graphly
    with context.graph_mode(), self.cached_session():
      with self.test_scope():
        three = constant_op.constant(3)
        five = constant_op.constant(5)
        product = three * five
        self.assertAllEqual(15, self.evaluate(product))

  def testDegenerateSlices(self):
    with self.test_scope():
      npt = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
      t = constant_op.constant(npt)
      # degenerate by offering a forward interval with a negative stride
      self.assertAllEqual(npt[0:-1:-1, :, :], t[0:-1:-1, :, :])
      # degenerate with a reverse interval with a positive stride
      self.assertAllEqual(npt[-1:0, :, :], t[-1:0, :, :])
      # empty interval in every dimension
      self.assertAllEqual(npt[-1:0, 2:2, 2:3:-1], t[-1:0, 2:2, 2:3:-1])

  def testIdentity(self):
    with self.test_scope():
      self.assertAllEqual(2, array_ops.identity(2))

  def testRandomOps(self):
    with self.test_scope():
      tensor = gen_random_ops.random_uniform((2, 2), dtypes.float32)
      row0 = tensor[0].numpy()
      row1 = tensor[1].numpy()
      # It should be very unlikely to rng to generate two equal rows.
      self.assertFalse((row0 == row1).all())

  def testIdentityOnVariable(self):
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(True)
      i = array_ops.identity(v)
    self.assertAllEqual(True, i.numpy())

  def testAssignAddVariable(self):
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)
      v.assign_add(2.0)
    self.assertEqual(3.0, v.numpy())

  def testReadAssignRead(self):
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)
      val1 = v.read_value()
      v.assign_add(2.0)
      val2 = v.read_value()
    self.assertEqual(1.0, val1.numpy())
    self.assertEqual(3.0, val2.numpy())

  def testGradient(self):
    def f(x):
      return x

    with self.test_scope():
      grad_fn = backprop.gradients_function(f)
      self.assertAllEqual(2., grad_fn(1., dy=2.)[0])

  def testVariableGradient(self):
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(1.0)

      def f():
        x = v0 * v0
        return x

      grads = backprop.implicit_grad(f)()
    self.assertEqual(2., grads[0][0].numpy())

  def testMultipleVariableReads(self):
    # This test makes sure consecutive variable reads don't copy
    # the underlying memory.
    with self.test_scope():
      # Create 128MiB variables
      var = resource_variable_ops.ResourceVariable(
          array_ops.ones([32, 1024, 1024]))

      # Read the same variable 100 times. If the underlying tensor
      # is not copied, this is a trivial operation. If it is copied,
      # this will eat over 13GB and OOM.
      values = []
      for _ in range(100):
        values.append(var.value())

  # The shape, shape_n, size, and rank are tested here because their
  # execution kernels (as opposed to compilation only tf2xla kernels)
  # are distincts from tf2xla kernels.

  def testShape(self):
    def const(value):
      return array_ops.shape(
          constant_op.constant(value)).numpy()

    def ones(value):
      return array_ops.shape(
          array_ops.ones(value)).numpy()

    with self.test_scope():
      # Shapes of directly constructed tensors
      self.assertAllEqual([], const(3))
      self.assertAllEqual([3], const([1.0, 2.0, 3.0]))
      self.assertAllEqual([2, 2], const([[1.0, 2.0], [3.0, 4.0]]))
      self.assertAllEqual([2, 1, 2], const([[[1.0, 2.0]], [[3.0, 4.0]]]))

      # Shapes of tensors created by op running on device
      # We make this distinction because directly constructed tensors
      # are treated differently in a few places that can influence shape:
      #  - they always have on_host_tensor
      #  - they and their shapes can be cached
      #  - they end up on device via a copy, instead of as program output
      self.assertAllEqual([], ones([]))
      self.assertAllEqual([3], ones([3]))
      self.assertAllEqual([2, 2], ones([2, 2]))
      self.assertAllEqual([2, 1, 2], ones([2, 1, 2]))

  def testShapeN(self):
    with self.test_scope():
      # Shapes of directly constructed tensors
      shapes = array_ops.shape_n([
          constant_op.constant(1.0),
          constant_op.constant([1.0, 2.0, 3.0]),
          constant_op.constant([[1.0, 2.0], [3.0, 4.0]])])
      self.assertAllEqual(
          [[], [3], [2, 2]],
          [x.numpy().tolist() for x in shapes])

      # Shapes of tensors created by op running on device
      shapes = array_ops.shape_n([
          array_ops.ones([]),
          array_ops.ones([3]),
          array_ops.ones([2, 2])])
      self.assertAllEqual(
          [[], [3], [2, 2]],
          [x.numpy().tolist() for x in shapes])

  def testSize(self):
    with self.test_scope():
      self.assertEqual(
          1, array_ops.size(constant_op.constant(1.0)).numpy())
      self.assertEqual(
          3, array_ops.size(constant_op.constant([1.0, 2.0, 3.0])).numpy())
      self.assertEqual(
          4, array_ops.size(
              constant_op.constant([[1.0, 2.0], [3.0, 4.0]])).numpy())

  def testRank(self):
    with self.test_scope():
      self.assertEqual(
          0, array_ops.rank(constant_op.constant(1.0)).numpy())
      self.assertEqual(
          1, array_ops.rank(constant_op.constant([1.0, 2.0, 3.0])).numpy())
      self.assertEqual(
          2, array_ops.rank(
              constant_op.constant([[1.0, 2.0], [3.0, 4.0]])).numpy())

  def testAdam(self):
    with self.test_scope():
      optimizer = adam.AdamOptimizer(0.1)
      x = resource_variable_ops.ResourceVariable(10.0)
      with backprop.GradientTape() as tape:
        y = x * x
      dy_dx = tape.gradient(y, x)
      optimizer.apply_gradients([(dy_dx, x)])
      self.assertAlmostEqual(9.9, x.numpy(), places=3)

  def testAdamSparse(self):
    with ops.device('/cpu:0'):
      # Create 2-D embedding for 3 objects on CPU because sparse/sliced updates
      # are not implemented on TPU.
      embedding_matrix = resource_variable_ops.ResourceVariable(
          array_ops.ones([3, 2]))

    with self.test_scope():
      with backprop.GradientTape() as tape:
        embedding = embedding_ops.embedding_lookup(embedding_matrix, [1])
        y = math_ops.reduce_sum(embedding)
      dy_dx = tape.gradient(y, embedding_matrix)
      self.assertIsInstance(dy_dx, ops.IndexedSlices)
      optimizer = adam.AdamOptimizer(0.1)
      # The gradient application operations will run on CPU because optimizer
      # updates are always collocated with the variable.
      optimizer.apply_gradients([(dy_dx, embedding_matrix)])

      # This assign_add will run on CPU because when an input to an
      # operation is a resource, this operation is placed on the resource's
      # device by the eager runtime.
      embedding_matrix.assign_add(array_ops.ones([3, 2]))

    self.assertAllClose([[2.0, 2.0],
                         [1.9, 1.9],
                         [2.0, 2.0]], embedding_matrix.numpy())


class EagerFunctionTest(xla_test.XLATestCase):

  def testBasic(self):
    with self.test_scope():
      matmul = function.defun(math_ops.matmul)
      t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
      sq = matmul(t, t, transpose_a=True)
      self.assertAllEqual(sq.numpy().reshape(-1), [10, 14, 14, 20])

  def testConv(self):
    if 'GPU' in self.device:
      # TODO(b/32333178)
      self.skipTest('Current implementation of RandomStandardNormal kernel '
                    'is very slow on GPU, and has been blacklisted.')
    with self.test_scope():
      data_format = 'channels_last'
      conv = convolutional.Conv2D(
          filters=1, kernel_size=2, padding='VALID',
          data_format=data_format, activation=nn_ops.relu,
          kernel_initializer=init_ops.ones_initializer(),
          bias_initializer=init_ops.zeros_initializer())
      pool = pooling.MaxPooling2D(2, 2, data_format=data_format)

      def model(x):
        x = conv(x)
        return pool(x)
      model = function.defun(model)

      x = array_ops.ones([1, 4, 4, 1])
      y = model(x)
      self.assertAllEqual(y.numpy(), [[[[4.]]]])

  def testReadVariable(self):
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)

      @function.defun
      def f():
        return v.read_value()

      var = f()
      self.assertEqual(1.0, var.numpy())

  def testUpdateVariable(self):
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)

      def f(v):
        v.assign_add(1.0)
        return v

      f = function.defun(f)

      var = f(v)
      self.assertEqual(2.0, var.numpy())

  def testReturnResourceHandle(self):
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable([[1.0, 2.0], [3.0, 4.0]])

      def f(v):
        return v.handle

      f = function.defun(f)
      handle = f(v)
      self.assertAllEqual(v.numpy(),
                          resource_variable_ops.read_variable_op(
                              handle, dtypes.float32).numpy())

  def testReturnMultipleResourceHandles(self):
    with self.test_scope():
      v1 = resource_variable_ops.ResourceVariable(1.25)
      v2 = resource_variable_ops.ResourceVariable(2.0)

      def f(v):
        return v.handle, 3.0 * v, v2.handle, v + v2

      f = function.defun(f)
      v1_handle, v1_times_3, v2_handle, variable_sum = f(v1)
      self.assertAllEqual(v1.numpy(),
                          resource_variable_ops.read_variable_op(
                              v1_handle, dtypes.float32).numpy())
      self.assertEqual(3.75, v1_times_3.numpy())
      self.assertAllEqual(v2.numpy(),
                          resource_variable_ops.read_variable_op(
                              v2_handle, dtypes.float32).numpy())
      self.assertEqual(3.25, variable_sum.numpy())

  def testAllArgumentKinds(self):
    """Test a complex function that takes different argument kinds.

    tf2xla machinery that translates, compiles, and runs defuns
    classifies arguments into: compile-time constants, regular tensors,
    and resources. This test creates a function with a mix of all these
    kinds. Moreover, the order of function arguments is intentionally mixed up.

    This also tests the case when the same argument is a compile-time constant
    as well as used in an operation that normally expects its inputs to be
    in device memory - addition in this case.
    """
    with self.test_scope():
      def foo(c1, r1, v1, c2, v2, r2):
        # c1 and c2 are compile-time constants
        # r1 and r2 are regular tensors
        # v1 and v2 are resource variables
        a = c1 + r1
        b = math_ops.cast(c2, dtypes.float32) + v2
        c = array_ops.slice(v1, c1, c2)
        d = r2 * v2
        return a, b, c, d

      foo = function.defun(foo)

      c1 = [0, 0]
      c2 = array_ops.ones([2], dtype=dtypes.int32)

      r1 = array_ops.ones([2])
      r2 = [[2., 2.], [3., 3.]]

      v1 = resource_variable_ops.ResourceVariable([[1., 2.], [3., 4.]])
      v2 = resource_variable_ops.ResourceVariable([[10., 20.], [30., 40.]])

      a, b, c, d = foo(c1, r1, v1, c2, v2, r2)

      self.assertAllEqual([1, 1], a.numpy())
      self.assertAllEqual([[11., 21.], [31., 41.]], b.numpy())
      self.assertAllEqual([[1.]], c.numpy())
      self.assertAllEqual([[20., 40.], [90., 120.]], d.numpy())

  def testDefunInGradientTape(self):
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @function.defun
      def f(x):
        x = v0 * v0 * x
        return x

      x = constant_op.constant(3.0)
      with backprop.GradientTape() as tape:
        y = f(x)
      dy = tape.gradient(y, v0)

    self.assertEqual(75, y.numpy())
    self.assertEqual(30, dy.numpy())

  def testGradientTapeInDefun(self):
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @function.defun
      def f():
        x = constant_op.constant(1.0)
        with backprop.GradientTape() as tape:
          y = v0 * x
        dy = tape.gradient(y, v0)
        return dy

      dy = f()
      self.assertEqual(1.0, dy.numpy())

  def testSliceInDefun(self):
    with self.test_scope():

      @function.defun
      def f(x, y):
        return x[0::2, y:, ...]

      x = array_ops.ones([2, 3, 4])
      y = array_ops.ones([], dtype=dtypes.int32)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        tape.watch(y)
        z = f(x, y)
      dz = tape.gradient(z, x)

      self.assertAllEqual(np.ones([1, 2, 4]), z.numpy())
      self.assertAllEqual((2, 3, 4), dz.shape.as_list())

  def testNestedDefun(self):
    with self.test_scope():

      @function.defun
      def times_two(x):
        return 2 * x

      @function.defun
      def two_x_plus_1(x):
        return times_two(x) + 1

      x = constant_op.constant([2, 3, 4])
      y = two_x_plus_1(x)
      self.assertAllEqual([5, 7, 9], y.numpy())

  def testNestedDefunWithVariable(self):
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @function.defun
      def g(x):
        x = v0 * x
        return x

      @function.defun
      def f(x):
        x = g(v0 * x)
        return x

      x = constant_op.constant(3.0)
      y = f(x)

    self.assertEqual(75, y.numpy())

  def testNestedDefunInGradientTape(self):
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)

      @function.defun
      def g(x):
        x = v0 * x
        return x

      @function.defun
      def f(x):
        x = g(v0 * x)
        return x

      x = constant_op.constant(3.0)
      with backprop.GradientTape() as tape:
        y = f(x)
      dy = tape.gradient(y, v0)

    self.assertEqual(75, y.numpy())
    self.assertEqual(30, dy.numpy())

  def testNestedDefunInGradientTapeDifferentVars(self):
    with self.test_scope():
      v0 = resource_variable_ops.ResourceVariable(5.0)
      v1 = resource_variable_ops.ResourceVariable(3.0)

      @function.defun
      def g(x):
        x = v1 * x
        return x

      @function.defun
      def f(x):
        x = g(v0 * x)
        return x

      x = constant_op.constant(3.0)
      with backprop.GradientTape(persistent=True) as tape:
        y = f(x)
      dy_v0 = tape.gradient(y, v0)
      dy_v1 = tape.gradient(y, v1)

    self.assertEqual(45, y.numpy())
    self.assertEqual(9, dy_v0.numpy())
    self.assertEqual(15, dy_v1.numpy())


class ExcessivePaddingTest(xla_test.XLATestCase):
  """Test that eager execution works with TPU flattened tensors.

  Tensors that would normally be excessively padded when written
  to TPU memory are reshaped to 1-D flat tensors.

  This test case verifies that such tensors work with eager execution.

  The flattening currently only happens on TPU, but tests should work
  fine with all backends as flattening is transparent.
  """

  def testFromConstant(self):
    with self.test_scope():
      # Create constant of shape [100, 2, 1]. This tensor would be
      # excessively padded on TPU.
      tensor = constant_op.constant(100 * [[[10.0], [2.0]]])
      # Use reduce_sum since it requires correctly working with
      # a particular dimension.
      reduced = math_ops.reduce_sum(tensor, axis=1)
      self.assertAllEqual(100 * [[12.0]], reduced)

  def testFromOperation(self):
    with self.test_scope():
      tensor = array_ops.ones([3, 100, 2, 2])
      reduced = math_ops.reduce_sum(tensor, axis=[0, 2, 3])
      self.assertAllEqual(100 * [12.0], reduced)

  def testAsFunctionInput(self):
    with self.test_scope():

      @function.defun
      def f(x):
        return math_ops.reduce_sum(x, axis=2)

      tensor = constant_op.constant(100 * [[[10.0, 2.0]]])
      reduced = f(tensor)
      self.assertAllEqual(100 * [[12.0]], reduced)

  def testAsFunctionOutput(self):
    with self.test_scope():

      @function.defun
      def f(x):
        return x * constant_op.constant(100 * [[[10.0, 2.0]]])

      y = f(3)
      reduced = math_ops.reduce_sum(y, axis=2)
      self.assertAllEqual(100 * [[36.0]], reduced)


def multiple_tpus():
  devices = context.context().devices()
  return len([d for d in devices if 'device:TPU:' in d]) > 1


class MultiDeviceTest(xla_test.XLATestCase):
  """Test running TPU computation on more than one core."""

  def testBasic(self):
    if not multiple_tpus():
      self.skipTest('MultiDeviceTest requires multiple TPU devices.')

    # Compute 10 on TPU core 0
    with ops.device('device:TPU:0'):
      two = constant_op.constant(2)
      five = constant_op.constant(5)
      ten = two * five
      self.assertAllEqual(10, ten)

    # Compute 6 on TPU core 1
    with ops.device('device:TPU:1'):
      two = constant_op.constant(2)
      three = constant_op.constant(3)
      six = two * three
      self.assertAllEqual(6, six)

    # Copy 10 and 6 to CPU and sum them
    self.assertAllEqual(16, ten + six)


if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(log_device_placement=True))
  googletest.main()
