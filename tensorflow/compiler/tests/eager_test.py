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

from tensorflow.compiler.tests.xla_test import XLATestCase
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
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import googletest


class EagerTest(XLATestCase):

  def testBasic(self):
    with self.test_scope():
      three = constant_op.constant(3)
      five = constant_op.constant(5)
      product = three * five
      self.assertAllEqual(15, product)

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
    with context.graph_mode(), self.test_session() as sess:
      with self.test_scope():
        three = constant_op.constant(3)
        five = constant_op.constant(5)
        product = three * five
        self.assertAllEqual(15, sess.run(product))

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


class EagerFunctionTest(XLATestCase):

  def testBasic(self):
    with self.test_scope():
      matmul = function.defun(math_ops.matmul, compiled=True)
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
      model = function.defun(model, compiled=True)

      x = array_ops.ones([1, 4, 4, 1])
      y = model(x)
      self.assertAllEqual(y.numpy(), [[[[4.]]]])

  def testReadVariable(self):
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)

      @function.defun(compiled=True)
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

      f = function.defun(f, compiled=True)

      var = f(v)
      self.assertEqual(2.0, var.numpy())

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

      foo = function.defun(foo, compiled=True)

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


if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(log_device_placement=True))
  googletest.main()
