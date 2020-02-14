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
"""Tests for low-level eager execution primitives."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python import keras


class Tests(test.TestCase):

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_MatMulCorrectResponse(self):
    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    a_100_by_784 = random_ops.random_uniform((100, 784))
    b_100_by_784 = random_ops.random_uniform((100, 784))

    ctx = context.context()
    ctx.ensure_initialized()

    self.assertAllClose(
        math_ops.matmul(a_2_by_2, b_2_by_2),
        pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                          "MatMul", None, None, a_2_by_2,
                                          b_2_by_2, "transpose_a", False,
                                          "transpose_b", False))
    self.assertAllClose(
        math_ops.matmul(a_100_by_784, b_100_by_784, transpose_b=True),
        pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                          "MatMul", None, None, a_100_by_784,
                                          b_100_by_784, "transpose_a", False,
                                          "transpose_b", True))

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_ResourceVariableMatMulCorrectResponse(self):
    ctx = context.context()
    ctx.ensure_initialized()

    a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
    m = resource_variable_ops.ResourceVariable(a_2_by_2)
    x = pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                          "MatMul", None, None, m, m,
                                          "transpose_a", False, "transpose_b",
                                          False)
    y = pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                          "MatMul", None, None, a_2_by_2,
                                          a_2_by_2, "transpose_a", False,
                                          "transpose_b", False)

    self.assertAllEqual(x, y)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_TapeWrite(self):
    ctx = context.context()
    ctx.ensure_initialized()

    with backprop.GradientTape(persistent=True) as tape:
      a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
      tape.watch(a_2_by_2)
      z = pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                            "MatMul", None, None, a_2_by_2,
                                            a_2_by_2, "transpose_a", False,
                                            "transpose_b", False)
    dz_dy = tape.gradient(z, [a_2_by_2])[0]
    self.assertAllEqual(dz_dy.numpy(),
                        constant_op.constant(4.0, shape=[2, 2]).numpy())

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_ResourceVariableTapeWrite(self):
    ctx = context.context()
    ctx.ensure_initialized()

    with backprop.GradientTape(persistent=True) as tape:
      a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
      m = resource_variable_ops.ResourceVariable(a_2_by_2)
      tape.watch(m)
      z = pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                            "MatMul", None, None, m, m,
                                            "transpose_a", False, "transpose_b",
                                            False)
    dz_dy = tape.gradient(z, [m])[0]
    self.assertAllEqual(dz_dy.numpy(),
                        constant_op.constant(4.0, shape=[2, 2]).numpy())

  # Tests homogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_AddNCorrectResponse(self):
    ctx = context.context()
    ctx.ensure_initialized()

    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    self.assertAllClose(
        math_ops.add_n([a_2_by_2, b_2_by_2]),
        pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name, "AddN",
                                          None, None, [a_2_by_2, b_2_by_2]))

  # Tests homogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_AddNTapeWrite(self):
    ctx = context.context()
    ctx.ensure_initialized()

    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(a_2_by_2)
      tape.watch(b_2_by_2)
      z1 = pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                             "AddN", None, None,
                                             [a_2_by_2, b_2_by_2])
      z2 = math_ops.add_n([a_2_by_2, b_2_by_2])
    dz1_dy = tape.gradient(z1, [a_2_by_2])[0]
    dz2_dy = tape.gradient(z2, [a_2_by_2])[0]
    self.assertAllEqual(dz1_dy.numpy(), dz2_dy.numpy())

  # Tests heterogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_IdentityNCorrectResponse(self):
    ctx = context.context()
    ctx.ensure_initialized()

    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    self.assertAllClose(
        array_ops.identity_n([a_2_by_2, b_2_by_2]),
        pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                          "IdentityN", None, None,
                                          [a_2_by_2, b_2_by_2]))

  # Tests heterogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_IdentityNTapeWrite(self):
    ctx = context.context()
    ctx.ensure_initialized()

    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(a_2_by_2)
      tape.watch(b_2_by_2)
      z1 = pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                             "IdentityN", None, None,
                                             [a_2_by_2, b_2_by_2])
      z2 = array_ops.identity_n([a_2_by_2, b_2_by_2])
    dz1_dy = tape.gradient(z1[0], [a_2_by_2])[0]
    dz2_dy = tape.gradient(z2[0], [a_2_by_2])[0]
    self.assertAllEqual(dz1_dy.numpy(), dz2_dy.numpy())

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_InvalidInputs(self):
    a_2_by_2 = random_ops.random_uniform((2, 2))
    ctx = context.context()
    ctx.ensure_initialized()

    assert ctx.executing_eagerly(
    ), "The prototype doesn't contain C code for graph construction"
    ctx_handle = ctx._handle  # pylint: disable=protected-access

    # Not enough base params
    with self.assertRaisesRegexp(ValueError,
                                 "at least 5 items in the input tuple"):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx_handle, ctx.device_name, "Identity")

    # Not enough inputs
    with self.assertRaisesRegexp(ValueError,
                                 "Expected to be at least 6, was 5"):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx_handle, ctx_handle, "Identity",
                                        None, [])

    # Bad type
    with self.assertRaisesRegexp(TypeError, "expected a string for op_name"):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx_handle, ctx.device_name, ctx_handle,
                                        None, [], a_2_by_2)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastPathExecute_InvalidAttributes(self):
    split_dim = constant_op.constant(0, dtype=dtypes.int32)
    value = constant_op.constant([0, 1, 2, 3], dtype=dtypes.float32)
    ctx = context.context()
    ctx.ensure_initialized()

    ctx_handle = ctx._handle
    with self.assertRaises(core._FallbackException):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx_handle, ctx.device_name, "Split",
                                        None, None, split_dim, value,
                                        "num_split", -1)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testInvalidNumOutputs(self):
    with self.assertRaisesRegexp(
        Exception,
        "Value for attr 'num_split' of -1 must be at least minimum 1"):
      array_ops.split(value=[1, 2, 3], num_or_size_splits=-1)

    with self.assertRaisesRegexp(
        Exception,
        "Value for attr 'num_split' of 0 must be at least minimum 1"):
      array_ops.split(value=[1, 2, 3], num_or_size_splits=0)

  def testIsFunction(self):
    ctx = context.context()
    self.assertFalse(ctx.has_function("not_a_function"))

    @def_function.function
    def f():
      return 1.

    self.assertTrue(ctx.has_function(f.get_concrete_function().name))

  def testEagerExecute_InvalidType(self):
    # Test case for GitHub issue 26879.
    value = keras.layers.Input((128, 128, 1), dtype="float32")
    with self.assertRaisesRegexp(TypeError,
                                 "Expected list for 'values' argument"):
      _ = array_ops.stack(value, axis=1)

  def testGraphResourceVariableRaisesFallback(self):
    with ops.Graph().as_default():
      a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
      m = resource_variable_ops.ResourceVariable(a_2_by_2)
    ctx = context.context()
    ctx.ensure_initialized()
    with self.assertRaises(core._FallbackException):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name, "MatMul",
                                        None, None, m, m, "transpose_a", False,
                                        "transpose_b", False)

  def testOpDefDefaultType(self):
    im = np.random.randint(
        low=0, high=65535, size=100, dtype=np.uint16).reshape(10, 10, 1)

    context.ensure_initialized()

    fastpath_dtype = test_ops.dtype_with_default_op(im).numpy()
    slowpath_dtype = test_ops.dtype_with_default_op_eager_fallback(
        im, None, context.context()).numpy()
    # Ensure the fastpath and slowpath eager paths work.
    self.assertEqual(fastpath_dtype, slowpath_dtype)

    with ops.Graph().as_default(), self.cached_session():
      graph_dtype_symbolic = test_ops.dtype_with_default_op(im)

      graph_dtype = self.evaluate(graph_dtype_symbolic)
    # Ensure the eager path matches the graph path.
    self.assertEqual(fastpath_dtype, graph_dtype)

    # Unfortunately, as of now, this doesn't work as expected on def_functions,
    # since we convert the numpy arrays to tensors pre-tracing (which won't get
    # overriddent by the default type).
    @def_function.function
    def func(im):
      return test_ops.dtype_with_default_op(im)

    function_dtype = func(im).numpy()
    self.assertNotEqual(fastpath_dtype, function_dtype)

    # Captures are OK, since they don't go through the conversion path.
    @def_function.function
    def func_captured():
      return test_ops.dtype_with_default_op(im)

    function_dtype = func_captured().numpy()
    self.assertEqual(fastpath_dtype, function_dtype)

  def testConvertFromArrayInterface(self):
    context.ensure_initialized()
    ctx = context.context()

    class MyArrayClass(object):

      def __init__(self):
        self.array = np.random.random(16)

      def __array__(self):
        return self.array

    a = MyArrayClass()
    t = ops.EagerTensor(a, device=ctx.device_name, dtype=None)
    self.assertAllEqual(t, a)

    # TODO(b/147830189): Converting from EagerTensor should work.
    # _ = ops.EagerTensor(t, device=ctx.device_name, dtype=None)

    # TODO(b/147828820): Converting with tensors should work.
    # _ = ops.EagerTensor([[t]], device=ctx.device_name, dtype=None)


if __name__ == "__main__":
  test.main()
