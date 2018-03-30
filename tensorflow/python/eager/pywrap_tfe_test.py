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

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops


class Tests(test.TestCase):

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_MatMulCorrectResponse(self):
    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    a_100_by_784 = random_ops.random_uniform((100, 784))
    b_100_by_784 = random_ops.random_uniform((100, 784))

    ctx = context.context()

    self.assertAllClose(
        math_ops.matmul(a_2_by_2, b_2_by_2),
        pywrap_tensorflow.TFE_Py_FastPathExecute(
            ctx._handle, ctx.device_name, "MatMul", None, None, a_2_by_2,
            b_2_by_2, "transpose_a", False, "transpose_b", False))
    self.assertAllClose(
        math_ops.matmul(a_100_by_784, b_100_by_784, transpose_b=True),
        pywrap_tensorflow.TFE_Py_FastPathExecute(
            ctx._handle, ctx.device_name, "MatMul", None, None, a_100_by_784,
            b_100_by_784, "transpose_a", False, "transpose_b", True))

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_ResourceVariableMatMulCorrectResponse(self):
    ctx = context.context()
    a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
    m = resource_variable_ops.ResourceVariable(a_2_by_2)
    x = pywrap_tensorflow.TFE_Py_FastPathExecute(
        ctx._handle, ctx.device_name, "MatMul", None, None, m, m, "transpose_a",
        False, "transpose_b", False)
    y = pywrap_tensorflow.TFE_Py_FastPathExecute(
        ctx._handle, ctx.device_name, "MatMul", None, None, a_2_by_2, a_2_by_2,
        "transpose_a", False, "transpose_b", False)

    self.assertAllEqual(x, y)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_TapeWrite(self):
    ctx = context.context()
    with backprop.GradientTape(persistent=True) as tape:
      a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
      tape.watch(a_2_by_2)
      z = pywrap_tensorflow.TFE_Py_FastPathExecute(
          ctx._handle, ctx.device_name, "MatMul", None, None, a_2_by_2,
          a_2_by_2, "transpose_a", False, "transpose_b", False)
    dz_dy = tape.gradient(z, [a_2_by_2])[0]
    self.assertAllEqual(dz_dy.numpy(),
                        constant_op.constant(4.0, shape=[2, 2]).numpy())

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_ResourceVariableTapeWrite(self):
    ctx = context.context()
    with backprop.GradientTape(persistent=True) as tape:
      a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
      m = resource_variable_ops.ResourceVariable(a_2_by_2)
      tape.watch(m)
      z = pywrap_tensorflow.TFE_Py_FastPathExecute(
          ctx._handle, ctx.device_name, "MatMul", None, None, m, m,
          "transpose_a", False, "transpose_b", False)
    dz_dy = tape.gradient(z, [m])[0]
    self.assertAllEqual(dz_dy.numpy(),
                        constant_op.constant(4.0, shape=[2, 2]).numpy())

  # Tests homogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_AddNCorrectResponse(self):
    ctx = context.context()
    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    self.assertAllClose(
        math_ops.add_n([a_2_by_2, b_2_by_2]),
        pywrap_tensorflow.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                                 "AddN", None, None,
                                                 [a_2_by_2, b_2_by_2]))

  # Tests homogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_AddNTapeWrite(self):
    ctx = context.context()
    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(a_2_by_2)
      tape.watch(b_2_by_2)
      z1 = pywrap_tensorflow.TFE_Py_FastPathExecute(
          ctx._handle, ctx.device_name, "AddN", None, None,
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
    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    self.assertAllClose(
        array_ops.identity_n([a_2_by_2, b_2_by_2]),
        pywrap_tensorflow.TFE_Py_FastPathExecute(ctx._handle, ctx.device_name,
                                                 "IdentityN", None, None,
                                                 [a_2_by_2, b_2_by_2]))

  # Tests heterogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_IdentityNTapeWrite(self):
    ctx = context.context()
    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(a_2_by_2)
      tape.watch(b_2_by_2)
      z1 = pywrap_tensorflow.TFE_Py_FastPathExecute(
          ctx._handle, ctx.device_name, "IdentityN", None, None,
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
    assert ctx.executing_eagerly(
    ), "The prototype doesn't contain C code for graph construction"
    ctx_handle = ctx._handle  # pylint: disable=protected-access

    # Not enough base params
    with self.assertRaisesRegexp(ValueError,
                                 "at least 5 items in the input tuple"):
      pywrap_tensorflow.TFE_Py_FastPathExecute(ctx_handle, ctx.device_name,
                                               "Identity")

    # Not enough inputs
    with self.assertRaisesRegexp(ValueError,
                                 "Expected to be at least 6, was 5"):
      pywrap_tensorflow.TFE_Py_FastPathExecute(ctx_handle, ctx_handle,
                                               "Identity", None, [])

    # Bad type
    with self.assertRaisesRegexp(TypeError, "expected a string for op_name"):
      pywrap_tensorflow.TFE_Py_FastPathExecute(ctx_handle, ctx.device_name,
                                               ctx_handle, None, [], a_2_by_2)


if __name__ == "__main__":
  test.main()
