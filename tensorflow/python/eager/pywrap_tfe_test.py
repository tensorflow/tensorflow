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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


def record_gradient_callback(inputs, attrs, results):
  return backprop._record_gradient("MatMul", inputs, attrs, results, None)


def c_tfe_py_fastpath_execute(a, b, transpose_a=False, transpose_b=False):
  ctx = context.context()
  assert not ctx.in_graph_mode(
  ), "The prototype doesn't contain C code for graph construction"
  ctx_handle = ctx._handle  # pylint: disable=protected-access

  return pywrap_tensorflow.TFE_Py_FastPathExecute(
      ctx_handle, ctx.device_name, "MatMul", record_gradient_callback, a, b,
      "transpose_a", transpose_a, "transpose_b", transpose_b)[0]


class Tests(test.TestCase):

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_MatMulCorrectResponse(self):
    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    a_100_by_784 = random_ops.random_uniform((100, 784))
    b_100_by_784 = random_ops.random_uniform((100, 784))

    self.assertAllClose(
        math_ops.matmul(a_2_by_2, b_2_by_2),
        c_tfe_py_fastpath_execute(a_2_by_2, b_2_by_2))
    self.assertAllClose(
        math_ops.matmul(a_100_by_784, b_100_by_784, transpose_b=True),
        c_tfe_py_fastpath_execute(a_100_by_784, b_100_by_784, transpose_b=True))

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_TapeWrite(self):
    with backprop.GradientTape(persistent=True) as tape:
      a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
      tape.watch(a_2_by_2)
      z = c_tfe_py_fastpath_execute(a_2_by_2, a_2_by_2)
    dz_dy = tape.gradient(z, [a_2_by_2])[0]
    self.assertAllEqual(dz_dy.numpy(),
                        constant_op.constant(4.0, shape=[2, 2]).numpy())

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_MatMulSlowPath(self):
    a_2_by_2 = random_ops.random_uniform((2, 2)).cpu().numpy()

    with self.assertRaises(NotImplementedError):
      c_tfe_py_fastpath_execute(a_2_by_2, a_2_by_2)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_InvalidInputs(self):
    a_2_by_2 = random_ops.random_uniform((2, 2))
    ctx = context.context()
    assert not ctx.in_graph_mode(
    ), "The prototype doesn't contain C code for graph construction"
    ctx_handle = ctx._handle  # pylint: disable=protected-access

    with self.assertRaisesRegexp(ValueError,
                                 "at least 4 items in the input tuple"):
      pywrap_tensorflow.TFE_Py_FastPathExecute(ctx_handle, ctx.device_name,
                                               "Identity")

    with self.assertRaisesRegexp(ValueError,
                                 "Expected to be at least 5, was 4"):
      pywrap_tensorflow.TFE_Py_FastPathExecute(
          ctx_handle, ctx_handle, "Identity", record_gradient_callback)

    with self.assertRaisesRegexp(TypeError, "expected a string for op_name"):
      pywrap_tensorflow.TFE_Py_FastPathExecute(
          ctx_handle, ctx.device_name, ctx_handle, record_gradient_callback,
          a_2_by_2)


if __name__ == "__main__":
  test.main()
