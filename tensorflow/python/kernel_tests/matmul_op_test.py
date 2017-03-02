# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.math_ops.matmul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as test_lib


def _AddTest(test, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class MatMulTest(test_lib.TestCase):
  pass  # Filled in below


def _GetMatMulTest(a_np_, b_np_, use_static_shape_, **kwargs_):

  def Test(self):
    np_val = np.matrix(a_np_) * np.matrix(b_np_)

    # Transpose and possibly conjugate a and b according to the attributes
    # such that tf.matmul(effective_a_np, effective_b_np, **kwargs) results in
    # a valid matrix multiplication and produces the same result as
    # np.matrix(a_np_) * np.matrix(b_np_)
    if kwargs_["transpose_a"] is True or kwargs_["adjoint_a"] is True:
      effective_a_np = a_np_.T
      if kwargs_["adjoint_a"] is True:
        effective_a_np = np.conj(effective_a_np)
    else:
      effective_a_np = a_np_

    if kwargs_["transpose_b"] is True or kwargs_["adjoint_b"] is True:
      effective_b_np = b_np_.T
      if kwargs_["adjoint_b"] is True:
        effective_b_np = np.conj(effective_b_np)
    else:
      effective_b_np = b_np_

    use_gpu = True
    if a_np_.dtype is np.float16 and (
        not test_util.CudaSupportsHalfMatMulAndConv()):
      use_gpu = False
      print("Built without fp16 matmul support for Cuda, running test on CPU.")

    with self.test_session(use_gpu=use_gpu) as sess:
      if use_static_shape_:
        a = constant_op.constant(effective_a_np)
        b = constant_op.constant(effective_b_np)
        res = math_ops.matmul(a, b, **kwargs_)
        tf_val = res.eval()
      else:
        a = array_ops.placeholder(a_np_.dtype)
        b = array_ops.placeholder(b_np_.dtype)
        res = math_ops.matmul(a, b, **kwargs_)
        tf_val = sess.run(res, {a: effective_a_np, b: effective_b_np})

    self.assertAllCloseAccordingToType(
        tf_val,
        np_val,
        float_rtol=1e-5,
        float_atol=1e-5,
        half_rtol=0.1,
        half_atol=0.1)

  return Test


# TODO(zhifengc): Figures out how to test matmul gradients on GPU.
class MatMulGradientTest(test_lib.TestCase):

  def testGradientInput0(self):
    with self.test_session(use_gpu=False):
      x = constant_op.constant(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          shape=[3, 2],
          dtype=dtypes.float64,
          name="x")
      y = constant_op.constant(
          [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
          shape=[2, 4],
          dtype=dtypes.float64,
          name="y")
      m = math_ops.matmul(x, y, name="matmul")
      err = gradient_checker.compute_gradient_error(x, [3, 2], m, [3, 4])
    print("matmul input0 gradient err = ", err)
    self.assertLess(err, 1e-10)

  def testGradientInput1(self):
    with self.test_session(use_gpu=False):
      x = constant_op.constant(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          shape=[3, 2],
          dtype=dtypes.float64,
          name="x")
      y = constant_op.constant(
          [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
          shape=[2, 4],
          dtype=dtypes.float64,
          name="y")
      m = math_ops.matmul(x, y, name="matmul")
      err = gradient_checker.compute_gradient_error(y, [2, 4], m, [3, 4])
    print("matmul input1 gradient err = ", err)
    self.assertLess(err, 1e-10)

  def _VerifyInput0(self, transpose_a, transpose_b):
    shape_x = [3, 2]
    shape_y = [2, 4]
    if transpose_a:
      shape_x = list(reversed(shape_x))
    if transpose_b:
      shape_y = list(reversed(shape_y))
    with self.test_session(use_gpu=False):
      x = constant_op.constant(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          shape=shape_x,
          dtype=dtypes.float64,
          name="x")
      y = constant_op.constant(
          [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
          shape=shape_y,
          dtype=dtypes.float64,
          name="y")
      m = math_ops.matmul(x, y, transpose_a, transpose_b, name="matmul")
      err = gradient_checker.compute_gradient_error(x, shape_x, m, [3, 4])
    print("matmul input0 gradient err = ", err)
    self.assertLess(err, 1e-10)

  def testGradientInput0WithTranspose(self):
    self._VerifyInput0(transpose_a=True, transpose_b=False)
    self._VerifyInput0(transpose_a=False, transpose_b=True)
    self._VerifyInput0(transpose_a=True, transpose_b=True)

  def _VerifyInput1(self, transpose_a, transpose_b):
    shape_x = [3, 2]
    shape_y = [2, 4]
    if transpose_a:
      shape_x = list(reversed(shape_x))
    if transpose_b:
      shape_y = list(reversed(shape_y))
    with self.test_session(use_gpu=False):
      x = constant_op.constant(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          shape=shape_x,
          dtype=dtypes.float64,
          name="x")
      y = constant_op.constant(
          [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
          shape=shape_y,
          dtype=dtypes.float64,
          name="y")
      m = math_ops.matmul(x, y, transpose_a, transpose_b, name="matmul")
      err = gradient_checker.compute_gradient_error(y, shape_y, m, [3, 4])
    print("matmul input1 gradient err = ", err)
    self.assertLess(err, 1e-10)

  def testGradientInput1WithTranspose(self):
    self._VerifyInput1(transpose_a=True, transpose_b=False)
    self._VerifyInput1(transpose_a=False, transpose_b=True)
    self._VerifyInput1(transpose_a=True, transpose_b=True)


class MatMulStatsTest(test_lib.TestCase):

  def testSimpleStatistics(self):
    g = ops.Graph()
    with g.as_default():
      a = variables.Variable(random_ops.random_normal([25, 16]))
      b = variables.Variable(random_ops.random_normal([16, 9]))
      math_ops.matmul(a, b)
      for op in g.get_operations():
        flops = ops.get_stats_for_node_def(g, op.node_def, "flops").value
        if op.name == "MatMul":
          self.assertEqual(7200, flops)

  def testTransposedStatistics(self):
    g = ops.Graph()
    with g.as_default():
      a = variables.Variable(random_ops.random_normal([16, 25]))
      b = variables.Variable(random_ops.random_normal([16, 9]))
      math_ops.matmul(a, b, transpose_a=True)
      for op in g.get_operations():
        flops = ops.get_stats_for_node_def(g, op.node_def, "flops").value
        if op.name == "MatMul":
          self.assertEqual(7200, flops)


if __name__ == "__main__":
  sizes = [1, 3, 5]
  trans_options = [[False, False], [True, False], [False, True]]
  for dtype in (np.int32, np.float16, np.float32, np.float64, np.complex64,
                np.complex128):
    for m in sizes:
      for n in sizes:
        for k in sizes:
          # Construct compatible random matrices a_np of size [m, k] and b_np
          # of size [k, n].
          a_np = np.random.normal(-5, 5, m * k).astype(dtype).reshape([m, k])
          if dtype in (np.complex64, np.complex128):
            a_np.imag = np.random.normal(-5, 5,
                                         m * k).astype(dtype).reshape([m, k])

          b_np = np.random.normal(-5, 5, k * n).astype(dtype).reshape([k, n])
          if dtype in (np.complex64, np.complex128):
            b_np.imag = np.random.normal(-5, 5,
                                         k * n).astype(dtype).reshape([k, n])
          for adjoint_a, transpose_a in trans_options:
            for adjoint_b, transpose_b in trans_options:
              for use_static_shape in [False, True]:
                name = "%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                    dtype.__name__, m, n, k, adjoint_a, transpose_a, adjoint_b,
                    transpose_b, use_static_shape)
                _AddTest(MatMulTest, "MatMulTest", name,
                         _GetMatMulTest(
                             a_np,
                             b_np,
                             use_static_shape,
                             adjoint_a=adjoint_a,
                             transpose_a=transpose_a,
                             adjoint_b=adjoint_b,
                             transpose_b=transpose_b))
  test_lib.main()
