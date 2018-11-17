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

import operator
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as test_lib

# TODO(yangzihao): Currently matmul autotuning is disabled by default. Use
# os.environ["TF_MATMUL_AUTOTUNE_ENABLE"] = "1" to enable it.


class MatVecTest(test_lib.TestCase):
  """Simple test for matvec, which is sugar on top of matmul."""

  def testTwoByTwoCase(self):
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    with self.cached_session():
      c = math_ops.matvec(a, b)
      self.assertAllEqual((2,), c.shape)
      c_ = self.evaluate(c)
    self.assertAllEqual([5 + 2 * 6, 3 * 5 + 4 * 6], c_)


def _AddTest(test, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


def _GetTransposedMatrices(x, x_name, kwargs):
  if kwargs["transpose_" + x_name] is True:
    return x.T
  elif kwargs["adjoint_" + x_name] is True:
    return np.conj(x.T)
  else:
    return x


class MatMulTest(test_lib.TestCase):
  pass  # Filled in below


def _GetMatMulTest(a_np_, b_np_, use_static_shape_, **kwargs_):

  def Test(self):
    np_val = np.matrix(a_np_) * np.matrix(b_np_)

    use_gpu = True
    if a_np_.dtype is np.float16 and (
        not test_util.CudaSupportsHalfMatMulAndConv()):
      use_gpu = False
      print("Built without fp16 matmul support for Cuda, running test on CPU.")

    # Transpose and possibly conjugate a_np_ and b_np_ according to the
    # attributes such that tf.matmul(effective_a_np, effective_b_np, **kwargs)
    # results in a valid matrix multiplication and produces the same result as
    # np.matrix(a_np_) * np.matrix(b_np_)
    effective_a_np = _GetTransposedMatrices(a_np_, "a", kwargs_)
    effective_b_np = _GetTransposedMatrices(b_np_, "b", kwargs_)
    with self.session(use_gpu=use_gpu) as sess:
      if use_static_shape_:
        a = constant_op.constant(effective_a_np)
        b = constant_op.constant(effective_b_np)
        res = math_ops.matmul(a, b, **kwargs_)
        tf_val = self.evaluate(res)
      else:
        a = array_ops.placeholder(a_np_.dtype)
        b = array_ops.placeholder(b_np_.dtype)
        res = math_ops.matmul(a, b, **kwargs_)
        tf_val = sess.run(res, feed_dict={a: effective_a_np, b: effective_b_np})

    self.assertAllCloseAccordingToType(
        tf_val,
        np_val,
        float_rtol=2e-5,
        float_atol=2e-5,
        half_rtol=0.2,
        half_atol=0.2)

  return Test


class MatMulGradientTest(test_lib.TestCase):
  pass  # Will be filled in below.


def _GetMatMulGradientTest(a_np_, b_np_, use_static_shape_, **kwargs_):

  def Test(self):
    if not use_static_shape_ or a_np_.dtype in (np.int32, np.int64, np.float16):
      self.skipTest("Skipping infeasible gradient test.")

    # Transpose and possibly conjugate a_np_ and b_np_ according to the
    # attributes such that tf.matmul(effective_a_np, effective_b_np, **kwargs)
    # results in a valid matrix multiplication and produces the same result as
    # np.matrix(a_np_) * np.matrix(b_np_)
    effective_a_np = _GetTransposedMatrices(a_np_, "a", kwargs_)
    effective_b_np = _GetTransposedMatrices(b_np_, "b", kwargs_)

    epsilon = np.finfo(a_np_.dtype).eps
    delta = epsilon**(1.0 / 3.0)
    tol = 20 * delta
    with self.session(use_gpu=True):
      a = constant_op.constant(effective_a_np)
      b = constant_op.constant(effective_b_np)
      res = math_ops.matmul(a, b, **kwargs_)
      for x, x_init in [a, effective_a_np], [b, effective_b_np]:
        theoretical, numerical = gradient_checker.compute_gradient(
            x,
            x_init.shape,
            res, [a_np_.shape[0], b_np_.shape[1]],
            x_init_value=x_init,
            delta=delta)
        self.assertAllClose(theoretical, numerical, rtol=tol, atol=tol)

  return Test


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


try:
  # @ operator supported since python 3.5.
  infix_matmul = operator.matmul
except AttributeError:

  # For earlier versions of python, emulate regular behavior.
  # Useful to build and test for 3.5+ on earlier versions.
  def infix_matmul(x, y):  # pylint: disable=invalid-name
    try:
      r = type(x).__matmul__(x, y)
    except AttributeError:
      r = NotImplemented
    if r is NotImplemented and type(x) is not type(y):
      try:
        r = type(y).__rmatmul__(y, x)
      except AttributeError:
        r = NotImplemented
    if r is NotImplemented:
      raise TypeError("unsupported operand type(s) for @: '{}' and '{}'"
                      .format(type(x).__name__, type(y).__name__))
    return r


class MatMulInfixOperatorTest(test_lib.TestCase):

  def testMismatchedShape(self):
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             lambda e: "Shape must" in str(e)):
      infix_matmul(
          ops.convert_to_tensor([10.0, 20.0, 30.0]),
          ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))

  def testMismatchedDimensions(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "Dimensions must" in str(e)):
      infix_matmul(
          ops.convert_to_tensor([[10.0, 20.0, 30.0]]),
          ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))

  def testInfixMatmulIsTfMatmul(self):
    a = ops.convert_to_tensor([[10.0, 20.0, 30.0]])
    b = ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0], [80.0, 90.0]])
    c = infix_matmul(a, b)
    self.assertEqual(c.op.type, "MatMul")

  def testInfixMatmulDoesDotProduct(self):
    a = ops.convert_to_tensor([[10.0, 20.0, 30.0]])
    b = ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0], [80.0, 90.0]])
    c = infix_matmul(a, b)
    d = math_ops.matmul(a, b)
    with self.cached_session():
      self.assertAllEqual(c.eval(), self.evaluate(d))


if __name__ == "__main__":
  sizes = [1, 3, 5]
  trans_options = [[False, False], [True, False], [False, True]]
  for use_static_shape in [False, True]:
    for dtype in (np.int32, np.int64, np.float16, np.float32, np.float64,
                  np.complex64, np.complex128):
      if not use_static_shape and (dtype == np.int32 or dtype == np.int64):
        # TODO(rmlarsen): Re-enable this test when we have fixed the underlying
        # bug in Windows (b/35935459).
        continue
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
                name = "%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                    use_static_shape, dtype.__name__, m, n, k, adjoint_a,
                    transpose_a, adjoint_b, transpose_b)
                _AddTest(MatMulTest, "MatMulTest", name,
                         _GetMatMulTest(
                             a_np,
                             b_np,
                             use_static_shape,
                             adjoint_a=adjoint_a,
                             transpose_a=transpose_a,
                             adjoint_b=adjoint_b,
                             transpose_b=transpose_b))
                _AddTest(MatMulGradientTest, "MatMulGradientTest", name,
                         _GetMatMulGradientTest(
                             a_np,
                             b_np,
                             use_static_shape,
                             adjoint_a=adjoint_a,
                             transpose_a=transpose_a,
                             adjoint_b=adjoint_b,
                             transpose_b=transpose_b))

  test_lib.main()
