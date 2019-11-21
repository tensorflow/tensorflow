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
"""Tests for preconditioned conjugate gradient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.ops.linalg.sparse import conjugate_gradient
from tensorflow.python.platform import test


def _add_test(test_class, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test_class, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test_class, test_name, fn)


class ConjugateGradientTest(test.TestCase):
  pass  # Filled in below.


def _get_conjugate_gradient_test(dtype_, use_static_shape_, shape_):

  def _test_conjugate_gradient(self):
    np.random.seed(1)
    a_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape_)).reshape(shape_).astype(dtype_)
    # Make a self-adjoint, positive definite.
    a_np = np.dot(a_np.T, a_np)
    # jacobi preconditioner
    jacobi_np = np.zeros_like(a_np)
    jacobi_np[range(a_np.shape[0]), range(a_np.shape[1])] = (
        1.0 / a_np.diagonal())
    rhs_np = np.random.uniform(
        low=-1.0, high=1.0, size=shape_[0]).astype(dtype_)
    x_np = np.zeros_like(rhs_np)
    tol = 1e-6 if dtype_ == np.float64 else 1e-3
    max_iter = 20
    if use_static_shape_:
      a = constant_op.constant(a_np)
      rhs = constant_op.constant(rhs_np)
      x = constant_op.constant(x_np)
      jacobi = constant_op.constant(jacobi_np)
    else:
      a = array_ops.placeholder_with_default(a_np, shape=None)
      rhs = array_ops.placeholder_with_default(rhs_np, shape=None)
      x = array_ops.placeholder_with_default(x_np, shape=None)
      jacobi = array_ops.placeholder_with_default(jacobi_np, shape=None)
    operator = linalg.LinearOperatorFullMatrix(
        a, is_positive_definite=True, is_self_adjoint=True)
    preconditioners = [
        None,
        # Preconditioner that does nothing beyond change shape.
        linalg.LinearOperatorIdentity(
            a_np.shape[-1],
            dtype=a_np.dtype,
            is_positive_definite=True,
            is_self_adjoint=True),
        # Jacobi preconditioner.
        linalg.LinearOperatorFullMatrix(
            jacobi,
            is_positive_definite=True,
            is_self_adjoint=True),
    ]
    cg_results = []
    for preconditioner in preconditioners:
      cg_graph = conjugate_gradient.conjugate_gradient(
          operator,
          rhs,
          preconditioner=preconditioner,
          x=x,
          tol=tol,
          max_iter=max_iter)
      cg_val = self.evaluate(cg_graph)
      norm_r0 = np.linalg.norm(rhs_np)
      norm_r = np.linalg.norm(cg_val.r)
      self.assertLessEqual(norm_r, tol * norm_r0)
      # Validate that we get an equally small residual norm with numpy
      # using the computed solution.
      r_np = rhs_np - np.dot(a_np, cg_val.x)
      norm_r_np = np.linalg.norm(r_np)
      self.assertLessEqual(norm_r_np, tol * norm_r0)
      cg_results.append(cg_val)
    # Validate that we get same results using identity_preconditioner
    # and None
    self.assertEqual(cg_results[0].i, cg_results[1].i)
    self.assertAlmostEqual(cg_results[0].gamma, cg_results[1].gamma)
    self.assertAllClose(cg_results[0].r, cg_results[1].r, rtol=tol)
    self.assertAllClose(cg_results[0].x, cg_results[1].x, rtol=tol)
    self.assertAllClose(cg_results[0].p, cg_results[1].p, rtol=tol)

  return [_test_conjugate_gradient]


if __name__ == "__main__":
  # Set up CG tests.
  for dtype in np.float32, np.float64:
    for size in 1, 4, 10:
      for use_static_shape in set([True, False]):
        shape = [size, size]
        arg_string = "%s_%s_staticshape_%s" % (dtype.__name__, size,
                                               use_static_shape)
        for test_fn in _get_conjugate_gradient_test(
            dtype, use_static_shape, shape):
          name = "_".join(["ConjugateGradient", test_fn.__name__, arg_string])
          _add_test(
              ConjugateGradientTest,
              "ConjugateGradient",
              test_fn.__name__ + arg_string,
              test_fn)

  test.main()
