# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import tf2
from tensorflow.contrib.solvers.python.ops import linear_equations
from tensorflow.contrib.solvers.python.ops import util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test as test_lib


def _add_test(test, test_name, fn):
  test_name = "_".join(["test", test_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class LinearEquationsTest(test_lib.TestCase):
  pass  # Filled in below.


def _get_linear_equations_tests(dtype_, use_static_shape_, shape_):

  def test_conjugate_gradient(self):
    np.random.seed(1)
    a_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape_)).reshape(shape_).astype(dtype_)
    # Make a selfadjoint, positive definite.
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
    with self.cached_session() as sess:
      if use_static_shape_:
        a = constant_op.constant(a_np)
        rhs = constant_op.constant(rhs_np)
        x = constant_op.constant(x_np)
        jacobi = constant_op.constant(jacobi_np)
      else:
        a = array_ops.placeholder(dtype_)
        rhs = array_ops.placeholder(dtype_)
        x = array_ops.placeholder(dtype_)
        jacobi = array_ops.placeholder(dtype_)
      operator = util.create_operator(a)
      preconditioners = [
          None, util.identity_operator(a),
          util.create_operator(jacobi)
      ]
      cg_results = []
      for preconditioner in preconditioners:
        cg_graph = linear_equations.conjugate_gradient(
            operator,
            rhs,
            preconditioner=preconditioner,
            x=x,
            tol=tol,
            max_iter=max_iter)
        if use_static_shape_:
          cg_val = sess.run(cg_graph)
        else:
          cg_val = sess.run(
              cg_graph,
              feed_dict={
                  a: a_np,
                  rhs: rhs_np,
                  x: x_np,
                  jacobi: jacobi_np
              })
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

  return [test_conjugate_gradient]


if __name__ == "__main__":
  for dtype in np.float32, np.float64:
    for size in 1, 4, 10:
      # TF2 does not support placeholders under eager so we skip it
      for use_static_shape in set([True, tf2.enabled()]):
        shape = [size, size]
        arg_string = "%s_%s_staticshape_%s" % (dtype.__name__, size,
                                               use_static_shape)
        for test_fn in _get_linear_equations_tests(dtype, use_static_shape,
                                                   shape):
          name = "_".join(["LinearEquations", test_fn.__name__, arg_string])
          _add_test(LinearEquationsTest, name, test_fn)

  test_lib.main()
