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
import tensorflow as tf

from tensorflow.contrib.solvers.python.ops import least_squares
from tensorflow.contrib.solvers.python.ops import util


def _add_test(test, test_name, fn):
  test_name = "_".join(["test", test_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class LeastSquaresTest(tf.test.TestCase):
  pass  # Filled in below.


def _get_least_squares_tests(dtype_, use_static_shape_, shape_):

  def test_cgls(self):
    np.random.seed(1)
    a_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape_)).reshape(shape_).astype(dtype_)
    rhs_np = np.random.uniform(
        low=-1.0, high=1.0, size=shape_[0]).astype(dtype_)
    tol = 1e-12 if dtype_ == np.float64 else 1e-6
    max_iter = 20
    with self.test_session() as sess:
      if use_static_shape_:
        a = tf.constant(a_np)
        rhs = tf.constant(rhs_np)
      else:
        a = tf.placeholder(dtype_)
        rhs = tf.placeholder(dtype_)
      operator = util.create_operator(a)
      cgls_graph = least_squares.cgls(operator, rhs, tol=tol, max_iter=max_iter)
      if use_static_shape_:
        cgls_val = sess.run(cgls_graph)
      else:
        cgls_val = sess.run(cgls_graph, feed_dict={a: a_np, rhs: rhs_np})
      # Below we use s = A^* (rhs - A x), s0 = A^* rhs
      norm_s0 = np.linalg.norm(np.dot(a_np.T, rhs_np))
      norm_s = np.sqrt(cgls_val.gamma)
      self.assertLessEqual(norm_s, tol * norm_s0)
      # Validate that we get an equally small residual norm with numpy
      # using the computed solution.
      r_np = rhs_np - np.dot(a_np, cgls_val.x)
      norm_s_np = np.linalg.norm(np.dot(a_np.T, r_np))
      self.assertLessEqual(norm_s_np, tol * norm_s0)

  return [test_cgls]


if __name__ == "__main__":
  for dtype in np.float32, np.float64:
    for shape in [[4, 4], [8, 5], [3, 7]]:
      for use_static_shape in True, False:
        arg_string = "%s_%s_staticshape_%s" % (dtype.__name__,
                                               "_".join(map(str, shape)),
                                               use_static_shape)
        for test_fn in _get_least_squares_tests(dtype, use_static_shape, shape):
          name = "_".join(["LeastSquares", test_fn.__name__, arg_string])
          _add_test(LeastSquaresTest, name, test_fn)

  tf.test.main()
