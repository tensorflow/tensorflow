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
from tensorflow.contrib.solvers.python.ops import lanczos
from tensorflow.contrib.solvers.python.ops import util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test as test_lib


def _add_test(test, test_name, fn):
  test_name = "_".join(["test", test_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class LanczosBidiagTest(test_lib.TestCase):
  pass  # Filled in below.


def _get_lanczos_tests(dtype_, use_static_shape_, shape_, orthogonalize_,
                       steps_):

  def test_lanczos_bidiag(self):
    np.random.seed(1)
    a_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape_)).reshape(shape_).astype(dtype_)
    tol = 1e-12 if dtype_ == np.float64 else 1e-5

    with self.cached_session() as sess:
      if use_static_shape_:
        a = constant_op.constant(a_np)
      else:
        a = array_ops.placeholder(dtype_)
      operator = util.create_operator(a)
      lbd = lanczos.lanczos_bidiag(
          operator, steps_, orthogonalize=orthogonalize_)

      # The computed factorization should satisfy the equations
      #  A * V = U * B
      #  A' * U[:, :-1] = V * B[:-1, :]'
      av = math_ops.matmul(a, lbd.v)
      ub = lanczos.bidiag_matmul(lbd.u, lbd.alpha, lbd.beta, adjoint_b=False)
      atu = math_ops.matmul(a, lbd.u[:, :-1], adjoint_a=True)
      vbt = lanczos.bidiag_matmul(lbd.v, lbd.alpha, lbd.beta, adjoint_b=True)

      if use_static_shape_:
        av_val, ub_val, atu_val, vbt_val = sess.run([av, ub, atu, vbt])
      else:
        av_val, ub_val, atu_val, vbt_val = sess.run([av, ub, atu, vbt],
                                                    feed_dict={a: a_np})
      self.assertAllClose(av_val, ub_val, atol=tol, rtol=tol)
      self.assertAllClose(atu_val, vbt_val, atol=tol, rtol=tol)

  return [test_lanczos_bidiag]


if __name__ == "__main__":
  for dtype in np.float32, np.float64:
    for shape in [[4, 4], [7, 4], [5, 8]]:
      for orthogonalize in True, False:
        for steps in range(1, min(shape) + 1):
          # TF2 does not support placeholders so we skip it
          for use_static_shape in set([True, tf2.enabled()]):
            arg_string = "%s_%s_%s_%s_staticshape_%s" % (
                dtype.__name__, "_".join(map(str, shape)), orthogonalize, steps,
                use_static_shape)
            for test_fn in _get_lanczos_tests(dtype, use_static_shape, shape,
                                              orthogonalize, steps):
              name = "_".join(["Lanczos", test_fn.__name__, arg_string])
              _add_test(LanczosBidiagTest, name, test_fn)

  test_lib.main()
