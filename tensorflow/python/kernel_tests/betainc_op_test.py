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
"""Functional tests for 3d convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


class BetaincTest(test.TestCase):
  use_gpu = False

  def _testBetaInc(self, dtype):
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      np_dt = dtype.as_numpy_dtype

      # Test random values
      a_s = np.abs(np.random.randn(10, 10) * 30).astype(np_dt)  # in (0, infty)
      b_s = np.abs(np.random.randn(10, 10) * 30).astype(np_dt)  # in (0, infty)
      x_s = np.random.rand(10, 10).astype(np_dt)  # in (0, 1)
      with self.test_session(use_gpu=self.use_gpu):
        tf_a_s = constant_op.constant(a_s, dtype=dtype)
        tf_b_s = constant_op.constant(b_s, dtype=dtype)
        tf_x_s = constant_op.constant(x_s, dtype=dtype)
        tf_out = math_ops.betainc(tf_a_s, tf_b_s, tf_x_s).eval()
      scipy_out = special.betainc(a_s, b_s, x_s).astype(np_dt)

      # the scipy version of betainc uses a double-only implementation.
      # TODO(ebrevdo): identify reasons for (sometime) precision loss
      # with doubles
      tol = 1e-4 if dtype == dtypes.float32 else 5e-5
      self.assertAllCloseAccordingToType(scipy_out, tf_out, rtol=tol, atol=tol)

      # Test out-of-range values (most should return nan output)
      combinations = list(itertools.product([-1, 0, 0.5, 1.0, 1.5], repeat=3))
      a_comb, b_comb, x_comb = np.asarray(list(zip(*combinations)), dtype=np_dt)
      with self.test_session(use_gpu=self.use_gpu):
        tf_comb = math_ops.betainc(a_comb, b_comb, x_comb).eval()
      scipy_comb = special.betainc(a_comb, b_comb, x_comb).astype(np_dt)
      self.assertAllCloseAccordingToType(scipy_comb, tf_comb)

      # Test broadcasting between scalars and other shapes
      with self.test_session(use_gpu=self.use_gpu):
        self.assertAllCloseAccordingToType(
            special.betainc(0.1, b_s, x_s).astype(np_dt),
            math_ops.betainc(0.1, b_s, x_s).eval(),
            rtol=tol,
            atol=tol)
        self.assertAllCloseAccordingToType(
            special.betainc(a_s, 0.1, x_s).astype(np_dt),
            math_ops.betainc(a_s, 0.1, x_s).eval(),
            rtol=tol,
            atol=tol)
        self.assertAllCloseAccordingToType(
            special.betainc(a_s, b_s, 0.1).astype(np_dt),
            math_ops.betainc(a_s, b_s, 0.1).eval(),
            rtol=tol,
            atol=tol)
        self.assertAllCloseAccordingToType(
            special.betainc(0.1, b_s, 0.1).astype(np_dt),
            math_ops.betainc(0.1, b_s, 0.1).eval(),
            rtol=tol,
            atol=tol)
        self.assertAllCloseAccordingToType(
            special.betainc(0.1, 0.1, 0.1).astype(np_dt),
            math_ops.betainc(0.1, 0.1, 0.1).eval(),
            rtol=tol,
            atol=tol)

      with self.assertRaisesRegexp(ValueError, "must be equal"):
        math_ops.betainc(0.5, [0.5], [[0.5]])

      with self.test_session(use_gpu=self.use_gpu):
        with self.assertRaisesOpError("Shapes of .* are inconsistent"):
          a_p = array_ops.placeholder(dtype)
          b_p = array_ops.placeholder(dtype)
          x_p = array_ops.placeholder(dtype)
          math_ops.betainc(a_p, b_p, x_p).eval(
              feed_dict={a_p: 0.5,
                         b_p: [0.5],
                         x_p: [[0.5]]})

    except ImportError as e:
      tf_logging.warn("Cannot test special functions: %s" % str(e))

  def testBetaIncFloat(self):
    self._testBetaInc(dtypes.float32)

  def testBetaIncDouble(self):
    self._testBetaInc(dtypes.float64)


class BetaincTestGPU(BetaincTest):
  use_gpu = True


if __name__ == "__main__":
  test.main()
