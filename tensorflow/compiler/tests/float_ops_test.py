# Copyright 2025 The OpenXLA Authors.
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
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


class FloatOpsTest(xla_test.XLATestCase):

  def test_float_ops(self):
    for dtype in self.float_types:
      x = np.arange(-0.90, 0.90, 0.25)
      self.assert_op_output_matches_expected(
          math_ops.acos, x.astype(dtype), expected=np.arccos(x).astype(dtype)
      )
      self.assert_op_output_matches_expected(
          math_ops.asin, x.astype(dtype), expected=np.arcsin(x).astype(dtype)
      )
      x = np.arange(-3, 3).reshape(1, 3, 2)
      self.assert_op_output_matches_expected(
          math_ops.atan, x.astype(dtype), expected=np.arctan(x).astype(dtype)
      )

      self.assert_op_output_matches_expected(
          math_ops.acosh,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [0, 1.3169579, 1.76274717, 2.06343707], dtype=dtype
          ),
      )

      self.assert_op_output_matches_expected(
          math_ops.asinh,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [0.88137359, 1.44363548, 1.81844646, 2.09471255], dtype=dtype
          ),
      )

      self.assert_op_output_matches_expected(
          math_ops.atanh,
          np.array([0.1, 0.2, 0.3, 0.4], dtype=dtype),
          expected=np.array(
              [0.10033535, 0.20273255, 0.3095196, 0.42364893], dtype=dtype
          ),
      )

      self.assert_op_output_matches_expected(
          math_ops.ceil,
          np.array([[-1.7, 1.2]], dtype=dtype),
          expected=np.array([[-1, 2]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          math_ops.cosh,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [1.54308063, 3.76219569, 10.067662, 27.30823284], dtype=dtype
          ),
      )

      # Disable float16 testing for now
      if dtype != np.float16:
        x = np.arange(-10, 10, 1).astype(dtype)
        with self.session() as session:
          erf_x = session.run(math_ops.erf(x))
          erfc_x = session.run(math_ops.erfc(x))

        self.assert_op_output_matches_expected(math_ops.erf, x, expected=erf_x)
        self.assert_op_output_matches_expected(
            math_ops.erfc, x, expected=erfc_x
        )

      self.assert_op_output_matches_expected(
          math_ops.exp,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([[0.36787945, 2.7182817]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          math_ops.expm1,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([[-0.63212056, 1.71828183]], dtype=dtype),
          rtol=1e-5,
      )

      self.assert_op_output_matches_expected(
          math_ops.floor,
          np.array([[-1.7, 1.2]], dtype=dtype),
          expected=np.array([[-2, 1]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          math_ops.is_finite,
          np.array(
              [[-np.inf, -2, -1, 0, 0.5, 1, 2, np.inf, np.nan]], dtype=dtype
          ),
          expected=np.array([[0, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=np.bool_),
      )

      # Tests for tf.nn ops.
      self.assert_op_output_matches_expected(
          nn_ops.l2_loss, np.array([[[]]], dtype=dtype), expected=dtype(0)
      )

      self.assert_op_output_matches_expected(nn_ops.l2_loss, dtype(4), dtype(8))

      self.assert_op_output_matches_expected(
          nn_ops.l2_loss, np.array([[-2, 4]], dtype=dtype), expected=dtype(10)
      )

      self.assert_op_output_matches_expected(
          math_ops.reciprocal,
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[1, 0.5]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          math_ops.log,
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[0, 0.69314718]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          math_ops.sin,
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[0.841478, 0.909302]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          math_ops.cos,
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[0.540297, -0.41614]], dtype=dtype),
      )

      # Confirm that log1p will remain precise across a range of small values.
      self.assert_op_output_matches_expected(
          math_ops.log1p,
          np.array(
              [[1e-14, 1e-15, 0.6, 2] + [x * 1e-5 for x in range(1, 20)]],
              dtype=dtype,
          ),
          expected=np.log1p(
              np.array(
                  [[1e-14, 1e-15, 0.6, 2] + [x * 1e-5 for x in range(1, 20)]],
                  dtype=dtype,
              )
          ).astype(dtype),
          rtol=1e-15 if dtype == np.float64 else 1e-4,
          atol=1e-15 if dtype == np.float64 else 1e-4,
      )

      self.assert_op_output_matches_expected(
          math_ops.rint,
          np.array(
              [
                  [-1.7, 1.2, 4.0, 0.0],
                  [-3.5, -2.5, -1.5, -0.5],
                  [0.5, 1.5, 2.5, 3.5],
              ],
              dtype=dtype,
          ),
          expected=np.array(
              [[-2, 1, 4, 0], [-4, -2, -2, 0], [0, 2, 2, 4]], dtype=dtype
          ),
      )
      self.assert_op_output_matches_expected(
          math_ops.round,
          np.array(
              [
                  [-1.7, 1.2, 4.0, 0.0],
                  [-3.5, -2.5, -1.5, -0.5],
                  [0.5, 1.5, 2.5, 3.5],
              ],
              dtype=dtype,
          ),
          expected=np.array(
              [[-2, 1, 4, 0], [-4, -2, -2, 0], [0, 2, 2, 4]], dtype=dtype
          ),
      )

      self.assert_op_output_matches_expected(
          math_ops.rsqrt,
          np.array([[4, 16]], dtype=dtype),
          expected=np.array([[0.5, 0.25]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          math_ops.sigmoid,
          np.array([[1, 1, 1, 1], [1, 2, 3, 4]], dtype=dtype),
          expected=np.array(
              [
                  [0.7310586, 0.7310586, 0.7310586, 0.7310586],
                  [0.7310586, 0.880797, 0.95257413, 0.98201376],
              ],
              dtype=dtype,
          ),
      )

      self.assert_op_output_matches_expected(
          math_ops.sigmoid,
          np.array([-300, -150, 0, 150, 300], dtype=dtype),
          expected=np.array([0, 0, 0.5, 1, 1], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          math_ops.sinh,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [1.17520119, 3.62686041, 10.01787493, 27.2899172], dtype=dtype
          ),
      )

      self.assert_op_output_matches_expected(
          math_ops.sqrt,
          np.array([[4, 9]], dtype=dtype),
          expected=np.array([[2, 3]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          math_ops.tan,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [1.55740772, -2.18503986, -0.14254654, 1.15782128], dtype=dtype
          ),
      )

      self.assert_op_output_matches_expected(
          math_ops.tanh,
          np.array(
              [[1, 2, 3, 4], [np.inf, -np.inf, np.nan, 20], [19, -19, 22, -22]],
              dtype=dtype,
          ),
          expected=np.array(
              [
                  [0.76159418, 0.96402758, 0.99505478, 0.99932933],
                  [1.0, -1.0, np.nan, 1.0],
                  [1.0, -1.0, 1.0, -1.0],
              ],
              dtype=dtype,
          ),
      )

      self.assert_op_output_matches_expected(
          nn_ops.log_softmax,
          np.array([[1, 1, 1, 1], [1, 2, 3, 4]], dtype=dtype),
          expected=np.array(
              [
                  [-1.3862944, -1.3862944, -1.3862944, -1.3862944],
                  [-3.4401896, -2.4401896, -1.4401897, -0.44018969],
              ],
              dtype=dtype,
          ),
      )

      self.assert_op_output_matches_expected(
          nn_ops.elu,
          np.array([[-1, 0, 1, -1e-6]], dtype=dtype),
          expected=np.array([[-0.63212056, 0, 1, -9.999995e-07]], dtype=dtype),
          rtol=1e-5,
          atol=1e-6,
      )

      self.assert_op_output_matches_expected(
          nn_ops.selu,
          np.array([[-1, 0, 1, -1e-5]], dtype=dtype),
          expected=np.array(
              [[-1.11133074, 0.0, 1.05070099, -1.758090550379974e-05]],
              dtype=dtype,
          ),
          rtol=1e-5,
          atol=1e-6,
      )

      self.assert_op_output_matches_expected(
          nn_ops.relu,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([[0, 1]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          nn_ops.relu6,
          np.array([[-0.05, 6.05, 5]], dtype=dtype),
          expected=np.array([[0, 6, 5]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          nn_ops.leaky_relu,
          np.array([[-2, -1, 0, 1, 2]], dtype=dtype),
          expected=np.array([[-0.4, -0.2, 0.0, 1.0, 2.0]], dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          nn_ops.softmax,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [0.032058604, 0.087144323, 0.23688284, 0.64391428], dtype=dtype
          ),
      )

      self.assert_op_output_matches_expected(
          nn_ops.softmax,
          np.array([[1, 1, 1, 1], [1, 2, 3, 4]], dtype=dtype),
          expected=np.array(
              [
                  [0.25, 0.25, 0.25, 0.25],
                  [0.032058604, 0.087144323, 0.23688284, 0.64391428],
              ],
              dtype=dtype,
          ),
      )

      self.assert_op_output_matches_expected(
          nn_ops.softmax,
          np.array([[[1, 1], [1, 1]], [[1, 2], [3, 4]]], dtype=dtype),
          expected=np.array(
              [
                  [[0.5, 0.5], [0.5, 0.5]],
                  [[0.26894142, 0.73105858], [0.26894142, 0.73105858]],
              ],
              dtype=dtype,
          ),
      )

      self.assert_op_output_matches_expected(
          nn_ops.softsign,
          np.array([[-2, -1, 0, 1, 2]], dtype=dtype),
          expected=np.array(
              [[-0.66666669, -0.5, 0, 0.5, 0.66666669]], dtype=dtype
          ),
      )

      self.assert_op_output_matches_expected(
          math_ops.sign,
          np.array(
              [[-2.0, -1.0, -0.0, +0.0, 1.0, 2.0, float("nan")]], dtype=dtype
          ),
          expected=np.array(
              [[-1.0, -1.0, -0.0, +0.0, 1.0, 1.0, float("nan")]], dtype=dtype
          ),
      )

      self.assert_op_output_matches_expected(
          math_ops.is_finite,
          np.array(
              [[42, float("inf"), -123], [float("nan"), 0, -0.0]], dtype=dtype
          ),
          expected=np.array(
              [[True, False, True], [False, True, True]], dtype=np.bool_
          ),
      )

      self.assert_op_output_matches_expected(
          math_ops.lgamma,
          np.array(0.5, dtype=dtype),
          expected=np.array(np.log(np.pi) / 2, dtype=dtype),
      )

      self.assert_op_output_matches_expected(
          math_ops.lgamma,
          np.array(
              [
                  [1, 2, 3],
                  [4, 5, 6],
                  [1 / 2, 3 / 2, 5 / 2],
                  [-3 / 2, -7 / 2, -11 / 2],
              ],
              dtype=dtype,
          ),
          expected=np.array(
              [
                  [0, 0, np.log(2.0)],
                  [np.log(6.0), np.log(24.0), np.log(120)],
                  [
                      np.log(np.pi) / 2,
                      np.log(np.pi) / 2 - np.log(2),
                      np.log(np.pi) / 2 - np.log(4) + np.log(3),
                  ],
                  [
                      np.log(np.pi) / 2 - np.log(3) + np.log(4),
                      np.log(np.pi) / 2 - np.log(105) + np.log(16),
                      np.log(np.pi) / 2 - np.log(10395) + np.log(64),
                  ],
              ],
              dtype=dtype,
          ),
      )

      # The actual result is complex. Take the real part.
      self.assert_op_output_matches_expected(
          math_ops.lgamma,
          np.array([-1 / 2, -5 / 2, -9 / 2], dtype=dtype),
          expected=np.array(
              [
                  np.log(np.pi) / 2 + np.log(2),
                  np.log(np.pi) / 2 - np.log(15) + np.log(8),
                  np.log(np.pi) / 2 - np.log(945) + np.log(32),
              ],
              dtype=dtype,
          ),
          atol=1e-4,
      )

      self.assert_op_output_matches_expected(
          math_ops.digamma,
          np.array(
              [
                  [1.0, 0.5, 1 / 3.0],
                  [0.25, 1 / 6.0, 0.125],
                  [2.0, 3.0, 4.0],
                  [6.0, 8.0, 9.0],
              ],
              dtype=dtype,
          ),
          expected=np.array(
              [
                  [
                      -np.euler_gamma,
                      -2 * np.log(2) - np.euler_gamma,
                      -np.pi / 2 / np.sqrt(3)
                      - 3 * np.log(3) / 2
                      - np.euler_gamma,
                  ],
                  [
                      -np.pi / 2 - 3 * np.log(2) - np.euler_gamma,
                      -np.pi * np.sqrt(3) / 2
                      - 2 * np.log(2)
                      - 3 * np.log(3) / 2
                      - np.euler_gamma,
                      -np.pi / 2
                      - 4 * np.log(2)
                      - (
                          np.pi
                          + np.log(2 + np.sqrt(2))
                          - np.log(2 - np.sqrt(2))
                      )
                      / np.sqrt(2)
                      - np.euler_gamma,
                  ],
                  [
                      1 - np.euler_gamma,
                      1.5 - np.euler_gamma,
                      11 / 6.0 - np.euler_gamma,
                  ],
                  [
                      137 / 60.0 - np.euler_gamma,
                      363 / 140.0 - np.euler_gamma,
                      761 / 280.0 - np.euler_gamma,
                  ],
              ],
              dtype=dtype,
          ),
      )


if __name__ == "__main__":
  googletest.main()
