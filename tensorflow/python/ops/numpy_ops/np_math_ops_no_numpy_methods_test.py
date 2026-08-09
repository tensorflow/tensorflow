# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf numpy math methods that must not depend on Tensor methods.

`np_math_ops.enable_numpy_methods_on_tensor()` adds numpy methods such as
`astype` to `Tensor`. The other numpy_ops test modules call it from their
`__main__`, so a code path that only works once those methods are installed
still passes there. This module deliberately does not call it, which is the
configuration a user gets from a plain `import tensorflow`.
"""

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test


# Unary ops that promote an integer argument to a floating point dtype.
_PROMOTING_UNARY_OPS = [
    ('log', np_math_ops.log, np.log),
    ('exp', np_math_ops.exp, np.exp),
    ('sqrt', np_math_ops.sqrt, np.sqrt),
    ('ceil', np_math_ops.ceil, np.ceil),
    ('floor', np_math_ops.floor, np.floor),
    ('sin', np_math_ops.sin, np.sin),
    ('cos', np_math_ops.cos, np.cos),
    ('tan', np_math_ops.tan, np.tan),
    ('sinh', np_math_ops.sinh, np.sinh),
    ('cosh', np_math_ops.cosh, np.cosh),
    ('tanh', np_math_ops.tanh, np.tanh),
    ('arcsin', np_math_ops.arcsin, np.arcsin),
    ('arccos', np_math_ops.arccos, np.arccos),
    ('arctan', np_math_ops.arctan, np.arctan),
    ('arcsinh', np_math_ops.arcsinh, np.arcsinh),
    ('arccosh', np_math_ops.arccosh, np.arccosh),
    ('arctanh', np_math_ops.arctanh, np.arctanh),
    ('expm1', np_math_ops.expm1, np.expm1),
    ('log1p', np_math_ops.log1p, np.log1p),
    ('log2', np_math_ops.log2, np.log2),
    ('log10', np_math_ops.log10, np.log10),
    ('cbrt', np_math_ops.cbrt, np.cbrt),
    ('sinc', np_math_ops.sinc, np.sinc),
    ('exp2', np_math_ops.exp2, np.exp2),
    ('deg2rad', np_math_ops.deg2rad, np.deg2rad),
    ('fix', np_math_ops.fix, np.fix),
    ('isnan', np_math_ops.isnan, np.isnan),
    ('isfinite', np_math_ops.isfinite, np.isfinite),
]


class MathWithoutNumpyMethodsOnTensorTest(test.TestCase):

  def testUnaryOpsAcceptIntegerInputs(self):
    # Integer inputs are promoted to a float dtype internally. That promotion
    # must not go through a `Tensor` method that only exists after
    # `enable_numpy_methods_on_tensor()`.
    expected_input = np.array([1, 2, 3], dtype=np.float64)
    for name, tf_fun, np_fun in _PROMOTING_UNARY_OPS:
      for arg in ([1, 2, 3],
                  np.array([1, 2, 3], dtype=np.int32),
                  np.array([1, 2, 3], dtype=np.int64)):
        actual = tf_fun(arg)
        expected = np_fun(expected_input)
        np.testing.assert_allclose(
            np.asarray(actual),
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg='{}({})'.format(name, arg))

  def testUnaryOpsAcceptIntegerScalars(self):
    for name, tf_fun, np_fun in _PROMOTING_UNARY_OPS:
      actual = tf_fun(2)
      expected = np_fun(np.float64(2))
      np.testing.assert_allclose(
          np.asarray(actual),
          expected,
          rtol=1e-6,
          atol=1e-6,
          err_msg='{}(2)'.format(name))

  def testFloatInputsAreUnchanged(self):
    arg = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    for name, tf_fun, np_fun in _PROMOTING_UNARY_OPS:
      actual = tf_fun(arg)
      np.testing.assert_allclose(
          np.asarray(actual),
          np_fun(arg),
          rtol=1e-6,
          atol=1e-6,
          err_msg='{}({})'.format(name, arg))


if __name__ == '__main__':
  ops.enable_eager_execution()
  # Intentionally not calling `np_math_ops.enable_numpy_methods_on_tensor()`.
  test.main()
