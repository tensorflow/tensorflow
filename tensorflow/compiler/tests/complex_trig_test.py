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
"""Regression test for XLA complex sin/cos overflow (issue #116944).

When |Im(z)| is near or above ~88 (float32) or ~709 (float64), the XLA
elemental IR emitter used to compute ``half_exp_neg_y`` as ``FDiv(0.5, exp_y)``
which overflows when ``exp_y`` underflows to 0 (negative imaginary side).
This caused ``tf.math.sin`` / ``tf.math.cos`` to return inf/nan for
representable inputs. Eager mode was correct. The fix replaces that division
with two independent ``exp(y + log(1/2))`` / ``exp(-y + log(1/2))`` calls,
mirroring the existing builder-level Cosh/Sinh formulation.
"""

import os

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

os.environ["XLA_FLAGS"] = "--xla_cpu_fast_math_honor_nans=true"


class ComplexTrigTest(xla_test.XLATestCase):
  """Regression tests for complex sin/cos with large imaginary part."""

  def _run(self, op, dtype, x):
    with self.session() as sess:
      with self.test_scope():
        x_ph = array_ops.placeholder(dtypes.as_dtype(dtype), x.shape)
        out = op(x_ph)
      return sess.run(out, {x_ph: x})

  def testSinComplexLargeImaginary(self):
    # Im(z) values spanning the float32 overflow boundary: |y| = 80 (safe),
    # 88 (just inside), and 89 (just at the boundary). Include both
    # positive and negative y to cover the sign asymmetry that the buggy
    # FDiv(0.5, exp_y) exhibited (e.g. 0 - 88j used to be nan - infj).
    imag_values = [80.0, -80.0, 88.0, -88.0]
    real_values = [0.0, 0.5, -1.0, 1.5]
    inputs = np.array(
        [complex(r, i) for i in imag_values for r in real_values],
        dtype=np.complex128,
    )
    for dtype in self.complex_types:
      x = inputs.astype(dtype)
      actual = self._run(math_ops.sin, dtype, x)
      expected = np.sin(x)
      self.assertAllCloseAccordingToType(actual, expected, rtol=1e-3)

  def testCosComplexLargeImaginary(self):
    imag_values = [80.0, -80.0, 88.0, -88.0]
    real_values = [0.0, 0.5, -1.0, 1.5]
    inputs = np.array(
        [complex(r, i) for i in imag_values for r in real_values],
        dtype=np.complex128,
    )
    for dtype in self.complex_types:
      x = inputs.astype(dtype)
      actual = self._run(math_ops.cos, dtype, x)
      expected = np.cos(x)
      self.assertAllCloseAccordingToType(actual, expected, rtol=1e-3)


if __name__ == "__main__":
  googletest.main()
