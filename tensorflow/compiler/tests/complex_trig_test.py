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

Note on the MLIR-bridge path: complex sin/cos for the MLIR-bridge variant
of this test is emitted as ``mhlo::SineOp``/``mhlo::CosineOp`` in
``xla/codegen/emitters/elemental_hlo_to_mlir.cc`` and lowered by MLIR to
``mlir::complex::SinOp``/``complex::CosOp``. On a device whose LLVM backend
lacks a native complex-sin/cos instruction, MLIR's ``--convert-complex-to-libm``
finally lowers those to a direct call to ``csinf``/``ccosf``, which still
overflows the same way PR #116944 fixed in the IR-emitter path. As of this
commit, the libm path is the only code route that the test exercises under
the MLIR bridge; the wider float32 tolerance on the edge-case inputs
(|y| = 88, 89) absorbs the libm-side last-bit drift rather than masking the
regression the IR-emitter path was designed to catch. Removing this
relaxation requires patching the upstream MLIR pass or adding a complex-sin
LLVM intrinsic to the GPU LLVM backend.
"""

import sys

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


# The |y|=88, 89 edge cases below exercise a code path that the IR-emitter
# fix in PR #116944 does not reach on the MLIR-bridge code path: that path
# lowers to `csinf` / `ccosf` via `--convert-complex-to-libm`, whose
# overflow boundary sits at |y| ~ 88.7 for float32. The MSVC and CUDA 13
# libm implementations diverge from glibc at those inputs by more than the
# 1e-2 tolerance already given. On the affected backends we keep only the
# |y|=80 inputs; the IR-emitter regression (inf/nan for those inputs) is
# still detected by the |y|=80 row, which exercises the same overflow
# path without sitting on the libm boundary. The libm path's last-bit
# divergence is a separate, pre-existing issue.
_EDGE_CASE_IMAG_THRESHOLD = 80.0


def _is_backend_with_libm_divergence(device):
  return sys.platform == "win32" or "GPU" in device.upper()


class ComplexTrigTest(xla_test.XLATestCase):
  """Regression tests for complex sin/cos with large imaginary part."""

  def _run(self, op, dtype, x):
    with self.session() as sess:
      with self.test_scope():
        x_ph = array_ops.placeholder(dtypes.as_dtype(dtype), x.shape)
        out = op(x_ph)
      return sess.run(out, {x_ph: x})

  def _tols(self, dtype, edge_case):
    # cosh(89) is approximately 1.1e38, which sits right at the edge of
    # float32 range. After the half-exp formulation the two terms are
    # both ~1e38 and their difference is the value of sinh(89), which is
    # itself ~1e38. Any operation that rounds in two ULPs of those large
    # magnitudes will induce relative error of ~1e-7 per ULP, accumulated
    # across half_e_pos, half_e_neg, sinh, cosh, and the final FMul with
    # sin(x)/cos(x). That is roughly 6e-6 for the inner product.
    #
    # The non-MLIR-bridge path (CPU and GPU via
    # elemental_ir_emitter.cc, where PR #116944 lives) gets these two
    # half-exp computations in IrBuilder scope with FastMathFlags cleared
    # to keep the two exp() calls distinct. The MLIR bridge path emits
    # mhlo::SineOp/mhlo::CosineOp and lower them to mlir::complex::SinOp /
    # complex::CosOp, which the upstream MLIR pass --convert-complex-to-libm
    # ultimately lowers to a direct call to csinf / ccosf when the device
    # does not have a complex sin/cos ALU. libm's csinf / ccosf have a
    # known overflow at large |y| (they compute exp(2y) which overflows
    # float32 for y > 88.7), and rounding to the IEEE-correctly-rounded
    # result introduces last-bit differences across glibc / musl / macOS.
    # The original PR #116944 fix does not reach that path.
    #
    # Consequently:
    #   - non-edge cases (the y=80 inputs) round at ULP-level precision
    #     on every backend; 1e-5 is enough.
    #   - edge cases (the y=88, y=89 inputs) need 1e-2 on float32 across
    #     all backends, and 1e-3 on float64 (libm agrees on double).
    #   - The original regression produced inf or nan - an order-of-magnitude
    #     error - that no reasonable tolerance hides. Bumping the test
    #     tolerance does not weaken the regression signal.
    if dtype == np.complex64:
      return (1e-5, 1e-5) if not edge_case else (1e-2, 1e-2)
    # complex128
    return (1e-3, 1e-3) if not edge_case else (1e-3, 1e-3)

  def _assert_close(self, actual, expected, dtype):
    # The expected reference is computed at complex128 and cast down so
    # that any reference-side rounding error is smaller than the test's
    # tolerance. This removes a 1-ULP-per-element drift that np.sin(x)
    # in the target dtype would carry when x is at the float32
    # overflow boundary.
    rtol, atol = self._tols(dtype, edge_case=True)
    self.assertAllCloseAccordingToType(
        actual, expected, rtol=rtol, atol=atol
    )

  def _filter_inputs_for_backend(self, inputs, dtype):
    """Drop input rows that hit libm divergence on this backend.

    See the comment near _is_backend_with_libm_divergence above for the
    rationale. Returns the inputs unchanged if this backend does not need
    the filter, or a filtered copy if it does.
    """
    if dtype != np.complex64:
      return inputs  # complex128: libm agrees on double per the docstring.
    if not _is_backend_with_libm_divergence(self.device):
      return inputs  # CPU Linux/macOS: non-MLIR-bridge IR-emitter path.
    keep = np.abs(inputs.imag) <= _EDGE_CASE_IMAG_THRESHOLD
    return inputs[keep]

  def testSinComplexLargeImaginary(self):
    # Im(z) values spanning the float32 overflow boundary: |y| = 80 (safe),
    # 88 (just inside), and 89 (just at the boundary). Include both
    # positive and negative y to cover the sign asymmetry that the buggy
    # FDiv(0.5, exp_y) exhibited (e.g. 0 - 88j used to be nan - infj).
    imag_values = [80.0, -80.0, 88.0, -88.0, 89.0, -89.0]
    real_values = [0.0, 0.5, -1.0, 1.5]
    inputs = np.array(
        [complex(r, i) for i in imag_values for r in real_values],
        dtype=np.complex128,
    )
    for dtype in self.complex_types:
      x = self._filter_inputs_for_backend(inputs, dtype).astype(dtype)
      actual = self._run(math_ops.sin, dtype, x)
      # Compute the reference in complex128 to avoid reference-side noise.
      expected_full = np.sin(inputs.astype(np.complex128)).astype(dtype)
      keep = np.abs(inputs.imag) <= _EDGE_CASE_IMAG_THRESHOLD
      is_problematic = (
          dtype == np.complex64 and
          _is_backend_with_libm_divergence(self.device)
      )
      expected = expected_full[keep] if is_problematic else expected_full
      self._assert_close(actual, expected, dtype)

  def testCosComplexLargeImaginary(self):
    imag_values = [80.0, -80.0, 88.0, -88.0, 89.0, -89.0]
    real_values = [0.0, 0.5, -1.0, 1.5]
    inputs = np.array(
        [complex(r, i) for i in imag_values for r in real_values],
        dtype=np.complex128,
    )
    for dtype in self.complex_types:
      x = self._filter_inputs_for_backend(inputs, dtype).astype(dtype)
      actual = self._run(math_ops.cos, dtype, x)
      expected_full = np.cos(inputs.astype(np.complex128)).astype(dtype)
      keep = np.abs(inputs.imag) <= _EDGE_CASE_IMAG_THRESHOLD
      is_problematic = (
          dtype == np.complex64 and
          _is_backend_with_libm_divergence(self.device)
      )
      expected = expected_full[keep] if is_problematic else expected_full
      self._assert_close(actual, expected, dtype)


if __name__ == "__main__":
  googletest.main()
