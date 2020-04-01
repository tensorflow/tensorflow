# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for XLA JIT compiler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class UnaryOpsTest(xla_test.XLATestCase):
  """Test cases for unary operators."""

  def __init__(self, method_name='runTest'):
    super(UnaryOpsTest, self).__init__(method_name)
    context.context().enable_mlir_bridge = True

  def _assertOpOutputMatchesExpected(self,
                                     op,
                                     inp,
                                     expected,
                                     equality_test=None,
                                     rtol=1e-3,
                                     atol=1e-5):
    """Verifies that 'op' produces 'expected' when fed input 'inp' .

    Args:
      op: operator to test
      inp: numpy input array to use as input to 'op'.
      expected: numpy array representing the expected output of 'op'.
      equality_test: either None, or a function that tests two numpy arrays for
        equality. If None, self.assertAllClose is used.
      rtol: relative tolerance for equality test.
      atol: absolute tolerance for equality test.
    """
    with self.session() as session:
      with self.test_scope():
        pinp = array_ops.placeholder(
            dtypes.as_dtype(inp.dtype), inp.shape, name='a')
        output = op(pinp)
      result = session.run(output, {pinp: inp})
      if equality_test is None:
        self.assertEqual(output.dtype, expected.dtype)
        self.assertAllCloseAccordingToType(
            expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03)
      else:
        equality_test(result, expected, rtol=rtol, atol=atol)

  def testNumericOps(self):
    # TODO(hinsu): Enable complex types after fixing the failure in export to
    # HLOModule.
    for dtype in self.numeric_types - {np.int8, np.uint8} - self.complex_types:
      self._assertOpOutputMatchesExpected(
          math_ops.abs,
          np.array([[2, -1]], dtype=dtype),
          expected=np.array([[2, 1]], dtype=np.real(dtype(0)).dtype))


if __name__ == '__main__':
  googletest.main()
