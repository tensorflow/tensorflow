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
"""Test cases for einsum op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.platform import googletest


class EinsumOpTest(xla_test.XLATestCase):
  """Test cases for einsum op."""

  def _testUnary(self, op, inp, expected):
    """Verifies that unary 'op' produces 'expected' when fed input 'inp'."""
    with self.session() as session:
      with self.test_scope():
        pinp = array_ops.placeholder(
            dtypes.as_dtype(inp.dtype), inp.shape, name='a')
        output = op(pinp)
      result = session.run(output, {pinp: inp})
      self.assertEqual(output.dtype, expected.dtype)
      self.assertAllCloseAccordingToType(
          expected, result, rtol=1e-3, atol=1e-5, bfloat16_rtol=0.03)

  def _testBinary(self, op, a, b, expected):
    """Verifies that binary 'op' produces 'expected' when fed 'a' and 'b'."""
    with self.session() as session:
      with self.test_scope():
        pa = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name='a')
        pb = array_ops.placeholder(dtypes.as_dtype(b.dtype), b.shape, name='b')
        output = op(pa, pb)
      result = session.run(output, {pa: a, pb: b})
      self.assertAllCloseAccordingToType(result, expected, rtol=1e-3)

  def testMatMul(self):
    for dtype in self.float_types:
      self._testBinary(
          lambda x, y: special_math_ops.einsum('ij,jk->ik', x, y),
          np.array([[-0.25]], dtype=dtype),
          np.array([[8]], dtype=dtype),
          expected=np.array([[-2]], dtype=dtype))

  def testImplicitForm(self):
    for dtype in self.float_types:
      self._testBinary(
          lambda x, y: special_math_ops.einsum('ijk,kji', x, y),
          np.array([[[1, 3], [2, 5], [6, 8]]], dtype=dtype),
          np.array([[[1], [3], [2]], [[5], [6], [8]]], dtype=dtype),
          expected=np.array(128, dtype=dtype))

  def testReducedIndices(self):
    for dtype in self.float_types:
      self._testBinary(
          lambda x, y: special_math_ops.einsum('ij,j->', x, y),
          np.array([[1, 3], [2, 5], [6, 8]], dtype=dtype),
          np.array([3, 2], dtype=dtype),
          expected=np.array(59, dtype=dtype))

  def testUnary(self):
    for dtype in self.float_types:
      self._testUnary(
          lambda x: special_math_ops.einsum('ijk->kji', x),
          np.array([[[1, 3], [2, 5], [6, 8]]], dtype=dtype),
          expected=np.array([[[1], [2], [6]], [[3], [5], [8]]], dtype=dtype))


if __name__ == '__main__':
  googletest.main()
