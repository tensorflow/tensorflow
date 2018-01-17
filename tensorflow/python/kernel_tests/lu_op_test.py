# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.lu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class LuOpTest(test.TestCase):

  def _verifyLU(self, x):
    for np_type in [np.float32, np.float64, np.complex64, np.complex128]:
      if np_type == np.float32 or np_type == np.complex64:
        tol = 1e-5
      else:
        tol = 1e-12
      for adjoint in False, True:
        if np_type is [np.float32, np.float64]:
          a = x.real().astype(np_type) 
        else:
          a = x.astype(np_type) 
        L, U, P, Q = linalg_ops.lu(a)
        pinv = linalg_ops.matrix_inverse(P)
        qinv = linalg_ops.matrix_inverse(Q)
        pL = math_ops.matmul(pinv, L)
        pLU = math_ops.matmul(pL, U)
        pLUq = math_ops.matmul(pLU, qinv)
        with self.test_session() as sess:
          out = pLUq.eval()
        self.assertEqual(a.shape, out.shape)
        self.assertAllClose(a, out, atol=tol, rtol=tol)

  def _generateMatrix(self, m, n):
    matrix = (np.random.normal(-5, 5,
                               m * n).astype(np.complex128).reshape([m, n]))
    matrix.imag = (np.random.normal(-5, 5, m * n).astype(np.complex128).reshape(
        [m, n]))
    return matrix

  def testLU(self):
    for n in 1, 2, 4, 9:
      matrix = self._generateMatrix(n, n)
      self._verifyLU(matrix)



if __name__ == "__main__":
  test.main()
