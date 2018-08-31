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
      rel_tol = 1e-3
      abs_tol = 1e-7
      a = x.astype(np_type) 
      l, u, p, info = math_ops.lu(a)

      # TODO(hzhuang): add test cases for the singular matrix.
      # "info" is the index for the first zero index or illegal (small) 
      # value, which might cause potential singular matrix problem for the 
      # downstream matrix solve.

      pl = math_ops.matmul(l, u) 
      #pinv = array_ops.invert_permutation(p);
      plu = array_ops.gather(pl, p) 
      udiag_ = array_ops.diag_part(u)
      with self.test_session() as sess:
        out = plu.eval()
        idx = info.eval()
        """
        udiag = udiag_.eval()
        udiag_0 = np.array(udiag)
        udiag_info = np.array(udiag)
        np.append(udiag_0, 0)
        np.append(udiag_info, info)
      self.assertAllClose(udiag_0, udiag_info, atol=abs_tol, rtol=rel_tol)
        """
      self.assertAllClose(a, out, atol=abs_tol, rtol=rel_tol)
      self.assertEqual(0, idx) # check the exit is successfully

  def _generateMatrix(self, m, n):
    # Generate random positive-definite matrix
     # TODO(hzhuang): use other matrix beyond PD matrix as factorization target 
    matrix = np.random.randint(100, size=(m, n))
    matrix = np.dot(matrix.T, matrix)
    return matrix

  def testLU(self):
    for n in 1, 4, 9, 16, 64, 128, 256, 512:
      matrix = self._generateMatrix(n, n)
      self._verifyLU(matrix)


if __name__ == "__main__":
  test.main()

