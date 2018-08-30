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
     # TODO(hzhuang): add test np.complex64, np.complex128
    for np_type in [np.float32, np.float64]: #, np.complex64, np.complex128]:
      if np_type == np.float32 or np_type == np.float64:
        rel_tol = 1e-3
        abs_tol = 1e-7
        a = x.astype(np_type) 
      l, u, p, info = math_ops.lu(a)
      # - add test cases for the singular matrix.
      # "info" is the index for the first zero index or illegal (small) 
      # value, which might causes potential singular matrix problem for the 
      # downstream matrix solve.
      # TODO(hzhuang): how to test here?
      pp = np.zeros(p.shape)
      pl = math_ops.matmul(l, u) 
      #pinv = array_ops.invert_permutation(p);
      plu = array_ops.gather(pl, p) 
      with self.test_session() as sess:
        out = plu.eval()
      self.assertAllClose(a, out, atol=abs_tol, rtol=rel_tol)

  def _generateMatrix(self, m, n):
    # Generate random positive-definite matrix
     # TODO(hzhuang): use other matrix beyond PD matrix as factorization target 
    matrix = np.random.rand(m, n)
    matrix = np.dot(matrix.T, matrix)
    return matrix

  def testLU(self):
    for n in 1, 4, 9, 16, 64, 128, 256, 512, 1024, 2048:
      matrix = self._generateMatrix(n, n)
      self._verifyLU(matrix)

if __name__ == "__main__":
  test.main()

