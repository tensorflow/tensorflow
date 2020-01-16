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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import googletest


class SLogDetOpTest(xla_test.XLATestCase):

  def testSimple(self):
    # 2x2 matrices
    matrix_np = np.array([[4., 6., 8., 10.], [6., 45., 54., 63.],
                          [8., 54., 146., 166.], [10., 63., 166., 310.]])

    with self.session() as sess:
      matrix = array_ops.placeholder(dtype=np.float32, shape=(4, 4))
      with self.test_scope():
        log_det = linalg_impl.slogdet(matrix)
      _, result = sess.run(log_det, {matrix: matrix_np})
    expected = 14.1601
    self.assertAllClose(result, expected, 1e-4)

  def testSimpleBatched(self):
    # 2x2 matrices
    matrix_np = np.array([[[4., 6., 8., 10.], [6., 45., 54., 63.],
                           [8., 54., 146., 166.], [10., 63., 166., 310.]],
                          [[16., 24., 8., 12.], [24., 61., 82., 48.],
                           [8., 82., 456., 106.], [12., 48., 106., 62.]]])

    with self.session() as sess:
      matrix = array_ops.placeholder(dtype=np.float32, shape=(2, 4, 4))
      with self.test_scope():
        log_det = linalg_impl.slogdet(matrix)
      _, result = sess.run(log_det, {matrix: matrix_np})
    expected = [14.1601, 14.3092]
    self.assertAllClose(result, expected, 1e-4)


if __name__ == "__main__":
  googletest.main()
