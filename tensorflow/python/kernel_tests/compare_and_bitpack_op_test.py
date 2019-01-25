# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.compare_and_bitpack_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class CompareAndBitpackTest(test.TestCase):

  def _testCompareAndBitpack(self,
                             x, threshold,
                             truth,
                             expected_err_re=None):
    with test_util.use_gpu():
      ans = math_ops.compare_and_bitpack(x, threshold)
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        self.assertShapeEqual(truth, ans)
        self.assertAllEqual(tf_ans, truth)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testBasic(self, dtype):
    rows = 371
    cols = 294
    x = np.random.randn(rows, cols * 8)
    if dtype == np.bool:
      x = x > 0
    else:
      x = x.astype(dtype)
    threshold = dtype(0)
    # np.packbits flattens the tensor, so we reshape it back to the
    # expected dimensions.
    truth = np.packbits(x > threshold).reshape(rows, cols)
    self._testCompareAndBitpack(x, threshold, truth)

  def testBasicFloat32(self):
    self._testBasic(np.float32)

  def testBasicFloat64(self):
    self._testBasic(np.float64)

  def testBasicFloat16(self):
    self._testBasic(np.float16)

  def testBasicBool(self):
    self._testBasic(np.bool)

  def testBasicInt8(self):
    self._testBasic(np.int8)

  def testBasicInt16(self):
    self._testBasic(np.int16)

  def testBasicInt32(self):
    self._testBasic(np.int32)

  def testBasicInt64(self):
    self._testBasic(np.int64)


if __name__ == "__main__":
  test.main()
