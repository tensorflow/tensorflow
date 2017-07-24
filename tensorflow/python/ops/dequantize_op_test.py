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
"""Tests for Dequantize Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DequantizeOpTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(DequantizeOpTest, self).__init__(method_name)

  def _testDequantizeOp(self, inputs, min_range, max_range, dtype):
    with self.test_session():
      input_op = constant_op.constant(inputs, shape=[len(inputs)], dtype=dtype)
      dequantized = array_ops.dequantize(input_op, min_range, max_range)
      tf_ans = dequantized.eval()

    # TODO(vrv): Add support for DT_QINT32 quantization if needed.
    type_dict = {
        dtypes.quint8: np.uint8,
        dtypes.qint8: np.int8,
        dtypes.quint16: np.uint16,
        dtypes.qint16: np.int16
    }
    self.assertTrue(dtype in type_dict.keys())
    v_max = np.iinfo(type_dict[dtype]).max
    v_min = np.iinfo(type_dict[dtype]).min
    self.assertTrue(min_range >= v_min)
    self.assertTrue(max_range <= v_max)
    type_range = v_max - v_min
    if v_min < 0:
      half_range = (type_range + 1) / 2
    else:
      half_range = 0.0

    np_ans = ((inputs.astype(np.float32) + half_range) *
              (max_range - min_range) / type_range) + min_range
    self.assertAllClose(tf_ans, np_ans, rtol=1e-5, atol=1e-5)

  def testBasicQuint8(self):
    self._testDequantizeOp(np.array([0, 128, 255]), 0.0, 6.0, dtypes.quint8)
    self._testDequantizeOp(np.array([0, 128, 255]), 0.0, 123.456, dtypes.quint8)
    self._testDequantizeOp(
        np.array([0, 4, 42, 108, 243]), 5.0, 200.2, dtypes.quint8)

  def testBasicQint8(self):
    self._testDequantizeOp(np.array([-128, 0, 127]), -1.0, 2.0, dtypes.qint8)
    self._testDequantizeOp(np.array([-2, 4, -17]), -5.0, -3.0, dtypes.qint8)
    self._testDequantizeOp(np.array([0, -4, 42, -108]), 5.0, 40.0, dtypes.qint8)


if __name__ == "__main__":
  test.main()
