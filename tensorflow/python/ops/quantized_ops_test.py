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
"""Functional tests for quantized operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class QuantizedOpsTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedOpsTest, self).__init__(method_name)

  def testQuantizeOp(self):
    expected_output = [1, 1, 2, 127, 255, 255]
    with self.test_session(use_gpu=False) as sess:
      x = constant_op.constant([1.0, 1.25, 1.75, 127.0, 255.0, 500.0], shape=[6], dtype=dtypes.float32)
      x_min = 0.0
      x_max = 255.0
      op = array_ops.quantize(x, x_min, x_max, dtypes.quint8, mode="MIN_FIRST")
      value = sess.run(op)
      self.assertArrayNear(expected_output, value.output, 0.1)

  def testDequantizeOp(self):
    expected_output = [1.0, 2.0, 4.0, 8.0, 16.0, 255.0]
    inp = np.array([1, 2, 4, 8, 16, 255]).astype(np.uint8)
    with self.test_session(use_gpu=False) as sess:
      x = constant_op.constant(inp, shape=[6], dtype=dtypes.quint8)
      x_min = 0.0
      x_max = 255.0
      op = array_ops.dequantize(x, x_min, x_max, mode="MIN_FIRST")
      value = sess.run(op)
      self.assertArrayNear(expected_output, value, 0.1)


if __name__ == "__main__":
  test.main()
