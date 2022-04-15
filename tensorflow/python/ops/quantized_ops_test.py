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
    with self.session(use_gpu=False) as sess:
      x = constant_op.constant(
          [1.0, 1.25, 1.75, 127.0, 255.0, 500.0],
          shape=[6],
          dtype=dtypes.float32)
      x_min = 0.0
      x_max = 255.0
      op = array_ops.quantize(x, x_min, x_max, dtypes.quint8, mode="MIN_FIRST")
      value = self.evaluate(op)
      self.assertArrayNear(expected_output, value.output, 0.1)

  def testDequantizeOp(self):
    expected_output = [1.0, 2.0, 4.0, 8.0, 16.0, 255.0]
    inp = np.array([1, 2, 4, 8, 16, 255]).astype(np.uint8)
    with self.session(use_gpu=False) as sess:
      x = constant_op.constant(inp, shape=[6], dtype=dtypes.quint8)
      x_min = 0.0
      x_max = 255.0
      op = array_ops.dequantize(x, x_min, x_max, mode="MIN_FIRST")
      value = self.evaluate(op)
      self.assertArrayNear(expected_output, value, 0.1)

  def testAxis(self):
    # Generates a tensor of the specified `shape` using values from `values`
    # scaled by (slice_idx + 1) along `axis` dimension.
    def scale_per_slice(shape, axis, values):
      # Note: repeats the values if the shape is larger than values.
      out = np.take(values, np.remainder(np.arange(np.prod(shape)),
                                         len(values))).reshape(shape)
      if axis is not None:
        scale_shape = [1] * len(shape)
        scale_shape[axis] = shape[axis]
        out *= np.arange(1, shape[axis] + 1).reshape(scale_shape)
      return out

    shape = np.array([2, 3, 4, 5])
    values = np.array([-1, -0.5, 0, 0.3, 0.8, 0.555, 0.5], dtype=np.float32)
    quant_values = np.array([-128, -64, 0, 38, 102, 71, 64], dtype=np.int32)
    for axis in [None, 0, 1, 2, 3]:
      inputs = constant_op.constant(scale_per_slice(shape, axis, values))
      expected_quantized = scale_per_slice(shape, None, quant_values)
      if axis is None:
        min_range, max_range = -1.0, 0.8
      else:
        num_slices = shape[axis]
        min_range, max_range = [], []
        for slice_idx in range(num_slices):
          min_range.append(-1.0 * (slice_idx + 1))
          max_range.append(0.8 * (slice_idx + 1))
      quantized = self.evaluate(
          array_ops.quantize(
              inputs,
              min_range,
              max_range,
              T=dtypes.qint8,
              mode="SCALED",
              round_mode="HALF_TO_EVEN",
              axis=axis)).output
      self.assertAllEqual(quantized, expected_quantized)
      if axis is not None:
        quantized = self.evaluate(
            array_ops.quantize(
                inputs,
                min_range,
                max_range,
                T=dtypes.qint8,
                mode="SCALED",
                round_mode="HALF_TO_EVEN",
                axis=(axis - 4))).output
        self.assertAllClose(quantized, expected_quantized)

if __name__ == "__main__":
  test.main()
