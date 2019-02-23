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
"""Tests for convolution related functionality in tensorflow.ops.nn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class Conv1DTest(test.TestCase):

  def testBasic(self):
    """Test that argument passing to conv1d is handled properly."""
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([1, 2, 3, 4], dtype=dtype)
      x = array_ops.expand_dims(x, 0)  # Add batch dimension
      x = array_ops.expand_dims(x, 2)  # And depth dimension
      filters = constant_op.constant([2, 1], dtype=dtype)
      filters = array_ops.expand_dims(filters, 1)  # in_channels
      filters = array_ops.expand_dims(filters, 2)  # out_channels
      # Filters is 2x1x1
      for stride in [1, 2]:
        with self.cached_session(use_gpu=test.is_gpu_available()):
          c = nn_ops.conv1d(x, filters, stride, padding="VALID")
          reduced = array_ops.squeeze(c)
          output = self.evaluate(reduced)
          if stride == 1:
            self.assertEqual(len(output), 3)
            self.assertAllClose(output,
                                [2 * 1 + 1 * 2, 2 * 2 + 1 * 3, 2 * 3 + 1 * 4])
          else:
            self.assertEqual(len(output), 2)
            self.assertAllClose(output, [2 * 1 + 1 * 2, 2 * 3 + 1 * 4])

  def testConv1DTranspose(self):
    with self.cached_session():
      stride = 2

      # Input, output: [batch, width, depth]
      x_shape = [2, 4, 3]
      y_shape = [2, 9, 2]

      # Filter: [kernel_width, output_depth, input_depth]
      f_shape = [3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv1d_transpose(
          x, f, y_shape, strides=stride, padding="VALID")
      value = self.evaluate(output)

      cache_values = np.zeros(y_shape, dtype=np.float32)

      # The amount of padding added
      pad = 1

      for n in xrange(x_shape[0]):
        for k in xrange(f_shape[1]):
          for w in xrange(pad, y_shape[1] - pad):
            target = 3.0
            # We add a case for locations divisible by the stride.
            w_in = w % stride == 0 and w > pad and w < y_shape[1] - 1 - pad
            if w_in:
              target += 3.0
            cache_values[n, w, k] = target

          # copy values in the border
          cache_values[n, 0, k] = cache_values[n, 1, k]
          cache_values[n, -1, k] = cache_values[n, -2, k]

    self.assertAllClose(cache_values, value)


if __name__ == "__main__":
  test.main()
