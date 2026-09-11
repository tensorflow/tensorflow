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
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

class Conv1DTest(test.TestCase):

  def testBasic(self):
    """Test that argument passing to conv1d is handled properly."""
    # double datatype is currently not supported for convolution ops
    # on the ROCm platform
    optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
    for dtype in [dtypes.float16, dtypes.float32] + optional_float64:
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

  def testExpandedBatch(self):
    """Test that argument passing to conv1d is handled properly."""
    # double datatype is currently not supported for convolution ops
    # on the ROCm platform
    x = constant_op.constant([1, 2, 3, 4], dtype=dtypes.float32)
    x = array_ops.expand_dims(x, 0)  # Add batch dimension
    x = array_ops.expand_dims(x, 2)  # And depth dimension
    x = array_ops_stack.stack([x, x])  # Make batch shape [2, 1]
    filters = constant_op.constant([2, 1], dtype=dtypes.float32)
    filters = array_ops.expand_dims(filters, 1)  # in_channels
    filters = array_ops.expand_dims(filters, 2)  # out_channels
    # Filters is 2x1x1
    for stride in [1, 2]:
      with self.cached_session(use_gpu=test.is_gpu_available()):
        c = nn_ops.conv1d(x, filters, stride, padding="VALID")
        reduced = array_ops.squeeze(c)  # Sequeeze out dims 1 and 3.
        output = self.evaluate(reduced)
        if stride == 1:
          self.assertAllClose(output,
                              [[2 * 1 + 1 * 2, 2 * 2 + 1 * 3, 2 * 3 + 1 * 4],
                               [2 * 1 + 1 * 2, 2 * 2 + 1 * 3, 2 * 3 + 1 * 4]])
        else:
          self.assertAllClose(
              output,
              [[2 * 1 + 1 * 2, 2 * 3 + 1 * 4], [2 * 1 + 1 * 2, 2 * 3 + 1 * 4]])

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

      for n in range(x_shape[0]):
        for k in range(f_shape[1]):
          for w in range(pad, y_shape[1] - pad):
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
    
  def testInvalidDilationValidPaddingRaises(self):
    # 1. Static shape validation fails during shape inference / graph construction.
    x = constant_op.constant(
        0.0, shape=[2, 10, 3], dtype=dtypes.float32)
    filters = constant_op.constant(
        0.0, shape=[2, 3, 1], dtype=dtypes.float32)

    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "(Negative dimension size|must be at least effective_filter_size)"):
      nn_ops.conv1d(
          x,
          filters,
          stride=1,
          padding="VALID",
          dilations=10)

  def testInvalidDilationValidPaddingRaisesDynamic(self):
    # XLA compilation bypasses the standard CPU/GPU kernels where the runtime
    # check is located.
    if test_util.is_xla_enabled():
      return
    # 2. Dynamic shape validation fails during execution (runtime).
    if context.executing_eagerly():

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[2, None, 3], dtype=dtypes.float32),
          tensor_spec.TensorSpec(shape=[2, 3, 1], dtype=dtypes.float32),
          tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32)
      ])
      def run_conv(x, filters, length):
        x_dynamic = x[:, :length, :]
        return nn_ops.conv1d(
            x_dynamic,
            filters,
            stride=1,
            padding="VALID",
            dilations=10)

      x_val = np.zeros([2, 15, 3], dtype=np.float32)
      filters_val = np.zeros([2, 3, 1], dtype=np.float32)

      dyn_len = array_ops.identity(constant_op.constant(10))

      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "(Negative dimension size|must be at least effective_filter_size)"):
        self.evaluate(run_conv(x_val, filters_val, dyn_len))

    else:
      with self.cached_session() as sess:
        x = array_ops.placeholder(dtypes.float32, shape=[2, None, 3])
        filters = array_ops.placeholder(dtypes.float32,
                                        shape=[2, 3, 1])

        output = nn_ops.conv1d(
            x,
            filters,
            stride=1,
            padding="VALID",
            dilations=10)

        x_val = np.zeros([2, 10, 3], dtype=np.float32)
        filters_val = np.zeros([2, 3, 1], dtype=np.float32)

        with self.assertRaisesRegex(
            errors.InvalidArgumentError,
            "(Negative dimension size|must be at least effective_filter_size)"):
          sess.run(output, feed_dict={
              x: x_val,
              filters: filters_val,
          })


if __name__ == "__main__":
  test.main()
