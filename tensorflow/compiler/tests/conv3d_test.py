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
"""Tests for 3D convolutions using the XLA JIT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest


# Test cloned from
# tensorflow/python/kernel_tests/conv3d_backprop_filter_v2_grad_test.py
class Conv3DBackpropFilterV2GradTest(xla_test.XLATestCase):

  def testGradient(self):
    with self.cached_session(), self.test_scope():
      for padding in ["SAME", "VALID"]:
        for stride in [1, 2]:
          np.random.seed(1)
          in_shape = [2, 4, 3, 3, 2]
          in_val = constant_op.constant(
              2 * np.random.random_sample(in_shape) - 1, dtype=dtypes.float32)
          filter_shape = [3, 3, 3, 2, 3]
          strides = [1, stride, stride, stride, 1]
          # Make a convolution op with the current settings, just to easily get
          # the shape of the output.
          conv_out = nn_ops.conv3d(in_val,
                                   array_ops.zeros(filter_shape), strides,
                                   padding)
          out_backprop_shape = conv_out.get_shape().as_list()
          out_backprop_val = constant_op.constant(
              2 * np.random.random_sample(out_backprop_shape) - 1,
              dtype=dtypes.float32)
          output = nn_ops.conv3d_backprop_filter_v2(in_val, filter_shape,
                                                    out_backprop_val, strides,
                                                    padding)
          err = gradient_checker.compute_gradient_error(
              [in_val, out_backprop_val], [in_shape, out_backprop_shape],
              output, filter_shape)
          print("conv3d_backprop_filter gradient err = %g " % err)
          err_tolerance = 1e-3
          self.assertLess(err, err_tolerance)


# Test cloned from tensorflow/python/kernel_tests/conv3d_transpose_test.py
class Conv3DTransposeTest(xla_test.XLATestCase):

  def testConv3DTransposeSingleStride(self):
    with self.cached_session(), self.test_scope():
      strides = [1, 1, 1, 1, 1]

      # Input, output: [batch, depth, height, width, channel]
      x_shape = [2, 5, 6, 4, 3]
      y_shape = [2, 5, 6, 4, 2]

      # Filter: [kernel_depth, kernel_height, kernel_width, out_depth, in_depth]
      f_shape = [3, 3, 3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv3d_transpose(
          x, f, y_shape, strides=strides, padding="SAME")
      value = output.eval()

      # We count the number of cells being added at the locations in the output.
      # At the center, #cells = kernel_depth * kernel_height * kernel_width
      # At the corners, #cells = ceil(kernel_depth/2) * ceil(kernel_height/2)
      #                          * ceil(kernel_width/2)
      # At the edges, #cells =
      #   kernel_depth * ceil(kernel_height/2) * ceil(kernel_width/2) or
      #   ceil(kernel_depth/2) * kernel_height * ceil(kernel_width/2) or
      #   ceil(kernel_depth/2) * ceil(kernel_height/2) * kernel_width
      # At the borders, #cells =
      #   ceil(kernel_depth/2) * kernel_height * kernel_width or
      #   kernel_depth * ceil(kernel_height/2) * kernel_width or
      #   kernel_depth * kernel_height * ceil(kernel_width/2)

      for n in xrange(x_shape[0]):
        for k in xrange(f_shape[3]):
          for w in xrange(y_shape[3]):
            for h in xrange(y_shape[2]):
              for d in xrange(y_shape[1]):
                d_in = d > 0 and d < y_shape[1] - 1
                h_in = h > 0 and h < y_shape[2] - 1
                w_in = w > 0 and w < y_shape[3] - 1
                if d_in + h_in + w_in == 3:
                  target = 27 * 3.0
                elif d_in + h_in + w_in == 2:
                  target = 18 * 3.0
                elif d_in or h_in or w_in:
                  target = 12 * 3.0
                else:
                  target = 8 * 3.0
                self.assertAllClose(target, value[n, d, h, w, k])

  def testConv3DTransposeSame(self):
    with self.cached_session(), self.test_scope():
      strides = [1, 2, 2, 2, 1]

      # Input, output: [batch, depth, height, width, depth]
      x_shape = [2, 5, 6, 4, 3]
      y_shape = [2, 10, 12, 8, 2]

      # Filter: [kernel_depth, kernel_height, kernel_width, out_depth, in_depth]
      f_shape = [3, 3, 3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv3d_transpose(
          x, f, y_shape, strides=strides, padding="SAME")
      value = output.eval()

      for n in xrange(x_shape[0]):
        for k in xrange(f_shape[3]):
          for w in xrange(y_shape[3]):
            for h in xrange(y_shape[2]):
              for d in xrange(y_shape[1]):
                # We add a case for locations divisible by the stride.
                d_in = d % strides[1] == 0 and 0 < d < y_shape[1] - 1
                h_in = h % strides[2] == 0 and 0 < h < y_shape[2] - 1
                w_in = w % strides[3] == 0 and 0 < w < y_shape[3] - 1
                if d_in + h_in + w_in == 3:
                  target = 8 * 3.0
                elif d_in + h_in + w_in == 2:
                  target = 4 * 3.0
                elif d_in or h_in or w_in:
                  target = 2 * 3.0
                else:
                  target = 3.0
                self.assertAllClose(target, value[n, d, h, w, k])

  def testConv3DTransposeValid(self):
    with self.cached_session(), self.test_scope():
      strides = [1, 2, 2, 2, 1]

      # Input, output: [batch, depth, height, width, depth]
      x_shape = [2, 5, 6, 4, 3]
      y_shape = [2, 11, 13, 9, 2]

      # Filter: [kernel_depth, kernel_height, kernel_width, out_depth, in_depth]
      f_shape = [3, 3, 3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv3d_transpose(
          x, f, y_shape, strides=strides, padding="VALID")
      value = output.eval()

      cache_values = np.zeros(y_shape, dtype=np.float32)

      # The amount of padding added
      pad = 1

      for n in xrange(x_shape[0]):
        for k in xrange(f_shape[3]):
          for w in xrange(y_shape[3]):
            for h in xrange(y_shape[2]):
              for d in xrange(y_shape[1]):
                # We add a case for locations divisible by the stride.
                d_in = d % strides[1] == 0 and pad < d < y_shape[1] - 1 - pad
                h_in = h % strides[2] == 0 and pad < h < y_shape[2] - 1 - pad
                w_in = w % strides[3] == 0 and pad < w < y_shape[3] - 1 - pad
                if d_in + h_in + w_in == 3:
                  target = 8 * 3.0
                elif d_in + h_in + w_in == 2:
                  target = 4 * 3.0
                elif d_in or h_in or w_in:
                  target = 2 * 3.0
                else:
                  target = 3.0
                cache_values[n, d, h, w, k] = target

          # copy values in the border
          cache_values[n, :, :, 0, k] = cache_values[n, :, :, 1, k]
          cache_values[n, :, :, -1, k] = cache_values[n, :, :, -2, k]
          cache_values[n, :, 0, :, k] = cache_values[n, :, 1, :, k]
          cache_values[n, :, -1, :, k] = cache_values[n, :, -2, :, k]
          cache_values[n, 0, :, :, k] = cache_values[n, 1, :, :, k]
          cache_values[n, -1, :, :, k] = cache_values[n, -2, :, :, k]

    self.assertAllClose(cache_values, value)

  def testGradient(self):
    x_shape = [2, 3, 4, 3, 2]
    f_shape = [3, 3, 3, 2, 2]
    y_shape = [2, 6, 8, 6, 2]
    strides = [1, 2, 2, 2, 1]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    f_val = np.random.random_sample(f_shape).astype(np.float64)
    with self.cached_session(), self.test_scope():
      x = constant_op.constant(x_val, name="x", dtype=dtypes.float32)
      f = constant_op.constant(f_val, name="f", dtype=dtypes.float32)
      output = nn_ops.conv3d_transpose(
          x, f, y_shape, strides=strides, padding="SAME")
      err = gradient_checker.compute_gradient_error([x, f], [x_shape, f_shape],
                                                    output, y_shape)
    print("conv3d_transpose gradient err = %g " % err)
    err_tolerance = 0.0005
    self.assertLess(err, err_tolerance)


if __name__ == "__main__":
  googletest.main()
