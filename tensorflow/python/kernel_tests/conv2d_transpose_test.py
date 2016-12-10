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
import tensorflow as tf
from tensorflow.python.client import device_lib


class Conv2DTransposeTest(tf.test.TestCase):

  def testConv2DTransposeSingleStride(self):
    with self.test_session():
      strides = [1, 1, 1, 1]

      # Input, output: [batch, height, width, depth]
      x_shape = [2, 6, 4, 3]
      y_shape = [2, 6, 4, 2]

      # Filter: [kernel_height, kernel_width, output_depth, input_depth]
      f_shape = [3, 3, 2, 3]

      x = tf.constant(1.0, shape=x_shape, name="x", dtype=tf.float32)
      f = tf.constant(1.0, shape=f_shape, name="filter", dtype=tf.float32)
      output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                      padding="SAME")
      value = output.eval()

      # We count the number of cells being added at the locations in the output.
      # At the center, #cells=kernel_height * kernel_width
      # At the corners, #cells=ceil(kernel_height/2) * ceil(kernel_width/2)
      # At the borders, #cells=ceil(kernel_height/2)*kernel_width or
      #                        kernel_height * ceil(kernel_width/2)

      for n in xrange(x_shape[0]):
        for k in xrange(f_shape[2]):
          for w in xrange(y_shape[2]):
            for h in xrange(y_shape[1]):
              target = 4 * 3.0
              h_in = h > 0 and h < y_shape[1] - 1
              w_in = w > 0 and w < y_shape[2] - 1
              if h_in and w_in:
                target += 5 * 3.0
              elif h_in or w_in:
                target += 2 * 3.0
              self.assertAllClose(target, value[n, h, w, k])

  def testConv2DTransposeSame(self):
    with self.test_session():
      strides = [1, 2, 2, 1]

      # Input, output: [batch, height, width, depth]
      x_shape = [2, 6, 4, 3]
      y_shape = [2, 12, 8, 2]

      # Filter: [kernel_height, kernel_width, output_depth, input_depth]
      f_shape = [3, 3, 2, 3]

      x = tf.constant(1.0, shape=x_shape, name="x", dtype=tf.float32)
      f = tf.constant(1.0, shape=f_shape, name="filter", dtype=tf.float32)
      output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                      padding="SAME")
      value = output.eval()

      for n in xrange(x_shape[0]):
        for k in xrange(f_shape[2]):
          for w in xrange(y_shape[2]):
            for h in xrange(y_shape[1]):
              target = 3.0
              # We add a case for locations divisible by the stride.
              h_in = h % strides[1] == 0 and h > 0 and h < y_shape[1] - 1
              w_in = w % strides[2] == 0 and w > 0 and w < y_shape[2] - 1
              if h_in and w_in:
                target += 9.0
              elif h_in or w_in:
                target += 3.0
              self.assertAllClose(target, value[n, h, w, k])

  def testConv2DTransposeValid(self):
    with self.test_session():
      strides = [1, 2, 2, 1]

      # Input, output: [batch, height, width, depth]
      x_shape = [2, 6, 4, 3]
      y_shape = [2, 13, 9, 2]

      # Filter: [kernel_height, kernel_width, output_depth, input_depth]
      f_shape = [3, 3, 2, 3]

      x = tf.constant(1.0, shape=x_shape, name="x", dtype=tf.float32)
      f = tf.constant(1.0, shape=f_shape, name="filter", dtype=tf.float32)
      output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                      padding="VALID")
      value = output.eval()

      cache_values = np.zeros(y_shape, dtype=np.float32)

      # The amount of padding added
      pad = 1

      for n in xrange(x_shape[0]):
        for k in xrange(f_shape[2]):
          for w in xrange(pad, y_shape[2] - pad):
            for h in xrange(pad, y_shape[1] - pad):
              target = 3.0
              # We add a case for locations divisible by the stride.
              h_in = h % strides[
                  1] == 0 and h > pad and h < y_shape[1] - 1 - pad
              w_in = w % strides[
                  2] == 0 and w > pad and w < y_shape[2] - 1 - pad
              if h_in and w_in:
                target += 9.0
              elif h_in or w_in:
                target += 3.0
              cache_values[n, h, w, k] = target

          # copy values in the border
          cache_values[n, :, 0, k] = cache_values[n, :, 1, k]
          cache_values[n, :, -1, k] = cache_values[n, :, -2, k]
          cache_values[n, 0, :, k] = cache_values[n, 1, :, k]
          cache_values[n, -1, :, k] = cache_values[n, -2, :, k]

    self.assertAllClose(cache_values, value)

  def testGradient(self):
    x_shape = [2, 6, 4, 3]
    f_shape = [3, 3, 2, 3]
    y_shape = [2, 12, 8, 2]
    strides = [1, 2, 2, 1]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    f_val = np.random.random_sample(f_shape).astype(np.float64)
    with self.test_session():
      x = tf.constant(x_val, name="x", dtype=tf.float32)
      f = tf.constant(f_val, name="f", dtype=tf.float32)
      output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                      padding="SAME")
      err = tf.test.compute_gradient_error(
          [x, f], [x_shape, f_shape], output, y_shape)
    print("conv2d_transpose gradient err = %g " % err)
    err_tolerance = 0.0005
    self.assertLess(err, err_tolerance)

  def testConv2DTransposeSingleStrideNCHW(self):
    # `NCHW` data fomat is only supported for CUDA device.
    if tf.test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        strides = [1, 1, 1, 1]

        # Input, output: [batch, depth, height, width, depth]
        x_shape = [2, 3, 6, 4]
        y_shape = [2, 2, 6, 4]

        # Filter: [kernel_height, kernel_width, output_depth, input_depth]
        f_shape = [3, 3, 2, 3]

        x = tf.constant(1.0, shape=x_shape, name="x", dtype=tf.float32)
        f = tf.constant(1.0, shape=f_shape, name="filter", dtype=tf.float32)

        output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                     padding="SAME", data_format='NCHW')

        value = output.eval()
        for n in xrange(x_shape[0]):
          for k in xrange(f_shape[2]):
            for w in xrange(y_shape[3]):
              for h in xrange(y_shape[2]):
                target = 4 * 3.0
                h_in = h > 0 and h < y_shape[2] - 1
                w_in = w > 0 and w < y_shape[3] - 1
                if h_in and w_in:
                  target += 5 * 3.0
                elif h_in or w_in:
                  target += 2 * 3.0
                self.assertAllClose(target, value[n, k, h, w])

  def testConv2DTransposeSameNCHW(self):
    # `NCHW` data fomat is only supported for CUDA device.
    if tf.test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        strides = [1, 1, 2, 2]

        # Input, output: [batch, depth, height, width]
        x_shape = [2, 3, 6, 4]
        y_shape = [2, 2, 12, 8]

        # Filter: [kernel_height, kernel_width, output_depth, input_depth]
        f_shape = [3, 3, 2, 3]

        x = tf.constant(1.0, shape=x_shape, name="x", dtype=tf.float32)
        f = tf.constant(1.0, shape=f_shape, name="filter", dtype=tf.float32)

        output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                          padding="SAME", data_format='NCHW')

        value = output.eval()
        for n in xrange(x_shape[0]):
          for k in xrange(f_shape[2]):
            for w in xrange(y_shape[3]):
              for h in xrange(y_shape[2]):
                target = 3.0
                # We add a case for locations divisible by the stride.
                h_in = h % strides[2] == 0 and h > 0 and h < y_shape[2] - 1
                w_in = w % strides[3] == 0 and w > 0 and w < y_shape[3] - 1
                if h_in and w_in:
                  target += 9.0
                elif h_in or w_in:
                  target += 3.0
                self.assertAllClose(target, value[n, k, h, w])

  def testConv2DTransposeValidNCHW(self):
    # `NCHW` data fomat is only supported for CUDA device.
    if tf.test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        strides = [1, 1, 2, 2]

        # Input, output: [batch, depth, height, width]
        x_shape = [2, 3, 6, 4]
        y_shape = [2, 2, 13, 9]

        # Filter: [kernel_height, kernel_width, output_depth, input_depth]
        f_shape = [3, 3, 2, 3]

        x = tf.constant(1.0, shape=x_shape, name="x", dtype=tf.float32)
        f = tf.constant(1.0, shape=f_shape, name="filter", dtype=tf.float32)
        output = tf.nn.conv2d_transpose(x, f, y_shape, strides=strides,
                                        padding="VALID", data_format='NCHW')

        value = output.eval()
        cache_values = np.zeros(y_shape, dtype=np.float32)
        # The amount of padding added
        pad = 1
        for n in xrange(x_shape[0]):
          for k in xrange(f_shape[2]):
            for w in xrange(pad, y_shape[3] - pad):
              for h in xrange(pad, y_shape[2] - pad):
                target = 3.0
                # We add a case for locations divisible by the stride.
                h_in = h % strides[
                    2] == 0 and h > pad and h < y_shape[2] - 1 - pad
                w_in = w % strides[
                    3] == 0 and w > pad and w < y_shape[3] - 1 - pad
                if h_in and w_in:
                  target += 9.0
                elif h_in or w_in:
                  target += 3.0
                cache_values[n, k, h, w] = target

            # copy values in the border
            cache_values[n, k, :, 0] = cache_values[n, k, :, 1]
            cache_values[n, k, :, -1] = cache_values[n, k, :, -2]
            cache_values[n, k, 0, :] = cache_values[n, k, 1, :]
            cache_values[n, k, -1, :] = cache_values[n, k, -2, :]

        self.assertAllClose(cache_values, value)


if __name__ == "__main__":
  tf.test.main()
