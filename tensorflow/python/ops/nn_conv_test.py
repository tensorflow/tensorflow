# Copyright 2015 Google Inc. All Rights Reserved.
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


class AtrousConv2DTest(tf.test.TestCase):

  def _upsample_filters(self, filters, rate):
    """Upsamples the filters by a factor of rate along the spatial dimensions.

    Args:
      filters: [h, w, in_depth, out_depth]. Original filters.
      rate: An int, specifying the upsampling rate.

    Returns:
      filters_up: [h_up, w_up, in_depth, out_depth]. Upsampled filters with
        h_up = h + (h - 1) * (rate - 1)
        w_up = w + (w - 1) * (rate - 1)
        containing (rate - 1) zeros between consecutive filter values along
        the filters' spatial dimensions.
    """
    if rate == 1:
      return filters
    # [h, w, in_depth, out_depth] -> [in_depth, out_depth, h, w]
    filters_up = np.transpose(filters, [2, 3, 0, 1])
    ker = np.zeros([rate, rate])
    ker[0, 0] = 1
    filters_up = np.kron(filters_up, ker)[:, :, :-(rate-1), :-(rate-1)]
    # [in_depth, out_depth, h_up, w_up] -> [h_up, w_up, in_depth, out_depth]
    filters_up = np.transpose(filters_up, [2, 3, 0, 1])
    self.assertEqual(np.sum(filters), np.sum(filters_up))
    return filters_up

  def testAtrousConv2DForward(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        # Input: [batch, height, width, input_depth]
        height = 15
        for width in [15, 16]:  # Test both odd and even width.
          x_shape = [2, height, width, 2]
          x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)

          # Filter: [kernel_height, kernel_width, input_depth, output_depth]
          for kernel_height in range(1, 5):
            for kernel_width in range(1, 5):
              f_shape = [kernel_height, kernel_width, 2, 2]
              f = np.arange(np.prod(f_shape), dtype=np.float32).reshape(f_shape)

              for rate in range(1, 5):
                f_up = self._upsample_filters(f, rate)

                for padding in ["SAME", "VALID"]:
                  y1 = tf.nn.atrous_conv2d(x, f, rate, padding=padding)
                  y2 = tf.nn.conv2d(x, f_up, strides=[1, 1, 1, 1],
                                    padding=padding)
                  self.assertAllClose(y1.eval(), y2.eval(), rtol=1e-2,
                                      atol=1e-2)

  def testAtrousSequence(self):
    """Tests optimization of sequence of atrous convolutions.

    Verifies that a sequence of `atrous_conv2d` operations with identical `rate`
    parameters, 'SAME' `padding`, and `filters` with odd heights/ widths:

        net = atrous_conv2d(net, filters1, rate, padding="SAME")
        net = atrous_conv2d(net, filters2, rate, padding="SAME")
        ...
        net = atrous_conv2d(net, filtersK, rate, padding="SAME")

    is equivalent to:

        pad = ...  # padding so that the input dims are multiples of rate
        net = space_to_batch(net, paddings=pad, block_size=rate)
        net = conv2d(net, filters1, strides=[1, 1, 1, 1], padding="SAME")
        net = conv2d(net, filters2, strides=[1, 1, 1, 1], padding="SAME")
        ...
        net = conv2d(net, filtersK, strides=[1, 1, 1, 1], padding="SAME")
        net = batch_to_space(net, crops=pad, block_size=rate)
    """
    padding = "SAME"  # The padding needs to be "SAME"
    np.random.seed(1)  # Make it reproducible.

    with self.test_session():
      # Input: [batch, height, width, input_depth]
      for height in range(15, 17):
        for width in range(15, 17):
          x_shape = [3, height, width, 2]
          x = np.random.random_sample(x_shape).astype(np.float32)

          for kernel in [1, 3, 5]:  # The kernel size needs to be odd.
            # Filter: [kernel_height, kernel_width, input_depth, output_depth]
            f_shape = [kernel, kernel, 2, 2]
            f = 1e-2 * np.random.random_sample(f_shape).astype(np.float32)

            for rate in range(2, 4):
              # y1: three atrous_conv2d in a row.
              y1 = tf.nn.atrous_conv2d(x, f, rate, padding=padding)
              y1 = tf.nn.atrous_conv2d(y1, f, rate, padding=padding)
              y1 = tf.nn.atrous_conv2d(y1, f, rate, padding=padding)
              # y2: space_to_batch, three conv2d in a row, batch_to_space
              pad_bottom = 0 if height % rate == 0 else rate - height % rate
              pad_right = 0 if width % rate == 0 else rate - width % rate
              pad = [[0, pad_bottom], [0, pad_right]]
              y2 = tf.space_to_batch(x, paddings=pad, block_size=rate)
              y2 = tf.nn.conv2d(y2, f, strides=[1, 1, 1, 1], padding=padding)
              y2 = tf.nn.conv2d(y2, f, strides=[1, 1, 1, 1], padding=padding)
              y2 = tf.nn.conv2d(y2, f, strides=[1, 1, 1, 1], padding=padding)
              y2 = tf.batch_to_space(y2, crops=pad, block_size=rate)
              self.assertAllClose(y1.eval(), y2.eval(), rtol=1e-2, atol=1e-2)

  def testGradient(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        # Input: [batch, height, width, input_depth]
        x_shape = [2, 5, 6, 2]
        # Filter: [kernel_height, kernel_width, input_depth, output_depth]
        f_shape = [3, 3, 2, 2]
        # Output: [batch, height, width, output_depth]
        y_shape = [2, 5, 6, 2]

        np.random.seed(1)  # Make it reproducible.
        x_val = np.random.random_sample(x_shape).astype(np.float32)
        f_val = np.random.random_sample(f_shape).astype(np.float32)
        x = tf.constant(x_val, name="x", dtype=tf.float32)
        f = tf.constant(f_val, name="f", dtype=tf.float32)

        for rate in range(1, 4):
          output = tf.nn.atrous_conv2d(x, f, rate=rate, padding="SAME")
          err = tf.test.compute_gradient_error(
              [x, f], [x_shape, f_shape], output, y_shape)
          print("atrous_conv2d gradient err = %g " % err)
          err_tolerance = 1e-3
          self.assertLess(err, err_tolerance)


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
    print("DeConv gradient err = %g " % err)
    err_tolerance = 0.0005
    self.assertLess(err, err_tolerance)


if __name__ == "__main__":
  tf.test.main()
