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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class Conv2DTransposeTest(test.TestCase):

  def testConv2DTransposeSingleStride(self):
    with self.cached_session():
      for dtype in (dtypes.float32, dtypes.int32):
        strides = [1, 1, 1, 1]

        # Input, output: [batch, height, width, depth]
        x_shape = [2, 6, 4, 3]
        y_shape = [2, 6, 4, 2]

        # Filter: [kernel_height, kernel_width, output_depth, input_depth]
        f_shape = [3, 3, 2, 3]

        x = constant_op.constant(1, shape=x_shape, name="x", dtype=dtype)
        f = constant_op.constant(1, shape=f_shape, name="filter", dtype=dtype)
        output = nn_ops.conv2d_transpose(
            x, f, y_shape, strides=strides, padding="SAME")
        value = self.evaluate(output)

        # We count the number of cells being added at the locations in the
        # output.
        # At the center, #cells=kernel_height * kernel_width
        # At the corners, #cells=ceil(kernel_height/2) * ceil(kernel_width/2)
        # At the borders, #cells=ceil(kernel_height/2)*kernel_width or
        #                        kernel_height * ceil(kernel_width/2)

        for n in range(x_shape[0]):
          for k in range(f_shape[2]):
            for w in range(y_shape[2]):
              for h in range(y_shape[1]):
                target = 4 * 3
                h_in = h > 0 and h < y_shape[1] - 1
                w_in = w > 0 and w < y_shape[2] - 1
                if h_in and w_in:
                  target += 5 * 3
                elif h_in or w_in:
                  target += 2 * 3
                if dtype.is_integer:
                  self.assertAllEqual(target, value[n, h, w, k])
                else:
                  self.assertAllClose(target, value[n, h, w, k])

  def testConv2DTransposeSame(self):
    with self.cached_session():
      for dtype in (dtypes.float32, dtypes.int32):
        strides = [1, 2, 2, 1]

        # Input, output: [batch, height, width, depth]
        x_shape = [2, 6, 4, 3]
        y_shape = [2, 12, 8, 2]

        # Filter: [kernel_height, kernel_width, output_depth, input_depth]
        f_shape = [3, 3, 2, 3]

        x = constant_op.constant(1, shape=x_shape, name="x", dtype=dtype)
        f = constant_op.constant(1, shape=f_shape, name="filter", dtype=dtype)
        output = nn_ops.conv2d_transpose(
            x, f, y_shape, strides=strides, padding="SAME")
        value = self.evaluate(output)

        for n in range(x_shape[0]):
          for k in range(f_shape[2]):
            for w in range(y_shape[2]):
              for h in range(y_shape[1]):
                target = 3
                # We add a case for locations divisible by the stride.
                h_in = h % strides[1] == 0 and h > 0 and h < y_shape[1] - 1
                w_in = w % strides[2] == 0 and w > 0 and w < y_shape[2] - 1
                if h_in and w_in:
                  target += 9
                elif h_in or w_in:
                  target += 3

                if dtype.is_integer:
                  self.assertAllEqual(target, value[n, h, w, k])
                else:
                  self.assertAllClose(target, value[n, h, w, k])

  def testConv2DTransposeValid(self):
    with self.cached_session():
      for dtype in (dtypes.float32, dtypes.int32):
        strides = [1, 2, 2, 1]

        # Input, output: [batch, height, width, depth]
        x_shape = [2, 6, 4, 3]
        y_shape = [2, 13, 9, 2]

        # Filter: [kernel_height, kernel_width, output_depth, input_depth]
        f_shape = [3, 3, 2, 3]

        x = constant_op.constant(1, shape=x_shape, name="x", dtype=dtype)
        f = constant_op.constant(1, shape=f_shape, name="filter", dtype=dtype)
        output = nn_ops.conv2d_transpose(
            x, f, y_shape, strides=strides, padding="VALID")
        value = self.evaluate(output)

        cache_values = np.zeros(y_shape, dtype=np.float32)

        # The amount of padding added
        pad = 1

        for n in range(x_shape[0]):
          for k in range(f_shape[2]):
            for w in range(pad, y_shape[2] - pad):
              for h in range(pad, y_shape[1] - pad):
                target = 3
                # We add a case for locations divisible by the stride.
                h_in = h % strides[1] == 0 and h > pad and h < y_shape[
                    1] - 1 - pad
                w_in = w % strides[2] == 0 and w > pad and w < y_shape[
                    2] - 1 - pad
                if h_in and w_in:
                  target += 9
                elif h_in or w_in:
                  target += 3
                cache_values[n, h, w, k] = target

            # copy values in the border
            cache_values[n, :, 0, k] = cache_values[n, :, 1, k]
            cache_values[n, :, -1, k] = cache_values[n, :, -2, k]
            cache_values[n, 0, :, k] = cache_values[n, 1, :, k]
            cache_values[n, -1, :, k] = cache_values[n, -2, :, k]

        if dtype.is_integer:
          self.assertAllEqual(cache_values, value)
        else:
          self.assertAllClose(cache_values, value)

  @test_util.run_deprecated_v1
  def testGradient(self):
    self.skipTest("b/262851489: Fix nightly build for GPU.")
    x_shape = [2, 6, 4, 3]
    f_shape = [3, 3, 2, 3]
    y_shape = [2, 12, 8, 2]
    strides = [1, 2, 2, 1]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    f_val = np.random.random_sample(f_shape).astype(np.float64)
    with self.cached_session():
      x = constant_op.constant(x_val, name="x", dtype=dtypes.float32)
      f = constant_op.constant(f_val, name="f", dtype=dtypes.float32)
      output = nn_ops.conv2d_transpose(
          x, f, y_shape, strides=strides, padding="SAME")
      err = gradient_checker.compute_gradient_error([x, f], [x_shape, f_shape],
                                                    output, y_shape)
    print("conv2d_transpose gradient err = %g " % err)
    err_tolerance = 0.0006
    self.assertLess(err, err_tolerance)

  def testConv2DTransposeSingleStrideNCHW(self):
    # `NCHW` data format is only supported for CUDA device.
    if test.is_gpu_available(cuda_only=True):
      with self.session():
        strides = [1, 1, 1, 1]

        # Input, output: [batch, depth, height, width, depth]
        x_shape = [2, 3, 6, 4]
        y_shape = [2, 2, 6, 4]

        # Filter: [kernel_height, kernel_width, output_depth, input_depth]
        f_shape = [3, 3, 2, 3]

        x = constant_op.constant(
            1.0, shape=x_shape, name="x", dtype=dtypes.float32)
        f = constant_op.constant(
            1.0, shape=f_shape, name="filter", dtype=dtypes.float32)

        output = nn_ops.conv2d_transpose(
            x, f, y_shape, strides=strides, padding="SAME", data_format="NCHW")

        value = self.evaluate(output)
        for n in range(x_shape[0]):
          for k in range(f_shape[2]):
            for w in range(y_shape[3]):
              for h in range(y_shape[2]):
                target = 4 * 3.0
                h_in = h > 0 and h < y_shape[2] - 1
                w_in = w > 0 and w < y_shape[3] - 1
                if h_in and w_in:
                  target += 5 * 3.0
                elif h_in or w_in:
                  target += 2 * 3.0
                self.assertAllClose(target, value[n, k, h, w])

  def testConv2DTransposeSameNCHW(self):
    # `NCHW` data format is only supported for CUDA device.
    if test.is_gpu_available(cuda_only=True):
      with self.session():
        strides = [1, 1, 2, 2]

        # Input, output: [batch, depth, height, width]
        x_shape = [2, 3, 6, 4]
        y_shape = [2, 2, 12, 8]

        # Filter: [kernel_height, kernel_width, output_depth, input_depth]
        f_shape = [3, 3, 2, 3]

        x = constant_op.constant(
            1.0, shape=x_shape, name="x", dtype=dtypes.float32)
        f = constant_op.constant(
            1.0, shape=f_shape, name="filter", dtype=dtypes.float32)

        output = nn_ops.conv2d_transpose(
            x, f, y_shape, strides=strides, padding="SAME", data_format="NCHW")

        value = self.evaluate(output)
        for n in range(x_shape[0]):
          for k in range(f_shape[2]):
            for w in range(y_shape[3]):
              for h in range(y_shape[2]):
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
    # `NCHW` data format is only supported for CUDA device.
    if test.is_gpu_available(cuda_only=True):
      with self.session():
        strides = [1, 1, 2, 2]

        # Input, output: [batch, depth, height, width]
        x_shape = [2, 3, 6, 4]
        y_shape = [2, 2, 13, 9]

        # Filter: [kernel_height, kernel_width, output_depth, input_depth]
        f_shape = [3, 3, 2, 3]

        x = constant_op.constant(
            1.0, shape=x_shape, name="x", dtype=dtypes.float32)
        f = constant_op.constant(
            1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
        output = nn_ops.conv2d_transpose(
            x, f, y_shape, strides=strides, padding="VALID", data_format="NCHW")

        value = self.evaluate(output)
        cache_values = np.zeros(y_shape, dtype=np.float32)
        # The amount of padding added
        pad = 1
        for n in range(x_shape[0]):
          for k in range(f_shape[2]):
            for w in range(pad, y_shape[3] - pad):
              for h in range(pad, y_shape[2] - pad):
                target = 3.0
                # We add a case for locations divisible by the stride.
                h_in = h % strides[2] == 0 and h > pad and h < y_shape[
                    2] - 1 - pad
                w_in = w % strides[3] == 0 and w > pad and w < y_shape[
                    3] - 1 - pad
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

  def testConv2DTransposeShapeInference(self):
    # Test case for 8972
    initializer = random_ops.truncated_normal(
        [3, 3, 5, 1], mean=0.0, stddev=0.01, dtype=dtypes.float32)
    x = variables.Variable(random_ops.random_normal([3, 10, 5, 1]))
    f = variable_scope.get_variable("f", initializer=initializer)
    f_shape = array_ops.stack([array_ops.shape(x)[0], 10, 5, 5])
    output = nn_ops.conv2d_transpose(
        x, f, f_shape, strides=[1, 1, 1, 1], padding="SAME")
    self.assertEqual(output.get_shape().as_list(), [3, 10, 5, 5])

  def testConv2DTransposeInvalidOutputShape(self):
    with self.session():
      with self.assertRaises((errors.InvalidArgumentError, ValueError)):
        op = nn_ops.conv2d_transpose(
            input=np.ones((1, 1, 1, 1)),
            filters=np.ones((1, 1, 1, 1)),
            output_shape=[2, -2],
            strides=[1])
        self.evaluate(op)

  def testConv2DTransposeLargeOutputShape(self):
    # On GPU, this test does try to allocate the output tensor and OOMs.
    with test_util.device(use_gpu=False):
      with self.assertRaises((errors.InvalidArgumentError, ValueError)):
        op = nn_ops.conv2d_transpose(
            input=np.ones((2, 2, 2, 2)),
            output_shape=[114078056, 179835296],
            strides=[10],
            filters=[[[[1]]]])
        self.evaluate(op)

if __name__ == "__main__":
  test.main()
