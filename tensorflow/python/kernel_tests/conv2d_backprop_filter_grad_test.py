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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class Conv2DBackpropFilterGradTest(test.TestCase):

  def testGradient(self):
    with self.cached_session():
      for padding in ["SAME", "VALID"]:
        for stride in [1, 2]:
          np.random.seed(1)
          in_shape = [5, 8, 6, 4]
          in_val = constant_op.constant(
              2 * np.random.random_sample(in_shape) - 1, dtype=dtypes.float32)
          filter_shape = [3, 3, 4, 6]
          # Make a convolution op with the current settings, just to easily get
          # the shape of the output.
          conv_out = nn_ops.conv2d(
              in_val,
              array_ops.zeros(filter_shape),
              strides=[1, stride, stride, 1],
              padding=padding)
          out_backprop_shape = conv_out.get_shape().as_list()
          out_backprop_val = constant_op.constant(
              2 * np.random.random_sample(out_backprop_shape) - 1,
              dtype=dtypes.float32)
          output = nn_ops.conv2d_backprop_filter(
              in_val,
              filter_shape,
              out_backprop_val,
              strides=[1, stride, stride, 1],
              padding=padding)
          err = gradient_checker.compute_gradient_error(
              [in_val, out_backprop_val], [in_shape, out_backprop_shape],
              output, filter_shape)
          print("conv2d_backprop_filter gradient err = %g " % err)
          err_tolerance = 2e-3
          self.assertLess(err, err_tolerance)

  def testGradientDilatedConv(self):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        for padding in ["SAME", "VALID"]:
          for stride in [1, 2]:
            np.random.seed(1)
            in_shape = [5, 8, 6, 4]
            in_val = constant_op.constant(
                2 * np.random.random_sample(in_shape) - 1, dtype=dtypes.float32)
            filter_shape = [3, 3, 4, 6]
            # Make a convolution op with the current settings,
            # just to easily get the shape of the output.
            conv_out = nn_ops.conv2d(
                in_val,
                array_ops.zeros(filter_shape),
                dilations=[1, 2, 2, 1],
                strides=[1, stride, stride, 1],
                padding=padding)
            out_backprop_shape = conv_out.get_shape().as_list()
            out_backprop_val = constant_op.constant(
                2 * np.random.random_sample(out_backprop_shape) - 1,
                dtype=dtypes.float32)
            output = nn_ops.conv2d_backprop_filter(
                in_val,
                filter_shape,
                out_backprop_val,
                dilations=[1, 2, 2, 1],
                strides=[1, stride, stride, 1],
                padding=padding)
            err = gradient_checker.compute_gradient_error(
                [in_val, out_backprop_val], [in_shape, out_backprop_shape],
                output, filter_shape)
            print("conv2d_backprop_filter gradient err = %g " % err)
            err_tolerance = 2e-3
            self.assertLess(err, err_tolerance)


if __name__ == "__main__":
  test.main()
