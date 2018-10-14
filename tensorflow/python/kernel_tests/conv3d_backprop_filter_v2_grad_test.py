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


class Conv3DBackpropFilterV2GradTest(test.TestCase):

  def testGradient(self):
    with self.cached_session():
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


if __name__ == "__main__":
  test.main()
