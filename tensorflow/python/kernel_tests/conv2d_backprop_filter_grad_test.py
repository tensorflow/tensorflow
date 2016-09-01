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
import tensorflow as tf


class Conv2DBackpropFilterGradTest(tf.test.TestCase):

  def testGradient(self):
    with self.test_session():
      for padding in ["SAME", "VALID"]:
        for stride in [1, 2]:
          np.random.seed(1)
          in_shape = [5, 8, 6, 4]
          in_val = tf.constant(
              2 * np.random.random_sample(in_shape) - 1,
              dtype=tf.float32)
          filter_shape = [3, 3, 4, 6]
          # Make a convolution op with the current settings, just to easily get
          # the shape of the output.
          conv_out = tf.nn.conv2d(in_val, tf.zeros(filter_shape),
                                  [1, stride, stride, 1], padding)
          out_backprop_shape = conv_out.get_shape().as_list()
          out_backprop_val = tf.constant(
              2 * np.random.random_sample(out_backprop_shape) - 1,
              dtype=tf.float32)
          output = tf.nn.conv2d_backprop_filter(in_val, filter_shape,
                                                out_backprop_val,
                                                [1, stride, stride, 1], padding)
          err = tf.test.compute_gradient_error([in_val, out_backprop_val],
                                               [in_shape, out_backprop_shape],
                                               output, filter_shape)
          print("conv2d_backprop_filter gradient err = %g " % err)
          err_tolerance = 2e-3
          self.assertLess(err, err_tolerance)


if __name__ == "__main__":
  tf.test.main()
