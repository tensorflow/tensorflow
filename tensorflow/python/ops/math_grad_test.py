# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Tests for Python ops defined in math_grad.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SquaredDifferenceOpTest(tf.test.TestCase):

  def _testGrad(self, left_shape, right_shape):

    if len(left_shape) > len(right_shape):
      output_shape = left_shape
    else:
      output_shape = right_shape
    l = np.random.randn(*left_shape)
    r = np.random.randn(*right_shape)

    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        left_tensor = tf.constant(l, shape=left_shape)
        right_tensor = tf.constant(r, shape=right_shape)
        output = tf.squared_difference(left_tensor, right_tensor)
        left_err = tf.test.compute_gradient_error(left_tensor,
                                                  left_shape,
                                                  output,
                                                  output_shape,
                                                  x_init_value=l)
        right_err = tf.test.compute_gradient_error(right_tensor,
                                                   right_shape,
                                                   output,
                                                   output_shape,
                                                   x_init_value=r)
      self.assertLess(left_err, 1e-10)
      self.assertLess(right_err, 1e-10)

  def testGrad(self):
    self._testGrad([1, 2, 3, 2], [3, 2])
    self._testGrad([2, 4], [3, 2, 4])


if __name__ == "__main__":
  tf.test.main()
