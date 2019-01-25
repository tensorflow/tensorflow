# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for resampler ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.contrib import resampler
from tensorflow.contrib.resampler.ops import gen_resampler_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ResamplerOpsTest(xla_test.XLATestCase):

  def _assertForwardOpMatchesExpected(self, image_np, warp_np, expected):
    with self.test_session() as sess, self.test_scope():
      input_image = array_ops.placeholder(image_np.dtype)
      warp = array_ops.placeholder(warp_np.dtype)
      resampled = resampler.resampler(input_image, warp, name='resampler')
      out = sess.run(resampled, {input_image: image_np, warp: warp_np})

      self.assertAllCloseAccordingToType(
          expected, out, rtol=5e-3, half_rtol=1e-2, bfloat16_rtol=3e-2)

  def _assertBackwardOpMatchesExpected(self, input_np, warp_np, grad_output_np,
                                       expected_grad_data, expected_grad_warp):
    with self.cached_session() as sess, self.test_scope():
      input_image = array_ops.placeholder(input_np.dtype)
      warp = array_ops.placeholder(warp_np.dtype)
      grad_output = array_ops.placeholder(grad_output_np.dtype)

      grad_data, grad_warp = gen_resampler_ops.resampler_grad(
          input_image, warp, grad_output)

      grad_data_tf, grad_warp_tf = sess.run([grad_data, grad_warp], {
          input_image: input_np,
          warp: warp_np,
          grad_output: grad_output_np
      })

      self.assertAllCloseAccordingToType(
          expected_grad_warp, grad_warp_tf, half_rtol=1e-2, bfloat16_rtol=3e-2)
      self.assertAllCloseAccordingToType(
          expected_grad_data, grad_data_tf, half_rtol=1e-2, bfloat16_rtol=3e-2)

  def testSimple(self):
    for dtype in self.float_types:
      input_shape = [1, 2, 2, 1]
      input_data = [0, 5, 13, 54]
      input_np = np.array(input_data, dtype=dtype).reshape(input_shape)

      warp_shape = [1, 2]
      warp_data = [0.7, 0.6]
      warp_np = np.array(warp_data, dtype=dtype).reshape(warp_shape)
      expected = [[26.42]]
      self._assertForwardOpMatchesExpected(input_np, warp_np, expected)

      grad_output = np.ones([1, 1], dtype=dtype)

      expected_grad_data = [[[[0.12], [0.27999997]], [[0.18000001],
                                                      [0.42000002]]]]

      expected_grad_warp = [[26.60000038, 38.20000076]]

      self._assertBackwardOpMatchesExpected(input_np, warp_np, grad_output,
                                            expected_grad_data,
                                            expected_grad_warp)

  def testMultiChannel(self):
    for dtype in self.float_types:
      input_shape = [1, 2, 2, 3]
      input_rgb_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
      input_np = np.array(input_rgb_data, dtype=dtype).reshape(input_shape)

      warp_shape = [1, 2]
      warp_data = [0.7, 0.6]
      warp_np = np.array(warp_data, dtype=dtype).reshape(warp_shape)
      expected = [[59.58000183, 146.94000244, 107.37999725]]
      self._assertForwardOpMatchesExpected(input_np, warp_np, expected)

      grad_output = np.ones([1, 3], dtype=dtype)

      expected_grad_data = [[[[0.12, 0.12, 0.12],
                              [0.27999997, 0.27999997, 0.27999997]],
                             [[0.18000001, 0.18000001, 0.18000001],
                              [0.42000002, 0.42000002, 0.42000002]]]]

      expected_grad_warp = [[199, 30]]

      self._assertBackwardOpMatchesExpected(input_np, warp_np, grad_output,
                                            expected_grad_data,
                                            expected_grad_warp)

  def testBatch2Height3byWidth3RGB(self):
    for dtype in self.float_types:
      input_shape = [2, 3, 3, 3]
      input_rgb_data = [
          0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1, 30, 105, 2, 40, 115,
          3, 50, 125, 4, 60, 135, 5, 70, 145, 6, 0, 5, 13, 54, 135, 226, 37, 8,
          234, 90, 255, 1, 30, 105, 2, 40, 115, 3, 50, 125, 4, 60, 135, 5, 70,
          145, 6
      ]
      input_np = np.array(input_rgb_data, dtype=dtype).reshape(input_shape)

      # 2 batches and 2 samples for each batch.
      warp_shape = [2, 2, 2]
      warp_data = [0.7, 0.6, 1, 0.7, 0.9, 1.2, 1.3, 1.6]
      warp_np = np.array(warp_data, dtype=dtype).reshape(warp_shape)

      expected_forward = [[[43.92, 128.4, 65.86], [37.2, 114., 69.2]],
                          [[40.6, 122.8, 2.5], [51., 126, 4.1]]]

      self._assertForwardOpMatchesExpected(input_np, warp_np, expected_forward)

      expected_grad_data = [[[[0.12, 0.12, 0.12],
                              [0.57999998, 0.57999998, 0.57999998],
                              [0., 0., 0.]],
                             [[0.18000001, 0.18000001, 0.18000001],
                              [1.12, 1.12, 1.12], [0., 0., 0.]],
                             [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                            [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                             [[0.08000001, 0.08000001, 0.08000001],
                              [0.99999988, 0.99999988, 0.99999988],
                              [0.11999997, 0.11999997, 0.11999997]],
                             [[0.02000001, 0.02000001, 0.02000001],
                              [0.60000008, 0.60000008, 0.60000008],
                              [0.17999998, 0.17999998, 0.17999998]]]]
      expected_grad_warp = [[[33.39999008, -96.20000458], [-26.10000229,
                                                           -278.]],
                            [[-162.99998474, 39.99999619], [21., 63.]]]

      grad_output = np.ones([2, 2, 3], dtype=dtype)
      self._assertBackwardOpMatchesExpected(input_np, warp_np, grad_output,
                                            expected_grad_data,
                                            expected_grad_warp)

  def testOutOfBoundWarps(self):
    # (x, y) are both less than 0.
    for dtype in self.float_types:
      input_shape = [1, 2, 2, 1]
      input_data = [10, 5, 13, 54]
      input_np = np.array(input_data, dtype=dtype).reshape(input_shape)

      warp_shape = [1, 2, 2]
      warp_data = [-1, -1, 0.7, 0.6]
      warp_np = np.array(warp_data, dtype=dtype).reshape(warp_shape)
      expected = [[[0.0], [27.62]]]
      self._assertForwardOpMatchesExpected(input_np, warp_np, expected)

      expected_grad_data = [[[[0.12], [0.27999997]], [[0.18000001],
                                                      [0.42000002]]]]
      expected_grad_warp = [[[0., 0.], [22.60000038, 35.20000076]]]

      grad_output = np.ones([1, 2, 1], dtype=dtype)
      self._assertBackwardOpMatchesExpected(input_np, warp_np, grad_output,
                                            expected_grad_data,
                                            expected_grad_warp)

    # One of (x, y) is less than 0.
    for dtype in self.float_types:
      input_shape = [1, 2, 2, 1]
      input_data = [10, 5, 13, 54]
      input_np = np.array(input_data, dtype=dtype).reshape(input_shape)

      warp_shape = [1, 2, 2]
      # -1 is out of bound for grad_warp.
      warp_data = [-1, 0.1, 0.7, 0.6]
      warp_np = np.array(warp_data, dtype=dtype).reshape(warp_shape)
      expected = [[[0.0], [27.62]]]
      self._assertForwardOpMatchesExpected(input_np, warp_np, expected)

      expected_grad_data = [[[[0.12], [0.27999997]], [[0.18000001],
                                                      [0.42000002]]]]
      expected_grad_warp = [[[0., 0.], [22.60000038, 35.20000076]]]

      grad_output = np.ones([1, 2, 1], dtype=dtype)
      self._assertBackwardOpMatchesExpected(input_np, warp_np, grad_output,
                                            expected_grad_data,
                                            expected_grad_warp)

    # Both of (x, y) are greater than image size.
    for dtype in self.float_types:
      input_shape = [1, 2, 2, 1]
      input_data = [10, 5, 13, 54]
      input_np = np.array(input_data, dtype=dtype).reshape(input_shape)

      warp_shape = [1, 2, 2]
      # -0.1 is *inbound* for grad_warp and grad_data, 2.1 is out of bound.
      warp_data = [-0.1, 0.1, 1.2, 2.1]
      warp_np = np.array(warp_data, dtype=dtype).reshape(warp_shape)
      expected = [[[0.0], [0.0]]]
      self._assertForwardOpMatchesExpected(input_np, warp_np, expected)

      expected_grad_data = [[[[0.81], [0.0]], [[0.09], [0.0]]]]
      expected_grad_warp = [[[10.30, 2.7], [0.0, 0.0]]]

      grad_output = np.ones([1, 2, 1], dtype=dtype)
      self._assertBackwardOpMatchesExpected(input_np, warp_np, grad_output,
                                            expected_grad_data,
                                            expected_grad_warp)

    # One of (x, y) is greater than image size.
    for dtype in self.float_types:
      input_shape = [1, 2, 2, 1]
      input_data = [10, 5, 13, 54]
      input_np = np.array(input_data, dtype=dtype).reshape(input_shape)

      warp_shape = [1, 2, 2]
      warp_data = [0.1, -0.1, 1.2, 0.1]
      warp_np = np.array(warp_data, dtype=dtype).reshape(warp_shape)
      expected = [[[0.0], [0.0]]]
      self._assertForwardOpMatchesExpected(input_np, warp_np, expected)

      expected_grad_data = [[[[0.81], [0.81]], [[0.0], [0.08]]]]
      expected_grad_warp = [[[-4.5, 9.5], [-9.9, 39.20]]]

      grad_output = np.ones([1, 2, 1], dtype=dtype)
      self._assertBackwardOpMatchesExpected(input_np, warp_np, grad_output,
                                            expected_grad_data,
                                            expected_grad_warp)


if __name__ == '__main__':
  test.main()
