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
"""Tests for Python ops defined in image_grad.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ResizeNearestNeighborOpTest(tf.test.TestCase):

  TYPES = [np.float32, np.float64]

  def testShapeIsCorrectAfterOp(self):
    in_shape = [1, 2, 2, 1]
    out_shape = [1, 4, 6, 1]

    for nptype in self.TYPES:
      x = np.arange(0, 4).reshape(in_shape).astype(nptype)

      for use_gpu in [False, True]:
        with self.test_session(use_gpu=use_gpu) as sess:
          input_tensor = tf.constant(x, shape=in_shape)
          resize_out = tf.image.resize_nearest_neighbor(input_tensor,
                                                      out_shape[1:3])
          self.assertEqual(out_shape, list(resize_out.get_shape()))

          resize_out = sess.run(resize_out)
        self.assertEqual(out_shape, list(resize_out.shape))

  def testGradFromResizeToLargerInBothDims(self):
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    for nptype in self.TYPES:
      x = np.arange(0, 6).reshape(in_shape).astype(nptype)

      for use_gpu in [False, True]:
        with self.test_session(use_gpu=use_gpu):
          input_tensor = tf.constant(x, shape=in_shape)
          resize_out = tf.image.resize_nearest_neighbor(input_tensor,
                                                      out_shape[1:3])
          err = tf.test.compute_gradient_error(input_tensor,
                                               in_shape,
                                               resize_out,
                                               out_shape,
                                               x_init_value=x)
        self.assertLess(err, 1e-3)

  def testGradFromResizeToSmallerInBothDims(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    for nptype in self.TYPES:
      x = np.arange(0, 24).reshape(in_shape).astype(nptype)

      for use_gpu in [False, True]:
        with self.test_session(use_gpu=use_gpu):
          input_tensor = tf.constant(x, shape=in_shape)
          resize_out = tf.image.resize_nearest_neighbor(input_tensor,
                                                      out_shape[1:3])
          err = tf.test.compute_gradient_error(input_tensor,
                                               in_shape,
                                               resize_out,
                                               out_shape,
                                               x_init_value=x)
        self.assertLess(err, 1e-3)

  def testCompareGpuVsCpu(self):
    in_shape = [1, 4, 6, 3]
    out_shape = [1, 8, 16, 3]

    for nptype in self.TYPES:
      x = np.arange(0, np.prod(in_shape)).reshape(in_shape).astype(nptype)
      for align_corners in [True, False]:
        with self.test_session(use_gpu=False):
          input_tensor = tf.constant(x, shape=in_shape)
          resize_out = tf.image.resize_nearest_neighbor(input_tensor,
                                                        out_shape[1:3],
                                                        align_corners=align_corners)
          grad_cpu = tf.test.compute_gradient(input_tensor,
                                              in_shape,
                                              resize_out,
                                              out_shape,
                                              x_init_value=x)

        with self.test_session(use_gpu=True):
          input_tensor = tf.constant(x, shape=in_shape)
          resize_out = tf.image.resize_nearest_neighbor(input_tensor,
                                                        out_shape[1:3],
                                                        align_corners=align_corners)
          grad_gpu = tf.test.compute_gradient(input_tensor,
                                              in_shape,
                                              resize_out,
                                              out_shape,
                                              x_init_value=x)
        self.assertAllClose(grad_cpu, grad_gpu, rtol=1e-5, atol=1e-5)

class ResizeBilinearOpTest(tf.test.TestCase):

  def testShapeIsCorrectAfterOp(self):
    in_shape = [1, 2, 2, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 4).reshape(in_shape).astype(np.float32)

    with self.test_session() as sess:
      input_tensor = tf.constant(x, shape=in_shape)
      resize_out = tf.image.resize_bilinear(input_tensor,
                                            out_shape[1:3])
      self.assertEqual(out_shape, list(resize_out.get_shape()))

      resize_out = sess.run(resize_out)
      self.assertEqual(out_shape, list(resize_out.shape))

  def testGradFromResizeToLargerInBothDims(self):
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]

    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)

    with self.test_session():
      input_tensor = tf.constant(x, shape=in_shape)
      resize_out = tf.image.resize_bilinear(input_tensor,
                                            out_shape[1:3])
      err = tf.test.compute_gradient_error(input_tensor,
                                           in_shape,
                                           resize_out,
                                           out_shape,
                                           x_init_value=x)
    self.assertLess(err, 1e-3)

  def testGradFromResizeToSmallerInBothDims(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    x = np.arange(0, 24).reshape(in_shape).astype(np.float32)

    with self.test_session():
      input_tensor = tf.constant(x, shape=in_shape)
      resize_out = tf.image.resize_bilinear(input_tensor,
                                            out_shape[1:3])
      err = tf.test.compute_gradient_error(input_tensor,
                                           in_shape,
                                           resize_out,
                                           out_shape,
                                           x_init_value=x)
    self.assertLess(err, 1e-3)

  def testGradOnUnsupportedType(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]

    x = np.arange(0, 24).reshape(in_shape).astype(np.uint8)

    with self.test_session():
      input_tensor = tf.constant(x, shape=in_shape)
      resize_out = tf.image.resize_bilinear(input_tensor, out_shape[1:3])
      grad = tf.gradients(input_tensor, [resize_out])
      self.assertEqual([None], grad)

if __name__ == "__main__":
  tf.test.main()
