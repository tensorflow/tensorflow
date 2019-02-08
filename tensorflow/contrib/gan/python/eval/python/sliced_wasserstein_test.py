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
"""Tests for Sliced Wasserstein Distance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import ndimage
from tensorflow.contrib.gan.python.eval.python import sliced_wasserstein_impl as swd
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ClassifierMetricsTest(test.TestCase):

  def test_laplacian_pyramid(self):
    # The numpy/scipy code for reference estimation comes from:
    # https://github.com/tkarras/progressive_growing_of_gans
    gaussian_filter = np.float32([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [
        6, 24, 36, 24, 6
    ], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256.0

    def np_pyr_down(minibatch):  # matches cv2.pyrDown()
      assert minibatch.ndim == 4
      return ndimage.convolve(
          minibatch,
          gaussian_filter[np.newaxis, np.newaxis, :, :],
          mode='mirror')[:, :, ::2, ::2]

    def np_pyr_up(minibatch):  # matches cv2.pyrUp()
      assert minibatch.ndim == 4
      s = minibatch.shape
      res = np.zeros((s[0], s[1], s[2] * 2, s[3] * 2), minibatch.dtype)
      res[:, :, ::2, ::2] = minibatch
      return ndimage.convolve(
          res,
          gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0,
          mode='mirror')

    def np_laplacian_pyramid(minibatch, num_levels):
      # Note: there's a bug in the original SWD, fixed repeatability.
      pyramid = [minibatch.astype('f').copy()]
      for _ in range(1, num_levels):
        pyramid.append(np_pyr_down(pyramid[-1]))
        pyramid[-2] -= np_pyr_up(pyramid[-1])
      return pyramid

    data = np.random.normal(size=[256, 3, 32, 32]).astype('f')
    pyramid = np_laplacian_pyramid(data, 3)
    data_tf = array_ops.placeholder(dtypes.float32, [256, 32, 32, 3])
    pyramid_tf = swd._laplacian_pyramid(data_tf, 3)
    with self.cached_session() as sess:
      pyramid_tf = sess.run(
          pyramid_tf, feed_dict={
              data_tf: data.transpose(0, 2, 3, 1)
          })
    for x in range(3):
      self.assertAllClose(
          pyramid[x].transpose(0, 2, 3, 1), pyramid_tf[x], atol=1e-6)

  def test_sliced_wasserstein_distance(self):
    """Test the distance."""
    d1 = random_ops.random_uniform([256, 32, 32, 3])
    d2 = random_ops.random_normal([256, 32, 32, 3])
    wfunc = swd.sliced_wasserstein_distance(d1, d2)
    with self.cached_session() as sess:
      wscores = [sess.run(x) for x in wfunc]
    self.assertAllClose(
        np.array([0.014, 0.014], 'f'),
        np.array([x[0] for x in wscores], 'f'),
        rtol=0.15)
    self.assertAllClose(
        np.array([0.014, 0.020], 'f'),
        np.array([x[1] for x in wscores], 'f'),
        rtol=0.15)

  def test_sliced_wasserstein_distance_svd(self):
    """Test the distance."""
    d1 = random_ops.random_uniform([256, 32, 32, 3])
    d2 = random_ops.random_normal([256, 32, 32, 3])
    wfunc = swd.sliced_wasserstein_distance(d1, d2, use_svd=True)
    with self.cached_session() as sess:
      wscores = [sess.run(x) for x in wfunc]
    self.assertAllClose(
        np.array([0.013, 0.013], 'f'),
        np.array([x[0] for x in wscores], 'f'),
        rtol=0.15)
    self.assertAllClose(
        np.array([0.014, 0.019], 'f'),
        np.array([x[1] for x in wscores], 'f'),
        rtol=0.15)

  def test_swd_mismatched(self):
    """Test the inputs mismatched shapes are detected."""
    d1 = random_ops.random_uniform([256, 32, 32, 3])
    d2 = random_ops.random_normal([256, 32, 31, 3])
    d3 = random_ops.random_normal([256, 31, 32, 3])
    d4 = random_ops.random_normal([255, 32, 32, 3])
    with self.assertRaises(ValueError):
      swd.sliced_wasserstein_distance(d1, d2)
    with self.assertRaises(ValueError):
      swd.sliced_wasserstein_distance(d1, d3)
    with self.assertRaises(ValueError):
      swd.sliced_wasserstein_distance(d1, d4)

  def test_swd_not_rgb(self):
    """Test that only RGB is supported."""
    d1 = random_ops.random_uniform([256, 32, 32, 1])
    d2 = random_ops.random_normal([256, 32, 32, 1])
    with self.assertRaises(ValueError):
      swd.sliced_wasserstein_distance(d1, d2)


if __name__ == '__main__':
  test.main()
