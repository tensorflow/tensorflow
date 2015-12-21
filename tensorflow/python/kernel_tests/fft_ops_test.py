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
"""Tests for fft operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class FFT2DOpsTest(tf.test.TestCase):

  def _tfFFT2D(self, x, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      return tf.fft2d(x).eval()

  def _npFFT2D(self, x):
    return np.fft.fft2(x)

  def _tfIFFT2D(self, x, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      return tf.ifft2d(x).eval()

  def _npIFFT2D(self, x):
    return np.fft.ifft2(x)

  def _Compare(self, x):
    if tf.test.IsBuiltWithCuda():
      # GPU/Forward
      self.assertAllClose(
          self._npFFT2D(x),
          self._tfFFT2D(x,
                        use_gpu=True),
          rtol=1e-4,
          atol=1e-4)
      # GPU/Backward
      self.assertAllClose(
          self._npIFFT2D(x),
          self._tfIFFT2D(x,
                         use_gpu=True),
          rtol=1e-4,
          atol=1e-4)

  def testBasic(self):
    self._Compare(np.arange(60).reshape([6, 10]))
    self._Compare(np.arange(60).reshape([10, 6]))

  def testRandom(self):
    np.random.seed(12345)

    def gen(shape):
      n = np.prod(shape)
      re = np.random.uniform(size=n)
      im = np.random.uniform(size=n)
      return (re + im * 1j).reshape(shape)

    for shape in [(1, 1), (5, 5), (5, 7), (7, 5), (100, 250)]:
      self._Compare(gen(shape))

  def testEmpty(self):
    x = np.zeros([40, 0]).astype(np.complex64)
    self.assertEqual(x.shape, self._tfFFT2D(x).shape)
    self.assertEqual(x.shape, self._tfIFFT2D(x).shape)

  def testError(self):
    x = np.zeros([1, 2, 3]).astype(np.complex64)
    with self.assertRaisesOpError("Input is not a matrix"):
      self._tfFFT2D(x)
    with self.assertRaisesOpError("Input is not a matrix"):
      self._tfIFFT2D(x)


if __name__ == "__main__":
  tf.test.main()
