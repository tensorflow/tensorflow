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

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin


VALID_FFT_RANKS = (1, 2, 3)


class BaseFFTOpsTest(tf.test.TestCase):

  def _tfFFT(self, x, rank, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      return self._tfFFTForRank(rank)(x).eval()

  def _npFFT(self, x, rank):
    if rank == 1:
      return np.fft.fft2(x, axes=(-1,))
    elif rank == 2:
      return np.fft.fft2(x, axes=(-2, -1))
    elif rank == 3:
      return np.fft.fft2(x, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _tfIFFT(self, x, rank, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      return self._tfIFFTForRank(rank)(x).eval()

  def _npIFFT(self, x, rank):
    if rank == 1:
      return np.fft.ifft2(x, axes=(-1,))
    elif rank == 2:
      return np.fft.ifft2(x, axes=(-2, -1))
    elif rank == 3:
      return np.fft.ifft2(x, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _Compare(self, x, rank):
    if tf.test.is_built_with_cuda():
      # GPU/Forward
      self.assertAllClose(
          self._npFFT(x, rank),
          self._tfFFT(x, rank, use_gpu=True),
          rtol=1e-4,
          atol=1e-4)
      # GPU/Backward
      self.assertAllClose(
          self._npIFFT(x, rank),
          self._tfIFFT(x, rank, use_gpu=True),
          rtol=1e-4,
          atol=1e-4)

  def _checkGrad(self, func, x, y, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      # func is a forward or inverse FFT function (batched or unbatched)
      z = func(tf.complex(inx, iny))
      # loss = sum(|z|^2)
      loss = tf.reduce_sum(tf.real(z * tf.conj(z)))
      ((x_jacob_t, x_jacob_n),
       (y_jacob_t, y_jacob_n)) = tf.test.compute_gradient(
           [inx, iny],
           [list(x.shape), list(y.shape)],
           loss,
           [1],
           x_init_value=[x, y],
           delta=1e-2)
    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=1e-2)
    self.assertAllClose(y_jacob_t, y_jacob_n, rtol=1e-2, atol=1e-2)


class BatchFFTOpsTest(BaseFFTOpsTest):

  def _tfFFTForRank(self, rank):
    if rank == 1:
      return tf.batch_fft
    elif rank == 2:
      return tf.batch_fft2d
    elif rank == 3:
      return tf.batch_fft3d
    else:
      raise ValueError("invalid rank")

  def _tfIFFTForRank(self, rank):
    if rank == 1:
      return tf.batch_ifft
    elif rank == 2:
      return tf.batch_ifft2d
    elif rank == 3:
      return tf.batch_ifft3d
    else:
      raise ValueError("invalid rank")

  def testEmpty(self):
    if tf.test.is_built_with_cuda():
      for rank in VALID_FFT_RANKS:
        for dims in xrange(rank, rank + 3):
          x = np.zeros((0,) * dims).astype(np.complex64)
          self.assertEqual(x.shape, self._tfFFT(x, rank).shape)
          self.assertEqual(x.shape, self._tfIFFT(x, rank).shape)

  def testBasic(self):
    for rank in VALID_FFT_RANKS:
      for dims in xrange(rank, rank + 3):
        self._Compare(
            np.mod(
                np.arange(np.power(4, dims)), 10).reshape((4,) * dims), rank)

  def testRandom(self):
    np.random.seed(12345)

    def gen(shape):
      n = np.prod(shape)
      re = np.random.uniform(size=n)
      im = np.random.uniform(size=n)
      return (re + im * 1j).reshape(shape)

    for rank in VALID_FFT_RANKS:
      for dims in xrange(rank, rank + 3):
        self._Compare(gen((4,) * dims), rank)

  def testError(self):
    if tf.test.is_built_with_cuda():
      for rank in VALID_FFT_RANKS:
        for dims in xrange(0, rank):
          x = np.zeros((1,) * dims).astype(np.complex64)
          with self.assertRaisesOpError(
              "Input must have rank of at least {}".format(rank)):
            self._tfFFT(x, rank)
          with self.assertRaisesOpError(
              "Input must have rank of at least {}".format(rank)):
            self._tfIFFT(x, rank)

  def testGrad_Simple(self):
    if tf.test.is_built_with_cuda():
      for rank in VALID_FFT_RANKS:
        for dims in xrange(rank, rank + 2):
          re = np.ones(shape=(4,) * dims, dtype=np.float32) / 10.0
          im = np.zeros(shape=(4,) * dims, dtype=np.float32)
          self._checkGrad(self._tfFFTForRank(rank), re, im, use_gpu=True)
          self._checkGrad(self._tfIFFTForRank(rank), re, im, use_gpu=True)

  def testGrad_Random(self):
    if tf.test.is_built_with_cuda():
      np.random.seed(54321)
      for rank in VALID_FFT_RANKS:
        for dims in xrange(rank, rank + 2):
          re = np.random.rand(*((3,) * dims)).astype(np.float32) * 2 - 1
          im = np.random.rand(*((3,) * dims)).astype(np.float32) * 2 - 1
          self._checkGrad(self._tfFFTForRank(rank), re, im, use_gpu=True)
          self._checkGrad(self._tfIFFTForRank(rank), re, im, use_gpu=True)


class FFTOpsTest(BaseFFTOpsTest):

  def _tfFFTForRank(self, rank):
    if rank == 1:
      return tf.fft
    elif rank == 2:
      return tf.fft2d
    elif rank == 3:
      return tf.fft3d
    else:
      raise ValueError("invalid rank")

  def _tfIFFTForRank(self, rank):
    if rank == 1:
      return tf.ifft
    elif rank == 2:
      return tf.ifft2d
    elif rank == 3:
      return tf.ifft3d
    else:
      raise ValueError("invalid rank")

  def testBasic(self):
    for rank in VALID_FFT_RANKS:
      self._Compare(
          np.mod(
              np.arange(np.power(4, rank)), 10).reshape((4,) * rank), rank)

  def testRandom(self):
    np.random.seed(12345)

    def gen(shape):
      n = np.prod(shape)
      re = np.random.uniform(size=n)
      im = np.random.uniform(size=n)
      return (re + im * 1j).reshape(shape)

    for rank in VALID_FFT_RANKS:
      for size in (1, 5, 7, 15):
        self._Compare(gen((size,) * rank), rank)

  def testEmpty(self):
    if tf.test.is_built_with_cuda():
      for rank in VALID_FFT_RANKS:
        x = np.zeros((0,) * rank).astype(np.complex64)
        self.assertEqual(x.shape, self._tfFFT(x, rank).shape)
        self.assertEqual(x.shape, self._tfIFFT(x, rank).shape)

  def testError(self):
    if tf.test.is_built_with_cuda():
      for rank in VALID_FFT_RANKS:
        for dims in list(range(0, rank)) + [rank + 1]:
          x = np.zeros((1,) * dims).astype(np.complex64)
          with self.assertRaisesOpError(
              "Input must be of rank {} but got:".format(rank)):
            self._tfFFT(x, rank)
          with self.assertRaisesOpError(
              "Input must be of rank {} but got:".format(rank)):
            self._tfIFFT(x, rank)

  def testGrad_Simple(self):
    if tf.test.is_built_with_cuda():
      for rank in VALID_FFT_RANKS:
        re = np.ones(shape=(4,) * rank, dtype=np.float32) / 10.0
        im = np.zeros(shape=(4,) * rank, dtype=np.float32)
        self._checkGrad(self._tfFFTForRank(rank), re, im, use_gpu=True)
        self._checkGrad(self._tfIFFTForRank(rank), re, im, use_gpu=True)

  def testGrad_Random(self):
    if tf.test.is_built_with_cuda():
      np.random.seed(54321)
      for rank in VALID_FFT_RANKS:
        re = np.random.rand(*((3,) * rank)).astype(np.float32) * 2 - 1
        im = np.random.rand(*((3,) * rank)).astype(np.float32) * 2 - 1
        self._checkGrad(self._tfFFTForRank(rank), re, im, use_gpu=True)
        self._checkGrad(self._tfIFFTForRank(rank), re, im, use_gpu=True)


if __name__ == "__main__":
  tf.test.main()
