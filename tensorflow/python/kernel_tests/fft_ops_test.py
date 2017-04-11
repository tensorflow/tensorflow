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
"""Tests for fft operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import spectral_ops
from tensorflow.python.platform import test

VALID_FFT_RANKS = (1, 2, 3)


class BaseFFTOpsTest(test.TestCase):

  def _Compare(self, x, rank, fft_length=None, use_placeholder=False):
    self._CompareForward(x, rank, fft_length, use_placeholder)
    self._CompareBackward(x, rank, fft_length, use_placeholder)

  def _CompareForward(self, x, rank, fft_length=None, use_placeholder=False):
    if test.is_gpu_available(cuda_only=True):
      x_np = self._npFFT(x, rank, fft_length)
      if use_placeholder:
        x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
        x_tf = self._tfFFT(x_ph, rank, fft_length, use_gpu=True,
                           feed_dict={x_ph: x})
      else:
        x_tf = self._tfFFT(x, rank, fft_length, use_gpu=True)

      # GPU/Forward
      self.assertAllClose(x_np, x_tf, rtol=1e-4, atol=1e-4)

  def _CompareBackward(self, x, rank, fft_length=None, use_placeholder=False):
    if test.is_gpu_available(cuda_only=True):
      x_np = self._npIFFT(x, rank, fft_length)
      if use_placeholder:
        x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
        x_tf = self._tfIFFT(x_ph, rank, fft_length, use_gpu=True,
                            feed_dict={x_ph: x})
      else:
        x_tf = self._tfIFFT(x, rank, fft_length, use_gpu=True)

      # GPU/Backward
      self.assertAllClose(x_np, x_tf, rtol=1e-4, atol=1e-4)

  def _checkGradComplex(self, func, x, y, result_is_complex=True,
                        use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      # func is a forward or inverse, real or complex, batched or unbatched FFT
      # function with a complex input.
      z = func(math_ops.complex(inx, iny))
      # loss = sum(|z|^2)
      loss = math_ops.reduce_sum(math_ops.real(z * math_ops.conj(z)))

      ((x_jacob_t, x_jacob_n),
       (y_jacob_t, y_jacob_n)) = gradient_checker.compute_gradient(
           [inx, iny], [list(x.shape), list(y.shape)],
           loss, [1],
           x_init_value=[x, y],
           delta=1e-2)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=1e-2)
    self.assertAllClose(y_jacob_t, y_jacob_n, rtol=1e-2, atol=1e-2)

  def _checkGradReal(self, func, x, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      # func is a forward RFFT function (batched or unbatched).
      z = func(inx)
      # loss = sum(|z|^2)
      loss = math_ops.reduce_sum(math_ops.real(z * math_ops.conj(z)))
      x_jacob_t, x_jacob_n = test.compute_gradient(
          inx, list(x.shape), loss, [1], x_init_value=x, delta=1e-2)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=1e-2)


class FFTOpsTest(BaseFFTOpsTest):

  def _tfFFT(self, x, rank, fft_length=None, use_gpu=False, feed_dict=None):
    # fft_length unused for complex FFTs.
    with self.test_session(use_gpu=use_gpu):
      return self._tfFFTForRank(rank)(x).eval(feed_dict=feed_dict)

  def _tfIFFT(self, x, rank, fft_length=None, use_gpu=False, feed_dict=None):
    # fft_length unused for complex FFTs.
    with self.test_session(use_gpu=use_gpu):
      return self._tfIFFTForRank(rank)(x).eval(feed_dict=feed_dict)

  def _npFFT(self, x, rank, fft_length=None):
    if rank == 1:
      return np.fft.fft2(x, s=fft_length, axes=(-1,))
    elif rank == 2:
      return np.fft.fft2(x, s=fft_length, axes=(-2, -1))
    elif rank == 3:
      return np.fft.fft2(x, s=fft_length, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _npIFFT(self, x, rank, fft_length=None):
    if rank == 1:
      return np.fft.ifft2(x, s=fft_length, axes=(-1,))
    elif rank == 2:
      return np.fft.ifft2(x, s=fft_length, axes=(-2, -1))
    elif rank == 3:
      return np.fft.ifft2(x, s=fft_length, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _tfFFTForRank(self, rank):
    if rank == 1:
      return spectral_ops.fft
    elif rank == 2:
      return spectral_ops.fft2d
    elif rank == 3:
      return spectral_ops.fft3d
    else:
      raise ValueError("invalid rank")

  def _tfIFFTForRank(self, rank):
    if rank == 1:
      return spectral_ops.ifft
    elif rank == 2:
      return spectral_ops.ifft2d
    elif rank == 3:
      return spectral_ops.ifft3d
    else:
      raise ValueError("invalid rank")

  def testEmpty(self):
    if test.is_gpu_available(cuda_only=True):
      for rank in VALID_FFT_RANKS:
        for dims in xrange(rank, rank + 3):
          x = np.zeros((0,) * dims).astype(np.complex64)
          self.assertEqual(x.shape, self._tfFFT(x, rank).shape)
          self.assertEqual(x.shape, self._tfIFFT(x, rank).shape)

  def testBasic(self):
    for rank in VALID_FFT_RANKS:
      for dims in xrange(rank, rank + 3):
        self._Compare(np.mod(np.arange(np.power(4, dims)), 10).reshape(
            (4,) * dims).astype(np.complex64), rank)

  def testBasicPlaceholder(self):
    for rank in VALID_FFT_RANKS:
      for dims in xrange(rank, rank + 3):
        self._Compare(np.mod(np.arange(np.power(4, dims)), 10).reshape(
            (4,) * dims).astype(np.complex64), rank, use_placeholder=True)

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
    if test.is_gpu_available(cuda_only=True):
      for rank in VALID_FFT_RANKS:
        for dims in xrange(0, rank):
          x = np.zeros((1,) * dims).astype(np.complex64)
          with self.assertRaisesWithPredicateMatch(
              ValueError, "Shape must be .*rank {}.*".format(rank)):
            self._tfFFT(x, rank)
          with self.assertRaisesWithPredicateMatch(
              ValueError, "Shape must be .*rank {}.*".format(rank)):
            self._tfIFFT(x, rank)

  def testGrad_Simple(self):
    if test.is_gpu_available(cuda_only=True):
      for rank in VALID_FFT_RANKS:
        for dims in xrange(rank, rank + 2):
          re = np.ones(shape=(4,) * dims, dtype=np.float32) / 10.0
          im = np.zeros(shape=(4,) * dims, dtype=np.float32)
          self._checkGradComplex(self._tfFFTForRank(rank), re, im, use_gpu=True)
          self._checkGradComplex(
              self._tfIFFTForRank(rank), re, im, use_gpu=True)

  def testGrad_Random(self):
    if test.is_gpu_available(cuda_only=True):
      np.random.seed(54321)
      for rank in VALID_FFT_RANKS:
        for dims in xrange(rank, rank + 2):
          re = np.random.rand(*((3,) * dims)).astype(np.float32) * 2 - 1
          im = np.random.rand(*((3,) * dims)).astype(np.float32) * 2 - 1
          self._checkGradComplex(self._tfFFTForRank(rank), re, im, use_gpu=True)
          self._checkGradComplex(
              self._tfIFFTForRank(rank), re, im, use_gpu=True)


class RFFTOpsTest(BaseFFTOpsTest):

  def _tfFFT(self, x, rank, fft_length=None, use_gpu=False, feed_dict=None):
    with self.test_session(use_gpu=use_gpu):
      return self._tfFFTForRank(rank)(x, fft_length).eval(feed_dict=feed_dict)

  def _tfIFFT(self, x, rank, fft_length=None, use_gpu=False, feed_dict=None):
    with self.test_session(use_gpu=use_gpu):
      return self._tfIFFTForRank(rank)(x, fft_length).eval(feed_dict=feed_dict)

  def _npFFT(self, x, rank, fft_length=None):
    if rank == 1:
      return np.fft.rfft2(x, s=fft_length, axes=(-1,))
    elif rank == 2:
      return np.fft.rfft2(x, s=fft_length, axes=(-2, -1))
    elif rank == 3:
      return np.fft.rfft2(x, s=fft_length, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _npIFFT(self, x, rank, fft_length=None):
    if rank == 1:
      return np.fft.irfft2(x, s=fft_length, axes=(-1,))
    elif rank == 2:
      return np.fft.irfft2(x, s=fft_length, axes=(-2, -1))
    elif rank == 3:
      return np.fft.irfft2(x, s=fft_length, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _tfFFTForRank(self, rank):
    if rank == 1:
      return spectral_ops.rfft
    elif rank == 2:
      return spectral_ops.rfft2d
    elif rank == 3:
      return spectral_ops.rfft3d
    else:
      raise ValueError("invalid rank")

  def _tfIFFTForRank(self, rank):
    if rank == 1:
      return spectral_ops.irfft
    elif rank == 2:
      return spectral_ops.irfft2d
    elif rank == 3:
      return spectral_ops.irfft3d
    else:
      raise ValueError("invalid rank")

  def testEmpty(self):
    if test.is_gpu_available(cuda_only=True):
      for rank in VALID_FFT_RANKS:
        for dims in xrange(rank, rank + 3):
          x = np.zeros((0,) * dims).astype(np.float32)
          self.assertEqual(x.shape, self._tfFFT(x, rank).shape)
          x = np.zeros((0,) * dims).astype(np.complex64)
          self.assertEqual(x.shape, self._tfIFFT(x, rank).shape)

  def testBasic(self):
    for rank in VALID_FFT_RANKS:
      for dims in xrange(rank, rank + 3):
        for size in (5, 6):
          inner_dim = size // 2 + 1
          r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
              (size,) * dims)
          self._CompareForward(r2c.astype(np.float32), rank, (size,) * rank)
          c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                       10).reshape((size,) * (dims - 1) + (inner_dim,))
          self._CompareBackward(c2r.astype(np.complex64), rank, (size,) * rank)

  def testBasicPlaceholder(self):
    for rank in VALID_FFT_RANKS:
      for dims in xrange(rank, rank + 3):
        for size in (5, 6):
          inner_dim = size // 2 + 1
          r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
              (size,) * dims)
          self._CompareForward(r2c.astype(np.float32), rank, (size,) * rank,
                               use_placeholder=True)
          c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                       10).reshape((size,) * (dims - 1) + (inner_dim,))
          self._CompareBackward(c2r.astype(np.complex64), rank, (size,) * rank,
                                use_placeholder=True)

  def testRandom(self):
    np.random.seed(12345)

    def gen_real(shape):
      n = np.prod(shape)
      re = np.random.uniform(size=n)
      ret = re.reshape(shape)
      return ret

    def gen_complex(shape):
      n = np.prod(shape)
      re = np.random.uniform(size=n)
      im = np.random.uniform(size=n)
      ret = (re + im * 1j).reshape(shape)
      return ret

    for rank in VALID_FFT_RANKS:
      for dims in xrange(rank, rank + 3):
        for size in (5, 6):
          inner_dim = size // 2 + 1
          self._CompareForward(gen_real((size,) * dims), rank, (size,) * rank)
          complex_dims = (size,) * (dims - 1) + (inner_dim,)
          self._CompareBackward(gen_complex(complex_dims), rank, (size,) * rank)

  def testError(self):
    if test.is_gpu_available(cuda_only=True):
      for rank in VALID_FFT_RANKS:
        for dims in xrange(0, rank):
          x = np.zeros((1,) * dims).astype(np.complex64)
          with self.assertRaisesWithPredicateMatch(
              ValueError, "Shape must be .*rank {}.*".format(rank)):
            self._tfFFT(x, rank)
          with self.assertRaisesWithPredicateMatch(
              ValueError, "Shape must be .*rank {}.*".format(rank)):
            self._tfIFFT(x, rank)
        for dims in xrange(rank, rank + 2):
          x = np.zeros((1,) * rank)

          # Test non-rank-1 fft_length produces an error.
          fft_length = np.zeros((1, 1)).astype(np.int32)
          with self.assertRaisesWithPredicateMatch(ValueError,
                                                   "Shape must be .*rank 1"):
            self._tfFFT(x, rank, fft_length)
          with self.assertRaisesWithPredicateMatch(ValueError,
                                                   "Shape must be .*rank 1"):
            self._tfIFFT(x, rank, fft_length)

          # Test wrong fft_length length.
          fft_length = np.zeros((rank + 1,)).astype(np.int32)
          with self.assertRaisesWithPredicateMatch(
              ValueError, "Dimension must be .*but is {}.*".format(rank + 1)):
            self._tfFFT(x, rank, fft_length)
          with self.assertRaisesWithPredicateMatch(
              ValueError, "Dimension must be .*but is {}.*".format(rank + 1)):
            self._tfIFFT(x, rank, fft_length)

  def testGrad_Simple(self):
    if test.is_gpu_available(cuda_only=True):
      for rank in VALID_FFT_RANKS:
        # rfft3d/irfft3d do not have gradients yet.
        if rank == 3:
          continue
        for dims in xrange(rank, rank + 2):
          for size in (
              5,
              6,):
            re = np.ones(shape=(size,) * dims, dtype=np.float32)
            im = -np.ones(shape=(size,) * dims, dtype=np.float32)
            self._checkGradReal(self._tfFFTForRank(rank), re, use_gpu=True)
            self._checkGradComplex(
                self._tfIFFTForRank(rank),
                re,
                im,
                result_is_complex=False,
                use_gpu=True)

  def testGrad_Random(self):
    if test.is_gpu_available(cuda_only=True):
      np.random.seed(54321)
      for rank in VALID_FFT_RANKS:
        # rfft3d/irfft3d do not have gradients yet.
        if rank == 3:
          continue
        for dims in xrange(rank, rank + 2):
          for size in (5, 6):
            re = np.random.rand(*((size,) * dims)).astype(np.float32) * 2 - 1
            im = np.random.rand(*((size,) * dims)).astype(np.float32) * 2 - 1
            self._checkGradReal(self._tfFFTForRank(rank), re, use_gpu=True)
            self._checkGradComplex(
                self._tfIFFTForRank(rank),
                re,
                im,
                result_is_complex=False,
                use_gpu=True)


if __name__ == "__main__":
  test.main()
