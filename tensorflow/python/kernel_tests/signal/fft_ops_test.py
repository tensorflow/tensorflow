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

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import spectral_ops_test_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test

VALID_FFT_RANKS = (1, 2, 3)


class BaseFFTOpsTest(test.TestCase):

  def _compare(self, x, rank, fft_length=None, use_placeholder=False,
               rtol=1e-4, atol=1e-4):
    self._compareForward(x, rank, fft_length, use_placeholder, rtol, atol)
    self._compareBackward(x, rank, fft_length, use_placeholder, rtol, atol)

  def _compareForward(self, x, rank, fft_length=None, use_placeholder=False,
                      rtol=1e-4, atol=1e-4):
    x_np = self._npFFT(x, rank, fft_length)
    if use_placeholder:
      x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
      x_tf = self._tfFFT(x_ph, rank, fft_length, feed_dict={x_ph: x})
    else:
      x_tf = self._tfFFT(x, rank, fft_length)

    self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

  def _compareBackward(self, x, rank, fft_length=None, use_placeholder=False,
                       rtol=1e-4, atol=1e-4):
    x_np = self._npIFFT(x, rank, fft_length)
    if use_placeholder:
      x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
      x_tf = self._tfIFFT(x_ph, rank, fft_length, feed_dict={x_ph: x})
    else:
      x_tf = self._tfIFFT(x, rank, fft_length)

    self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

  def _checkMemoryFail(self, x, rank):
    config = config_pb2.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1e-2
    with self.cached_session(config=config, force_gpu=True):
      self._tfFFT(x, rank, fft_length=None)

  def _checkGradComplex(self, func, x, y, result_is_complex=True,
                        rtol=1e-2, atol=1e-2):
    with self.cached_session(use_gpu=True):
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

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=rtol, atol=atol)
    self.assertAllClose(y_jacob_t, y_jacob_n, rtol=rtol, atol=atol)

  def _checkGradReal(self, func, x, rtol=1e-2, atol=1e-2):
    with self.cached_session(use_gpu=True):
      inx = ops.convert_to_tensor(x)
      # func is a forward RFFT function (batched or unbatched).
      z = func(inx)
      # loss = sum(|z|^2)
      loss = math_ops.reduce_sum(math_ops.real(z * math_ops.conj(z)))
      x_jacob_t, x_jacob_n = test.compute_gradient(
          inx, list(x.shape), loss, [1], x_init_value=x, delta=1e-2)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=rtol, atol=atol)


class FFTOpsTest(BaseFFTOpsTest):

  def _tfFFT(self, x, rank, fft_length=None, feed_dict=None):
    # fft_length unused for complex FFTs.
    with self.cached_session(use_gpu=True) as sess:
      return sess.run(self._tfFFTForRank(rank)(x), feed_dict=feed_dict)

  def _tfIFFT(self, x, rank, fft_length=None, feed_dict=None):
    # fft_length unused for complex FFTs.
    with self.cached_session(use_gpu=True) as sess:
      return sess.run(self._tfIFFTForRank(rank)(x), feed_dict=feed_dict)

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
      return fft_ops.fft
    elif rank == 2:
      return fft_ops.fft2d
    elif rank == 3:
      return fft_ops.fft3d
    else:
      raise ValueError("invalid rank")

  def _tfIFFTForRank(self, rank):
    if rank == 1:
      return fft_ops.ifft
    elif rank == 2:
      return fft_ops.ifft2d
    elif rank == 3:
      return fft_ops.ifft3d
    else:
      raise ValueError("invalid rank")

  @test_util.run_deprecated_v1
  def testEmpty(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for np_type in (np.complex64, np.complex128):
        for rank in VALID_FFT_RANKS:
          for dims in xrange(rank, rank + 3):
            x = np.zeros((0,) * dims).astype(np_type)
            self.assertEqual(x.shape, self._tfFFT(x, rank).shape)
            self.assertEqual(x.shape, self._tfIFFT(x, rank).shape)

  @test_util.run_deprecated_v1
  def testBasic(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for np_type, tol in ((np.complex64, 1e-4), (np.complex128, 1e-8)):
        for rank in VALID_FFT_RANKS:
          for dims in xrange(rank, rank + 3):
            self._compare(
                np.mod(np.arange(np.power(4, dims)), 10).reshape(
                    (4,) * dims).astype(np_type), rank, rtol=tol, atol=tol)

  def testLargeBatch(self):
    if test.is_gpu_available(cuda_only=True):
      rank = 1
      for dims in xrange(rank, rank + 3):
        for np_type, tol in ((np.complex64, 1e-4), (np.complex128, 1e-5)):
          self._compare(
              np.mod(np.arange(np.power(128, dims)), 10).reshape(
                  (128,) * dims).astype(np_type), rank, rtol=tol, atol=tol)

  # TODO(yangzihao): Disable before we can figure out a way to
  # properly test memory fail for large batch fft.
  # def testLargeBatchMemoryFail(self):
  #   if test.is_gpu_available(cuda_only=True):
  #     rank = 1
  #     for dims in xrange(rank, rank + 3):
  #       self._checkMemoryFail(
  #           np.mod(np.arange(np.power(128, dims)), 64).reshape(
  #               (128,) * dims).astype(np.complex64), rank)

  @test_util.run_deprecated_v1
  def testBasicPlaceholder(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for np_type, tol in ((np.complex64, 1e-4), (np.complex128, 1e-8)):
        for rank in VALID_FFT_RANKS:
          for dims in xrange(rank, rank + 3):
            self._compare(
                np.mod(np.arange(np.power(4, dims)), 10).reshape(
                    (4,) * dims).astype(np_type),
                rank, use_placeholder=True, rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testRandom(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for np_type, tol in ((np.complex64, 1e-4), (np.complex128, 5e-6)):
        def gen(shape):
          n = np.prod(shape)
          re = np.random.uniform(size=n)
          im = np.random.uniform(size=n)
          return (re + im * 1j).reshape(shape)

        for rank in VALID_FFT_RANKS:
          for dims in xrange(rank, rank + 3):
            self._compare(gen((4,) * dims).astype(np_type), rank,
                          rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testRandom1D(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for np_type in (np.complex64, np.complex128):
        has_gpu = test.is_gpu_available(cuda_only=True)
        tol = {(np.complex64, True): 1e-4,
               (np.complex64, False): 1e-2,
               (np.complex128, True): 1e-4,
               (np.complex128, False): 1e-2}[(np_type, has_gpu)]
        def gen(shape):
          n = np.prod(shape)
          re = np.random.uniform(size=n)
          im = np.random.uniform(size=n)
          return (re + im * 1j).reshape(shape)

        # Check a variety of power-of-2 FFT sizes.
        for dim in (128, 256, 512, 1024):
          self._compare(gen((dim,)).astype(np_type), 1, rtol=tol, atol=tol)

        # Check a variety of non-power-of-2 FFT sizes.
        for dim in (127, 255, 511, 1023):
          self._compare(gen((dim,)).astype(np_type), 1, rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testError(self):
    for rank in VALID_FFT_RANKS:
      for dims in xrange(0, rank):
        x = np.zeros((1,) * dims).astype(np.complex64)
        with self.assertRaisesWithPredicateMatch(
            ValueError, "Shape must be .*rank {}.*".format(rank)):
          self._tfFFT(x, rank)
        with self.assertRaisesWithPredicateMatch(
            ValueError, "Shape must be .*rank {}.*".format(rank)):
          self._tfIFFT(x, rank)

  @test_util.run_deprecated_v1
  def testGrad_Simple(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for np_type, tol in ((np.float32, 1e-4), (np.float64, 1e-10)):
        for rank in VALID_FFT_RANKS:
          for dims in xrange(rank, rank + 2):
            re = np.ones(shape=(4,) * dims, dtype=np_type) / 10.0
            im = np.zeros(shape=(4,) * dims, dtype=np_type)
            self._checkGradComplex(self._tfFFTForRank(rank), re, im,
                                   rtol=tol, atol=tol)
            self._checkGradComplex(self._tfIFFTForRank(rank), re, im,
                                   rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testGrad_Random(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for np_type, tol in ((np.float32, 1e-2), (np.float64, 1e-10)):
        for rank in VALID_FFT_RANKS:
          for dims in xrange(rank, rank + 2):
            re = np.random.rand(*((3,) * dims)).astype(np_type) * 2 - 1
            im = np.random.rand(*((3,) * dims)).astype(np_type) * 2 - 1
            self._checkGradComplex(self._tfFFTForRank(rank), re, im,
                                   rtol=tol, atol=tol)
            self._checkGradComplex(self._tfIFFTForRank(rank), re, im,
                                   rtol=tol, atol=tol)


class RFFTOpsTest(BaseFFTOpsTest):

  def _compareBackward(self, x, rank, fft_length=None, use_placeholder=False):
    super(RFFTOpsTest, self)._compareBackward(x, rank, fft_length,
                                              use_placeholder)

  def _tfFFT(self, x, rank, fft_length=None, feed_dict=None):
    with self.cached_session(use_gpu=True) as sess:
      return sess.run(
          self._tfFFTForRank(rank)(x, fft_length), feed_dict=feed_dict)

  def _tfIFFT(self, x, rank, fft_length=None, feed_dict=None):
    with self.cached_session(use_gpu=True) as sess:
      return sess.run(
          self._tfIFFTForRank(rank)(x, fft_length), feed_dict=feed_dict)

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
      return fft_ops.rfft
    elif rank == 2:
      return fft_ops.rfft2d
    elif rank == 3:
      return fft_ops.rfft3d
    else:
      raise ValueError("invalid rank")

  def _tfIFFTForRank(self, rank):
    if rank == 1:
      return fft_ops.irfft
    elif rank == 2:
      return fft_ops.irfft2d
    elif rank == 3:
      return fft_ops.irfft3d
    else:
      raise ValueError("invalid rank")

  @test_util.run_deprecated_v1
  def testEmpty(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for rank in VALID_FFT_RANKS:
        for dims in xrange(rank, rank + 3):
          x = np.zeros((0,) * dims).astype(np.float32)
          self.assertEqual(x.shape, self._tfFFT(x, rank).shape)
          x = np.zeros((0,) * dims).astype(np.complex64)
          self.assertEqual(x.shape, self._tfIFFT(x, rank).shape)

  @test_util.run_deprecated_v1
  def testBasic(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for rank in VALID_FFT_RANKS:
        for dims in xrange(rank, rank + 3):
          for size in (5, 6):
            inner_dim = size // 2 + 1
            r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
                (size,) * dims)
            self._compareForward(r2c.astype(np.float32), rank, (size,) * rank)
            c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                         10).reshape((size,) * (dims - 1) + (inner_dim,))
            self._compareBackward(
                c2r.astype(np.complex64), rank, (size,) * rank)

  def testLargeBatch(self):
    if test.is_gpu_available(cuda_only=True):
      rank = 1
      for dims in xrange(rank, rank + 3):
        for size in (64, 128):
          inner_dim = size // 2 + 1
          r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
              (size,) * dims)
          self._compareForward(r2c.astype(np.float32), rank, (size,) * rank)
          c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                       10).reshape((size,) * (dims - 1) + (inner_dim,))
          self._compareBackward(c2r.astype(np.complex64), rank, (size,) * rank)

  @test_util.run_deprecated_v1
  def testBasicPlaceholder(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for rank in VALID_FFT_RANKS:
        for dims in xrange(rank, rank + 3):
          for size in (5, 6):
            inner_dim = size // 2 + 1
            r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
                (size,) * dims)
            self._compareForward(
                r2c.astype(np.float32),
                rank, (size,) * rank,
                use_placeholder=True)
            c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                         10).reshape((size,) * (dims - 1) + (inner_dim,))
            self._compareBackward(
                c2r.astype(np.complex64),
                rank, (size,) * rank,
                use_placeholder=True)

  @test_util.run_deprecated_v1
  def testFftLength(self):
    if test.is_gpu_available(cuda_only=True):
      with spectral_ops_test_util.fft_kernel_label_map():
        for rank in VALID_FFT_RANKS:
          for dims in xrange(rank, rank + 3):
            for size in (5, 6):
              inner_dim = size // 2 + 1
              r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
                  (size,) * dims)
              c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                           10).reshape((size,) * (dims - 1) + (inner_dim,))
              # Test truncation (FFT size < dimensions).
              fft_length = (size - 2,) * rank
              self._compareForward(r2c.astype(np.float32), rank, fft_length)
              self._compareBackward(c2r.astype(np.complex64), rank, fft_length)
              # Confirm it works with unknown shapes as well.
              self._compareForward(
                  r2c.astype(np.float32),
                  rank,
                  fft_length,
                  use_placeholder=True)
              self._compareBackward(
                  c2r.astype(np.complex64),
                  rank,
                  fft_length,
                  use_placeholder=True)
              # Test padding (FFT size > dimensions).
              fft_length = (size + 2,) * rank
              self._compareForward(r2c.astype(np.float32), rank, fft_length)
              self._compareBackward(c2r.astype(np.complex64), rank, fft_length)
              # Confirm it works with unknown shapes as well.
              self._compareForward(
                  r2c.astype(np.float32),
                  rank,
                  fft_length,
                  use_placeholder=True)
              self._compareBackward(
                  c2r.astype(np.complex64),
                  rank,
                  fft_length,
                  use_placeholder=True)

  @test_util.run_deprecated_v1
  def testRandom(self):
    with spectral_ops_test_util.fft_kernel_label_map():
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
            self._compareForward(gen_real((size,) * dims), rank, (size,) * rank)
            complex_dims = (size,) * (dims - 1) + (inner_dim,)
            self._compareBackward(
                gen_complex(complex_dims), rank, (size,) * rank)

  @test_util.run_deprecated_v1
  def testError(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for rank in VALID_FFT_RANKS:
        for dims in xrange(0, rank):
          x = np.zeros((1,) * dims).astype(np.complex64)
          with self.assertRaisesWithPredicateMatch(
              ValueError, "Shape .* must have rank at least {}".format(rank)):
            self._tfFFT(x, rank)
          with self.assertRaisesWithPredicateMatch(
              ValueError, "Shape .* must have rank at least {}".format(rank)):
            self._tfIFFT(x, rank)
        for dims in xrange(rank, rank + 2):
          x = np.zeros((1,) * rank)

          # Test non-rank-1 fft_length produces an error.
          fft_length = np.zeros((1, 1)).astype(np.int32)
          with self.assertRaisesWithPredicateMatch(ValueError,
                                                   "Shape .* must have rank 1"):
            self._tfFFT(x, rank, fft_length)
          with self.assertRaisesWithPredicateMatch(ValueError,
                                                   "Shape .* must have rank 1"):
            self._tfIFFT(x, rank, fft_length)

          # Test wrong fft_length length.
          fft_length = np.zeros((rank + 1,)).astype(np.int32)
          with self.assertRaisesWithPredicateMatch(
              ValueError, "Dimension must be .*but is {}.*".format(rank + 1)):
            self._tfFFT(x, rank, fft_length)
          with self.assertRaisesWithPredicateMatch(
              ValueError, "Dimension must be .*but is {}.*".format(rank + 1)):
            self._tfIFFT(x, rank, fft_length)

        # Test that calling the kernel directly without padding to fft_length
        # produces an error.
        rffts_for_rank = {
            1: [gen_spectral_ops.rfft, gen_spectral_ops.irfft],
            2: [gen_spectral_ops.rfft2d, gen_spectral_ops.irfft2d],
            3: [gen_spectral_ops.rfft3d, gen_spectral_ops.irfft3d]
        }
        rfft_fn, irfft_fn = rffts_for_rank[rank]
        with self.assertRaisesWithPredicateMatch(
            errors.InvalidArgumentError,
            "Input dimension .* must have length of at least 6 but got: 5"):
          x = np.zeros((5,) * rank).astype(np.float32)
          fft_length = [6] * rank
          with self.cached_session():
            self.evaluate(rfft_fn(x, fft_length))

        with self.assertRaisesWithPredicateMatch(
            errors.InvalidArgumentError,
            "Input dimension .* must have length of at least .* but got: 3"):
          x = np.zeros((3,) * rank).astype(np.complex64)
          fft_length = [6] * rank
          with self.cached_session():
            self.evaluate(irfft_fn(x, fft_length))

  @test_util.run_deprecated_v1
  def testGrad_Simple(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for rank in VALID_FFT_RANKS:
        # rfft3d/irfft3d do not have gradients yet.
        if rank == 3:
          continue
        for dims in xrange(rank, rank + 2):
          for size in (5, 6):
            re = np.ones(shape=(size,) * dims, dtype=np.float32)
            im = -np.ones(shape=(size,) * dims, dtype=np.float32)
            self._checkGradReal(self._tfFFTForRank(rank), re)
            self._checkGradComplex(
                self._tfIFFTForRank(rank), re, im, result_is_complex=False)

  @test_util.run_deprecated_v1
  def testGrad_Random(self):
    with spectral_ops_test_util.fft_kernel_label_map():
      for rank in VALID_FFT_RANKS:
        # rfft3d/irfft3d do not have gradients yet.
        if rank == 3:
          continue
        for dims in xrange(rank, rank + 2):
          for size in (5, 6):
            re = np.random.rand(*((size,) * dims)).astype(np.float32) * 2 - 1
            im = np.random.rand(*((size,) * dims)).astype(np.float32) * 2 - 1
            self._checkGradReal(self._tfFFTForRank(rank), re)
            self._checkGradComplex(
                self._tfIFFTForRank(rank), re, im, result_is_complex=False)


if __name__ == "__main__":
  test.main()
