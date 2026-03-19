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
"""Tests for FFT via the XLA JIT."""

import itertools

import numpy as np
import scipy.signal as sps

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops.signal import signal
from tensorflow.python.platform import googletest

BATCH_DIMS = (3, 5)
RTOL = 0.009  # Eigen/cuFFT differ widely from np, especially for FFT3D
ATOL = 1e-4
RTOL_3D = 0.07
ATOL_3D = 4e-4


def pick_10(x):
  x = list(x)
  np.random.seed(123)
  np.random.shuffle(x)
  return x[:10]


def to_32bit(x):
  if x.dtype == np.complex128:
    return x.astype(np.complex64)
  if x.dtype == np.float64:
    return x.astype(np.float32)
  return x


POWS_OF_2 = 2**np.arange(3, 12)
INNER_DIMS_1D = list((x,) for x in POWS_OF_2)
POWS_OF_2 = 2**np.arange(3, 8)  # To avoid OOM on GPU.
INNER_DIMS_2D = pick_10(itertools.product(POWS_OF_2, POWS_OF_2))
INNER_DIMS_3D = pick_10(itertools.product(POWS_OF_2, POWS_OF_2, POWS_OF_2))


class FFTTest(xla_test.XLATestCase):

  def _VerifyFftMethod(self,
                       inner_dims,
                       complex_to_input,
                       input_to_expected,
                       tf_method,
                       atol=ATOL,
                       rtol=RTOL):
    for indims in inner_dims:
      print("nfft =", indims)
      shape = BATCH_DIMS + indims
      data = np.arange(np.prod(shape) * 2) / np.prod(indims)
      np.random.seed(123)
      np.random.shuffle(data)
      data = np.reshape(data.astype(np.float32).view(np.complex64), shape)
      data = to_32bit(complex_to_input(data))
      expected = to_32bit(input_to_expected(data))
      with self.session() as sess:
        with self.test_scope():
          ph = array_ops.placeholder(
              dtypes.as_dtype(data.dtype), shape=data.shape)
          out = tf_method(ph)
        value = sess.run(out, {ph: data})
        self.assertAllClose(expected, value, rtol=rtol, atol=atol)

  def testContribSignalSTFT(self):
    ws = 512
    hs = 128
    dims = (ws * 20,)
    shape = BATCH_DIMS + dims
    data = np.arange(np.prod(shape)) / np.prod(dims)
    np.random.seed(123)
    np.random.shuffle(data)
    data = np.reshape(data.astype(np.float32), shape)
    window = sps.get_window("hann", ws)
    expected = sps.stft(
        data, nperseg=ws, noverlap=ws - hs, boundary=None, window=window)[2]
    expected = np.swapaxes(expected, -1, -2)
    expected *= window.sum()  # scipy divides by window sum
    with self.session() as sess:
      with self.test_scope():
        ph = array_ops.placeholder(
            dtypes.as_dtype(data.dtype), shape=data.shape)
        out = signal.stft(ph, ws, hs)
        grad = gradients_impl.gradients(out, ph,
                                        grad_ys=array_ops.ones_like(out))

      # For gradients, we simply verify that they compile & execute.
      value, _ = sess.run([out, grad], {ph: data})
      self.assertAllClose(expected, value, rtol=RTOL, atol=ATOL)

  def testFFT(self):
    self._VerifyFftMethod(INNER_DIMS_1D, lambda x: x, np.fft.fft,
                          signal.fft)

  def testFFT2D(self):
    self._VerifyFftMethod(INNER_DIMS_2D, lambda x: x, np.fft.fft2,
                          signal.fft2d)

  def testFFT3D(self):
    self._VerifyFftMethod(INNER_DIMS_3D, lambda x: x,
                          lambda x: np.fft.fftn(x, axes=(-3, -2, -1)),
                          signal.fft3d, ATOL_3D, RTOL_3D)

  def testIFFT(self):
    self._VerifyFftMethod(INNER_DIMS_1D, lambda x: x, np.fft.ifft,
                          signal.ifft)

  def testIFFT2D(self):
    self._VerifyFftMethod(INNER_DIMS_2D, lambda x: x, np.fft.ifft2,
                          signal.ifft2d)

  def testIFFT3D(self):
    self._VerifyFftMethod(INNER_DIMS_3D, lambda x: x,
                          lambda x: np.fft.ifftn(x, axes=(-3, -2, -1)),
                          signal.ifft3d, ATOL_3D, RTOL_3D)

  def testRFFT(self):

    def _to_expected(x):
      return np.fft.rfft(x, n=x.shape[-1])

    def _tf_fn(x):
      return signal.rfft(x, fft_length=[x.shape[-1]])

    self._VerifyFftMethod(INNER_DIMS_1D, np.real, _to_expected, _tf_fn)

  def testRFFT2D(self):

    def _tf_fn(x):
      return signal.rfft2d(x, fft_length=[x.shape[-2], x.shape[-1]])

    self._VerifyFftMethod(
        INNER_DIMS_2D, np.real,
        lambda x: np.fft.rfft2(x, s=[x.shape[-2], x.shape[-1]]), _tf_fn)

  def testRFFT3D(self):

    def _to_expected(x):
      return np.fft.rfftn(
          x, axes=(-3, -2, -1), s=[x.shape[-3], x.shape[-2], x.shape[-1]])

    def _tf_fn(x):
      return signal.rfft3d(
          x, fft_length=[x.shape[-3], x.shape[-2], x.shape[-1]])

    self._VerifyFftMethod(INNER_DIMS_3D, np.real, _to_expected, _tf_fn, ATOL_3D,
                          RTOL_3D)

  def testRFFT3DMismatchedSize(self):

    def _to_expected(x):
      return np.fft.rfftn(
          x,
          axes=(-3, -2, -1),
          s=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

    def _tf_fn(x):
      return signal.rfft3d(
          x, fft_length=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

    self._VerifyFftMethod(INNER_DIMS_3D, np.real, _to_expected, _tf_fn)

  def testIRFFT(self):

    def _tf_fn(x):
      return signal.irfft(x, fft_length=[2 * (x.shape[-1] - 1)])

    self._VerifyFftMethod(
        INNER_DIMS_1D, lambda x: np.fft.rfft(np.real(x), n=x.shape[-1]),
        lambda x: np.fft.irfft(x, n=2 * (x.shape[-1] - 1)), _tf_fn)

  def testIRFFT2D(self):

    def _tf_fn(x):
      return signal.irfft2d(x, fft_length=[x.shape[-2], 2 * (x.shape[-1] - 1)])

    self._VerifyFftMethod(
        INNER_DIMS_2D,
        lambda x: np.fft.rfft2(np.real(x), s=[x.shape[-2], x.shape[-1]]),
        lambda x: np.fft.irfft2(x, s=[x.shape[-2], 2 * (x.shape[-1] - 1)]),
        _tf_fn)

  def testIRFFT3D(self):

    def _to_input(x):
      return np.fft.rfftn(
          np.real(x),
          axes=(-3, -2, -1),
          s=[x.shape[-3], x.shape[-2], x.shape[-1]])

    def _to_expected(x):
      return np.fft.irfftn(
          x,
          axes=(-3, -2, -1),
          s=[x.shape[-3], x.shape[-2], 2 * (x.shape[-1] - 1)])

    def _tf_fn(x):
      return signal.irfft3d(
          x, fft_length=[x.shape[-3], x.shape[-2], 2 * (x.shape[-1] - 1)])

    self._VerifyFftMethod(INNER_DIMS_3D, _to_input, _to_expected, _tf_fn,
                          ATOL_3D, RTOL_3D)

  def testIRFFT3DMismatchedSize(self):

    def _to_input(x):
      return np.fft.rfftn(
          np.real(x),
          axes=(-3, -2, -1),
          s=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

    def _to_expected(x):
      return np.fft.irfftn(
          x,
          axes=(-3, -2, -1),
          s=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

    def _tf_fn(x):
      return signal.irfft3d(
          x, fft_length=[x.shape[-3] // 2, x.shape[-2], x.shape[-1] * 2])

    self._VerifyFftMethod(INNER_DIMS_3D, _to_input, _to_expected, _tf_fn,
                          ATOL_3D, RTOL_3D)



if __name__ == "__main__":
  googletest.main()
