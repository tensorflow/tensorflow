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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import scipy.signal as sps

from tensorflow.compiler.tests import xla_test
from tensorflow.contrib.signal.python.ops import spectral_ops as signal
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import spectral_ops
from tensorflow.python.platform import googletest

BATCH_DIMS = (3, 5)
RTOL = 0.02  # Eigen/cuFFT differ widely from np, especially for FFT3D
ATOL = 1e-3


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

  def _VerifyFftMethod(self, inner_dims, complex_to_input, input_to_expected,
                       tf_method):
    for indims in inner_dims:
      print("nfft =", indims)
      shape = BATCH_DIMS + indims
      data = np.arange(np.prod(shape) * 2) / np.prod(indims)
      np.random.seed(123)
      np.random.shuffle(data)
      data = np.reshape(data.astype(np.float32).view(np.complex64), shape)
      data = to_32bit(complex_to_input(data))
      expected = to_32bit(input_to_expected(data))
      with self.cached_session() as sess:
        with self.test_scope():
          ph = array_ops.placeholder(
              dtypes.as_dtype(data.dtype), shape=data.shape)
          out = tf_method(ph)
        value = sess.run(out, {ph: data})
        self.assertAllClose(expected, value, rtol=RTOL, atol=ATOL)

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
    with self.cached_session() as sess:
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
                          spectral_ops.fft)

  def testFFT2D(self):
    self._VerifyFftMethod(INNER_DIMS_2D, lambda x: x, np.fft.fft2,
                          spectral_ops.fft2d)

  def testFFT3D(self):
    self._VerifyFftMethod(INNER_DIMS_3D, lambda x: x,
                          lambda x: np.fft.fftn(x, axes=(-3, -2, -1)),
                          spectral_ops.fft3d)

  def testIFFT(self):
    self._VerifyFftMethod(INNER_DIMS_1D, lambda x: x, np.fft.ifft,
                          spectral_ops.ifft)

  def testIFFT2D(self):
    self._VerifyFftMethod(INNER_DIMS_2D, lambda x: x, np.fft.ifft2,
                          spectral_ops.ifft2d)

  def testIFFT3D(self):
    self._VerifyFftMethod(INNER_DIMS_3D, lambda x: x,
                          lambda x: np.fft.ifftn(x, axes=(-3, -2, -1)),
                          spectral_ops.ifft3d)

  def testRFFT(self):
    self._VerifyFftMethod(
        INNER_DIMS_1D, np.real, lambda x: np.fft.rfft(x, n=x.shape[-1]),
        lambda x: spectral_ops.rfft(x, fft_length=[x.shape[-1].value]))

  def testRFFT2D(self):

    def _tf_fn(x):
      return spectral_ops.rfft2d(
          x, fft_length=[x.shape[-2].value, x.shape[-1].value])

    self._VerifyFftMethod(
        INNER_DIMS_2D, np.real,
        lambda x: np.fft.rfft2(x, s=[x.shape[-2], x.shape[-1]]), _tf_fn)

  def testRFFT3D(self):

    def _to_expected(x):
      return np.fft.rfftn(
          x, axes=(-3, -2, -1), s=[x.shape[-3], x.shape[-2], x.shape[-1]])

    def _tf_fn(x):
      return spectral_ops.rfft3d(
          x,
          fft_length=[x.shape[-3].value, x.shape[-2].value, x.shape[-1].value])

    self._VerifyFftMethod(INNER_DIMS_3D, np.real, _to_expected, _tf_fn)

  def testIRFFT(self):

    def _tf_fn(x):
      return spectral_ops.irfft(x, fft_length=[2 * (x.shape[-1].value - 1)])

    self._VerifyFftMethod(
        INNER_DIMS_1D, lambda x: np.fft.rfft(np.real(x), n=x.shape[-1]),
        lambda x: np.fft.irfft(x, n=2 * (x.shape[-1] - 1)), _tf_fn)

  def testIRFFT2D(self):

    def _tf_fn(x):
      return spectral_ops.irfft2d(
          x, fft_length=[x.shape[-2].value, 2 * (x.shape[-1].value - 1)])

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
      return spectral_ops.irfft3d(
          x,
          fft_length=[
              x.shape[-3].value, x.shape[-2].value, 2 * (x.shape[-1].value - 1)
          ])

    self._VerifyFftMethod(INNER_DIMS_3D, _to_input, _to_expected, _tf_fn)


if __name__ == "__main__":
  googletest.main()
