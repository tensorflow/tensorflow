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
"""Tests for mfcc_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

import numpy as np


from tensorflow.contrib.signal.python.ops import mfcc_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import spectral_ops_test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


# TODO(rjryan): Add scipy.fftpack to the TensorFlow build.
def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


fftpack = try_import("scipy.fftpack")


class DCTTest(test.TestCase):

  def _np_dct2(self, signals, norm=None):
    """Computes the DCT-II manually with NumPy."""
    # X_k = sum_{n=0}^{N-1} x_n * cos(\frac{pi}{N} * (n + 0.5) * k)  k=0,...,N-1
    dct_size = signals.shape[-1]
    dct = np.zeros_like(signals)
    for k in range(dct_size):
      phi = np.cos(np.pi * (np.arange(dct_size) + 0.5) * k / dct_size)
      dct[..., k] = np.sum(signals * phi, axis=-1)
    # SciPy's `dct` has a scaling factor of 2.0 which we follow.
    # https://github.com/scipy/scipy/blob/v0.15.1/scipy/fftpack/src/dct.c.src
    if norm == "ortho":
      # The orthogonal scaling includes a factor of 0.5 which we combine with
      # the overall scaling of 2.0 to cancel.
      dct[..., 0] *= np.sqrt(1.0 / dct_size)
      dct[..., 1:] *= np.sqrt(2.0 / dct_size)
    else:
      dct *= 2.0
    return dct

  def test_compare_to_numpy(self):
    """Compare dct against a manual DCT-II implementation."""
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session(use_gpu=True):
        for size in range(1, 23):
          signals = np.random.rand(size).astype(np.float32)
          actual_dct = mfcc_ops._dct2_1d(signals).eval()
          expected_dct = self._np_dct2(signals)
          self.assertAllClose(expected_dct, actual_dct, atol=5e-4, rtol=5e-4)

  def test_compare_to_fftpack(self):
    """Compare dct against scipy.fftpack.dct."""
    if not fftpack:
      return
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session(use_gpu=True):
        for size in range(1, 23):
          signal = np.random.rand(size).astype(np.float32)
          actual_dct = mfcc_ops._dct2_1d(signal).eval()
          expected_dct = fftpack.dct(signal, type=2)
          self.assertAllClose(expected_dct, actual_dct, atol=5e-4, rtol=5e-4)


# TODO(rjryan): We have no open source tests for MFCCs at the moment. Internally
# at Google, this code is tested against a reference implementation that follows
# HTK conventions.
class MFCCTest(test.TestCase):

  def test_error(self):
    # num_mel_bins must be positive.
    with self.assertRaises(ValueError):
      signal = array_ops.zeros((2, 3, 0))
      mfcc_ops.mfccs_from_log_mel_spectrograms(signal)

    # signal must be float32
    with self.assertRaises(ValueError):
      signal = array_ops.zeros((2, 3, 5), dtype=dtypes.float64)
      mfcc_ops.mfccs_from_log_mel_spectrograms(signal)

  def test_basic(self):
    """A basic test that the op runs on random input."""
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session(use_gpu=True):
        signal = random_ops.random_normal((2, 3, 5))
        mfcc_ops.mfccs_from_log_mel_spectrograms(signal).eval()


if __name__ == "__main__":
  test.main()
