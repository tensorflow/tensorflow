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
"""Tests for DCT operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

import numpy as np

from tensorflow.python.ops import spectral_ops
from tensorflow.python.ops import spectral_ops_test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


fftpack = try_import("scipy.fftpack")


def _np_dct2(signals, norm=None):
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
    # The orthonormal scaling includes a factor of 0.5 which we combine with
    # the overall scaling of 2.0 to cancel.
    dct[..., 0] *= np.sqrt(1.0 / dct_size)
    dct[..., 1:] *= np.sqrt(2.0 / dct_size)
  else:
    dct *= 2.0
  return dct


def _np_dct3(signals, norm=None):
  """Computes the DCT-III manually with NumPy."""
  # SciPy's `dct` has a scaling factor of 2.0 which we follow.
  # https://github.com/scipy/scipy/blob/v0.15.1/scipy/fftpack/src/dct.c.src
  dct_size = signals.shape[-1]
  signals = np.array(signals)  # make a copy so we can modify
  if norm == "ortho":
    signals[..., 0] *= np.sqrt(4.0 / dct_size)
    signals[..., 1:] *= np.sqrt(2.0 / dct_size)
  else:
    signals *= 2.0
  dct = np.zeros_like(signals)
  # X_k = 0.5 * x_0 +
  #       sum_{n=1}^{N-1} x_n * cos(\frac{pi}{N} * n * (k + 0.5))  k=0,...,N-1
  half_x0 = 0.5 * signals[..., 0]
  for k in range(dct_size):
    phi = np.cos(np.pi * np.arange(1, dct_size) * (k + 0.5) / dct_size)
    dct[..., k] = half_x0 + np.sum(signals[..., 1:] * phi, axis=-1)
  return dct


NP_DCT = {2: _np_dct2, 3: _np_dct3}
NP_IDCT = {2: _np_dct3, 3: _np_dct2}


class DCTOpsTest(test.TestCase):

  def _compare(self, signals, norm, dct_type, atol=5e-4, rtol=5e-4):
    """Compares (I)DCT to SciPy (if available) and a NumPy implementation."""
    np_dct = NP_DCT[dct_type](signals, norm)
    tf_dct = spectral_ops.dct(signals, type=dct_type, norm=norm).eval()
    self.assertAllClose(np_dct, tf_dct, atol=atol, rtol=rtol)
    np_idct = NP_IDCT[dct_type](signals, norm)
    tf_idct = spectral_ops.idct(signals, type=dct_type, norm=norm).eval()
    self.assertAllClose(np_idct, tf_idct, atol=atol, rtol=rtol)
    if fftpack:
      scipy_dct = fftpack.dct(signals, type=dct_type, norm=norm)
      self.assertAllClose(scipy_dct, tf_dct, atol=atol, rtol=rtol)
      scipy_idct = fftpack.idct(signals, type=dct_type, norm=norm)
      self.assertAllClose(scipy_idct, tf_idct, atol=atol, rtol=rtol)
    # Verify inverse(forward(s)) == s, up to a normalization factor.
    tf_idct_dct = spectral_ops.idct(
        tf_dct, type=dct_type, norm=norm).eval()
    tf_dct_idct = spectral_ops.dct(
        tf_idct, type=dct_type, norm=norm).eval()
    if norm is None:
      tf_idct_dct *= 0.5 / signals.shape[-1]
      tf_dct_idct *= 0.5 / signals.shape[-1]
    self.assertAllClose(signals, tf_idct_dct, atol=atol, rtol=rtol)
    self.assertAllClose(signals, tf_dct_idct, atol=atol, rtol=rtol)

  def test_random(self):
    """Test randomly generated batches of data."""
    with spectral_ops_test_util.fft_kernel_label_map():
      with self.test_session(use_gpu=True):
        for shape in ([1], [2], [3], [10], [2, 20], [2, 3, 25]):
          signals = np.random.rand(*shape).astype(np.float32)
          for norm in (None, "ortho"):
            self._compare(signals, norm, 2)
            self._compare(signals, norm, 3)

  def test_error(self):
    signals = np.random.rand(10)
    # Unsupported type.
    with self.assertRaises(ValueError):
      spectral_ops.dct(signals, type=1)
    # Unknown normalization.
    with self.assertRaises(ValueError):
      spectral_ops.dct(signals, norm="bad")
    with self.assertRaises(NotImplementedError):
      spectral_ops.dct(signals, n=10)
    with self.assertRaises(NotImplementedError):
      spectral_ops.dct(signals, axis=0)


if __name__ == "__main__":
  test.main()
