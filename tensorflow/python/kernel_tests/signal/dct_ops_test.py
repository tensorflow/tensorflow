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
import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops.signal import dct_ops
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


def _modify_input_for_dct(signals, n=None):
  """Pad or trim the provided NumPy array's innermost axis to length n."""
  signal = np.array(signals)
  if n is None or n == signal.shape[-1]:
    signal_mod = signal
  elif n >= 1:
    signal_len = signal.shape[-1]
    if n <= signal_len:
      signal_mod = signal[..., 0:n]
    else:
      output_shape = list(signal.shape)
      output_shape[-1] = n
      signal_mod = np.zeros(output_shape)
      signal_mod[..., 0:signal.shape[-1]] = signal
  if n:
    assert signal_mod.shape[-1] == n
  return signal_mod


def _np_dct1(signals, n=None, norm=None):
  """Computes the DCT-I manually with NumPy."""
  # X_k = (x_0 + (-1)**k * x_{N-1} +
  #       2 * sum_{n=0}^{N-2} x_n * cos(\frac{pi}{N-1} * n * k)  k=0,...,N-1
  del norm
  signals_mod = _modify_input_for_dct(signals, n=n)
  dct_size = signals_mod.shape[-1]
  dct = np.zeros_like(signals_mod)
  for k in range(dct_size):
    phi = np.cos(np.pi * np.arange(1, dct_size - 1) * k / (dct_size - 1))
    dct[..., k] = 2 * np.sum(
        signals_mod[..., 1:-1] * phi, axis=-1) + (
            signals_mod[..., 0] + (-1)**k * signals_mod[..., -1])
  return dct


def _np_dct2(signals, n=None, norm=None):
  """Computes the DCT-II manually with NumPy."""
  # X_k = sum_{n=0}^{N-1} x_n * cos(\frac{pi}{N} * (n + 0.5) * k)  k=0,...,N-1
  signals_mod = _modify_input_for_dct(signals, n=n)
  dct_size = signals_mod.shape[-1]
  dct = np.zeros_like(signals_mod)
  for k in range(dct_size):
    phi = np.cos(np.pi * (np.arange(dct_size) + 0.5) * k / dct_size)
    dct[..., k] = np.sum(signals_mod * phi, axis=-1)
  # SciPy's `dct` has a scaling factor of 2.0 which we follow.
  # https://github.com/scipy/scipy/blob/v1.2.1/scipy/fftpack/src/dct.c.src
  if norm == "ortho":
    # The orthonormal scaling includes a factor of 0.5 which we combine with
    # the overall scaling of 2.0 to cancel.
    dct[..., 0] *= np.sqrt(1.0 / dct_size)
    dct[..., 1:] *= np.sqrt(2.0 / dct_size)
  else:
    dct *= 2.0
  return dct


def _np_dct3(signals, n=None, norm=None):
  """Computes the DCT-III manually with NumPy."""
  # SciPy's `dct` has a scaling factor of 2.0 which we follow.
  # https://github.com/scipy/scipy/blob/v1.2.1/scipy/fftpack/src/dct.c.src
  signals_mod = _modify_input_for_dct(signals, n=n)
  dct_size = signals_mod.shape[-1]
  signals_mod = np.array(signals_mod)  # make a copy so we can modify
  if norm == "ortho":
    signals_mod[..., 0] *= np.sqrt(4.0 / dct_size)
    signals_mod[..., 1:] *= np.sqrt(2.0 / dct_size)
  else:
    signals_mod *= 2.0
  dct = np.zeros_like(signals_mod)
  # X_k = 0.5 * x_0 +
  #       sum_{n=1}^{N-1} x_n * cos(\frac{pi}{N} * n * (k + 0.5))  k=0,...,N-1
  half_x0 = 0.5 * signals_mod[..., 0]
  for k in range(dct_size):
    phi = np.cos(np.pi * np.arange(1, dct_size) * (k + 0.5) / dct_size)
    dct[..., k] = half_x0 + np.sum(signals_mod[..., 1:] * phi, axis=-1)
  return dct


def _np_dct4(signals, n=None, norm=None):
  """Computes the DCT-IV manually with NumPy."""
  # SciPy's `dct` has a scaling factor of 2.0 which we follow.
  # https://github.com/scipy/scipy/blob/v1.2.1/scipy/fftpack/src/dct.c.src
  signals_mod = _modify_input_for_dct(signals, n=n)
  dct_size = signals_mod.shape[-1]
  signals_mod = np.array(signals_mod)  # make a copy so we can modify
  if norm == "ortho":
    signals_mod *= np.sqrt(2.0 / dct_size)
  else:
    signals_mod *= 2.0
  dct = np.zeros_like(signals_mod)
  # X_k = sum_{n=0}^{N-1}
  #            x_n * cos(\frac{pi}{4N} * (2n + 1) * (2k + 1))  k=0,...,N-1
  for k in range(dct_size):
    phi = np.cos(np.pi *
                 (2 * np.arange(0, dct_size) + 1) * (2 * k + 1) /
                 (4.0 * dct_size))
    dct[..., k] = np.sum(signals_mod * phi, axis=-1)
  return dct


NP_DCT = {1: _np_dct1, 2: _np_dct2, 3: _np_dct3, 4: _np_dct4}
NP_IDCT = {1: _np_dct1, 2: _np_dct3, 3: _np_dct2, 4: _np_dct4}


@test_util.run_all_in_graph_and_eager_modes
class DCTOpsTest(parameterized.TestCase, test.TestCase):

  def _compare(self, signals, n, norm, dct_type, atol, rtol):
    """Compares (I)DCT to SciPy (if available) and a NumPy implementation."""
    np_dct = NP_DCT[dct_type](signals, n=n, norm=norm)
    tf_dct = dct_ops.dct(signals, n=n, type=dct_type, norm=norm)
    self.assertEqual(tf_dct.dtype.as_numpy_dtype, signals.dtype)
    self.assertAllClose(np_dct, tf_dct, atol=atol, rtol=rtol)
    np_idct = NP_IDCT[dct_type](signals, n=None, norm=norm)
    tf_idct = dct_ops.idct(signals, type=dct_type, norm=norm)
    self.assertEqual(tf_idct.dtype.as_numpy_dtype, signals.dtype)
    self.assertAllClose(np_idct, tf_idct, atol=atol, rtol=rtol)
    if fftpack and dct_type != 4:
      scipy_dct = fftpack.dct(signals, n=n, type=dct_type, norm=norm)
      self.assertAllClose(scipy_dct, tf_dct, atol=atol, rtol=rtol)
      scipy_idct = fftpack.idct(signals, type=dct_type, norm=norm)
      self.assertAllClose(scipy_idct, tf_idct, atol=atol, rtol=rtol)
    # Verify inverse(forward(s)) == s, up to a normalization factor.
    # Since `n` is not implemented for IDCT operation, re-calculating tf_dct
    # without n.
    tf_dct = dct_ops.dct(signals, type=dct_type, norm=norm)
    tf_idct_dct = dct_ops.idct(tf_dct, type=dct_type, norm=norm)
    tf_dct_idct = dct_ops.dct(tf_idct, type=dct_type, norm=norm)
    if norm is None:
      if dct_type == 1:
        tf_idct_dct *= 0.5 / (signals.shape[-1] - 1)
        tf_dct_idct *= 0.5 / (signals.shape[-1] - 1)
      else:
        tf_idct_dct *= 0.5 / signals.shape[-1]
        tf_dct_idct *= 0.5 / signals.shape[-1]
    self.assertAllClose(signals, tf_idct_dct, atol=atol, rtol=rtol)
    self.assertAllClose(signals, tf_dct_idct, atol=atol, rtol=rtol)

  @parameterized.parameters(itertools.product(
      [1, 2, 3, 4],
      [None, "ortho"],
      [[2], [3], [10], [2, 20], [2, 3, 25]],
      [np.float32, np.float64]))
  def test_random(self, dct_type, norm, shape, dtype):
    """Test randomly generated batches of data."""
    # "ortho" normalization is not implemented for type I.
    if dct_type == 1 and norm == "ortho":
      return
    with self.session():
      tol = 5e-4 if dtype == np.float32 else 1e-7
      signals = np.random.rand(*shape).astype(dtype)
      n = np.random.randint(1, 2 * signals.shape[-1])
      n = np.random.choice([None, n])
      self._compare(signals, n, norm=norm, dct_type=dct_type,
                    rtol=tol, atol=tol)

  def test_error(self):
    signals = np.random.rand(10)
    # Unsupported type.
    with self.assertRaises(ValueError):
      dct_ops.dct(signals, type=5)
    # Invalid n.
    with self.assertRaises(ValueError):
      dct_ops.dct(signals, n=-2)
    # DCT-I normalization not implemented.
    with self.assertRaises(ValueError):
      dct_ops.dct(signals, type=1, norm="ortho")
    # DCT-I requires at least two inputs.
    with self.assertRaises(ValueError):
      dct_ops.dct(np.random.rand(1), type=1)
    # Unknown normalization.
    with self.assertRaises(ValueError):
      dct_ops.dct(signals, norm="bad")
    with self.assertRaises(NotImplementedError):
      dct_ops.dct(signals, axis=0)


if __name__ == "__main__":
  test.main()
