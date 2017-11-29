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
"""Tests for window_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.contrib.signal.python.ops import window_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


def _scipy_raised_cosine(length, symmetric=True, a=0.5, b=0.5):
  """A simple implementation of a raised cosine window that matches SciPy.

  https://en.wikipedia.org/wiki/Window_function#Hann_window
  https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/windows.py#L615

  Args:
    length: The window length.
    symmetric: Whether to create a symmetric window.
    a: The alpha parameter of the raised cosine window.
    b: The beta parameter of the raised cosine window.

  Returns:
    A raised cosine window of length `length`.
  """
  if length == 1:
    return np.ones(1)
  odd = length % 2
  if not symmetric and not odd:
    length += 1
  window = a - b * np.cos(2.0 * np.pi * np.arange(length) / (length - 1))
  if not symmetric and not odd:
    window = window[:-1]
  return window


class WindowOpsTest(test.TestCase):

  def setUp(self):
    self._window_lengths = [1, 2, 3, 4, 5, 31, 64, 128]
    self._dtypes = [(dtypes.float16, 1e-2),
                    (dtypes.float32, 1e-6),
                    (dtypes.float64, 1e-9)]

  def _compare_window_fns(self, np_window_fn, tf_window_fn):
    with self.test_session(use_gpu=True):
      for window_length in self._window_lengths:
        for periodic in [False, True]:
          for tf_dtype, tol in self._dtypes:
            np_dtype = tf_dtype.as_numpy_dtype
            expected = np_window_fn(window_length,
                                    symmetric=not periodic).astype(np_dtype)
            actual = tf_window_fn(window_length, periodic=periodic,
                                  dtype=tf_dtype).eval()
            self.assertAllClose(expected, actual, tol, tol)

  def test_hann_window(self):
    """Check that hann_window matches scipy.signal.hann behavior."""
    # The Hann window is a raised cosine window with parameters alpha=0.5 and
    # beta=0.5.
    # https://en.wikipedia.org/wiki/Window_function#Hann_window
    self._compare_window_fns(
        functools.partial(_scipy_raised_cosine, a=0.5, b=0.5),
        window_ops.hann_window)

  def test_hamming_window(self):
    """Check that hamming_window matches scipy.signal.hamming's behavior."""
    # The Hamming window is a raised cosine window with parameters alpha=0.54
    # and beta=0.46.
    # https://en.wikipedia.org/wiki/Window_function#Hamming_window
    self._compare_window_fns(
        functools.partial(_scipy_raised_cosine, a=0.54, b=0.46),
        window_ops.hamming_window)


if __name__ == '__main__':
  test.main()
