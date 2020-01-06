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
import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.kernel_tests.signal import test_util
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.platform import test


_TF_DTYPE_TOLERANCE = [(dtypes.float16, 1e-2),
                       (dtypes.float32, 1e-6),
                       (dtypes.float64, 1e-9)]
_WINDOW_LENGTHS = [1, 2, 3, 4, 5, 31, 64, 128]


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


@tf_test_util.run_all_in_graph_and_eager_modes
class WindowOpsTest(test.TestCase, parameterized.TestCase):

  def _compare_window_fns(self, np_window_fn, tf_window_fn, window_length,
                          periodic, tf_dtype_tol):
    tf_dtype, tol = tf_dtype_tol
    np_dtype = tf_dtype.as_numpy_dtype
    expected = np_window_fn(window_length,
                            symmetric=not periodic).astype(np_dtype)
    actual = tf_window_fn(window_length, periodic=periodic,
                          dtype=tf_dtype)
    self.assertAllClose(expected, actual, tol, tol)

  @parameterized.parameters(
      itertools.product(
          _WINDOW_LENGTHS,
          (False, True),
          _TF_DTYPE_TOLERANCE))
  def test_hann_window(self, window_length, periodic, tf_dtype_tol):
    """Check that hann_window matches scipy.signal.hann behavior."""
    # The Hann window is a raised cosine window with parameters alpha=0.5 and
    # beta=0.5.
    # https://en.wikipedia.org/wiki/Window_function#Hann_window
    self._compare_window_fns(
        functools.partial(_scipy_raised_cosine, a=0.5, b=0.5),
        window_ops.hann_window, window_length, periodic, tf_dtype_tol)

  @parameterized.parameters(
      itertools.product(
          _WINDOW_LENGTHS,
          (False, True),
          _TF_DTYPE_TOLERANCE))
  def test_hamming_window(self, window_length, periodic, tf_dtype_tol):
    """Check that hamming_window matches scipy.signal.hamming's behavior."""
    # The Hamming window is a raised cosine window with parameters alpha=0.54
    # and beta=0.46.
    # https://en.wikipedia.org/wiki/Window_function#Hamming_window
    self._compare_window_fns(
        functools.partial(_scipy_raised_cosine, a=0.54, b=0.46),
        window_ops.hamming_window, window_length, periodic, tf_dtype_tol)

  @parameterized.parameters(
      itertools.product(
          (window_ops.hann_window, window_ops.hamming_window),
          (False, True),
          _TF_DTYPE_TOLERANCE))
  def test_constant_folding(self, window_fn, periodic, tf_dtype_tol):
    """Window functions should be constant foldable for constant inputs."""
    if context.executing_eagerly():
      return
    g = ops.Graph()
    with g.as_default():
      window = window_fn(100, periodic=periodic, dtype=tf_dtype_tol[0])
      rewritten_graph = test_util.grappler_optimize(g, [window])
      self.assertLen(rewritten_graph.node, 1)

  @parameterized.parameters(
      # Due to control flow, only MLIR is supported.
      # Only float32 is supported.
      (window_ops.hann_window, 10, False, dtypes.float32, True),
      (window_ops.hann_window, 10, True, dtypes.float32, True),
      (window_ops.hamming_window, 10, False, dtypes.float32, True),
      (window_ops.hamming_window, 10, True, dtypes.float32, True))
  def test_tflite_convert(self, window_fn, window_length, periodic, dtype,
                          use_mlir):
    def fn(window_length):
      return window_fn(window_length, periodic, dtype=dtype)

    tflite_model = test_util.tflite_convert(
        fn, [tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32)], use_mlir)
    window_length = np.array(window_length).astype(np.int32)
    actual_output, = test_util.evaluate_tflite_model(
        tflite_model, [window_length])

    expected_output = self.evaluate(fn(window_length))
    self.assertAllClose(actual_output, expected_output, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  test.main()
