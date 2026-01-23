# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""NumPy-compatible signal processing functions.

This module provides NumPy-compatible signal processing functions built on top
of TensorFlow operations.
"""
# pylint: disable=g-direct-tensorflow-import

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export


@tf_export.tf_export('experimental.numpy.convolve', v1=[])
@np_utils.np_doc('convolve')
def convolve(a, v, mode='full'):
  """Returns the discrete, linear convolution of two one-dimensional sequences.

  Args:
    a: First one-dimensional input array.
    v: Second one-dimensional input array.
    mode: str. One of:
      - 'full': Returns the full discrete linear convolution (default).
        Output size: N + M - 1
      - 'same': Returns output of length max(M, N).
      - 'valid': Returns output of length max(M, N) - min(M, N) + 1.

  Returns:
    Discrete, linear convolution of a and v.

  Raises:
    ValueError: If mode is not one of 'full', 'same', or 'valid'.
  """
  if mode not in ('full', 'same', 'valid'):
    raise ValueError(f"mode must be 'full', 'same', or 'valid', got '{mode}'")

  a = np_array_ops.asarray(a)
  v = np_array_ops.asarray(v)

  # Ensure 1-D
  a = np_array_ops.ravel(a)
  v = np_array_ops.ravel(v)

  # Promote dtypes
  result_dtype = np_utils.result_type(a, v)
  a = math_ops.cast(a, result_dtype)
  v = math_ops.cast(v, result_dtype)

  len_a = array_ops.shape(a)[0]
  len_v = array_ops.shape(v)[0]

  # For convolution, we flip the second array
  v_flipped = v[::-1]

  # Prepare for tf.nn.conv1d: [batch, in_width, in_channels]
  a_expanded = array_ops.reshape(a, [1, -1, 1])
  v_expanded = array_ops.reshape(v_flipped, [-1, 1, 1])

  if mode == 'full':
    # Pad input to get full convolution
    pad_size = len_v - 1
    a_padded = array_ops.pad(
        a_expanded,
        [[0, 0], [pad_size, pad_size], [0, 0]]
    )
    result = nn_ops.conv1d(
        a_padded,
        v_expanded,
        stride=1,
        padding='VALID'
    )
  elif mode == 'same':
    # Pad to get 'same' output length
    pad_left = (len_v - 1) // 2
    pad_right = len_v - 1 - pad_left
    a_padded = array_ops.pad(
        a_expanded,
        [[0, 0], [pad_left, pad_right], [0, 0]]
    )
    result = nn_ops.conv1d(
        a_padded,
        v_expanded,
        stride=1,
        padding='VALID'
    )
  else:  # 'valid'
    result = nn_ops.conv1d(
        a_expanded,
        v_expanded,
        stride=1,
        padding='VALID'
    )

  return array_ops.squeeze(result)


@tf_export.tf_export('experimental.numpy.correlate', v1=[])
@np_utils.np_doc('correlate')
def correlate(a, v, mode='valid'):
  """Cross-correlation of two 1-dimensional sequences.

  This function computes the correlation as generally understood in signal
  processing texts.

  Args:
    a: First one-dimensional input array.
    v: Second one-dimensional input array.
    mode: str. One of 'full', 'same', or 'valid'. Default is 'valid'.

  Returns:
    Discrete cross-correlation of a and v.
  """
  # Correlation is convolution with the second sequence conjugated and reversed
  v = np_array_ops.asarray(v)

  # Handle complex arrays by taking conjugate
  if v.dtype in (dtypes.complex64, dtypes.complex128):
    v = math_ops.conj(v)

  # Reverse v (correlate uses reversed v, then we flip again in convolve)
  # Since convolve already flips, we pass v as-is for correlation
  v_reversed = v[::-1]

  return convolve(a, v_reversed, mode=mode)


@tf_export.tf_export('experimental.numpy.searchsorted', v1=[])
@np_utils.np_doc('searchsorted')
def searchsorted(a, v, side='left', sorter=None):
  """Find indices where elements should be inserted to maintain order.

  Find the indices into a sorted array a such that, if the corresponding
  elements in v were inserted before the indices, the order of a would
  be preserved.

  Args:
    a: 1-D sorted array.
    v: Values to insert into a.
    side: If 'left', the index of the first suitable location found is
      given. If 'right', return the last such index.
    sorter: Not supported. Array of indices that sort a.

  Returns:
    Array of insertion points with the same shape as v.

  Raises:
    ValueError: If side is not 'left' or 'right'.
    NotImplementedError: If sorter is provided.
  """
  if sorter is not None:
    raise NotImplementedError('sorter parameter is not currently supported.')

  if side not in ('left', 'right'):
    raise ValueError(f"side must be 'left' or 'right', got '{side}'")

  a = np_array_ops.asarray(a)
  v = np_array_ops.asarray(v)

  return array_ops.searchsorted(a, v, side=side)


@tf_export.tf_export('experimental.numpy.interp', v1=[])
@np_utils.np_doc_only('interp')
def interp(x, xp, fp, left=None, right=None, period=None):
  """One-dimensional linear interpolation.

  Returns the one-dimensional piecewise linear interpolant to a function
  with given discrete data points (xp, fp), evaluated at x.

  Args:
    x: The x-coordinates at which to evaluate the interpolated values.
    xp: The x-coordinates of the data points, must be increasing.
    fp: The y-coordinates of the data points, same length as xp.
    left: Value to return for x < xp[0], default is fp[0].
    right: Value to return for x > xp[-1], default is fp[-1].
    period: Not supported. A period for the x-coordinates.

  Returns:
    The interpolated values, same shape as x.

  Raises:
    NotImplementedError: If period is provided.
  """
  if period is not None:
    raise NotImplementedError('period parameter is not currently supported.')

  x = np_array_ops.asarray(x)
  xp = np_array_ops.asarray(xp)
  fp = np_array_ops.asarray(fp)

  # Promote to common dtype
  result_dtype = np_utils.result_type(x, xp, fp)
  x = math_ops.cast(x, result_dtype)
  xp = math_ops.cast(xp, result_dtype)
  fp = math_ops.cast(fp, result_dtype)

  # Find indices where x would be inserted
  indices = array_ops.searchsorted(xp, x, side='right')

  # Clip indices to valid range [1, len(xp)-1] for interpolation
  n = array_ops.shape(xp)[0]
  indices = math_ops.maximum(indices, 1)
  indices = math_ops.minimum(indices, n - 1)

  # Get neighboring points
  x0 = array_ops.gather(xp, indices - 1)
  x1 = array_ops.gather(xp, indices)
  f0 = array_ops.gather(fp, indices - 1)
  f1 = array_ops.gather(fp, indices)

  # Linear interpolation formula: f0 + (x - x0) * (f1 - f0) / (x1 - x0)
  # Handle potential division by zero when x0 == x1
  dx = x1 - x0
  dx_safe = array_ops.where(math_ops.equal(dx, 0), math_ops.ones_like(dx), dx)
  t = (x - x0) / dx_safe
  result = f0 + t * (f1 - f0)

  # Handle out of bounds
  left_val = fp[0] if left is None else math_ops.cast(left, result_dtype)
  right_val = fp[-1] if right is None else math_ops.cast(right, result_dtype)

  result = array_ops.where(x < xp[0], left_val, result)
  result = array_ops.where(x > xp[-1], right_val, result)

  return result


@tf_export.tf_export('experimental.numpy.piecewise', v1=[])
@np_utils.np_doc_only('piecewise')
def piecewise(x, condlist, funclist):
  """Evaluate a piecewise-defined function.

  Given a set of conditions and corresponding functions, evaluate each
  function on the input data wherever its condition is true.

  Args:
    x: The input array.
    condlist: List of boolean arrays. The length must match funclist or
      be one element shorter.
    funclist: List of scalars or callables. If one element longer than
      condlist, the extra element is used as a default value.

  Returns:
    An array with the same shape as x, with values determined by
    the conditions.

  Raises:
    ValueError: If the lengths of condlist and funclist are incompatible.
  """
  x = np_array_ops.asarray(x)

  if len(funclist) == len(condlist) + 1:
    # Last element is default
    default = funclist[-1]
    funclist = funclist[:-1]
  elif len(funclist) == len(condlist):
    default = 0
  else:
    raise ValueError(
        f"funclist must be same length as condlist or one element longer. "
        f"Got {len(funclist)} funclist and {len(condlist)} condlist."
    )

  # Initialize result with default value
  if callable(default):
    result = default(x)
  else:
    result = np_array_ops.full_like(x, default)

  # Apply conditions in reverse order (later conditions can override)
  for cond, func in zip(reversed(condlist), reversed(funclist)):
    cond = np_array_ops.asarray(cond)
    cond = math_ops.cast(cond, dtypes.bool)

    if callable(func):
      values = func(x)
    else:
      values = np_array_ops.full_like(x, func)

    result = array_ops.where(cond, values, result)

  return result


@tf_export.tf_export('experimental.numpy.digitize', v1=[])
@np_utils.np_doc('digitize')
def digitize(x, bins, right=False):
  """Return the indices of the bins to which each value in input array belongs.

  Args:
    x: Input array to be binned.
    bins: Array of bins. It has to be 1-dimensional and monotonic.
    right: Indicating whether the intervals include the right or the left
      bin edge.

  Returns:
    Output array of indices, same shape as x.
  """
  x = np_array_ops.asarray(x)
  bins = np_array_ops.asarray(bins)

  # searchsorted with side='left' gives indices where x would be inserted
  # to maintain sorted order
  if right:
    indices = array_ops.searchsorted(bins, x, side='left')
  else:
    indices = array_ops.searchsorted(bins, x, side='right')

  return indices
