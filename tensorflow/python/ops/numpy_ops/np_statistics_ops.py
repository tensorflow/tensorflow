# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""NumPy-compatible statistical functions.

This module provides NumPy-compatible statistical functions built on top of
TensorFlow operations.
"""
# pylint: disable=g-direct-tensorflow-import

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export


@tf_export.tf_export('experimental.numpy.histogram', v1=[])
@np_utils.np_doc('histogram')
def histogram(a, bins=10, range=None, density=None, weights=None):  # pylint: disable=redefined-builtin
  """Compute the histogram of a dataset.

  Args:
    a: Input data. The histogram is computed over the flattened array.
    bins: int. The number of equal-width bins in the given range.
    range: (float, float). The lower and upper range of the bins. If not
      provided, range is simply (a.min(), a.max()).
    density: bool. If True, the result is the value of the probability
      density function at the bin, normalized such that the integral over
      the range is 1.
    weights: Not supported. Included for NumPy API compatibility.

  Returns:
    hist: The values of the histogram.
    bin_edges: The bin edges (length(hist)+1).

  Raises:
    ValueError: If weights is provided (not supported).
  """
  if weights is not None:
    raise ValueError('weights parameter is not currently supported.')

  a = np_array_ops.asarray(a)
  a = np_array_ops.ravel(a)

  # Determine range
  if range is None:
    a_min = math_ops.reduce_min(a)
    a_max = math_ops.reduce_max(a)
    # Handle edge case where all values are the same
    range_width = a_max - a_min
    range_val = [a_min, a_max + math_ops.cast(
        math_ops.equal(range_width, 0), a.dtype)]
  else:
    range_val = [
        math_ops.cast(range[0], a.dtype),
        math_ops.cast(range[1], a.dtype)
    ]

  range_tensor = ops.convert_to_tensor(range_val, dtype=a.dtype)

  # Use TensorFlow's histogram implementation
  # pylint: disable=protected-access
  hist = gen_math_ops._histogram_fixed_width(
      a, range_tensor, bins, dtype=dtypes.int32)
  # pylint: enable=protected-access

  # Compute bin edges
  bin_edges = math_ops.linspace(range_tensor[0], range_tensor[1], bins + 1)

  if density:
    # Normalize to form a probability density
    hist_float = math_ops.cast(hist, a.dtype)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    total = math_ops.reduce_sum(hist_float)
    hist = hist_float / (total * bin_widths)

  return hist, bin_edges


@tf_export.tf_export('experimental.numpy.histogram_bin_edges', v1=[])
@np_utils.np_doc('histogram_bin_edges')
def histogram_bin_edges(a, bins=10, range=None, weights=None):  # pylint: disable=redefined-builtin
  """Compute the bin edges used by histogram.

  Args:
    a: Input data. The histogram is computed over the flattened array.
    bins: int. The number of equal-width bins in the given range.
    range: (float, float). The lower and upper range of the bins.
    weights: Not supported.

  Returns:
    bin_edges: Array of dtype float defining the bin edges.
  """
  if weights is not None:
    raise ValueError('weights parameter is not currently supported.')

  a = np_array_ops.asarray(a)
  a = np_array_ops.ravel(a)

  if range is None:
    a_min = math_ops.reduce_min(a)
    a_max = math_ops.reduce_max(a)
    range_width = a_max - a_min
    range_val = [a_min, a_max + math_ops.cast(
        math_ops.equal(range_width, 0), a.dtype)]
  else:
    range_val = [
        math_ops.cast(range[0], a.dtype),
        math_ops.cast(range[1], a.dtype)
    ]

  range_tensor = ops.convert_to_tensor(range_val, dtype=a.dtype)
  bin_edges = math_ops.linspace(range_tensor[0], range_tensor[1], bins + 1)

  return bin_edges


@tf_export.tf_export('experimental.numpy.quantile', v1=[])
@np_utils.np_doc_only('quantile')
def quantile(a, q, axis=None, interpolation='linear', keepdims=False):
  """Compute the q-th quantile of the data along the specified axis.

  Args:
    a: Input array or object that can be converted to an array.
    q: Quantile or sequence of quantiles to compute, in the range [0, 1].
    axis: Axis or axes along which the quantiles are computed. If None,
      the array is flattened before computation.
    interpolation: Specifies the interpolation method. Options: 'linear',
      'lower', 'higher', 'midpoint', 'nearest'.
    keepdims: If True, the axes which are reduced are left in the result
      as dimensions with size one.

  Returns:
    Quantile(s) of the array elements.

  Raises:
    ValueError: If interpolation method is not supported.
  """
  valid_interpolations = ('linear', 'lower', 'higher', 'midpoint', 'nearest')
  if interpolation not in valid_interpolations:
    raise ValueError(
        f"interpolation must be one of {valid_interpolations}, "
        f"got '{interpolation}'"
    )

  a = np_array_ops.asarray(a)
  q = np_array_ops.asarray(q)

  # Ensure q is in valid range
  q = math_ops.cast(q, a.dtype)

  # Store original shape info for keepdims
  original_shape = array_ops.shape(a)
  original_ndim = a.ndim

  # Flatten if no axis specified
  if axis is None:
    a = np_array_ops.ravel(a)
    axis = 0
  elif axis < 0:
    axis = axis + original_ndim

  # Sort along the specified axis
  sorted_a = sort_ops.sort(a, axis=axis)

  # Get the size along the axis
  n = math_ops.cast(array_ops.shape(sorted_a)[axis], a.dtype)

  # Compute virtual indices
  virtual_index = q * (n - 1)

  if interpolation == 'lower':
    indices = math_ops.cast(math_ops.floor(virtual_index), dtypes.int32)
    result = array_ops.gather(sorted_a, indices, axis=axis)
  elif interpolation == 'higher':
    indices = math_ops.cast(math_ops.ceil(virtual_index), dtypes.int32)
    result = array_ops.gather(sorted_a, indices, axis=axis)
  elif interpolation == 'nearest':
    indices = math_ops.cast(math_ops.round(virtual_index), dtypes.int32)
    result = array_ops.gather(sorted_a, indices, axis=axis)
  elif interpolation == 'midpoint':
    lower_idx = math_ops.cast(math_ops.floor(virtual_index), dtypes.int32)
    upper_idx = math_ops.cast(math_ops.ceil(virtual_index), dtypes.int32)
    lower_val = array_ops.gather(sorted_a, lower_idx, axis=axis)
    upper_val = array_ops.gather(sorted_a, upper_idx, axis=axis)
    result = (lower_val + upper_val) / 2.0
  else:  # linear
    lower_idx = math_ops.cast(math_ops.floor(virtual_index), dtypes.int32)
    upper_idx = math_ops.cast(math_ops.ceil(virtual_index), dtypes.int32)
    # Ensure indices are in valid range
    max_idx = math_ops.cast(n - 1, dtypes.int32)
    lower_idx = math_ops.minimum(lower_idx, max_idx)
    upper_idx = math_ops.minimum(upper_idx, max_idx)

    lower_val = array_ops.gather(sorted_a, lower_idx, axis=axis)
    upper_val = array_ops.gather(sorted_a, upper_idx, axis=axis)

    # Compute interpolation weight
    weight = virtual_index - math_ops.floor(virtual_index)
    result = lower_val + weight * (upper_val - lower_val)

  if keepdims:
    # Expand dims to restore original dimensionality
    if axis is not None:
      result = array_ops.expand_dims(result, axis=axis)

  return result


@tf_export.tf_export('experimental.numpy.percentile', v1=[])
@np_utils.np_doc_only('percentile')
def percentile(a, q, axis=None, interpolation='linear', keepdims=False):
  """Compute the q-th percentile of the data along the specified axis.

  This is equivalent to quantile(a, q/100, ...).

  Args:
    a: Input array or object that can be converted to an array.
    q: Percentile or sequence of percentiles to compute, in the range [0, 100].
    axis: Axis or axes along which the percentiles are computed.
    interpolation: Specifies the interpolation method.
    keepdims: If True, the axes which are reduced are left in the result.

  Returns:
    Percentile(s) of the array elements.
  """
  q = np_array_ops.asarray(q)
  q_normalized = q / 100.0
  return quantile(a, q_normalized, axis=axis, interpolation=interpolation,
                  keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.median', v1=[])
@np_utils.np_doc('median')
def median(a, axis=None, keepdims=False):
  """Compute the median along the specified axis.

  Returns the median of the array elements.

  Args:
    a: Input array or object that can be converted to an array.
    axis: Axis or axes along which the medians are computed. If None,
      the array is flattened before computation.
    keepdims: If True, the axes which are reduced are left in the result.

  Returns:
    Median of the array elements.
  """
  return quantile(a, 0.5, axis=axis, interpolation='linear', keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.nanmedian', v1=[])
@np_utils.np_doc_only('nanmedian')
def nanmedian(a, axis=None, keepdims=False):
  """Compute the median along the specified axis, ignoring NaNs.

  Args:
    a: Input array or object that can be converted to an array.
    axis: Axis or axes along which the medians are computed.
    keepdims: If True, the axes which are reduced are left in the result.

  Returns:
    Median of the array elements, ignoring NaNs.

  Note:
    This implementation uses masking to handle NaN values.
  """
  a = np_array_ops.asarray(a)

  # Create mask for non-NaN values
  mask = math_ops.logical_not(math_ops.is_nan(a))

  # Replace NaN with a large value for sorting (they will be at the end)
  max_val = a.dtype.max if hasattr(a.dtype, 'max') else np.finfo(
      a.dtype.as_numpy_dtype).max
  a_masked = array_ops.where(mask, a, max_val)

  # Use quantile on the masked array
  return quantile(a_masked, 0.5, axis=axis, interpolation='linear',
                  keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.nanpercentile', v1=[])
@np_utils.np_doc_only('nanpercentile')
def nanpercentile(a, q, axis=None, interpolation='linear', keepdims=False):
  """Compute the qth percentile of the data along the specified axis,
  ignoring NaN values.

  Args:
    a: Input array.
    q: Percentile(s) to compute, in the range [0, 100].
    axis: Axis along which the percentiles are computed.
    interpolation: Interpolation method.
    keepdims: If True, reduced axes are left as dimensions with size one.

  Returns:
    Percentile(s) of the array elements.
  """
  a = np_array_ops.asarray(a)
  q = np_array_ops.asarray(q)

  # Create mask for non-NaN values
  mask = math_ops.logical_not(math_ops.is_nan(a))

  # Replace NaN with a large value
  max_val = a.dtype.max if hasattr(a.dtype, 'max') else np.finfo(
      a.dtype.as_numpy_dtype).max
  a_masked = array_ops.where(mask, a, max_val)

  q_normalized = q / 100.0
  return quantile(a_masked, q_normalized, axis=axis,
                  interpolation=interpolation, keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.nanquantile', v1=[])
@np_utils.np_doc_only('nanquantile')
def nanquantile(a, q, axis=None, interpolation='linear', keepdims=False):
  """Compute the qth quantile of the data along the specified axis,
  ignoring NaN values.

  Args:
    a: Input array.
    q: Quantile(s) to compute, in the range [0, 1].
    axis: Axis along which the quantiles are computed.
    interpolation: Interpolation method.
    keepdims: If True, reduced axes are left as dimensions with size one.

  Returns:
    Quantile(s) of the array elements.
  """
  a = np_array_ops.asarray(a)

  # Create mask for non-NaN values
  mask = math_ops.logical_not(math_ops.is_nan(a))

  # Replace NaN with a large value
  max_val = a.dtype.max if hasattr(a.dtype, 'max') else np.finfo(
      a.dtype.as_numpy_dtype).max
  a_masked = array_ops.where(mask, a, max_val)

  return quantile(a_masked, q, axis=axis, interpolation=interpolation,
                  keepdims=keepdims)
