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
# pylint: disable=g-short-docstring-punctuation
"""Histograms.

Please see @{$python/histogram_ops} guide.

@@histogram_fixed_width_bins
@@histogram_fixed_width
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util.tf_export import tf_export


@tf_export('histogram_fixed_width_bins')
def histogram_fixed_width_bins(values,
                               value_range,
                               nbins=100,
                               dtype=dtypes.int32,
                               name=None):
  """Bins the given values for use in a histogram.

  Given the tensor `values`, this operation returns a rank 1 `Tensor`
  representing the indices of a histogram into which each element
  of `values` would be binned. The bins are equal width and
  determined by the arguments `value_range` and `nbins`.

  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram_fixed_width').

  Returns:
    A `Tensor` holding the indices of the binned values whose shape matches
    `values`.

  Examples:

  ```python
  # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
  nbins = 5
  value_range = [0.0, 5.0]
  new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]

  with tf.get_default_session() as sess:
    indices = tf.histogram_fixed_width_bins(new_values, value_range, nbins=5)
    variables.global_variables_initializer().run()
    sess.run(indices) => [0, 0, 1, 2, 4]
  ```
  """
  with ops.name_scope(name, 'histogram_fixed_width_bins',
                      [values, value_range, nbins]):
    values = ops.convert_to_tensor(values, name='values')
    shape = array_ops.shape(values)

    values = array_ops.reshape(values, [-1])
    value_range = ops.convert_to_tensor(value_range, name='value_range')
    nbins = ops.convert_to_tensor(nbins, dtype=dtypes.int32, name='nbins')
    nbins_float = math_ops.cast(nbins, values.dtype)

    # Map tensor values that fall within value_range to [0, 1].
    scaled_values = math_ops.truediv(
        values - value_range[0],
        value_range[1] - value_range[0],
        name='scaled_values')

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins_float * scaled_values, name='indices')

    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)
    return array_ops.reshape(indices, shape)


@tf_export('histogram_fixed_width')
def histogram_fixed_width(values,
                          value_range,
                          nbins=100,
                          dtype=dtypes.int32,
                          name=None):
  """Return histogram of values.

  Given the tensor `values`, this operation returns a rank 1 histogram counting
  the number of entries in `values` that fell into every bin.  The bins are
  equal width and determined by the arguments `value_range` and `nbins`.

  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram_fixed_width').

  Returns:
    A 1-D `Tensor` holding histogram of values.

  Examples:

  ```python
  # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
  nbins = 5
  value_range = [0.0, 5.0]
  new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]

  with tf.get_default_session() as sess:
    hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
    variables.global_variables_initializer().run()
    sess.run(hist) => [2, 1, 1, 0, 2]
  ```
  """
  with ops.name_scope(name, 'histogram_fixed_width',
                      [values, value_range, nbins]) as name:
    return gen_math_ops._histogram_fixed_width(  # pylint: disable=protected-access
        values, value_range, nbins, dtype=dtype, name=name)
