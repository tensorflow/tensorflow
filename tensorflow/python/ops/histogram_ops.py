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
"""## Histograms

@@histogram_fixed_width
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops


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
    value_range:  Shape [2] `Tensor`.  new_values <= value_range[0] will be
      mapped to hist[0], values >= value_range[1] will be mapped to hist[-1].
      Must be same dtype as new_values.
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

  with tf.default_session() as sess:
    hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
    variables.initialize_all_variables().run()
    sess.run(hist) => [2, 1, 1, 0, 2]
  ```
  """
  with ops.op_scope([values, value_range, nbins], name,
                    'histogram_fixed_width') as scope:
    values = ops.convert_to_tensor(values, name='values')
    values = array_ops.reshape(values, [-1])
    value_range = ops.convert_to_tensor(value_range, name='value_range')
    nbins = ops.convert_to_tensor(nbins, dtype=dtypes.int32, name='nbins')
    nbins_float = math_ops.to_float(nbins)

    # Map tensor values that fall within value_range to [0, 1].
    scaled_values = math_ops.truediv(values - value_range[0],
                                     value_range[1] - value_range[0],
                                     name='scaled_values')

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins_float * scaled_values, name='indices')

    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)

    # TODO(langmore) This creates an array of ones to add up and place in the
    # bins.  This is inefficient, so replace when a better Op is available.
    return math_ops.unsorted_segment_sum(
        array_ops.ones_like(indices, dtype=dtype),
        indices,
        nbins,
        name=scope)
