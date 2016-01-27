# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Operations for histograms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


def histogram_fixed_width(hist,
                          new_values,
                          value_range,
                          use_locking=False,
                          name='histogram_fixed_width'):
  """Update histogram Variable with new values.

  This Op fills histogram with counts of values falling within fixed-width,
  half-open bins.

  Args:
    hist:  1-D mutable `Tensor`, e.g. a `Variable`.
    new_values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor`.  new_values <= value_range[0] will be
      mapped to hist[0], values >= value_range[1] will be mapped to hist[-1].
      Must be same dtype as new_values.
    use_locking:  Boolean.
      If `True`, use locking during the operation (optional).
    name:  A name for this operation (optional).

  Returns:
    An op that updates `hist` with `new_values` when evaluated.

  Examples:
  ```python
  # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
  nbins = 5
  value_range = [0.0, 5.0]
  new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]

  with tf.default_session() as sess:
    hist = variables.Variable(array_ops.zeros(nbins, dtype=tf.int32))
    hist_update = histogram_ops.histogram_fixed_width(hist, new_values,
                                                      value_range)
    variables.initialize_all_variables().run()
    sess.run(hist_update) => [2, 1, 1, 0, 2]
  ```
  """
  with ops.op_scope([hist, new_values, value_range], name) as scope:
    new_values = ops.convert_to_tensor(new_values, name='new_values')
    new_values = array_ops.reshape(new_values, [-1])
    value_range = ops.convert_to_tensor(value_range, name='value_range')
    dtype = hist.dtype

    # Map tensor values that fall within value_range to [0, 1].
    scaled_values = math_ops.truediv(new_values - value_range[0],
                                     value_range[1] - value_range[0],
                                     name='scaled_values')
    nbins = math_ops.cast(hist.get_shape()[0], scaled_values.dtype)

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins * scaled_values, name='indices')

    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins - 1), dtypes.int32)

    # Dummy vector to scatter.
    # TODO(langmore) Replace non-ideal creation of large dummy vector once an
    # alternative to scatter is available.
    updates = array_ops.ones([indices.get_shape()[0]], dtype=dtype)
    return state_ops.scatter_add(hist,
                                 indices,
                                 updates,
                                 use_locking=use_locking,
                                 name=scope)
