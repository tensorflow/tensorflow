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
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope


def histogram_fixed_width(values,
                          value_range,
                          nbins=100,
                          use_locking=True,
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
    nbins:  Integer number of bins in this histogram.
    use_locking:  Boolean.
      If `True`, use locking during the operation (optional).
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram_fixed_width').

  Returns:
    A `Variable` holding histogram of values.

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
  with variable_scope.variable_op_scope(
      [values, value_range], name, 'histogram_fixed_width') as scope:
    values = ops.convert_to_tensor(values, name='values')
    values = array_ops.reshape(values, [-1])
    value_range = ops.convert_to_tensor(value_range, name='value_range')

    # Map tensor values that fall within value_range to [0, 1].
    scaled_values = math_ops.truediv(values - value_range[0],
                                     value_range[1] - value_range[0],
                                     name='scaled_values')

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins * scaled_values, name='indices')

    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins - 1), dtypes.int32)

    # Dummy vector to scatter.
    # TODO(langmore) Replace non-ideal creation of large dummy vector once an
    # alternative to scatter is available.
    updates = array_ops.ones_like(indices, dtype=dtype)

    hist = variable_scope.get_variable('hist',
                                       initializer=array_ops.zeros_initializer(
                                           [nbins],
                                           dtype=dtype),
                                       trainable=False)
    hist_assign_zero = hist.assign(array_ops.zeros_like(hist))

    with ops.control_dependencies([hist_assign_zero]):
      return state_ops.scatter_add(hist,
                                   indices,
                                   updates,
                                   use_locking=use_locking,
                                   name=scope.name)
