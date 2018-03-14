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
"""Batch Data Using a Rolling Window"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.array_ops import shape

def rolling_window(window_size=1, stride=1):
  """A transformation that batches the given dataset using a rolling window.

  For example:

  ```python
  # NOTE: The following example uses `{ ... }` to represent the
  # contents of a dataset.
  a = { 1, 2, 3, 4, 5, 6, 7 }

  a.apply(tf.contrib.data.rolling_window(window_size=3, stride=2))
    == { ( 1, 2, 3 ), ( 3, 4, 5 ), ( 5, 6, 7 ) }
  ```

  Args:
    window_size: A `tf.int64` scalar `tf.Tensor`, representing the size
      of the rolling window to be applied.
    stride: A `tf.int64` scalar `tf.Tensor`, representing the number of
      elements the window moves over each iteration.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """

  def get_slices(x):
    num_slices = shape(x, out_type=dtypes.int64)[0] - window_size + 1
    slices = dataset_ops.Dataset.range(0, num_slices, stride)
    return slices.map(lambda i: x[i:i + window_size])

  def _apply_fn(dataset):
    return dataset.flat_map(get_slices)

  return _apply_fn
