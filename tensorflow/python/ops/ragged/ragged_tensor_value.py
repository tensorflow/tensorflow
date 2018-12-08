# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Value for RaggedTensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class RaggedTensorValue(object):
  """Represents the value of a `RaggedTensor`.

  See `RaggedTensor` for a description of ragged tensors.
  """

  def __init__(self, values, row_splits):
    """Creates a `RaggedTensorValue`.

    Args:
      values: A numpy array of any type and shape; or a RaggedTensorValue.
      row_splits: A 1-D int64 numpy array.
    """
    if not (isinstance(row_splits, (np.ndarray, np.generic)) and
            row_splits.dtype == np.int64 and row_splits.ndim == 1):
      raise TypeError("row_splits must be a 1D int64 numpy array")
    if not isinstance(values, (np.ndarray, np.generic, RaggedTensorValue)):
      raise TypeError("values must be a numpy array or a RaggedTensorValue")
    self._values = values
    self._row_splits = row_splits

  row_splits = property(
      lambda self: self._row_splits,
      doc="""The split indices for the ragged tensor value.""")
  values = property(
      lambda self: self._values,
      doc="""The concatenated values for all rows in this tensor.""")
  dtype = property(
      lambda self: self._values.dtype,
      doc="""The numpy dtype of values in this tensor.""")

  @property
  def flat_values(self):
    """The innermost `values` array for this ragged tensor value."""
    rt_values = self.values
    while isinstance(rt_values, RaggedTensorValue):
      rt_values = rt_values.values
    return rt_values

  @property
  def nested_row_splits(self):
    """The row_splits for all ragged dimensions in this ragged tensor value."""
    rt_nested_splits = [self.row_splits]
    rt_values = self.values
    while isinstance(rt_values, RaggedTensorValue):
      rt_nested_splits.append(rt_values.row_splits)
      rt_values = rt_values.values
    return tuple(rt_nested_splits)

  @property
  def ragged_rank(self):
    """The number of ragged dimensions in this ragged tensor value."""
    values_is_ragged = isinstance(self._values, RaggedTensorValue)
    return self._values.ragged_rank + 1 if values_is_ragged else 1

  @property
  def shape(self):
    """A tuple indicating the shape of this RaggedTensorValue."""
    return (self._row_splits.shape[0] - 1,) + (None,) + self._values.shape[1:]

  def __str__(self):
    return "<tf.RaggedTensorValue %s>" % self.to_list()

  def __repr__(self):
    return "tf.RaggedTensorValue(values=%r, row_splits=%r)" % (self._values,
                                                               self._row_splits)

  def to_list(self):
    """Returns this ragged tensor value as a nested Python list."""
    if isinstance(self._values, RaggedTensorValue):
      values_as_list = self._values.to_list()
    else:
      values_as_list = self._values.tolist()
    return [
        values_as_list[self._row_splits[i]:self._row_splits[i + 1]]
        for i in range(len(self._row_splits) - 1)
    ]

  def value_rowids(self, name=None):
    del name
    row_lengths = self._row_splits[1:] - self._row_splits[:-1]
    nrows = self._row_splits.shape[-1] - 1
    indices = np.arange(nrows)
    return np.repeat(indices, repeats=row_lengths, axis=0)
