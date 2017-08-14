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
"""Experimental API for TensorFlow's "Eager" mode of execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# TODO(agarwal): get rid of this import and change callers to use the classes in
# ops.py.
# pylint: disable=unused-import
from tensorflow.python.framework.ops import _tensor_from_handle
from tensorflow.python.framework.ops import convert_n_to_eager_tensor
from tensorflow.python.framework.ops import convert_to_eager_tensor
from tensorflow.python.framework.ops import EagerTensor as Tensor
# pylint: enable=unused-import


class IndexedSlices(object):
  """A sparse representation of a set of tensor slices at given indices.

  This class is a simple wrapper for a pair of `Tensor` objects:

  * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
  * `indices`: A 1-D integer `Tensor` with shape `[D0]`.

  An `IndexedSlices` is typically used to represent a subset of a larger
  tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
  The values in `indices` are the indices in the first dimension of
  the slices that have been extracted from the larger tensor.

  The dense tensor `dense` represented by an `IndexedSlices` `slices` has

  ```python
  dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
  ```

  The `IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients
  (e.g. @{tf.gather}).
  """

  def __init__(self, values, indices, dense_shape):
    """Creates an `IndexedSlices`."""
    self._values = values
    self._indices = indices
    assert indices.shape[0] == values.shape[0]
    self._dense_shape = dense_shape

  @property
  def values(self):
    """A `Tensor` containing the values of the slices."""
    return self._values

  @property
  def indices(self):
    """A 1-D `Tensor` containing the indices of the slices."""
    return self._indices

  @property
  def dense_shape(self):
    """A 1-D `Tensor` containing the shape of the corresponding dense tensor."""
    return self._dense_shape


class _Op(object):
  """Fake op for _LazyZero to make its python API tf.Tensor-like."""

  def __init__(self):
    self.type = "Zeros"


class LazyZero(object):
  """Lazily-instantiated zero-valued Tensor used as autograd accumulator."""

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype
    self.op = _Op()

  def __add__(self, other):
    return other

  def __radd__(self, other):
    return other

  def numpy(self):
    return np.zeros(self.shape, self.dtype)



