# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Helpers for handling composite tensors and composite tensor values."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value


def is_composite_or_composite_value(tensor):
  """Returns true if 'tensor' is a CompositeTensor or a CT Value object."""
  # TODO(b/125094323): This should be isinstance(CompositeTensor) or
  # isinstance(CompositeTensorValue) once we support that.
  return isinstance(
      tensor,
      (composite_tensor.CompositeTensor, sparse_tensor.SparseTensorValue,
       ragged_tensor_value.RaggedTensorValue))


def _append_sparse_tensor_value(target, to_append):
  """Append sparse tensor value objects."""
  # Make sure the sparse tensors are of the same size (except for the 0th dim).
  if len(target.dense_shape) != len(to_append.dense_shape):
    raise RuntimeError(
        'Unable to concatenate %s and %s. The inner dense shapes do not '
        'have the same number of dimensions (%s vs %s)' %
        (target, to_append, target.dense_shape, to_append.dense_shape))

  if target.dense_shape[1:] != to_append.dense_shape[1:]:
    raise RuntimeError(
        'Unable to concatenate %s and %s. The inner dense shapes do not '
        'match inner dimensions (%s vs %s)' %
        (target, to_append, target.dense_shape[1:], to_append.dense_shape[1:]))

  # Add the to_append indices to target, updating the 0th value, and keeping
  # track of the maximum so we know the final dense_shape of this tensor.
  base_dim0_value = target.dense_shape[0]
  max_dim0_value = target.dense_shape[0]
  new_indices = target.indices
  for index in to_append.indices:
    # Here, we iterate through the sparse indices of the tensor to append. For
    # each index, we update its zeroth value (the batch index) by adding the
    # number of batch items in the tensor we are appending to (so an index
    # of [0, 0, 1] for a value that is being appended to a tensor with 0th dim
    # size 3 would become [3, 0, 1].)
    index[0] += base_dim0_value
    max_dim0_value = max(max_dim0_value, index[0])
    new_indices = np.append(new_indices, [index], axis=0)

  # Extend the values array to contain all of the appended values. These will
  # be in the same order as the indices added above.
  new_values = np.concatenate((target.values, to_append.values), axis=0)

  # Create a new dense shape by replacing the value for the 0th dimension
  # with the new max dim0 value.
  new_dense_shape = list(target.dense_shape)
  new_dense_shape[0] = max_dim0_value + 1
  new_dense_shape = tuple(new_dense_shape)

  return sparse_tensor.SparseTensorValue(
      indices=new_indices, values=new_values, dense_shape=new_dense_shape)


def _append_ragged_tensor_value(target, to_append):
  """Append ragged tensor value objects."""
  # Make sure the ragged tensors are of the same size (save for the 0th dim).
  if len(target.shape) != len(to_append.shape):
    raise RuntimeError('Unable to concatenate %s and %s' % (target, to_append))

  if target.shape[1:] != to_append.shape[1:]:
    raise RuntimeError('Unable to concatenate %s and %s' % (target, to_append))

  adjusted_row_splits = to_append.row_splits[1:] + target.row_splits[-1]
  new_row_splits = np.append(target.row_splits, adjusted_row_splits)
  if isinstance(target.values, ragged_tensor_value.RaggedTensorValue):
    new_values = _append_ragged_tensor_value(target.values, to_append.values)
  else:
    new_values = np.concatenate((target.values, to_append.values), axis=0)

  return ragged_tensor_value.RaggedTensorValue(new_values, new_row_splits)


def append_composite_tensor(target, to_append):
  """Helper function to append composite tensors to each other in the 0 axis.

  In order to support batching within a fit/evaluate/predict call, we need
  to be able to aggregate within a CompositeTensor. Unfortunately, the CT
  API currently does not make this easy - especially in V1 mode, where we're
  working with CompositeTensor Value objects that have no connection with the
  CompositeTensors that created them.

  Arguments:
    target: CompositeTensor or CompositeTensor value object that will be
      appended to.
    to_append: CompositeTensor or CompositeTensor value object to append to.
      'target'.

  Returns:
    A CompositeTensor or CompositeTensor value object.

  Raises:
    RuntimeError: if concatenation is not possible.
  """
  if type(target) is not type(to_append):
    raise RuntimeError('Unable to concatenate %s and %s' %
                       (type(target), type(to_append)))

  # Perform type-specific concatenation.
  # TODO(b/125094323): This should be replaced by a simple call to
  # target.append() that should work on all of the below classes.

  # If we're seeing a CompositeTensor here, we know it's because we're in
  # Eager mode (or else we'd have evaluated the CT to a CT Value object
  # already). Therefore, it's safe to call concat() on it without evaluating
  # the result any further. If not - that is, if we're seeing a
  # SparseTensorValue or a RaggedTensorValue - we need to hand-update it
  # since we're outside of the graph anyways.
  if isinstance(target, sparse_tensor.SparseTensor):
    # We need to invoke the sparse version of concatenate here - tf.concat
    # won't work.
    return sparse_ops.sparse_concat(sp_inputs=[target, to_append], axis=0)
  elif isinstance(target, ragged_tensor.RaggedTensor):
    return ragged_concat_ops.concat([target, to_append], axis=0)
  elif isinstance(target, sparse_tensor.SparseTensorValue):
    return _append_sparse_tensor_value(target, to_append)
  elif isinstance(target, ragged_tensor_value.RaggedTensorValue):
    return _append_ragged_tensor_value(target, to_append)
  else:
    raise RuntimeError('Attempted to concatenate unsupported object %s.' %
                       type(target))
