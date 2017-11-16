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
"""Python dataset sparse tensor utility functitons."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops


def any_sparse(types):
  """Checks for sparse tensor types.

  Args:
    types: a structure with tensor types.

  Returns:
    `True` if `types` contains a sparse tensor type and `False` otherwise.
  """
  return any([isinstance(ty, SparseType) for ty in nest.flatten(types)])


def deserialize_sparse_tensors(tensors, types):
  """Deserializes sparse tensors.

  Args:
    tensors: a structure of tensors to deserialize.
    types: a structure object the holds information about which tensors in
      `tensors` represent serialized sparse tensors

  Returns:
    `tensors` with any serialized sparse tensors replaced by their deserialized
    version.
  """
  # TODO(b/63669786): support batching of sparse tensors
  ret = nest.pack_sequence_as(types, [
      sparse_ops.deserialize_sparse(tensor, ty.dtype)
      if isinstance(ty, SparseType) else tensor
      for (tensor, ty) in zip(nest.flatten(tensors), nest.flatten(types))
  ])
  return ret


def get_sparse_types(tensors):
  """Gets sparse types for a structure of tensors.

  Args:
    tensors: the tensor structure to get sparse types for.

  Returns:
    a structure matching the nested structure of `tensors`, containing
    `SparseType` at positions where `tensors` contains a sparse tensor and
    `None` otherwise
  """
  return nest.pack_sequence_as(tensors, [
      SparseType(tensor.dtype)
      if isinstance(tensor, sparse_tensor.SparseTensor) else None
      for tensor in nest.flatten(tensors)
  ])


def serialize_sparse_tensors(tensors):
  """Serializes sparse tensors.

  Args:
    tensors: a tensor structure to serialize.

  Returns:
    `tensors` with any sparse tensors replaced by the their serialized version.
  """

  ret = nest.pack_sequence_as(tensors, [
      sparse_ops.serialize_sparse(tensor)
      if isinstance(tensor, sparse_tensor.SparseTensor) else tensor
      for tensor in nest.flatten(tensors)
  ])
  return ret


def unwrap_sparse_types(types):
  """Unwraps sparse tensor types as `dtypes.string`.

  Args:
    types: a structure of types to unwrap.

  Returns:
    a structure matching the nested structure of `types`, containing
    `dtypes.string` at positions where `types` contains a sparse tensor and
    matching contents of `types` otherwise
  """
  ret = nest.pack_sequence_as(types, [
      dtypes.string if isinstance(ty, SparseType) else ty
      for ty in nest.flatten(types)
  ])
  return ret


def wrap_sparse_types(tensors, types):
  """Wraps sparse tensor types in `SparseType`.

  Args:
    tensors: a structure of tensors for which to wrap types.
    types: a structure that holds information about which tensors in
      `tensors` represent serialized sparse tensors

  Returns:
    a structure matching the nested structure of `tensors`, containing
    `SparseType` at positions where `tensors` contains a sparse tensor and
    `DType` otherwise
  """
  ret = nest.pack_sequence_as(types, [
      tensor.dtype if ty is None else ty
      for tensor, ty in zip(nest.flatten(tensors), nest.flatten(types))
  ])
  return ret


class SparseType(object):
  """Wrapper class for representing types of sparse tensors in tf.data."""

  def __init__(self, dtype):
    """Creates a new instace of `SparseType`.

    Args:
      dtype: the sparse tensor type to wrap.
    """
    self._dtype = dtype

  def __repr__(self):
    return "SparseType({0!r})".format(self._dtype)

  def __eq__(self, other):
    """Returns `True` iff `self == other`."""
    if not isinstance(other, SparseType):
      return False
    return self._dtype == other.dtype

  def __ne__(self, other):
    """Returns `True` iff `self != other`."""
    return not self.__eq__(other)

  def __hash__(self):
    return self._dtype.__hash__()

  @property
  def dtype(self):
    """Returns the wrapped sparse tensor type."""
    return self._dtype
