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
"""Python dataset sparse tensor utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import sparse_ops


def any_sparse(classes):
  """Checks for sparse tensor.

  Args:
    classes: a structure of objects that identify the dataset item classes

  Returns:
    `True` if `classes` contains a sparse tensor type and `False` otherwise.
  """
  return any(c is sparse_tensor.SparseTensor for c in nest.flatten(classes))


def as_dense_shapes(shapes, classes):
  """Converts sparse tensor shapes to their physical shapes.

  Args:
    shapes: a structure of shapes to convert.
    classes: a structure of objects that identify the dataset item classes

  Returns:
    a structure matching the nested structure of `shapes`, containing
    `tensor_shape.unknown_shape()` at positions where `classes` contains
    `tf.sparse.SparseTensor` and matching contents of `shapes` otherwise
  """
  ret = nest.pack_sequence_as(shapes, [
      tensor_shape.unknown_shape() if c is sparse_tensor.SparseTensor else shape
      for shape, c in zip(nest.flatten(shapes), nest.flatten(classes))
  ])
  return ret


def as_dense_types(types, classes):
  """Converts sparse tensor types to `dtypes.variant`.

  Args:
    types: a structure of types to convert.
    classes: a structure of objects that identify the dataset item classes

  Returns:
    a structure matching the nested structure of `types`, containing
    `dtypes.variant` at positions where `classes` contains
    `tf.sparse.SparseTensor` and matching contents of `types` otherwise
  """
  ret = nest.pack_sequence_as(types, [
      dtypes.variant if c is sparse_tensor.SparseTensor else ty
      for ty, c in zip(nest.flatten(types), nest.flatten(classes))
  ])
  return ret


def deserialize_sparse_tensors(tensors, types, shapes, classes):
  """Deserializes sparse tensors.

  Args:
    tensors: a structure of tensors to deserialize.
    types: a structure that holds information about types of `tensors`
    shapes: a structure that holds information about shapes of `tensors`
    classes: a structure of objects that identify the dataset item classes

  Returns:
    `tensors` with any serialized sparse tensors replaced by their deserialized
    version.
  """
  ret = nest.pack_sequence_as(types, [
      sparse_ops.deserialize_sparse(tensor, dtype=ty, rank=shape.ndims)
      if c is sparse_tensor.SparseTensor else tensor
      for (tensor, ty, shape, c) in zip(
          nest.flatten(tensors), nest.flatten(types), nest.flatten(shapes),
          nest.flatten(classes))
  ])
  return ret


def get_classes(tensors):
  """Gets classes for a structure of tensors.

  Args:
    tensors: the tensor structure to get classes for.

  Returns:
    a structure matching the nested structure of `tensors`, containing
    `tf.sparse.SparseTensor` at positions where `tensors` contains a sparse
    tensor and `tf.Tensor` otherwise.
  """
  return nest.pack_sequence_as(tensors, [
      sparse_tensor.SparseTensor
      if isinstance(tensor, sparse_tensor.SparseTensor) else ops.Tensor
      for tensor in nest.flatten(tensors)
  ])


def serialize_many_sparse_tensors(tensors):
  """Serializes many sparse tensors into a batch.

  Args:
    tensors: a tensor structure to serialize.

  Returns:
    `tensors` with any sparse tensors replaced by the serialized batch.
  """

  ret = nest.pack_sequence_as(tensors, [
      sparse_ops.serialize_many_sparse(tensor, out_type=dtypes.variant)
      if sparse_tensor.is_sparse(tensor) else tensor
      for tensor in nest.flatten(tensors)
  ])
  return ret


def serialize_sparse_tensors(tensors):
  """Serializes sparse tensors.

  Args:
    tensors: a tensor structure to serialize.

  Returns:
    `tensors` with any sparse tensors replaced by their serialized version.
  """

  ret = nest.pack_sequence_as(tensors, [
      sparse_ops.serialize_sparse(tensor, out_type=dtypes.variant)
      if isinstance(tensor, sparse_tensor.SparseTensor) else tensor
      for tensor in nest.flatten(tensors)
  ])
  return ret
