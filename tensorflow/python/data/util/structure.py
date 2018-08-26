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
"""Utilities for describing the structure of a `tf.data` type."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import sparse_ops


class Structure(object):
  """Represents structural information, such as type and shape, about a value.

  A `Structure` generalizes the `tf.Tensor.dtype` and `tf.Tensor.shape`
  properties, so that we can define generic containers of objects including:

  * `tf.Tensor`
  * `tf.SparseTensor`
  * Nested structures of the above.

  TODO(b/110122868): In the future, a single `Structure` will replace the
  `tf.data.Dataset.output_types`, `tf.data.Dataset.output_shapes`,
  and `tf.data.Dataset.output_classes`, and similar properties and arguments in
  the `tf.data.Iterator` and `Optional` classes.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def _flat_shapes(self):
    """A list of shapes matching the shapes of `self._to_tensor_list()`.

    Returns:
      A list of `tf.TensorShape` objects.
    """
    raise NotImplementedError("Structure._flat_shapes")

  @abc.abstractproperty
  def _flat_types(self):
    """A list of types matching the types of `self._to_tensor_list()`.

    Returns:
      A list of `tf.DType` objects.
    """
    raise NotImplementedError("Structure._flat_shapes")

  @abc.abstractmethod
  def is_compatible_with(self, value):
    """Returns `True` if `value` is compatible with this structure.

    A value `value` is compatible with a structure `s` if
    `Structure.from_value(value)` would return a structure `t` that is a
    "subtype" of `s`. A structure `t` is a "subtype" of `s` if:

    * `s` and `t` are instances of the same `Structure` subclass.
    * The nested structures (if any) of `s` and `t` are the same, according to
      `tf.contrib.framework.nest.assert_same_structure`, and each nested
      structure of `t` is a "subtype" of the corresponding nested structure of
      `s`.
    * Any `tf.DType` components of `t` are the same as the corresponding
      components in `s`.
    * Any `tf.TensorShape` components of `t` are compatible with the
      corresponding components in `s`, according to
      `tf.TensorShape.is_compatible_with`.

    Args:
      value: A potentially structured value.

    Returns:
      `True` if `value` matches this structure, otherwise `False`.
    """
    raise NotImplementedError("Structure.is_compatible_with()")

  @abc.abstractmethod
  def _to_tensor_list(self, value):
    """Returns a flat list of `tf.Tensor` representing `value`.

    This method can be used, along with `self._flat_shapes` and
    `self._flat_types` to represent structured values in lower level APIs
    (such as plain TensorFlow operations) that do not understand structure.

    Requires: `self.is_compatible_with(value)`.

    Args:
      value: A value with compatible structure.

    Returns:
      A flat list of `tf.Tensor` representing `value`.
    """
    raise NotImplementedError("Structure._to_tensor_list()")

  @abc.abstractmethod
  def _from_tensor_list(self, flat_value):
    """Builds a flat list of `tf.Tensor` into a value matching this structure.

    Requires: The shapes and types of the tensors in `flat_value` must be
    compatible with `self._flat_shapes` and `self._flat_types` respectively.

    Args:
      flat_value: A list of `tf.Tensor` with compatible flat structure.

    Returns:
      A structured object matching this structure.
    """
    raise NotImplementedError("Structure._from_tensor_list()")

  @staticmethod
  def from_value(value):
    """Returns a `Structure` that represents the given `value`.

    Args:
      value: A potentially structured value.

    Returns:
      A `Structure` that is compatible with `value`.

    Raises:
      TypeError: If a structure cannot be built for `value`, because its type
        or one of its component types is not supported.
    """

    # TODO(b/110122868): Add support for custom types, Dataset, and Optional
    # to this method.
    if isinstance(
        value,
        (sparse_tensor_lib.SparseTensor, sparse_tensor_lib.SparseTensorValue)):
      return SparseTensorStructure.from_value(value)
    elif isinstance(value, (tuple, dict)):
      return NestedStructure.from_value(value)
    else:
      try:
        tensor = ops.convert_to_tensor(value)
      except (ValueError, TypeError):
        raise TypeError("Could not build a structure for %r" % value)
      return TensorStructure.from_value(tensor)


# NOTE(mrry): The following classes make extensive use of non-public methods of
# their base class, so we disable the protected-access lint warning once here.
# pylint: disable=protected-access
class NestedStructure(Structure):
  """Represents a nested structure in which each leaf is a `Structure`."""

  def __init__(self, nested_structure):
    self._nested_structure = nested_structure
    self._flat_shapes_list = []
    self._flat_types_list = []
    for s in nest.flatten(nested_structure):
      if not isinstance(s, Structure):
        raise TypeError("nested_structure must be a (potentially nested) tuple "
                        "or dictionary of Structure objects.")
      self._flat_shapes_list.extend(s._flat_shapes)
      self._flat_types_list.extend(s._flat_types)

  @property
  def _flat_shapes(self):
    return self._flat_shapes_list

  @property
  def _flat_types(self):
    return self._flat_types_list

  def is_compatible_with(self, value):
    try:
      nest.assert_shallow_structure(self._nested_structure, value)
    except (ValueError, TypeError):
      return False

    return all(
        s.is_compatible_with(v) for s, v in zip(
            nest.flatten(self._nested_structure),
            nest.flatten_up_to(self._nested_structure, value)))

  def _to_tensor_list(self, value):
    ret = []

    try:
      flat_value = nest.flatten_up_to(self._nested_structure, value)
    except (ValueError, TypeError):
      raise ValueError("The value %r is not compatible with the nested "
                       "structure %r." % (value, self._nested_structure))

    for sub_value, structure in zip(flat_value,
                                    nest.flatten(self._nested_structure)):
      if not structure.is_compatible_with(sub_value):
        raise ValueError("Component value %r is not compatible with the nested "
                         "structure %r." % (sub_value, structure))
      ret.extend(structure._to_tensor_list(sub_value))
    return ret

  def _from_tensor_list(self, flat_value):
    if len(flat_value) != len(self._flat_types):
      raise ValueError("Expected %d flat values in NestedStructure but got %d."
                       % (len(self._flat_types), len(flat_value)))

    flat_ret = []
    for sub_value, structure in zip(flat_value,
                                    nest.flatten(self._nested_structure)):
      flat_ret.append(structure._from_tensor_list([sub_value]))

    return nest.pack_sequence_as(self._nested_structure, flat_ret)

  @staticmethod
  def from_value(value):
    flat_nested_structure = [
        Structure.from_value(sub_value) for sub_value in nest.flatten(value)
    ]
    return NestedStructure(nest.pack_sequence_as(value, flat_nested_structure))


class TensorStructure(Structure):
  """Represents structural information about a `tf.Tensor`."""

  def __init__(self, dtype, shape):
    self._dtype = dtypes.as_dtype(dtype)
    self._shape = tensor_shape.as_shape(shape)

  @property
  def _flat_shapes(self):
    return [self._shape]

  @property
  def _flat_types(self):
    return [self._dtype]

  def is_compatible_with(self, value):
    try:
      value = ops.convert_to_tensor(value, dtype=self._dtype)
    except (ValueError, TypeError):
      return False

    return (self._dtype.is_compatible_with(value.dtype) and
            self._shape.is_compatible_with(value.shape))

  def _to_tensor_list(self, value):
    if not self.is_compatible_with(value):
      raise ValueError("Value %r is not convertible to a tensor with dtype %s "
                       "and shape %s." % (value, self._dtype, self._shape))
    return [value]

  def _from_tensor_list(self, flat_value):
    if len(flat_value) != 1:
      raise ValueError("TensorStructure corresponds to a single tf.Tensor.")
    if not self.is_compatible_with(flat_value[0]):
      raise ValueError("Cannot convert %r to a tensor with dtype %s and shape "
                       "%s." % (flat_value[0], self._dtype, self._shape))
    return flat_value[0]

  @staticmethod
  def from_value(value):
    return TensorStructure(value.dtype, value.shape)


class SparseTensorStructure(Structure):
  """Represents structural information about a `tf.SparseTensor`."""

  def __init__(self, dtype, dense_shape):
    self._dtype = dtypes.as_dtype(dtype)
    self._dense_shape = tensor_shape.as_shape(dense_shape)

  @property
  def _flat_shapes(self):
    return [tensor_shape.vector(3)]

  @property
  def _flat_types(self):
    return [dtypes.variant]

  def is_compatible_with(self, value):
    try:
      value = sparse_tensor_lib.SparseTensor.from_value(value)
    except TypeError:
      return False
    return (isinstance(value, (sparse_tensor_lib.SparseTensor,
                               sparse_tensor_lib.SparseTensorValue)) and
            self._dtype.is_compatible_with(value.dtype) and
            self._dense_shape.is_compatible_with(
                tensor_util.constant_value_as_shape(value.dense_shape)))

  def _to_tensor_list(self, value):
    return [sparse_ops.serialize_sparse(value, out_type=dtypes.variant)]

  def _from_tensor_list(self, flat_value):
    if (len(flat_value) != 1 or flat_value[0].dtype != dtypes.variant or
        not flat_value[0].shape.is_compatible_with(tensor_shape.vector(3))):
      raise ValueError("SparseTensorStructure corresponds to a single "
                       "tf.variant vector of length 3.")
    return sparse_ops.deserialize_sparse(
        flat_value[0], dtype=self._dtype, rank=self._dense_shape.ndims)

  @staticmethod
  def from_value(value):
    sparse_tensor = sparse_tensor_lib.SparseTensor.from_value(value)
    return SparseTensorStructure(
        sparse_tensor.dtype,
        tensor_util.constant_value_as_shape(sparse_tensor.dense_shape))
