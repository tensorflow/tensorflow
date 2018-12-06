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
"""An Optional type for representing potentially missing values."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util.tf_export import tf_export


@six.add_metaclass(abc.ABCMeta)
class Optional(object):
  """Wraps a nested structure of tensors that may/may not be present at runtime.

  An `Optional` can represent the result of an operation that may fail as a
  value, rather than raising an exception and halting execution. For example,
  `tf.data.experimental.get_next_as_optional` returns an `Optional` that either
  contains the next value from a `tf.data.Iterator` if one exists, or a "none"
  value that indicates the end of the sequence has been reached.
  """

  @abc.abstractmethod
  def has_value(self, name=None):
    """Returns a tensor that evaluates to `True` if this optional has a value.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A scalar `tf.Tensor` of type `tf.bool`.
    """
    raise NotImplementedError("Optional.has_value()")

  @abc.abstractmethod
  def get_value(self, name=None):
    """Returns a nested structure of values wrapped by this optional.

    If this optional does not have a value (i.e. `self.has_value()` evaluates
    to `False`), this operation will raise `tf.errors.InvalidArgumentError`
    at runtime.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A nested structure of `tf.Tensor` and/or `tf.SparseTensor` objects.
    """
    raise NotImplementedError("Optional.get_value()")

  @abc.abstractproperty
  def value_structure(self):
    """The structure of the components of this optional.

    Returns:
      A `Structure` object representing the structure of the components of this
        optional.
    """
    raise NotImplementedError("Optional.value_structure")

  @staticmethod
  def from_value(value):
    """Returns an `Optional` that wraps the given value.

    Args:
      value: A nested structure of `tf.Tensor` and/or `tf.SparseTensor` objects.

    Returns:
      An `Optional` that wraps `value`.
    """
    with ops.name_scope("optional") as scope:
      with ops.name_scope("value"):
        value_structure = structure.Structure.from_value(value)
        encoded_value = value_structure._to_tensor_list(value)  # pylint: disable=protected-access

    return _OptionalImpl(
        gen_dataset_ops.optional_from_value(encoded_value, name=scope),
        value_structure)

  @staticmethod
  def none_from_structure(value_structure):
    """Returns an `Optional` that has no value.

    NOTE: This method takes an argument that defines the structure of the value
    that would be contained in the returned `Optional` if it had a value.

    Args:
      value_structure: A `Structure` object representing the structure of the
        components of this optional.

    Returns:
      An `Optional` that has no value.
    """
    return _OptionalImpl(gen_dataset_ops.optional_none(), value_structure)


class _OptionalImpl(Optional):
  """Concrete implementation of `tf.data.experimental.Optional`.

  NOTE(mrry): This implementation is kept private, to avoid defining
  `Optional.__init__()` in the public API.
  """

  def __init__(self, variant_tensor, value_structure):
    self._variant_tensor = variant_tensor
    self._value_structure = value_structure

  def has_value(self, name=None):
    return gen_dataset_ops.optional_has_value(self._variant_tensor, name=name)

  def get_value(self, name=None):
    # TODO(b/110122868): Consolidate the restructuring logic with similar logic
    # in `Iterator.get_next()` and `StructuredFunctionWrapper`.
    with ops.name_scope(name, "OptionalGetValue",
                        [self._variant_tensor]) as scope:
      # pylint: disable=protected-access
      return self._value_structure._from_tensor_list(
          gen_dataset_ops.optional_get_value(
              self._variant_tensor,
              name=scope,
              output_types=self._value_structure._flat_types,
              output_shapes=self._value_structure._flat_shapes))

  @property
  def value_structure(self):
    return self._value_structure


@tf_export("data.experimental.OptionalStructure")
class OptionalStructure(structure.Structure):
  """Represents an optional potentially containing a structured value."""

  def __init__(self, value_structure):
    self._value_structure = value_structure

  @property
  def _flat_shapes(self):
    return [tensor_shape.scalar()]

  @property
  def _flat_types(self):
    return [dtypes.variant]

  def is_compatible_with(self, other):
    # pylint: disable=protected-access
    return (isinstance(other, OptionalStructure) and
            self._value_structure.is_compatible_with(other._value_structure))

  def _to_tensor_list(self, value):
    return [value._variant_tensor]  # pylint: disable=protected-access

  def _from_tensor_list(self, flat_value):
    if (len(flat_value) != 1 or flat_value[0].dtype != dtypes.variant or
        not flat_value[0].shape.is_compatible_with(tensor_shape.scalar())):
      raise ValueError(
          "OptionalStructure corresponds to a single tf.variant scalar.")
    return self._from_compatible_tensor_list(flat_value)

  def _from_compatible_tensor_list(self, flat_value):
    # pylint: disable=protected-access
    return _OptionalImpl(flat_value[0], self._value_structure)

  @staticmethod
  def from_value(value):
    return OptionalStructure(value.value_structure)

  def _to_legacy_output_types(self):
    return self

  def _to_legacy_output_shapes(self):
    return self

  def _to_legacy_output_classes(self):
    return self

  def _batch(self, batch_size):
    raise NotImplementedError(
        "Batching for `tf.data.experimental.Optional` objects.")


# pylint: disable=protected-access
structure.Structure._register_custom_converter(Optional,
                                               OptionalStructure.from_value)
# pylint: enable=protected-access
