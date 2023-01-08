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
"""A type for representing values that may or may not exist."""
import abc

from tensorflow.python.data.util import structure
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export("experimental.Optional", "data.experimental.Optional")
@deprecation.deprecated_endpoints("data.experimental.Optional")
class Optional(composite_tensor.CompositeTensor, metaclass=abc.ABCMeta):
  """Represents a value that may or may not be present.

  A `tf.experimental.Optional` can represent the result of an operation that may
  fail as a value, rather than raising an exception and halting execution. For
  example, `tf.data.Iterator.get_next_as_optional()` returns a
  `tf.experimental.Optional` that either contains the next element of an
  iterator if one exists, or an "empty" value that indicates the end of the
  sequence has been reached.

  `tf.experimental.Optional` can only be used with values that are convertible
  to `tf.Tensor` or `tf.CompositeTensor`.

  One can create a `tf.experimental.Optional` from a value using the
  `from_value()` method:

  >>> optional = tf.experimental.Optional.from_value(42)
  >>> print(optional.has_value())
  tf.Tensor(True, shape=(), dtype=bool)
  >>> print(optional.get_value())
  tf.Tensor(42, shape=(), dtype=int32)

  or without a value using the `empty()` method:

  >>> optional = tf.experimental.Optional.empty(
  ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))
  >>> print(optional.has_value())
  tf.Tensor(False, shape=(), dtype=bool)
  """

  @abc.abstractmethod
  def has_value(self, name=None):
    """Returns a tensor that evaluates to `True` if this optional has a value.

    >>> optional = tf.experimental.Optional.from_value(42)
    >>> print(optional.has_value())
    tf.Tensor(True, shape=(), dtype=bool)

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A scalar `tf.Tensor` of type `tf.bool`.
    """
    raise NotImplementedError("Optional.has_value()")

  @abc.abstractmethod
  def get_value(self, name=None):
    """Returns the value wrapped by this optional.

    If this optional does not have a value (i.e. `self.has_value()` evaluates to
    `False`), this operation will raise `tf.errors.InvalidArgumentError` at
    runtime.

    >>> optional = tf.experimental.Optional.from_value(42)
    >>> print(optional.get_value())
    tf.Tensor(42, shape=(), dtype=int32)

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      The wrapped value.
    """
    raise NotImplementedError("Optional.get_value()")

  @abc.abstractproperty
  def element_spec(self):
    """The type specification of an element of this optional.

    >>> optional = tf.experimental.Optional.from_value(42)
    >>> print(optional.element_spec)
    tf.TensorSpec(shape=(), dtype=tf.int32, name=None)

    Returns:
      A (nested) structure of `tf.TypeSpec` objects matching the structure of an
      element of this optional, specifying the type of individual components.
    """
    raise NotImplementedError("Optional.element_spec")

  @staticmethod
  def empty(element_spec):
    """Returns an `Optional` that has no value.

    NOTE: This method takes an argument that defines the structure of the value
    that would be contained in the returned `Optional` if it had a value.

    >>> optional = tf.experimental.Optional.empty(
    ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))
    >>> print(optional.has_value())
    tf.Tensor(False, shape=(), dtype=bool)

    Args:
      element_spec: A (nested) structure of `tf.TypeSpec` objects matching the
        structure of an element of this optional.

    Returns:
      A `tf.experimental.Optional` with no value.
    """
    return _OptionalImpl(gen_dataset_ops.optional_none(), element_spec)

  @staticmethod
  def from_value(value):
    """Returns a `tf.experimental.Optional` that wraps the given value.

    >>> optional = tf.experimental.Optional.from_value(42)
    >>> print(optional.has_value())
    tf.Tensor(True, shape=(), dtype=bool)
    >>> print(optional.get_value())
    tf.Tensor(42, shape=(), dtype=int32)

    Args:
      value: A value to wrap. The value must be convertible to `tf.Tensor` or
        `tf.CompositeTensor`.

    Returns:
      A `tf.experimental.Optional` that wraps `value`.
    """
    with ops.name_scope("optional") as scope:
      with ops.name_scope("value"):
        element_spec = structure.type_spec_from_value(value)
        encoded_value = structure.to_tensor_list(element_spec, value)

    return _OptionalImpl(
        gen_dataset_ops.optional_from_value(encoded_value, name=scope),
        element_spec)


class _OptionalImpl(Optional):
  """Concrete implementation of `tf.experimental.Optional`.

  NOTE(mrry): This implementation is kept private, to avoid defining
  `Optional.__init__()` in the public API.
  """

  def __init__(self, variant_tensor, element_spec):
    super().__init__()
    self._variant_tensor = variant_tensor
    self._element_spec = element_spec

  def has_value(self, name=None):
    with ops.colocate_with(self._variant_tensor):
      return gen_dataset_ops.optional_has_value(self._variant_tensor, name=name)

  def get_value(self, name=None):
    # TODO(b/110122868): Consolidate the restructuring logic with similar logic
    # in `Iterator.get_next()` and `StructuredFunctionWrapper`.
    with ops.name_scope(name, "OptionalGetValue",
                        [self._variant_tensor]) as scope:
      with ops.colocate_with(self._variant_tensor):
        result = gen_dataset_ops.optional_get_value(
            self._variant_tensor,
            name=scope,
            output_types=structure.get_flat_tensor_types(self._element_spec),
            output_shapes=structure.get_flat_tensor_shapes(self._element_spec))
      # NOTE: We do not colocate the deserialization of composite tensors
      # because not all ops are guaranteed to have non-GPU kernels.
      return structure.from_tensor_list(self._element_spec, result)

  @property
  def element_spec(self):
    return self._element_spec

  @property
  def _type_spec(self):
    return OptionalSpec.from_value(self)


@tf_export(
    "OptionalSpec", v1=["OptionalSpec", "data.experimental.OptionalStructure"])
class OptionalSpec(type_spec.TypeSpec):
  """Type specification for `tf.experimental.Optional`.

  For instance, `tf.OptionalSpec` can be used to define a tf.function that takes
  `tf.experimental.Optional` as an input argument:

  >>> @tf.function(input_signature=[tf.OptionalSpec(
  ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))])
  ... def maybe_square(optional):
  ...   if optional.has_value():
  ...     x = optional.get_value()
  ...     return x * x
  ...   return -1
  >>> optional = tf.experimental.Optional.from_value(5)
  >>> print(maybe_square(optional))
  tf.Tensor(25, shape=(), dtype=int32)

  Attributes:
    element_spec: A (nested) structure of `TypeSpec` objects that represents the
      type specification of the optional element.
  """

  __slots__ = ["_element_spec"]

  def __init__(self, element_spec):
    super().__init__()
    self._element_spec = element_spec

  @property
  def value_type(self):
    return _OptionalImpl

  def _serialize(self):
    return (self._element_spec,)

  @property
  def _component_specs(self):
    return [tensor_spec.TensorSpec((), dtypes.variant)]

  def _to_components(self, value):
    return [value._variant_tensor]  # pylint: disable=protected-access

  def _from_components(self, flat_value):
    # pylint: disable=protected-access
    return _OptionalImpl(flat_value[0], self._element_spec)

  @staticmethod
  def from_value(value):
    return OptionalSpec(value.element_spec)

  def _to_legacy_output_types(self):
    return self

  def _to_legacy_output_shapes(self):
    return self

  def _to_legacy_output_classes(self):
    return self
