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
"""Type specifications for TensorFlow APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import re

import numpy as np
import six

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export

# Use LazyLoader to avoid circular dependencies.
tensor_spec = LazyLoader(
    "tensor_spec", globals(),
    "tensorflow.python.framework.tensor_spec")
ops = LazyLoader(
    "ops", globals(),
    "tensorflow.python.framework.ops")


@tf_export("TypeSpec", v1=["TypeSpec", "data.experimental.Structure"])
@six.add_metaclass(abc.ABCMeta)
class TypeSpec(object):
  """Specifies a TensorFlow value type.

  A `tf.TypeSpec` provides metadata describing an object accepted or returned
  by TensorFlow APIs.  Concrete subclasses, such as `tf.TensorSpec` and
  `tf.RaggedTensorSpec`, are used to describe different value types.

  For example, `tf.function`'s `input_signature` argument accepts a list
  (or nested structure) of `TypeSpec`s.

  Creating new subclasses of `TypeSpec` (outside of TensorFlow core) is not
  currently supported.  In particular, we may make breaking changes to the
  private methods and properties defined by this base class.

  Example:

  >>> spec = tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32)
  >>> @tf.function(input_signature=[spec])
  ... def double(x):
  ...   return x * 2
  >>> print(double(tf.ragged.constant([[1, 2], [3]])))
  <tf.RaggedTensor [[2, 4], [6]]>
  """
  # === Subclassing ===
  #
  # Each `TypeSpec` subclass must define:
  #
  #   * A "component encoding" for values.
  #   * A "serialization" for types.
  #
  # The component encoding for a value is a nested structure of `tf.Tensor`
  # or `CompositeTensor` that can be used by the `TypeSpec` to reconstruct
  # the value.  Each individual `TypeSpec` must use the same nested structure
  # for all values -- this structure is defined by the `component_specs`
  # attribute.  Decomposing values into components, and reconstructing them
  # from those components, should be inexpensive.  In particular, it should
  # *not* require any TensorFlow ops.
  #
  # The serialization for a `TypeSpec` is a nested tuple of values that can
  # be used to reconstruct the `TypeSpec`.  See the documentation for
  # `_serialize()` for more information.

  __slots__ = []

  @abc.abstractproperty
  def value_type(self):
    """The Python type for values that are compatible with this TypeSpec.

    In particular, all values that are compatible with this TypeSpec must be an
    instance of this type.
    """
    raise NotImplementedError("%s.value_type" % type(self).__name__)

  def is_compatible_with(self, spec_or_value):
    """Returns true if `spec_or_value` is compatible with this TypeSpec."""
    # === Subclassing ===
    # If not overridden by subclasses, the default behavior is to convert
    # `spec_or_value` to a `TypeSpec` (if it isn't already); and then to
    # consider two `TypeSpec`s compatible if they have the same type, and
    # the values returned by `_serialize` are compatible (where
    # `tf.TensorShape`, `tf.TensorSpec`, and `tf.DType` are checked for
    # compatibility using their `is_compatible_with` method; and all other
    # types are considered compatible if they are equal).
    if not isinstance(spec_or_value, TypeSpec):
      spec_or_value = type_spec_from_value(spec_or_value)
    if type(self) is not type(spec_or_value):
      return False
    return self.__is_compatible(self._serialize(),
                                spec_or_value._serialize())  # pylint: disable=protected-access

  def most_specific_compatible_type(self, other):
    """Returns the most specific TypeSpec compatible with `self` and `other`.

    Args:
      other: A `TypeSpec`.

    Raises:
      ValueError: If there is no TypeSpec that is compatible with both `self`
        and `other`.
    """
    # === Subclassing ===
    # If not overridden by a subclass, the default behavior is to raise a
    # `ValueError` if `self` and `other` have different types, or if their type
    # serializations differ by anything other than `TensorShape`s.  Otherwise,
    # the two type serializations are combined (using
    # `most_specific_compatible_shape` to combine `TensorShape`s), and the
    # result is used to construct and return a new `TypeSpec`.
    if type(self) is not type(other):
      raise ValueError("No TypeSpec is compatible with both %s and %s" %
                       (self, other))
    merged = self.__most_specific_compatible_type_serialization(
        self._serialize(), other._serialize())  # pylint: disable=protected-access
    return self._deserialize(merged)

  def _with_tensor_ranks_only(self):
    """Returns a TypeSpec compatible with `self`, with tensor shapes relaxed.

    Returns:
      A `TypeSpec` that is compatible with `self`, where any `TensorShape`
      information has been relaxed to include only tensor rank (and not
      the dimension sizes for individual axes).
    """

    # === Subclassing ===
    # If not overridden by a subclass, the default behavior is to serialize
    # this TypeSpec, relax any TensorSpec or TensorShape values, and
    # deserialize the result.

    def relax(value):
      if isinstance(value, TypeSpec):
        return value._with_tensor_ranks_only()  # pylint: disable=protected-access
      elif (isinstance(value, tensor_shape.TensorShape) and
            value.rank is not None):
        return tensor_shape.TensorShape([None] * value.rank)
      else:
        return value

    return self._deserialize(nest.map_structure(relax, self._serialize()))

  # === Component encoding for values ===

  @abc.abstractmethod
  def _to_components(self, value):
    """Encodes `value` as a nested structure of `Tensor` or `CompositeTensor`.

    Args:
      value: A value compatible with this `TypeSpec`.  (Caller is responsible
        for ensuring compatibility.)

    Returns:
      A nested structure of `tf.Tensor` or `tf.CompositeTensor` compatible with
      `self._component_specs`, which can be used to reconstruct `value`.
    """
    # === Subclassing ===
    # This method must be inexpensive (do not call TF ops).
    raise NotImplementedError("%s._to_components()" % type(self).__name__)

  @abc.abstractmethod
  def _from_components(self, components):
    """Reconstructs a value from a nested structure of Tensor/CompositeTensor.

    Args:
      components: A nested structure of `tf.Tensor` or `tf.CompositeTensor`,
        compatible with `self._component_specs`.  (Caller is responsible for
        ensuring compatibility.)

    Returns:
      A value that is compatible with this `TypeSpec`.
    """
    # === Subclassing ===
    # This method must be inexpensive (do not call TF ops).
    raise NotImplementedError("%s._from_components()" % type(self).__name__)

  @abc.abstractproperty
  def _component_specs(self):
    """A nested structure of TypeSpecs for this type's components.

    Returns:
      A nested structure describing the component encodings that are returned
      by this TypeSpec's `_to_components` method.  In particular, for a
      TypeSpec `spec` and a compatible value `value`:

      ```
      nest.map_structure(lambda t, c: assert t.is_compatible_with(c),
                         spec._component_specs, spec._to_components(value))
      ```
    """
    raise NotImplementedError("%s._component_specs()" % type(self).__name__)

  # === Tensor list encoding for values ===

  def _to_tensor_list(self, value):
    """Encodes `value` as a flat list of `tf.Tensor`.

    By default, this just flattens `self._to_components(value)` using
    `nest.flatten`.  However, subclasses may override this to return a
    different tensor encoding for values.  In particular, some subclasses
    of `BatchableTypeSpec` override this method to return a "boxed" encoding
    for values, which then can be batched or unbatched.  See
    `BatchableTypeSpec` for more details.

    Args:
      value: A value with compatible this `TypeSpec`.  (Caller is responsible
        for ensuring compatibility.)

    Returns:
      A list of `tf.Tensor`, compatible with `self._flat_tensor_specs`, which
      can be used to reconstruct `value`.
    """
    return nest.flatten(self._to_components(value), expand_composites=True)

  def _from_tensor_list(self, tensor_list):
    """Reconstructs a value from a flat list of `tf.Tensor`.

    Args:
      tensor_list: A flat list of `tf.Tensor`, compatible with
        `self._flat_tensor_specs`.

    Returns:
      A value that is compatible with this `TypeSpec`.

    Raises:
      ValueError: If `tensor_list` is not compatible with
      `self._flat_tensor_specs`.
    """
    self.__check_tensor_list(tensor_list)
    return self._from_compatible_tensor_list(tensor_list)

  def _from_compatible_tensor_list(self, tensor_list):
    """Reconstructs a value from a compatible flat list of `tf.Tensor`.

    Args:
      tensor_list: A flat list of `tf.Tensor`, compatible with
        `self._flat_tensor_specs`.  (Caller is responsible for ensuring
        compatibility.)

    Returns:
      A value that is compatible with this `TypeSpec`.
    """
    return self._from_components(nest.pack_sequence_as(
        self._component_specs, tensor_list, expand_composites=True))

  @property
  def _flat_tensor_specs(self):
    """A list of TensorSpecs compatible with self._to_tensor_list(v)."""
    return nest.flatten(self._component_specs, expand_composites=True)

  # === Serialization for types ===

  @abc.abstractmethod
  def _serialize(self):
    """Returns a nested tuple containing the state of this TypeSpec.

    The serialization may contain the following value types: boolean,
    integer, string, float, None, `TensorSpec`, `tf.TensorShape`, `tf.DType`,
    `np.ndarray`, `TypeSpec`, and nested tuples, namedtuples, dicts, and
    OrderedDicts of any of the above.

    This method is used to provide default definitions for: equality
    testing (__eq__, __ne__), hashing (__hash__), pickling (__reduce__),
    string representation (__repr__), `self.is_compatible_with()`,
    `self.most_specific_compatible_type()`, and protobuf serialization
    (e.g. TensorInfo and StructuredValue).
    """
    raise NotImplementedError("%s._serialize()" % type(self).__name__)

  @classmethod
  def _deserialize(cls, serialization):
    """Reconstructs a TypeSpec from a value returned by `serialize`.

    Args:
      serialization: A value returned by _serialize.  In some contexts,
        `namedtuple`s in `serialization` may not have the identical type
        that was returned by `_serialize` (but its type will still be a
        `namedtuple` type with the same type name and field names).  For
        example, the code that loads a SavedModel does not have access to
        the original `namedtuple` type, so it dynamically creates a new
        `namedtuple` type with the same type name and field names as the
        original one.  If necessary, you can check `serialization` for these
        duck-typed `nametuple` types, and restore them to the original type.
        (E.g., this would be necessary if you rely on type checks such as
        `isinstance` for this `TypeSpec`'s member variables).

    Returns:
      A `TypeSpec` of type `cls`.
    """
    return cls(*serialization)

  # === Operators ===

  def __eq__(self, other):
    # pylint: disable=protected-access
    return (type(other) is type(self) and
            self.__get_cmp_key() == other.__get_cmp_key())

  def __ne__(self, other):
    return not self == other

  def __hash__(self):
    return hash(self.__get_cmp_key())

  def __reduce__(self):
    return type(self), self._serialize()

  def __repr__(self):
    return "%s%r" % (type(self).__name__, self._serialize())

  # === Legacy Output ===
  # TODO(b/133606651) Document and/or deprecate the legacy_output methods.
  # (These are used by tf.data.)

  def _to_legacy_output_types(self):
    raise NotImplementedError("%s._to_legacy_output_types()" %
                              type(self).__name__)

  def _to_legacy_output_shapes(self):
    raise NotImplementedError("%s._to_legacy_output_shapes()" %
                              type(self).__name__)

  def _to_legacy_output_classes(self):
    return self.value_type

  # === Private Helper Methods ===

  def __check_tensor_list(self, tensor_list):
    expected = self._flat_tensor_specs
    specs = [type_spec_from_value(t) for t in tensor_list]
    if len(specs) != len(expected):
      raise ValueError("Incompatible input: wrong number of tensors")
    for i, (s1, s2) in enumerate(zip(specs, expected)):
      if not s1.is_compatible_with(s2):
        raise ValueError("Incompatible input: tensor %d (%s) is incompatible "
                         "with %s" % (i, tensor_list[i], s2))

  def __get_cmp_key(self):
    """Returns a hashable eq-comparable key for `self`."""
    # TODO(b/133606651): Decide whether to cache this value.
    return (type(self), self.__make_cmp_key(self._serialize()))

  def __make_cmp_key(self, value):
    """Converts `value` to a hashable key."""
    if isinstance(value,
                  (int, float, bool, np.generic, dtypes.DType, TypeSpec)):
      return value
    if isinstance(value, compat.bytes_or_text_types):
      return value
    if value is None:
      return value
    if isinstance(value, dict):
      return tuple([
          tuple([self.__make_cmp_key(key),
                 self.__make_cmp_key(value[key])])
          for key in sorted(value.keys())
      ])
    if isinstance(value, tuple):
      return tuple([self.__make_cmp_key(v) for v in value])
    if isinstance(value, list):
      return (list, tuple([self.__make_cmp_key(v) for v in value]))
    if isinstance(value, tensor_shape.TensorShape):
      if value.ndims is None:
        # Note: we include a type object in the tuple, to ensure we can't get
        # false-positive matches (since users can't include type objects).
        return (tensor_shape.TensorShape, None)
      return (tensor_shape.TensorShape, tuple(value.as_list()))
    if isinstance(value, np.ndarray):
      return (np.ndarray, value.shape,
              TypeSpec.__nested_list_to_tuple(value.tolist()))
    raise ValueError("Unsupported value type %s returned by "
                     "%s._serialize" %
                     (type(value).__name__, type(self).__name__))

  @staticmethod
  def __nested_list_to_tuple(value):
    """Converts a nested list to a corresponding nested tuple."""
    if isinstance(value, list):
      return tuple(TypeSpec.__nested_list_to_tuple(v) for v in value)
    return value

  @staticmethod
  def __same_types(a, b):
    """Returns whether a and b have the same type, up to namedtuple equivalence.

    Consistent with tf.nest.assert_same_structure(), two namedtuple types
    are considered the same iff they agree in their class name (without
    qualification by module name) and in their sequence of field names.
    This makes namedtuples recreated by StructureCoder compatible with their
    original Python definition.

    Args:
      a: a Python object.
      b: a Python object.

    Returns:
      A boolean that is true iff type(a) and type(b) are the same object
      or equivalent namedtuple types.
    """
    if nest.is_namedtuple(a) and nest.is_namedtuple(b):
      return nest.same_namedtuples(a, b)
    else:
      return type(a) is type(b)

  @staticmethod
  def __is_compatible(a, b):
    """Returns true if the given type serializations compatible."""
    if isinstance(a, TypeSpec):
      return a.is_compatible_with(b)
    if not TypeSpec.__same_types(a, b):
      return False
    if isinstance(a, (list, tuple)):
      return (len(a) == len(b) and
              all(TypeSpec.__is_compatible(x, y) for (x, y) in zip(a, b)))
    if isinstance(a, dict):
      return (len(a) == len(b) and sorted(a.keys()) == sorted(b.keys()) and all(
          TypeSpec.__is_compatible(a[k], b[k]) for k in a.keys()))
    if isinstance(a, (tensor_shape.TensorShape, dtypes.DType)):
      return a.is_compatible_with(b)
    return a == b

  @staticmethod
  def __most_specific_compatible_type_serialization(a, b):
    """Helper for most_specific_compatible_type.

    Combines two type serializations as follows:

    * If they are both tuples of the same length, then recursively combine
      the respective tuple elements.
    * If they are both dicts with the same keys, then recursively combine
      the respective dict elements.
    * If they are both TypeSpecs, then combine using
      TypeSpec.most_specific_compatible_type.
    * If they are both TensorShapes, then combine using
      TensorShape.most_specific_compatible_shape.
    * If they are both TensorSpecs with the same dtype, then combine using
      TensorShape.most_specific_compatible_shape to combine shapes.
    * If they are equal, then return a.
    * If none of the above, then raise a ValueError.

    Args:
      a: A serialized TypeSpec or nested component from a serialized TypeSpec.
      b: A serialized TypeSpec or nested component from a serialized TypeSpec.

    Returns:
      A value with the same type and structure as `a` and `b`.

    Raises:
      ValueError: If `a` and `b` are incompatible.
    """
    if not TypeSpec.__same_types(a, b):
      raise ValueError("Types are not compatible: %r vs %r" % (a, b))
    if nest.is_namedtuple(a):
      assert a._fields == b._fields  # Implied by __same_types(a, b).
      return type(a)(*[
          TypeSpec.__most_specific_compatible_type_serialization(x, y)
          for (x, y) in zip(a, b)])
    if isinstance(a, (list, tuple)):
      if len(a) != len(b):
        raise ValueError("Types are not compatible: %r vs %r" % (a, b))
      return tuple(TypeSpec.__most_specific_compatible_type_serialization(x, y)
                   for (x, y) in zip(a, b))
    if isinstance(a, collections.OrderedDict):
      a_keys, b_keys = a.keys(), b.keys()
      if len(a) != len(b) or a_keys != b_keys:
        raise ValueError("Types are not compatible: %r vs %r" % (a, b))
      return collections.OrderedDict([
          (k,
           TypeSpec.__most_specific_compatible_type_serialization(a[k], b[k]))
          for k in a_keys
      ])
    if isinstance(a, dict):
      a_keys, b_keys = sorted(a.keys()), sorted(b.keys())
      if len(a) != len(b) or a_keys != b_keys:
        raise ValueError("Types are not compatible: %r vs %r" % (a, b))
      return {
          k: TypeSpec.__most_specific_compatible_type_serialization(a[k], b[k])
          for k in a_keys
      }
    if isinstance(a, tensor_shape.TensorShape):
      return a.most_specific_compatible_shape(b)
    if isinstance(a, list):
      raise AssertionError("_serialize() should not return list values.")
    if isinstance(a, TypeSpec):
      return a.most_specific_compatible_type(b)
    if a != b:
      raise ValueError("Types are not compatible: %r vs %r" % (a, b))
    return a


class BatchableTypeSpec(TypeSpec):
  """TypeSpec with a batchable tensor encoding.

  The batchable tensor encoding is a list of `tf.Tensor`s that supports
  batching and unbatching.  In particular, stacking (or unstacking)
  values with the same `TypeSpec` must be equivalent to stacking (or
  unstacking) each of their tensor lists.  Unlike the component encoding
  (returned by `self._to_components)`, the batchable tensor encoding
  may require using encoding/decoding ops.

  If a subclass's batchable tensor encoding is not simply a flattened version
  of the component encoding, then the subclass must override `_to_tensor_list`,
  `_from_tensor_list`, and _flat_tensor_specs`.
  """

  __slots__ = []

  @abc.abstractmethod
  def _batch(self, batch_size):
    """Returns a TypeSpec representing a batch of objects with this TypeSpec.

    Args:
      batch_size: An `int` representing the number of elements in a batch,
        or `None` if the batch size may vary.

    Returns:
      A `TypeSpec` representing a batch of objects with this TypeSpec.
    """
    raise NotImplementedError("%s._batch" % type(self).__name__)

  @abc.abstractmethod
  def _unbatch(self):
    """Returns a TypeSpec representing a single element this TypeSpec.

    Returns:
      A `TypeSpec` representing a single element of objects with this TypeSpec.
    """
    raise NotImplementedError("%s._unbatch" % type(self).__name__)

  def _to_batched_tensor_list(self, value):
    """Returns a tensor list encoding for value with rank>0."""
    tensor_list = self._to_tensor_list(value)
    if any(t.shape.ndims == 0 for t in tensor_list):
      raise ValueError("Value %s has insufficient rank for batching." % value)
    return tensor_list


@tf_export("type_spec_from_value")
def type_spec_from_value(value):
  """Returns a `tf.TypeSpec` that represents the given `value`.

  Examples:

    >>> tf.type_spec_from_value(tf.constant([1, 2, 3]))
    TensorSpec(shape=(3,), dtype=tf.int32, name=None)
    >>> tf.type_spec_from_value(np.array([4.0, 5.0], np.float64))
    TensorSpec(shape=(2,), dtype=tf.float64, name=None)
    >>> tf.type_spec_from_value(tf.ragged.constant([[1, 2], [3, 4, 5]]))
    RaggedTensorSpec(TensorShape([2, None]), tf.int32, 1, tf.int64)

    >>> example_input = tf.ragged.constant([[1, 2], [3]])
    >>> @tf.function(input_signature=[tf.type_spec_from_value(example_input)])
    ... def f(x):
    ...   return tf.reduce_sum(x, axis=1)

  Args:
    value: A value that can be accepted or returned by TensorFlow APIs.
      Accepted types for `value` include `tf.Tensor`, any value that can be
      converted to `tf.Tensor` using `tf.convert_to_tensor`, and any subclass
      of `CompositeTensor` (such as `tf.RaggedTensor`).

  Returns:
    A `TypeSpec` that is compatible with `value`.

  Raises:
    TypeError: If a TypeSpec cannot be built for `value`, because its type
      is not supported.
  """
  spec = _type_spec_from_value(value)
  if spec is not None:
    return spec

  # Fallback: try converting value to a tensor.
  try:
    tensor = ops.convert_to_tensor(value)
    spec = _type_spec_from_value(tensor)
    if spec is not None:
      return spec
  except (ValueError, TypeError) as e:
    logging.vlog(
        3, "Failed to convert %r to tensor: %s" % (type(value).__name__, e))

  raise TypeError("Could not build a TypeSpec for %r with type %s" %
                  (value, type(value).__name__))


def _type_spec_from_value(value):
  """Returns a `TypeSpec` that represents the given `value`."""
  if isinstance(value, ops.Tensor):
    # Note: we do not include Tensor names when constructing TypeSpecs.
    return tensor_spec.TensorSpec(value.shape, value.dtype)

  if isinstance(value, composite_tensor.CompositeTensor):
    return value._type_spec  # pylint: disable=protected-access

  # If `value` is a list and all of its elements can be represented by the same
  # batchable type spec, then we can represent the entire list using a single
  # type spec that captures the type accurately (unlike the `convert_to_tensor`
  # fallback).
  if isinstance(value, list) and value:
    subspecs = [_type_spec_from_value(v) for v in value]
    if isinstance(subspecs[0], BatchableTypeSpec):
      merged_subspec = subspecs[0]
      try:
        for subspec in subspecs[1:]:
          merged_subspec = merged_subspec.most_specific_compatible_type(subspec)
        return merged_subspec._batch(len(subspecs))  # pylint: disable=protected-access
      except (ValueError, TypeError):
        pass  # incompatible subspecs

  for entry in reversed(_TYPE_CONVERSION_FUNCTION_REGISTRY):
    type_object, converter_fn, allow_subclass = entry
    if ((type(value) is type_object) or  # pylint: disable=unidiomatic-typecheck
        (allow_subclass and isinstance(value, type_object))):
      return converter_fn(value)

  return None

_TYPE_CONVERSION_FUNCTION_REGISTRY = []


def register_type_spec_from_value_converter(type_object, converter_fn,
                                            allow_subclass=False):
  """Registers a function for converting values with a given type to TypeSpecs.

  If multiple registered `type_object`s match a value, then the most recent
  registration takes precedence.  Custom converters should not be defined for
  `CompositeTensor`s; use `CompositeTensor._type_spec` instead.

  Args:
    type_object: A Python `type` object representing the type of values
      accepted by `converter_fn`.
    converter_fn: A function that takes one argument (an instance of the
      type represented by `type_object`) and returns a `TypeSpec`.
    allow_subclass: If true, then use `isinstance(value, type_object)` to
      check for matches.  If false, then use `type(value) is type_object`.
  """
  _, type_object = tf_decorator.unwrap(type_object)
  _TYPE_CONVERSION_FUNCTION_REGISTRY.append(
      (type_object, converter_fn, allow_subclass))


_pywrap_utils.RegisterType("TypeSpec", TypeSpec)


_TYPE_SPEC_TO_NAME = {}
_NAME_TO_TYPE_SPEC = {}


# Regular expression for valid TypeSpec names.
_REGISTERED_NAME_RE = re.compile(r"^(\w+\.)+\w+$")


# TODO(b/173744905) tf_export this as "tf.register_type_spec".  (And add a
# usage example to the docstring, once the API is public.)
#
# TODO(b/173744905) Update this decorator to apply to ExtensionType rather than
# TypeSpec (once we do refactoring to move to_components/from_components from
# TypeSpec to ExtensionType).
def register(name):
  """Decorator used to register a globally unique name for a TypeSpec subclass.

  Args:
    name: The name of the type spec.  Must be globally unique.  Must have
      the form `"{project_name}.{type_name}"`.  E.g. `"my_project.MyTypeSpec"`.

  Returns:
    A class decorator that registers the decorated class with the given name.
  """
  if not isinstance(name, str):
    raise TypeError("Expected `name` to be a string; got %r" % (name,))
  if not _REGISTERED_NAME_RE.match(name):
    raise ValueError(
        "Registered name must have the form '{project_name}.{type_name}' "
        "(e.g. 'my_project.MyTypeSpec'); got %r." % name)

  def decorator_fn(cls):
    if not (isinstance(cls, type) and issubclass(cls, TypeSpec)):
      raise TypeError("Expected `cls` to be a TypeSpec; got %r" % (cls,))
    if cls in _TYPE_SPEC_TO_NAME:
      raise ValueError("Class %s.%s has already been registered with name %s."
                       % (cls.__module__, cls.__name__,
                          _TYPE_SPEC_TO_NAME[cls]))
    if name in _NAME_TO_TYPE_SPEC:
      raise ValueError("Name %s has already been registered for class %s.%s."
                       % (name, _NAME_TO_TYPE_SPEC[name].__module__,
                          _NAME_TO_TYPE_SPEC[name].__name__))
    _TYPE_SPEC_TO_NAME[cls] = name
    _NAME_TO_TYPE_SPEC[name] = cls
    return cls

  return decorator_fn


# TODO(edloper) tf_export this as "tf.get_type_spec_name" (or some similar name)
def get_name(cls):
  """Returns the registered name for TypeSpec `cls`."""
  if not (isinstance(cls, type) and issubclass(cls, TypeSpec)):
    raise TypeError("Expected `cls` to be a TypeSpec; got %r" % (cls,))
  if cls not in _TYPE_SPEC_TO_NAME:
    raise ValueError("TypeSpec %s.%s has not been registered." %
                     (cls.__module__, cls.__name__))
  return _TYPE_SPEC_TO_NAME[cls]


# TODO(edloper) tf_export this as "tf.lookup_type_spec" (or some similar name)
def lookup(name):
  """Returns the TypeSpec that has been registered with name `name`."""
  if not isinstance(name, str):
    raise TypeError("Expected `name` to be a string; got %r" % (name,))
  if name not in _NAME_TO_TYPE_SPEC:
    raise ValueError("No TypeSpec has been registered with name %r" % (name,))
  return _NAME_TO_TYPE_SPEC[name]
