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

import collections
import functools

import six

from tensorflow.python.data.util import nest
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util.tf_export import tf_export


# Define backwards-compatiblity wrappers for using TypeSpec and its subclasses
# to replace Structure and its subclasses.  Note that the constructor argument
# order is different in many cases -- in particular, TypeSpec follows TensorSpec
# and uses the order (shape, dtype); but most Structure subclasses use the
# order (dtype, shape).
#
# TODO(b/133606651) Update tf.data to use TypeSpec directly, and then remove
# these compatibility wrappers.


Structure = type_spec.TypeSpec


# pylint: disable=invalid-name


@tf_export("data.experimental.TensorStructure")
def TensorStructure(dtype, shape):
  return tensor_spec.TensorSpec(shape, dtype)


@tf_export("data.experimental.SparseTensorStructure")
def SparseTensorStructure(dtype, shape):
  return sparse_tensor.SparseTensorSpec(shape, dtype)


@tf_export("data.experimental.TensorArrayStructure")
def TensorArrayStructure(dtype, element_shape, dynamic_size, infer_shape):
  return tensor_array_ops.TensorArraySpec(element_shape, dtype,
                                          dynamic_size, infer_shape)


@tf_export("data.experimental.RaggedTensorStructure")
def RaggedTensorStructure(dtype, shape, ragged_rank):
  return ragged_tensor.RaggedTensorSpec(shape, dtype, ragged_rank)


# TODO(jsimsa): Remove the special-case for `TensorArray` pass-through once
# it is a subclass of `CompositeTensor`.
def normalize_element(element):
  """Normalizes a nested structure of element components.

  * Components matching `SparseTensorSpec` are converted to `SparseTensor`.
  * Components matching `RaggedTensorSpec` are converted to `RaggedTensor`.
  * Components matching `DatasetSpec` or `TensorArraySpec` are passed through.
  * `CompositeTensor` components are passed through.
  * All other components are converted to `Tensor`.

  Args:
    element: A nested structure of individual components.

  Returns:
    A nested structure of `Tensor`, `Dataset`, `SparseTensor`, `RaggedTensor`,
    or `TensorArray` objects.
  """
  components = nest.flatten(element)
  normalized_components = []
  with ops.name_scope("normalize_tensors"):
    # Imported here to avoid circular dependency
    from tensorflow.python.data.ops import dataset_ops  # pylint: disable=g-import-not-at-top
    for i, t in enumerate(components):
      spec = type_spec_from_value(t)
      if isinstance(spec, sparse_tensor.SparseTensorSpec):
        normalized_components.append(sparse_tensor.SparseTensor.from_value(t))
      elif isinstance(spec, ragged_tensor.RaggedTensorSpec):
        normalized_components.append(
            ragged_tensor.convert_to_tensor_or_ragged_tensor(
                t, name="component_%d" % i))
      elif isinstance(
          spec,
          (tensor_array_ops.TensorArraySpec, dataset_ops.DatasetStructure)):
        normalized_components.append(t)
      elif isinstance(t, composite_tensor.CompositeTensor):
        normalized_components.append(t)
      else:
        normalized_components.append(
            ops.convert_to_tensor(t, name="component_%d" % i))
  return nest.pack_sequence_as(element, normalized_components)


def convert_legacy_structure(output_types, output_shapes, output_classes):
  """Returns a `Structure` that represents the given legacy structure.

  This method provides a way to convert from the existing `Dataset` and
  `Iterator` structure-related properties to a `Structure` object. A "legacy"
  structure is represented by the `tf.data.Dataset.output_types`,
  `tf.data.Dataset.output_shapes`, and `tf.data.Dataset.output_classes`
  properties.

  TODO(b/110122868): Remove this function once `Structure` is used throughout
  `tf.data`.

  Args:
    output_types: A nested structure of `tf.DType` objects corresponding to
      each component of a structured value.
    output_shapes: A nested structure of `tf.TensorShape` objects
      corresponding to each component a structured value.
    output_classes: A nested structure of Python `type` objects corresponding
      to each component of a structured value.

  Returns:
    A `Structure`.

  Raises:
    TypeError: If a structure cannot be built from the arguments, because one of
      the component classes in `output_classes` is not supported.
  """
  flat_types = nest.flatten(output_types)
  flat_shapes = nest.flatten(output_shapes)
  flat_classes = nest.flatten(output_classes)
  flat_ret = []
  for flat_type, flat_shape, flat_class in zip(flat_types, flat_shapes,
                                               flat_classes):
    if isinstance(flat_class, type_spec.TypeSpec):
      flat_ret.append(flat_class)
    elif issubclass(flat_class, sparse_tensor.SparseTensor):
      flat_ret.append(SparseTensorStructure(flat_type, flat_shape))
    elif issubclass(flat_class, ops.Tensor):
      flat_ret.append(TensorStructure(flat_type, flat_shape))
    elif issubclass(flat_class, tensor_array_ops.TensorArray):
      # We sneaked the dynamic_size and infer_shape into the legacy shape.
      flat_ret.append(
          TensorArrayStructure(
              flat_type, flat_shape[2:],
              dynamic_size=tensor_shape.dimension_value(flat_shape[0]),
              infer_shape=tensor_shape.dimension_value(flat_shape[1])))
    else:
      # NOTE(mrry): Since legacy structures produced by iterators only
      # comprise Tensors, SparseTensors, and nests, we do not need to
      # support all structure types here.
      raise TypeError(
          "Could not build a structure for output class %r" % (flat_class,))

  return nest.pack_sequence_as(output_classes, flat_ret)


# TODO(b/133606651) Once cl/253160704 is part of tf-nightly, remove the use
# of NestedStructure from TFF and remove this class.
@tf_export("data.experimental.NestedStructure")
class NestedStructure(type_spec.BatchableTypeSpec):
  """Represents a nested structure in which each leaf is a `TypeSpec`."""

  # NOTE(edloper): This class makes extensive use of non-public TypeSpec
  # methods, so we disable the protected-access lint warning once here.
  # pylint: disable=protected-access

  __slots__ = ["_nested_structure", "_flat_nested_structure",
               "__flat_tensor_specs"]

  def __init__(self, nested_structure):
    self._nested_structure = nested_structure
    self._flat_nested_structure = nest.flatten(nested_structure)
    self.__flat_tensor_specs = []
    for s in self._flat_nested_structure:
      if not isinstance(s, type_spec.TypeSpec):
        raise TypeError("nested_structure must be a (potentially nested) tuple "
                        "or dictionary of TypeSpec objects.")
      self.__flat_tensor_specs.extend(s._flat_tensor_specs)

  value_type = property(lambda self: type(self._nested_structure))

  def _serialize(self):
    return self._nested_structure

  @classmethod
  def _deserialize(cls, nested_structure):
    return cls(nested_structure)

  def most_specific_compatible_type(self, other):
    if type(self) is not type(other):
      raise ValueError("Incompatible types")
    return self._deserialize(
        nest.map_structure(lambda a, b: a.most_specific_compatible_type(b),
                           self._nested_structure, other._nested_structure))

  def __eq__(self, other):
    if not isinstance(other, NestedStructure):
      return False
    try:
      nest.assert_same_structure(self._nested_structure,
                                 other._nested_structure)
    except (ValueError, TypeError):
      return False
    return (nest.flatten(self._nested_structure) ==
            nest.flatten(other._nested_structure))

  def __hash__(self):
    return hash(tuple(nest.flatten(self._nested_structure)))

  def is_compatible_with(self, other):
    if not isinstance(other, NestedStructure):
      return False
    try:
      nest.assert_same_structure(self._nested_structure,
                                 other._nested_structure)
    except (ValueError, TypeError):
      return False

    # pylint: disable=g-complex-comprehension
    return all(
        substructure.is_compatible_with(other_substructure)
        for substructure, other_substructure in zip(
            nest.flatten(self._nested_structure),
            nest.flatten(other._nested_structure)))

  _component_specs = property(lambda self: self._nested_structure)
  _flat_tensor_specs = property(lambda self: self.__flat_tensor_specs)

  def _to_components(self, value):
    return nest.map_structure_up_to(
        self._nested_structure, lambda t, v: t._to_components(v),
        self._nested_structure, value)

  def _from_components(self, value):
    return nest.map_structure_up_to(
        self._nested_structure, lambda t, v: t._from_components(v),
        self._nested_structure, value)

  def _to_tensor_list(self, value):
    return self.__value_to_tensors(
        value, lambda struct, val: struct._to_tensor_list(val))

  def _to_batched_tensor_list(self, value):
    return self.__value_to_tensors(
        value, lambda struct, val: struct._to_batched_tensor_list(val))

  def __value_to_tensors(self, value, to_tensor_list_fn):
    ret = []

    try:
      flat_value = nest.flatten_up_to(self._nested_structure, value)
    except (ValueError, TypeError):
      raise ValueError("The value %r is not compatible with the nested "
                       "structure %r." % (value, self._nested_structure))

    for sub_value, structure in zip(flat_value, self._flat_nested_structure):
      if not structure.is_compatible_with(
          type_spec.type_spec_from_value(sub_value)):
        raise ValueError("Component value %r is not compatible with the nested "
                         "structure %r." % (sub_value, structure))
      ret.extend(to_tensor_list_fn(structure, sub_value))
    return ret

  def _from_tensor_list(self, value):
    return self.__tensors_to_value(
        value, lambda struct, val: struct._from_tensor_list(val))

  def _from_compatible_tensor_list(self, value):
    return self.__tensors_to_value(
        value, lambda struct, val: struct._from_compatible_tensor_list(val))

  def __tensors_to_value(self, flat_value, from_tensor_list_fn):
    if len(flat_value) != len(self._flat_tensor_specs):
      raise ValueError("Expected %d flat values in NestedStructure but got %d."
                       % (len(self._flat_tensor_specs), len(flat_value)))
    flat_ret = []
    i = 0
    for structure in self._flat_nested_structure:
      num_flat_values = len(structure._flat_tensor_specs)
      sub_value = flat_value[i:i + num_flat_values]
      flat_ret.append(from_tensor_list_fn(structure, sub_value))
      i += num_flat_values

    return nest.pack_sequence_as(self._nested_structure, flat_ret)

  @staticmethod
  def from_value(value):
    flat_nested_structure = [
        type_spec.type_spec_from_value(sub_value)
        for sub_value in nest.flatten(value)
    ]
    return NestedStructure(nest.pack_sequence_as(value, flat_nested_structure))

  def _to_legacy_output_types(self):
    return nest.map_structure(
        lambda s: s._to_legacy_output_types(), self._nested_structure)

  def _to_legacy_output_shapes(self):
    return nest.map_structure(
        lambda s: s._to_legacy_output_shapes(), self._nested_structure)

  def _to_legacy_output_classes(self):
    return nest.map_structure(
        lambda s: s._to_legacy_output_classes(), self._nested_structure)

  def _batch(self, batch_size):
    return NestedStructure(nest.map_structure(
        lambda s: s._batch(batch_size), self._nested_structure))

  def _unbatch(self):
    return NestedStructure(nest.map_structure(
        lambda s: s._unbatch(), self._nested_structure))


def _from_tensor_list_helper(decode_fn, element_spec, tensor_list):
  """Returns an element constructed from the given spec and tensor list.

  Args:
    decode_fn: Method that constructs an element component from the element spec
      component and a tensor list.
    element_spec: A nested structure of `tf.TypeSpec` objects representing to
      element type specification.
    tensor_list: A list of tensors to use for constructing the value.

  Returns:
    An element constructed from the given spec and tensor list.

  Raises:
    ValueError: If the number of tensors needed to construct an element for
      the given spec does not match the given number of tensors.
  """

  # pylint: disable=protected-access

  flat_specs = nest.flatten(element_spec)
  flat_spec_lengths = [len(spec._flat_tensor_specs) for spec in flat_specs]
  if sum(flat_spec_lengths) != len(tensor_list):
    raise ValueError("Expected %d tensors but got %d." %
                     (sum(flat_spec_lengths), len(tensor_list)))

  i = 0
  flat_ret = []
  for (component_spec, num_flat_values) in zip(flat_specs, flat_spec_lengths):
    value = tensor_list[i:i + num_flat_values]
    flat_ret.append(decode_fn(component_spec, value))
    i += num_flat_values
  return nest.pack_sequence_as(element_spec, flat_ret)


def from_compatible_tensor_list(element_spec, tensor_list):
  """Returns an element constructed from the given spec and tensor list.

  Args:
    element_spec: A nested structure of `tf.TypeSpec` objects representing to
      element type specification.
    tensor_list: A list of tensors to use for constructing the value.

  Returns:
    An element constructed from the given spec and tensor list.

  Raises:
    ValueError: If the number of tensors needed to construct an element for
      the given spec does not match the given number of tensors.
  """

  # pylint: disable=protected-access
  # pylint: disable=g-long-lambda
  return _from_tensor_list_helper(
      lambda spec, value: spec._from_compatible_tensor_list(value),
      element_spec, tensor_list)


def from_tensor_list(element_spec, tensor_list):
  """Returns an element constructed from the given spec and tensor list.

  Args:
    element_spec: A nested structure of `tf.TypeSpec` objects representing to
      element type specification.
    tensor_list: A list of tensors to use for constructing the value.

  Returns:
    An element constructed from the given spec and tensor list.

  Raises:
    ValueError: If the number of tensors needed to construct an element for
      the given spec does not match the given number of tensors or the given
      spec is not compatible with the tensor list.
  """

  # pylint: disable=protected-access
  # pylint: disable=g-long-lambda
  return _from_tensor_list_helper(
      lambda spec, value: spec._from_tensor_list(value), element_spec,
      tensor_list)


def get_flat_tensor_specs(element_spec):
  """Returns a list `tf.TypeSpec`s for the element tensor representation.

  Args:
    element_spec: A nested structure of `tf.TypeSpec` objects representing to
      element type specification.

  Returns:
    A list `tf.TypeSpec`s for the element tensor representation.
  """

  # pylint: disable=protected-access
  return functools.reduce(lambda state, value: state + value._flat_tensor_specs,
                          nest.flatten(element_spec), [])


def get_flat_tensor_shapes(element_spec):
  """Returns a list `tf.TensorShapes`s for the element tensor representation.

  Args:
    element_spec: A nested structure of `tf.TypeSpec` objects representing to
      element type specification.

  Returns:
    A list `tf.TensorShapes`s for the element tensor representation.
  """
  return [spec.shape for spec in get_flat_tensor_specs(element_spec)]


def get_flat_tensor_types(element_spec):
  """Returns a list `tf.DType`s for the element tensor representation.

  Args:
    element_spec: A nested structure of `tf.TypeSpec` objects representing to
      element type specification.

  Returns:
    A list `tf.DType`s for the element tensor representation.
  """
  return [spec.dtype for spec in get_flat_tensor_specs(element_spec)]


def _to_tensor_list_helper(encode_fn, element_spec, element):
  """Returns a tensor list representation of the element.

  Args:
    encode_fn: Method that constructs a tensor list representation from the
      given element spec and element.
    element_spec: A nested structure of `tf.TypeSpec` objects representing to
      element type specification.
    element: The element to convert to tensor list representation.

  Returns:
    A tensor list representation of `element`.

  Raises:
    ValueError: If `element_spec` and `element` do not have the same number of
      elements or if the two structures are not nested in the same way.
    TypeError: If `element_spec` and `element` differ in the type of sequence
      in any of their substructures.
  """

  nest.assert_same_structure(element_spec, element)

  def reduce_fn(state, value):
    spec, component = value
    return encode_fn(state, spec, component)

  return functools.reduce(
      reduce_fn, zip(nest.flatten(element_spec), nest.flatten(element)), [])


def to_batched_tensor_list(element_spec, element):
  """Returns a tensor list representation of the element.

  Args:
    element_spec: A nested structure of `tf.TypeSpec` objects representing to
      element type specification.
    element: The element to convert to tensor list representation.

  Returns:
    A tensor list representation of `element`.

  Raises:
    ValueError: If `element_spec` and `element` do not have the same number of
      elements or if the two structures are not nested in the same way or the
      rank of any of the tensors in the tensor list representation is 0.
    TypeError: If `element_spec` and `element` differ in the type of sequence
      in any of their substructures.
  """

  # pylint: disable=protected-access
  # pylint: disable=g-long-lambda
  return _to_tensor_list_helper(
      lambda state, spec, component: state + spec._to_batched_tensor_list(
          component), element_spec, element)


def to_tensor_list(element_spec, element):
  """Returns a tensor list representation of the element.

  Args:
    element_spec: A nested structure of `tf.TypeSpec` objects representing to
      element type specification.
    element: The element to convert to tensor list representation.

  Returns:
    A tensor list representation of `element`.

  Raises:
    ValueError: If `element_spec` and `element` do not have the same number of
      elements or if the two structures are not nested in the same way.
    TypeError: If `element_spec` and `element` differ in the type of sequence
      in any of their substructures.
  """

  # pylint: disable=protected-access
  # pylint: disable=g-long-lambda
  return _to_tensor_list_helper(
      lambda state, spec, component: state + spec._to_tensor_list(component),
      element_spec, element)


def are_compatible(spec1, spec2):
  """Indicates whether two type specifications are compatible.

  Two type specifications are compatible if they have the same nested structure
  and the their individual components are pair-wise compatible.

  Args:
    spec1: A `tf.TypeSpec` object to compare.
    spec2: A `tf.TypeSpec` object to compare.

  Returns:
    `True` if the two type specifications are compatible and `False` otherwise.
  """

  try:
    nest.assert_same_structure(spec1, spec2)
  except TypeError:
    return False
  except ValueError:
    return False

  for s1, s2 in zip(nest.flatten(spec1), nest.flatten(spec2)):
    if not s1.is_compatible_with(s2) or not s2.is_compatible_with(s1):
      return False
  return True


def type_spec_from_value(element):
  """Creates a type specification for the given value.

  Args:
    element: The element to create the type specification for.

  Returns:
    A nested structure of `TypeSpec`s that represents the type specification
    of `element`.

  Raises:
    TypeError: If a `TypeSpec` cannot be built for `element`, because its type
      is not supported.
  """
  spec = type_spec._type_spec_from_value(element)  # pylint: disable=protected-access
  if spec is not None:
    return spec

  if isinstance(element, dict):
    # We create a shallow copy in an attempt to preserve the key order.
    #
    # Note that we do not guarantee that the key order is preserved, which is
    # a limitation inherited from `copy()`. As a consequence, callers of
    # `type_spec_from_value` should not assume that the key order of a `dict`
    # in the returned nested structure matches the key order of the
    # corresponding `dict` in the input value.
    result = element.copy()
    for k in element:
      result[k] = type_spec_from_value(element[k])
    return result

  if isinstance(element, tuple):
    if hasattr(element, "_fields") and isinstance(
        element._fields, collections.Sequence) and all(
            isinstance(f, six.string_types) for f in element._fields):
      # `element` is a namedtuple
      return type(element)(*[type_spec_from_value(v) for v in element])
    # `element` is not a namedtuple
    return tuple([type_spec_from_value(v) for v in element])

  # Fallback: try converting value to a tensor.
  try:
    tensor = ops.convert_to_tensor(element)
    spec = type_spec_from_value(tensor)
    if spec is not None:
      return spec
  except (ValueError, TypeError):
    pass

  raise TypeError("Could not build a TypeSpec for %r with type %s" %
                  (element, type(element).__name__))
