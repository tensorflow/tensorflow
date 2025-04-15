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
import collections
import functools
import itertools

import wrapt

from tensorflow.python.data.util import nest
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import none_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import internal
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.nest_util import CustomNestProtocol
from tensorflow.python.util.tf_export import tf_export


# pylint: disable=invalid-name
@tf_export(v1=["data.experimental.TensorStructure"])
@deprecation.deprecated(None, "Use `tf.TensorSpec` instead.")
def _TensorStructure(dtype, shape):
  return tensor_lib.TensorSpec(shape, dtype)


@tf_export(v1=["data.experimental.SparseTensorStructure"])
@deprecation.deprecated(None, "Use `tf.SparseTensorSpec` instead.")
def _SparseTensorStructure(dtype, shape):
  return sparse_tensor.SparseTensorSpec(shape, dtype)


@tf_export(v1=["data.experimental.TensorArrayStructure"])
@deprecation.deprecated(None, "Use `tf.TensorArraySpec` instead.")
def _TensorArrayStructure(dtype, element_shape, dynamic_size, infer_shape):
  return tensor_array_ops.TensorArraySpec(element_shape, dtype,
                                          dynamic_size, infer_shape)


@tf_export(v1=["data.experimental.RaggedTensorStructure"])
@deprecation.deprecated(None, "Use `tf.RaggedTensorSpec` instead.")
def _RaggedTensorStructure(dtype, shape, ragged_rank):
  return ragged_tensor.RaggedTensorSpec(shape, dtype, ragged_rank)
# pylint: enable=invalid-name


# TODO(jsimsa): Remove the special-case for `TensorArray` pass-through once
# it is a subclass of `CompositeTensor`.
def normalize_element(element, element_signature=None):
  """Normalizes a nested structure of element components.

  * Components matching `SparseTensorSpec` are converted to `SparseTensor`.
  * Components matching `RaggedTensorSpec` are converted to `RaggedTensor`.
  * Components matching `VariableSpec` are converted to `Tensor`.
  * Components matching `DatasetSpec` or `TensorArraySpec` are passed through.
  * `CompositeTensor` components are passed through.
  * All other components are converted to `Tensor`.

  Args:
    element: A nested structure of individual components.
    element_signature: (Optional.) A nested structure of `tf.DType` objects
      corresponding to each component of `element`. If specified, it will be
      used to set the exact type of output tensor when converting input
      components which are not tensors themselves (e.g. numpy arrays, native
      python types, etc.)

  Returns:
    A nested structure of `Tensor`, `Variable`, `Dataset`, `SparseTensor`,
    `RaggedTensor`, or `TensorArray` objects.
  """
  normalized_components = []
  if element_signature is None:
    components = nest.flatten(element)
    flattened_signature = [None] * len(components)
    pack_as = element
  else:
    flattened_signature = nest.flatten(element_signature)
    components = nest.flatten_up_to(element_signature, element)
    pack_as = element_signature
  with ops.name_scope("normalize_element"):
    for i, (t, spec) in enumerate(zip(components, flattened_signature)):
      try:
        if spec is None:
          spec = type_spec_from_value(t, use_fallback=False)
      except TypeError:
        # TypeError indicates it was not possible to compute a `TypeSpec` for
        # the value. As a fallback try converting the value to a tensor.
        normalized_components.append(
            ops.convert_to_tensor(t, name="component_%d" % i))
      else:
        # To avoid a circular dependency between dataset_ops and structure,
        # we check the class name instead of using `isinstance`.
        if spec.__class__.__name__ == "DatasetSpec":
          normalized_components.append(t)
        elif isinstance(spec, sparse_tensor.SparseTensorSpec):
          normalized_components.append(sparse_tensor.SparseTensor.from_value(t))
        elif isinstance(spec, ragged_tensor.RaggedTensorSpec):
          normalized_components.append(
              ragged_tensor.convert_to_tensor_or_ragged_tensor(
                  t, name="component_%d" % i))
        elif isinstance(spec, (tensor_array_ops.TensorArraySpec)):
          normalized_components.append(t)
        elif isinstance(spec, none_tensor.NoneTensorSpec):
          normalized_components.append(none_tensor.NoneTensor())
        elif isinstance(spec, resource_variable_ops.VariableSpec):
          normalized_components.append(
              ops.convert_to_tensor(t, name=f"component_{i}", dtype=spec.dtype))
        elif isinstance(t, composite_tensor.CompositeTensor):
          normalized_components.append(t)
        else:
          dtype = getattr(spec, "dtype", None)
          normalized_components.append(
              ops.convert_to_tensor(t, name="component_%d" % i, dtype=dtype))
  return nest.pack_sequence_as(pack_as, normalized_components)


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
      flat_ret.append(sparse_tensor.SparseTensorSpec(flat_shape, flat_type))
    elif issubclass(flat_class, tensor_lib.Tensor):
      flat_ret.append(tensor_lib.TensorSpec(flat_shape, flat_type))
    elif issubclass(flat_class, tensor_array_ops.TensorArray):
      # We sneaked the dynamic_size and infer_shape into the legacy shape.
      flat_ret.append(
          tensor_array_ops.TensorArraySpec(
              flat_shape[2:], flat_type,
              dynamic_size=tensor_shape.dimension_value(flat_shape[0]),
              infer_shape=tensor_shape.dimension_value(flat_shape[1])))
    else:
      # NOTE(mrry): Since legacy structures produced by iterators only
      # comprise Tensors, SparseTensors, and nests, we do not need to
      # support all structure types here.
      raise TypeError(
          "Could not build a structure for output class {}. Make sure any "
          "component class in `output_classes` inherits from one of the "
          "following classes: `tf.TypeSpec`, `tf.sparse.SparseTensor`, "
          "`tf.Tensor`, `tf.TensorArray`.".format(flat_class.__name__))

  return nest.pack_sequence_as(output_classes, flat_ret)


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
    raise ValueError("Expected {} tensors but got {}.".format(
        sum(flat_spec_lengths), len(tensor_list)))

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
  return list(
      itertools.chain.from_iterable(
          spec._flat_tensor_specs for spec in nest.flatten(element_spec)))


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
    if isinstance(spec, internal.TensorSpec):
      try:
        component = ops.convert_to_tensor(component, spec.dtype)
      except (TypeError, ValueError):
        raise ValueError(
            f"Value {component} is not convertible to a tensor with "
            f"dtype {spec.dtype} and shape {spec.shape}."
        )
      if not component.shape.is_compatible_with(spec.shape):
        raise ValueError(
            f"Value {component} is not convertible to a tensor with "
            f"dtype {spec.dtype} and shape {spec.shape}."
        )
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


def type_spec_from_value(element, use_fallback=True):
  """Creates a type specification for the given value.

  Args:
    element: The element to create the type specification for.
    use_fallback: Whether to fall back to converting the element to a tensor
      in order to compute its `TypeSpec`.

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

  if isinstance(element, collections_abc.Mapping):
    # We create a shallow copy in an attempt to preserve the key order.
    #
    # Note that we do not guarantee that the key order is preserved, which is
    # a limitation inherited from `copy()`. As a consequence, callers of
    # `type_spec_from_value` should not assume that the key order of a `dict`
    # in the returned nested structure matches the key order of the
    # corresponding `dict` in the input value.
    if isinstance(element, collections.defaultdict):
      ctor = lambda items: type(element)(element.default_factory, items)
    else:
      ctor = type(element)
    return ctor([(k, type_spec_from_value(v)) for k, v in element.items()])

  if isinstance(element, tuple):
    if hasattr(element, "_fields") and isinstance(
        element._fields, collections_abc.Sequence) and all(
            isinstance(f, str) for f in element._fields):
      if isinstance(element, wrapt.ObjectProxy):
        element_type = type(element.__wrapped__)
      else:
        element_type = type(element)
      # `element` is a namedtuple
      return element_type(*[type_spec_from_value(v) for v in element])
    # `element` is not a namedtuple
    return tuple([type_spec_from_value(v) for v in element])

  if hasattr(element.__class__, "__attrs_attrs__"):
    # `element` is an `attr.s` decorated class
    attrs = getattr(element.__class__, "__attrs_attrs__")
    return type(element)(*[
        type_spec_from_value(getattr(element, a.name)) for a in attrs
    ])

  if isinstance(element, CustomNestProtocol):
    # pylint: disable=protected-access
    metadata, children = element.__tf_flatten__()
    return element.__tf_unflatten__(metadata, type_spec_from_value(children))
    # pylint: enable=protected-access

  if use_fallback:
    # As a fallback try converting the element to a tensor.
    try:
      tensor = ops.convert_to_tensor(element)
      spec = type_spec_from_value(tensor)
      if spec is not None:
        return spec
    except (ValueError, TypeError) as e:
      logging.vlog(
          3, "Failed to convert %r to tensor: %s" % (type(element).__name__, e))

  raise TypeError("Could not build a `TypeSpec` for {} with type {}".format(
      element,
      type(element).__name__))


# TODO(b/149584798): remove legacy forwarding references
NoneTensor = none_tensor.NoneTensor
NoneTensorSpec = none_tensor.NoneTensorSpec
