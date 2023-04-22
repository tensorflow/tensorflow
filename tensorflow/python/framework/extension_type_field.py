# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Meatadata about fields for user-defined ExtensionType classes."""

import collections
import collections.abc
import typing

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec

# These names may not be used as the name for a ExtensionType field (to prevent
# name clashes).  All names beginning with `'_tf_extension_type'` are also
# reserved.
RESERVED_FIELD_NAMES = [
    'self',
    # Name of the nested TypeSpec class.
    'Spec',
    # Names defined by the CompositeTensor base class.
    '_type_spec',
    '_shape_invariant_to_type_spec',
    '_consumers',
    # Names defined by the TypeSpec base class.
    'value_type',
    'is_compatible_with',
    'most_specific_compatible_type',
    '_with_tensor_ranks_only',
    '_to_components',
    '_from_components',
    '_component_specs',
    '_to_tensor_list',
    '_from_tensor_list',
    '_from_compatible_tensor_list',
    '_flat_tensor_specs',
    '_serialize',
    '_deserialize',
    '_to_legacy_output_types',
    '_to_legacy_output_shapes',
    '_to_legacy_output_classes',
]


class Sentinel(object):
  """Sentinel value that's not equal (w/ `is`) to any user value."""

  def __init__(self, name):
    self._name = name

  def __repr__(self):
    return self._name


# ==============================================================================
# ExtensionTypeField
# ==============================================================================
class ExtensionTypeField(
    collections.namedtuple('ExtensionTypeField',
                           ['name', 'value_type', 'default'])):
  """Metadata about a single field in a `tf.ExtensionType` object."""

  NO_DEFAULT = Sentinel('ExtensionTypeField.NO_DEFAULT')

  def __new__(cls, name, value_type, default=NO_DEFAULT):
    """Constructs a new ExtensionTypeField containing metadata for a single field.

    Args:
      name: The name of the new field (`str`).  May not be a reserved name.
      value_type: A python type expression constraining what values this field
        can take.
      default: The default value for the new field, or `NO_DEFAULT` if this
        field has no default value.

    Returns:
      A new `ExtensionTypeField`.

    Raises:
      TypeError: If the type described by `value_type` is not currently
          supported by `tf.ExtensionType`.
      TypeError: If `default` is specified and its type does not match
        `value_type`.
    """
    try:
      validate_field_value_type(value_type, allow_forward_references=True)
    except TypeError as e:
      raise TypeError(f'In field {name!r}: {e}')

    if default is not cls.NO_DEFAULT:
      default = _convert_value(default, value_type,
                               (f'default value for {name}',))
    return super(ExtensionTypeField, cls).__new__(cls, name, value_type,
                                                  default)

  @staticmethod
  def is_reserved_name(name):
    """Returns true if `name` is a reserved name."""
    return name in RESERVED_FIELD_NAMES or name.lower().startswith(
        '_tf_extension_type')


def validate_field_value_type(value_type,
                              in_mapping_key=False,
                              allow_forward_references=False):
  """Checks that `value_type` contains only supported type annotations.

  Args:
    value_type: The type annotation to check.
    in_mapping_key: True if `value_type` is nested in the key of a mapping.
    allow_forward_references: If false, then raise an exception if a
      `value_type` contains a forward reference (i.e., a string literal).

  Raises:
    TypeError: If `value_type` contains an unsupported type annotation.
  """
  if isinstance(value_type, str) or is_forward_ref(value_type):
    if allow_forward_references:
      return
    else:
      raise TypeError(f'Unresolved forward reference {value_type!r}')

  if value_type in (int, float, str, bytes, bool, None, _NoneType,
                    dtypes.DType):
    return
  elif (value_type in (ops.Tensor, tensor_shape.TensorShape) or
        isinstance(value_type, type_spec.TypeSpec) or
        (isinstance(value_type, type) and
         issubclass(value_type, composite_tensor.CompositeTensor))):
    if in_mapping_key:
      raise TypeError('Key must be hashable.')
  elif is_generic_tuple(value_type) or is_generic_union(value_type):
    type_args = get_generic_type_args(value_type)
    if (len(type_args) == 2 and type_args[1] is Ellipsis and
        is_generic_tuple(value_type)):  # `Tuple[X, ...]`
      validate_field_value_type(type_args[0], in_mapping_key,
                                allow_forward_references)
    else:
      for arg in get_generic_type_args(value_type):
        validate_field_value_type(arg, in_mapping_key, allow_forward_references)
  elif is_generic_mapping(value_type):
    key_type, value_type = get_generic_type_args(value_type)
    validate_field_value_type(key_type, True, allow_forward_references)
    validate_field_value_type(value_type, in_mapping_key,
                              allow_forward_references)
  elif isinstance(value_type, type):
    raise TypeError(f'Unsupported type annotation `{value_type.__name__}`')
  else:
    raise TypeError(f'Unsupported type annotation {value_type!r}')


# ==============================================================================
# Type-checking & conversion for ExtensionTypeField values
# ==============================================================================


def convert_fields(fields, field_values):
  """Type-checks and converts each field in `field_values` (in place).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.

  Raises:
    ValueError: If the keys of `field_values` do not match the names of
      the fields in `fields`.
    TypeError: If any value in `field_values` does not have the type indicated
      by the corresponding `ExtensionTypeField` object.
  """
  _convert_fields(fields, field_values, for_spec=False)


def convert_fields_for_spec(fields, field_values):
  """Type-checks and converts field values for a TypeSpec (in place).

  This is similar to `convert_fields`, except that we expect a TypeSpec
  for tensor-like types.  In particular, if the `value_type` of a field
  specifies a tensor-like type (tf.Tensor, CompositeTensor, or TypeSpec),
  then the corresponding value in `fields` is expected to contain a TypeSpec
  (rather than a value described by that TypeSpec).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.

  Raises:
    ValueError: If the keys of `field_values` do not match the names of
      the fields in `fields`.
    TypeError: If any value in `field_values` does not have the type indicated
      by the corresponding `ExtensionTypeField` object.
  """
  _convert_fields(fields, field_values, for_spec=True)


def _convert_fields(fields, field_values, for_spec):
  """Type-checks and converts each field in `field_values` (in place).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.
    for_spec: If false, then expect a value for tensor-like types; if true, then
      expect a TypeSpec for tensor-like types.

  Raises:
    ValueError: If the keys of `field_values` do not match the names of
      the fields in `fields`.
    TypeError: If any value in `field_values` does not have the type indicated
      by the corresponding `ExtensionTypeField` object.
  """
  converted = {}
  if len(fields) != len(field_values):
    _report_field_mismatches(fields, field_values)
  for field in fields:
    if field.name not in field_values:
      _report_field_mismatches(fields, field_values)
    field_value = field_values[field.name]
    converted[field.name] = _convert_value(field_value, field.value_type,
                                           (field.name,), for_spec)
  field_values.update(converted)


def _convert_value(value, expected_type, path, for_spec=False):
  """Type-checks and converts a value.

  Args:
    value: The value to type-check.
    expected_type: The expected type for the value.
    path: Tuple of `str` naming the value (used for exception messages).
    for_spec: If false, then expect a value for tensor-like types; if true, then
      expect a TensorSpec for tensor-like types.

  Returns:
    A copy of `value`, converted to the expected type.

  Raises:
    TypeError: If `value` can not be converted to the expected type.
  """
  assert isinstance(path, tuple)

  if expected_type is None:
    expected_type = _NoneType

  if expected_type is ops.Tensor:
    return _convert_tensor(value, path, for_spec)
  elif isinstance(expected_type, tensor_spec.TensorSpec):
    return _convert_tensor_spec(value, expected_type, path, for_spec)
  elif isinstance(expected_type, type_spec.TypeSpec):
    return _convert_type_spec(value, expected_type, path, for_spec)
  elif (isinstance(expected_type, type) and
        issubclass(expected_type, composite_tensor.CompositeTensor)):
    return _convert_composite_tensor(value, expected_type, path, for_spec)
  elif expected_type in (int, float, bool, str, bytes, _NoneType, dtypes.DType,
                         tensor_shape.TensorShape):
    if not isinstance(value, expected_type):
      raise TypeError(f'{"".join(path)}: expected '
                      f'{expected_type.__name__}, got {value!r}')
    return value
  elif is_generic_tuple(expected_type):
    return _convert_tuple(value, expected_type, path, for_spec)
  elif is_generic_mapping(expected_type):
    return _convert_mapping(value, expected_type, path, for_spec)
  elif is_generic_union(expected_type):
    return _convert_union(value, expected_type, path, for_spec)
  else:
    raise TypeError(f'{"".join(path)}: Unsupported type annotation '
                    f'{expected_type!r}')


def _convert_tensor(value, path, for_spec):
  """Converts `value` to a `Tensor`."""
  if for_spec:
    if not isinstance(value, tensor_spec.TensorSpec):
      raise TypeError(f'{"".join(path)}: expected a TensorSpec, got {value!r}')
    return value

  if not isinstance(value, ops.Tensor):
    try:
      value = ops.convert_to_tensor(value)
    except (ValueError, TypeError) as e:
      raise TypeError(f'{"".join(path)}: expected a Tensor, '
                      f'got {value!r}') from e
  return value


def _convert_tensor_spec(value, expected_type, path, for_spec):
  """Converts `value` to a Tensor comptible with TensorSpec expected_type."""
  if for_spec:
    if not (isinstance(value, tensor_spec.TensorSpec) and
            expected_type.is_compatible_with(value)):
      raise TypeError(f'{"".join(path)}: expected a TensorSpec compatible '
                      f'with {expected_type}, got {value!r}')
    return value

  if not isinstance(value, ops.Tensor):
    try:
      value = ops.convert_to_tensor(value, expected_type.dtype)
    except (ValueError, TypeError):
      raise TypeError(f'{"".join(path)}: expected a {expected_type.dtype!r} '
                      f'Tensor, got {value!r}')
  if not expected_type.is_compatible_with(value):
    raise TypeError(f'{"".join(path)}: expected a Tensor compatible with '
                    f'{expected_type}, got {value!r}')
  return value


def _convert_type_spec(value, expected_type, path, for_spec):
  """Converts `value` to a value comptible with TypeSpec `expected_type`."""
  if for_spec:
    if not (isinstance(value, type_spec.TypeSpec) and
            expected_type.is_compatible_with(value)):
      raise TypeError(f'{"".join(path)}: expected a TypeSpec compatible '
                      f'with {expected_type}, got {value!r}')
    return value

  if (isinstance(value, type_spec.TypeSpec) or
      not expected_type.is_compatible_with(value)):
    raise TypeError(f'{"".join(path)}: expected {expected_type!r}, '
                    f'got {value!r}')
  return value


def _convert_composite_tensor(value, expected_type, path, for_spec):
  """Converts `value` to a value of type `expected_type`."""
  if for_spec:
    if not (isinstance(value, type_spec.TypeSpec) and
            issubclass(value.value_type, expected_type)):
      raise TypeError(f'{"".join(path)}: expected a TypeSpec for '
                      f'{expected_type.__name__}, got {value!r}')
    return value

  if not isinstance(value, expected_type):
    raise TypeError(f'{"".join(path)}: expected {expected_type.__name__}, '
                    f'got {value!r}')
  return value


def _convert_tuple(value, expected_type, path, for_spec):
  """Converts `value` to a tuple with type `expected_type`."""
  if not isinstance(value, typing.Sequence):
    raise TypeError(f'{"".join(path)}: expected tuple, got {value!r}')
  element_types = get_generic_type_args(expected_type)
  if len(element_types) == 2 and element_types[1] is Ellipsis:
    return tuple([
        _convert_value(v, element_types[0], path + (f'[{i}]',), for_spec)
        for (i, v) in enumerate(value)
    ])
  else:
    if len(value) != len(element_types):
      raise TypeError(f'{"".join(path)}: expected tuple with length '
                      f'{len(element_types)}, got {value!r})')
    return tuple([
        _convert_value(v, t, path + (f'[{i}]',), for_spec)
        for (i, (v, t)) in enumerate(zip(value, element_types))
    ])


def _convert_mapping(value, expected_type, path, for_spec):
  """Converts `value` to a mapping with type `expected_type`."""
  if not isinstance(value, typing.Mapping):
    raise TypeError(f'{"".join(path)}: expected mapping, got {value!r}')
  key_type, value_type = get_generic_type_args(expected_type)
  return immutable_dict.ImmutableDict([
      (_convert_value(k, key_type, path + ('[<key>]',), for_spec),
       _convert_value(v, value_type, path + (f'[{k!r}]',), for_spec))
      for (k, v) in value.items()
  ])


def _convert_union(value, expected_type, path, for_spec):
  """Converts `value` to a value with any of the types in `expected_type`."""
  for type_option in get_generic_type_args(expected_type):
    try:
      return _convert_value(value, type_option, path, for_spec)
    except TypeError:
      pass
  raise TypeError(f'{"".join(path)}: expected {expected_type}, got {value!r}')


def _report_field_mismatches(fields, field_values):
  """Raises an exception with mismatches between fields and field_values."""
  expected = set(f.name for f in fields)
  actual = set(field_values)
  extra = actual - expected
  if extra:
    raise ValueError(f'Got unexpected fields: {extra}')
  missing = expected - actual
  if missing:
    raise ValueError(f'Missing required fields: {missing}')


# ==============================================================================
# Utilities for accessing Python generic type annotations (typing.*)
# ==============================================================================
def is_generic_union(tp):
  """Returns true if `tp` is a parameterized typing.Union value."""
  return (tp is not typing.Union and
          getattr(tp, '__origin__', None) is typing.Union)


def is_generic_tuple(tp):
  """Returns true if `tp` is a parameterized typing.Tuple value."""
  return (tp not in (tuple, typing.Tuple) and
          getattr(tp, '__origin__', None) in (tuple, typing.Tuple))


def is_generic_mapping(tp):
  """Returns true if `tp` is a parameterized typing.Mapping value."""
  return (tp not in (collections.abc.Mapping, typing.Mapping) and getattr(
      tp, '__origin__', None) in (collections.abc.Mapping, typing.Mapping))


def is_forward_ref(tp):
  """Returns true if `tp` is a typing forward reference."""
  if hasattr(typing, 'ForwardRef'):
    return isinstance(tp, typing.ForwardRef)
  elif hasattr(typing, '_ForwardRef'):
    return isinstance(tp, typing._ForwardRef)  # pylint: disable=protected-access
  else:
    return False


# Note: typing.get_args was added in Python 3.8.
if hasattr(typing, 'get_args'):
  get_generic_type_args = typing.get_args
else:
  get_generic_type_args = lambda tp: tp.__args__

_NoneType = type(None)
