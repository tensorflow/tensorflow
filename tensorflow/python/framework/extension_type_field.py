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
"""Metadata about fields for user-defined ExtensionType classes."""

import collections
import collections.abc
import enum
import typing

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.util import type_annotations

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
    # Used by Keras
    '_keras_mask'
]


class Sentinel(object):
  """Sentinel value that's not equal (w/ `is`) to any user value."""

  def __init__(self, name):
    self._name = name

  def __repr__(self):
    return self._name


_NoneType = type(None)


def _issubclass(cls, clsinfo):
  """Internal issubclass that doesn't raise TypeError."""
  try:
    return issubclass(cls, clsinfo)
  except TypeError:
    # issubclass with GenericAlias instances raises TypeError. For example,
    # `issubclass(tuple[int], composite_tensor.CompositeTensor)`.
    return False


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
      raise TypeError(f'In field {name!r}: {e}') from e

    if default is not cls.NO_DEFAULT:
      default = _convert_value(default, value_type,
                               (f'default value for {name}',),
                               _ConversionContext.DEFAULT)
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
  if isinstance(value_type, str) or type_annotations.is_forward_ref(value_type):
    if allow_forward_references:
      return
    else:
      raise TypeError(f'Unresolved forward reference {value_type!r}')

  if value_type in (int, float, str, bytes, bool, None, _NoneType,
                    dtypes.DType):
    return
  elif (value_type in (tensor.Tensor, tensor_shape.TensorShape) or
        (isinstance(value_type, type) and
         _issubclass(value_type, composite_tensor.CompositeTensor))):
    if in_mapping_key:
      raise TypeError(f'Mapping had a key {value_type.__name__!r} with type '
                      f'{type(value_type).__name__!r}')
  elif (type_annotations.is_generic_tuple(value_type) or
        type_annotations.is_generic_union(value_type)):
    type_args = type_annotations.get_generic_type_args(value_type)
    if (len(type_args) == 2 and type_args[1] is Ellipsis and
        type_annotations.is_generic_tuple(value_type)):  # `Tuple[X, ...]`
      validate_field_value_type(type_args[0], in_mapping_key,
                                allow_forward_references)
    else:
      for arg in type_annotations.get_generic_type_args(value_type):
        validate_field_value_type(arg, in_mapping_key, allow_forward_references)
  elif type_annotations.is_generic_mapping(value_type):
    key_type, value_type = type_annotations.get_generic_type_args(value_type)
    validate_field_value_type(key_type, True, allow_forward_references)
    validate_field_value_type(value_type, in_mapping_key,
                              allow_forward_references)
  elif isinstance(value_type, type):
    raise TypeError(f'Unsupported type annotation {value_type.__name__!r}')
  else:
    raise TypeError(f'Unsupported type annotation {value_type!r}')


# ==============================================================================
# Type-checking & conversion for ExtensionTypeField values
# ==============================================================================


class _ConversionContext(enum.Enum):
  """Enum to indicate what kind of value is being converted.

  Used by `_convert_fields` and `_convert_value` and their helper methods.
  """
  VALUE = 1  # Converting an ExtensionType field
  SPEC = 2  # Converting an ExtensionType.Spec field
  DEFAULT = 3  # Converting a default value for __init__


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
  _convert_fields(fields, field_values, context=_ConversionContext.VALUE)


def convert_fields_for_spec(fields, field_values):
  """Type-checks and converts field values for a TypeSpec (in place).

  This is similar to `convert_fields`, except that we expect a `TypeSpec` for
  tensor-like types.  In particular, if the `value_type` of a field is
  `tf.Tensor` or a `CompositeTensor` subclass, then the corresponding value in
  `fields` is expected to contain a `TypeSpec` (rather than a value described by
  that `TypeSpec`).

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
  _convert_fields(fields, field_values, context=_ConversionContext.SPEC)


def _convert_fields(fields, field_values, context):
  """Type-checks and converts each field in `field_values` (in place).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.
    context: _ConversionContext, indicates what kind of value we are converting.

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
                                           (field.name,), context)
  field_values.update(converted)


def _convert_value(value, expected_type, path,
                   context=_ConversionContext.VALUE):
  """Type-checks and converts a value.

  Args:
    value: The value to type-check.
    expected_type: The expected type for the value.
    path: Tuple of `str` naming the value (used for exception messages).
    context: _ConversionContext, indicates what kind of value we are converting.

  Returns:
    A copy of `value`, converted to the expected type.

  Raises:
    TypeError: If `value` can not be converted to the expected type.
  """
  assert isinstance(path, tuple)

  if expected_type is None:
    expected_type = _NoneType

  if expected_type is tensor.Tensor:
    return _convert_tensor(value, path, context)
  elif (isinstance(expected_type, type) and
        _issubclass(expected_type, composite_tensor.CompositeTensor)):
    return _convert_composite_tensor(value, expected_type, path, context)
  elif expected_type is tensor_shape.TensorShape:
    try:
      return tensor_shape.as_shape(value)
    except TypeError as e:
      raise TypeError(f"{''.join(path)}: expected 'tf.TensorShape', got "
                      f'{type(value).__name__!r}') from e
  elif expected_type is dtypes.DType:
    try:
      return dtypes.as_dtype(value)
    except TypeError as e:
      raise TypeError(f"{''.join(path)}: expected 'tf.DType', got "
                      f'{type(value).__name__!r}') from e
  elif expected_type in (int, float, bool, str, bytes, _NoneType):
    if not isinstance(value, expected_type):
      raise TypeError(f'{"".join(path)}: expected {expected_type.__name__!r}, '
                      f'got {type(value).__name__!r}')
    return value
  elif type_annotations.is_generic_tuple(expected_type):
    return _convert_tuple(value, expected_type, path, context)
  elif type_annotations.is_generic_mapping(expected_type):
    return _convert_mapping(value, expected_type, path, context)
  elif type_annotations.is_generic_union(expected_type):
    return _convert_union(value, expected_type, path, context)
  else:
    raise TypeError(f'{"".join(path)}: Unsupported type annotation '
                    f'{expected_type!r}')


def _convert_tensor(value, path, context):
  """Converts `value` to a `Tensor`."""
  if context == _ConversionContext.SPEC:
    if not (isinstance(value, type_spec.TypeSpec) and
            value.value_type is tensor.Tensor):
      raise TypeError(
          f'{"".join(path)}: expected a TensorSpec, got '
          f'{type(value).__name__!r}')
    return value

  if not isinstance(value, tensor.Tensor):
    if context == _ConversionContext.DEFAULT:
      # TODO(edloper): Convert the value to a numpy array?  (Note: we can't just
      # use `np.array(value)`, since the default dtypes for TF and numpy are
      # different -- e.g., int->np.int64 but int->tf.int32.
      return value
    try:
      value = ops.convert_to_tensor(value)
    except (ValueError, TypeError) as e:
      raise TypeError(f'{"".join(path)}: expected a Tensor, '
                      f'got {type(value).__name__!r}') from e
  return value


def _convert_composite_tensor(value, expected_type, path, context):
  """Converts `value` to a value of type `expected_type`."""
  if context == _ConversionContext.SPEC:
    if not (isinstance(value, type_spec.TypeSpec) and
            _issubclass(value.value_type, expected_type)):
      raise TypeError(f'{"".join(path)}: expected a TypeSpec for '
                      f'{expected_type.__name__!r}, got '
                      f'{type(value).__name__!r}')
    return value

  if not isinstance(value, expected_type):
    raise TypeError(f'{"".join(path)}: expected {expected_type.__name__!r}, '
                    f'got {type(value).__name__!r}')
  return value


def _convert_tuple(value, expected_type, path, context):
  """Converts `value` to a tuple with type `expected_type`."""
  if not isinstance(value, typing.Sequence):
    raise TypeError(f'{"".join(path)}: expected tuple, got '
                    f'{type(value).__name__!r}')
  element_types = type_annotations.get_generic_type_args(expected_type)
  if len(element_types) == 2 and element_types[1] is Ellipsis:
    return tuple([
        _convert_value(v, element_types[0], path + (f'[{i}]',), context)
        for (i, v) in enumerate(value)
    ])
  else:
    if len(value) != len(element_types):
      raise TypeError(f'{"".join(path)}: expected tuple with length '
                      f'{len(element_types)}, got {type(value).__name__!r})')
    return tuple([
        _convert_value(v, t, path + (f'[{i}]',), context)
        for (i, (v, t)) in enumerate(zip(value, element_types))
    ])


def _convert_mapping(value, expected_type, path, context):
  """Converts `value` to a mapping with type `expected_type`."""
  if not isinstance(value, typing.Mapping):
    raise TypeError(f'{"".join(path)}: expected mapping, got '
                    f'{type(value).__name__!r}')
  key_type, value_type = type_annotations.get_generic_type_args(expected_type)
  return immutable_dict.ImmutableDict([
      (_convert_value(k, key_type, path + ('[<key>]',), context),
       _convert_value(v, value_type, path + (f'[{k!r}]',), context))
      for (k, v) in value.items()
  ])


def _convert_union(value, expected_type, path, context):
  """Converts `value` to a value with any of the types in `expected_type`."""
  for type_option in type_annotations.get_generic_type_args(expected_type):
    try:
      return _convert_value(value, type_option, path, context)
    except TypeError:
      pass
  raise TypeError(f'{"".join(path)}: expected {expected_type!r}, got '
                  f'{type(value).__name__!r}')


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
