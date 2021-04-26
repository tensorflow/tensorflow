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
"""User-defined Struct classes."""

# Note: this module is not named `struct` to avoid a name clash with the
# standard Python `struct` module.

import abc
import typing

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import struct_field
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

# Attribute used to keep track of when we're inside a user-defined constructor
# (in which case the fields of `self` may be modified).
_IN_CONSTRUCTOR = '_tf_struct_in_constructor'


# ==============================================================================
# Utility functions
# ==============================================================================
def _create_object_from_type_and_dict(cls, obj_dict):
  """Creates an object, bypassing the constructor.

  Creates an object of type `cls`, whose `__dict__` is updated to contain
  `obj_dict`.

  Args:
    cls: The type of the new object.
    obj_dict: A `Mapping` that should be used to initialize the new object's
      `__dict__`.

  Returns:
    An object of type `cls`.
  """
  value = object.__new__(cls)
  value.__dict__.update(obj_dict)
  return value


# ==============================================================================
# Metaclass for tf.Struct
# ==============================================================================
class StructMetaclass(abc.ABCMeta):
  """Metaclass for tf.Struct types."""

  def __init__(cls, name, bases, namespace):
    # Don't transform the `Struct` base class itself (only transform
    # subclasses). We can't just check if `cls is Struct`, because the `Struct`
    # type hasn't been added to globals yet when this constructor is called. So
    # we instead check that `cls` has 3 ancestor classes (object,
    # CompositeTensor, and Struct).
    cls_is_struct = name == 'Struct' and len(cls.__mro__) == 3
    if not cls_is_struct:
      _check_field_annotations(cls)
      _add_struct_constructor(cls)
      _add_type_spec(cls)
    super(StructMetaclass, cls).__init__(name, bases, namespace)


# ==============================================================================
# Base class for user-defined structures
# ==============================================================================
class Struct(composite_tensor.CompositeTensor, metaclass=StructMetaclass):
  """Base class for TensorFlow `Struct` classes.

  Tensorflow `Struct` classes are specialized Python classes that can be
  used transparently with TensorFlow -- e.g., they can be used with ops
  such as `tf.cond` or `tf.while_loop` and used as inputs or outputs for
  `tf.function` and Keras layers.

  New `Struct` classes are defined by creating a subclass of `tf.Struct` that
  contains type annotations for all instance variables.  The following type
  annotations are supported:

  Type                 | Example
  -------------------- | --------------------------------------------
  Python integers      | `i: int`
  Python floats        | `f: float`
  Python strings       | `s: str`
  Python booleans      | `b: bool`
  Python None          | `n: None`
  Tensors              | `t: tf.Tensor`
  Struct types         | `rt: tf.RaggdTensor`
  Tensor shapes        | `shape: tf.TensorShape`
  Tensor dtypes        | `dtype: tf.DType`
  Type unions          | `length: typing.Union[int, float]`
  Tuples               | `params: typing.Tuple[int, float, int, int]`
  Tuples w/ Ellipsis   | `lengths: typing.Tuple[int, ...]`
  Mappings             | `tags: typing.Mapping[str, str]`
  TensorSpec instances | `t2: tf.TensorSpec(shape=[8, None], dtype=tf.int32)`
  TypeSpec instances   | `rt2: tf.RaggedTensorSpec(ragged_rank=2)`

  Fields annotated with `typing.Mapping` will be stored using an immutable
  mapping type.

  Due to technical limitations of Python's `typing` module, `TensorSpec`
  and `TypeSpec` instances may not currently be nested inside generic types
  (such as `typing.Union` or `typing.Tuple`).  TODO(b/184564088) Define
  tf generic types to avoid this limitation.

  Struct values are immutable -- i.e., once constructed, you can not modify or
  delete any of their instance members.

  ### Examples

  >>> class MaskedTensor(Struct):
  ...   values: tf.Tensor
  ...   mask: tf.TensorSpec(shape=None, dtype=tf.bool)

  >>> class Toy(Struct):
  ...   name: str
  ...   price: ops.Tensor
  ...   features: typing.Mapping[str, ops.Tensor]

  >>> class ToyStore(Struct):
  ...   name: str
  ...   toys: typing.Tuple[Toy, ...]
  """

  def __init__(self, *args, **kwargs):
    raise AssertionError('Struct is an abstract base class.')

  # This class variable is used to cache the return value for _tf_struct_fields.
  _tf_struct_cached_fields = None

  @classmethod
  def _tf_struct_fields(cls):  # pylint: disable=no-self-argument
    """An ordered list of `StructField`s describing the fields of this struct.

    Returns:
      A list of `StructField` objects.  Forward references are resolved if
      possible, or left unresolved otherwise.
    """
    if cls._tf_struct_cached_fields is not None:
      return cls._tf_struct_cached_fields

    try:
      type_hints = typing.get_type_hints(cls)
      ok_to_cache = True  # all forward references have been resolved.
    except (NameError, AttributeError):
      # Unresolved forward reference -- gather type hints manually.
      # * NameError comes from an annotation like `Foo` where class
      #   `Foo` hasn't been defined yet.
      # * AttributeError comes from an annotation like `foo.Bar`, where
      #   the module `foo` exists but `Bar` hasn't been defined yet.
      # Note: If a user attempts to instantiate a `Struct` type that still
      # has unresolved forward references (e.g., because of a typo or a
      # missing import), then the constructor will raise an exception.
      type_hints = {}
      for base in reversed(cls.__mro__):
        type_hints.update(base.__dict__.get('__annotations__', {}))
      ok_to_cache = False

    fields = []
    for (name, value_type) in type_hints.items():
      default = getattr(cls, name, struct_field.StructField.NO_DEFAULT)
      fields.append(struct_field.StructField(name, value_type, default))
    fields = tuple(fields)

    if ok_to_cache:
      cls._tf_struct_cached_fields = fields

    return fields

  @classmethod
  def _tf_struct_has_field(cls, name):
    return any(name == field.name for field in cls._tf_struct_fields())

  def _tf_struct_convert_fields(self):
    struct_field.convert_fields(self._tf_struct_fields(), self.__dict__)

  def __repr__(self):
    fields = ', '.join([
        f'{field.name}={getattr(self, field.name)!r}'
        for field in self._tf_struct_fields()
    ])
    return f'{type(self).__name__}({fields})'

  def __setattr__(self, name, value):
    if hasattr(self, _IN_CONSTRUCTOR) and self._tf_struct_has_field(name):
      self.__dict__[name] = value
    else:
      raise AttributeError('cannot assign to field %r' % name)

  def __delattr__(self, name):
    if hasattr(self, _IN_CONSTRUCTOR) and self._tf_struct_has_field(name):
      del self.__dict__[name]
    else:
      raise AttributeError('cannot delete field %r' % name)

  def __eq__(self, other):
    if type(self) is not type(other):
      return False

    if self._type_spec != other._type_spec:
      return False

    self_tensors = nest.flatten(self, expand_composites=True)
    other_tensors = nest.flatten(other, expand_composites=True)
    if len(self_tensors) != len(other_tensors):
      return False
    conditions = []
    for t1, t2 in zip(self_tensors, other_tensors):
      conditions.append(
          math_ops.reduce_all(
              gen_math_ops.equal(
                  array_ops.shape(t1),
                  array_ops.shape(t2),
                  incompatible_shape_error=False)))
      # Explicitly check shape (values that have different shapes but broadcast
      # to the same value are considered non-equal).
      conditions.append(
          math_ops.reduce_all(
              gen_math_ops.equal(t1, t2, incompatible_shape_error=False)))
    return math_ops.reduce_all(array_ops.stack(conditions))

  def __ne__(self, other):
    eq = self.__eq__(other)
    if isinstance(eq, ops.Tensor):
      return math_ops.logical_not(eq)
    else:
      return not eq

  def __validate__(self):
    """Perform post-construction validation."""

  # This instance variable is used to cache the value for the _type_spec
  # property.
  _tf_struct_cached_type_spec = None

  @property
  def _type_spec(self):  # CompositeTensor API.
    # Note: the TypeSpec contains all static (non-tensor) data from `self`.
    if self._tf_struct_cached_type_spec is None:
      self.__dict__['_tf_struct_cached_type_spec'] = self.Spec.from_value(self)
    return self._tf_struct_cached_type_spec


# ==============================================================================
# Base class for the tf.Struct TypeSpecs
# ==============================================================================
# TODO(b/184565242) Support custom TypeSpec constructors.
# TODO(b/184565242) Support custom TypeSpec methods & properties.
# TODO(b/184565242) Support custom TypeSpec validation.
# TODO(b/184565242) Support custom TypeSpec repr.
# TODO(b/184565242) Support customizing type relaxation for tracing.
# TODO(b/184565242) Support conversion to/from FullType


class StructSpec(type_spec.TypeSpec):
  """Base class for tf.Struct TypeSpec."""

  def _serialize(self):  # TypeSpec API.
    # TODO(b/184565242) Preserve the order of the fields in the TypeSpec?
    return _change_nested_mappings_to(self.__dict__, dict)

  @classmethod
  def _deserialize(cls, state):  # TypeSpec API.
    state = _change_nested_mappings_to(state, immutable_dict.ImmutableDict)
    return _create_object_from_type_and_dict(cls, state)

  def _to_components(self, value):  # TypeSpec API.
    tensor_or_composite = (ops.Tensor, composite_tensor.CompositeTensor)
    return tuple(
        x for x in nest.flatten(value.__dict__)
        if isinstance(x, tensor_or_composite))

  def _from_components(self, components):  # TypeSpec API.
    components_iter = iter(components)
    flat = [
        next(components_iter) if isinstance(x, type_spec.TypeSpec) else x
        for x in nest.flatten(self.__dict__)
    ]
    if list(components_iter):
      raise ValueError('Components do not match spec.')
    fields = nest.pack_sequence_as(self.__dict__, flat)

    # Build the new value.  Bypass the constructor (__init__), in case the user
    # who defined the struct type used a custom constructor.
    return _create_object_from_type_and_dict(self.value_type, fields)

  @property
  def _component_specs(self):  # TypeSpec API.
    components = []

    def push_if_type_spec(x):
      if isinstance(x, type_spec.TypeSpec):
        components.append(x)

    nest.map_structure(push_if_type_spec, self.__dict__)
    return tuple(components)

  @classmethod
  def from_value(cls, value):
    value_fields = value.__dict__
    spec_fields = nest.map_structure(_replace_tensor_with_spec, value_fields)
    return _create_object_from_type_and_dict(cls, spec_fields)

  def __setattr__(self, name, value):
    if hasattr(self, _IN_CONSTRUCTOR) and self._tf_struct_has_field(name):
      self.__dict__[name] = value
    else:
      raise AttributeError('cannot assign to field %r' % name)

  def __delattr__(self, name):
    if hasattr(self, _IN_CONSTRUCTOR) and self._tf_struct_has_field(name):
      del self.__dict__[name]
    else:
      raise AttributeError('cannot delete field %r' % name)

  def __validate__(self):
    """Perform post-construction validation."""

  @classmethod
  def _tf_struct_fields(cls):
    return cls.value_type._tf_struct_fields()  # pylint: disable=protected-access

  @classmethod
  def _tf_struct_has_field(cls, name):
    return any(name == field.name for field in cls._tf_struct_fields())

  def _tf_struct_convert_fields(self):
    struct_field.convert_fields_for_spec(self._tf_struct_fields(),
                                         self.__dict__)


def _replace_tensor_with_spec(value):
  if isinstance(value, ops.Tensor):
    # Note: we intentionally exclude `value.name` from the `TensorSpec`.
    return tensor_spec.TensorSpec(value.shape, value.dtype)
  if hasattr(value, '_type_spec'):
    return value._type_spec  # pylint: disable=protected-access
  return value


def _change_nested_mappings_to(value, new_type):
  """Recursively replace mappings with `new_type`."""
  if isinstance(value, (dict, immutable_dict.ImmutableDict)):
    return new_type([(k, _change_nested_mappings_to(v, new_type))
                     for (k, v) in value.items()])
  elif isinstance(value, tuple):
    return tuple(_change_nested_mappings_to(elt, new_type) for elt in value)
  else:
    return value


# ==============================================================================
# Helper methods for tf.StructMetaclass
# ==============================================================================


def _check_field_annotations(cls):
  """Validates the field annotations for tf.Struct subclass `cls`."""
  # Check that no fields use reserved names.
  for name in cls.__dict__:
    if struct_field.StructField.is_reserved_name(name):
      raise ValueError(f"The field name '{name}' is reserved.")

  # Check that all fields have type annotaitons.
  annotations = getattr(cls, '__annotations__', {})
  for (key, value) in cls.__dict__.items():
    if not (key in annotations or callable(value) or key.startswith('_abc_') or
            key == '_tf_struct_fields' or key.startswith('__') and
            key.endswith('__') or isinstance(value, property)):
      raise ValueError('Field %s must have a type annotation' % key)


def _add_struct_constructor(cls):
  """Creates a constructor for a Struct or StructSpec subclass."""
  if '__init__' in cls.__dict__:
    _wrap_user_constructor(cls)
  else:
    _build_struct_constructor(cls)


def _wrap_user_constructor(cls):
  """Wraps a user-defined constructor for tf.Struct subclass `cls`."""
  user_constructor = cls.__init__

  def wrapped_init(self, *args, **kwargs):
    self.__dict__[_IN_CONSTRUCTOR] = True
    user_constructor(self, *args, **kwargs)
    del self.__dict__[_IN_CONSTRUCTOR]

    self._tf_struct_convert_fields()  # pylint: disable=protected-access
    self.__validate__()

  cls.__init__ = tf_decorator.make_decorator(user_constructor, wrapped_init)


# TODO(b/184565242) Consider using the templating system from autograph here.
def _build_struct_constructor(cls):
  """Builds a constructor for tf.Struct subclass `cls`."""

  params = []
  kind = tf_inspect.Parameter.POSITIONAL_OR_KEYWORD
  for field in cls._tf_struct_fields():  # pylint: disable=protected-access
    if field.default is struct_field.StructField.NO_DEFAULT:
      default = tf_inspect.Parameter.empty
    else:
      default = field.default
    params.append(
        tf_inspect.Parameter(
            field.name, kind, default=default, annotation=field.value_type))

  signature = tf_inspect.Signature(params, return_annotation=cls.__name__)

  def __init__(self, *args, **kwargs):  # pylint: disable=invalid-name
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    self.__dict__.update(bound_args.arguments)
    self._tf_struct_convert_fields()  # pylint: disable=protected-access
    self.__validate__()

  # __signature__ is supported by some inspection/documentation tools
  # (but note: typing.get_type_hints does not respect __signature__).
  __init__.__signature__ = tf_inspect.Signature(
      [
          tf_inspect.Parameter('self',
                               tf_inspect.Parameter.POSITIONAL_OR_KEYWORD)
      ] + params,
      return_annotation=cls)

  cls.__init__ = __init__


def _build_spec_constructor(cls):
  """Builds a constructor for StructSpec subclass `cls`."""
  params = []
  kind = tf_inspect.Parameter.POSITIONAL_OR_KEYWORD
  for field in cls._tf_struct_fields():  # pylint: disable=protected-access
    params.append(tf_inspect.Parameter(field.name, kind))

  signature = tf_inspect.Signature(params, return_annotation=cls.__name__)

  def __init__(self, *args, **kwargs):  # pylint: disable=invalid-name
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    self.__dict__.update(bound_args.arguments)
    self._tf_struct_convert_fields()  # pylint: disable=protected-access
    self.__validate__()

  # __signature__ is supported by some inspection/documentation tools.
  __init__.__signature__ = tf_inspect.Signature(
      [
          tf_inspect.Parameter('self',
                               tf_inspect.Parameter.POSITIONAL_OR_KEYWORD)
      ] + params,
      return_annotation=cls)

  cls.__init__ = __init__


def _add_type_spec(cls):
  """Creates a nested TypeSpec class for tf.Struct subclass `cls`."""
  # Build the TypeSpec class for this struct type, and add it as a
  # nested class.
  spec_name = cls.__name__ + '.Spec'
  spec_dict = {'value_type': cls}
  spec = type(spec_name, (StructSpec,), spec_dict)
  setattr(cls, 'Spec', spec)

  # Build a constructor for the TypeSpec class.
  _build_spec_constructor(spec)

  cls.__abstractmethods__ -= {'_type_spec'}
