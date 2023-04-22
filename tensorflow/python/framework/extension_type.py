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
"""User-defined ExtensionType classes."""

import abc
import typing

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
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
_IN_CONSTRUCTOR = '_tf_extension_type_in_constructor'


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
# Metaclass for tf.ExtensionType
# ==============================================================================
class ExtensionTypeMetaclass(abc.ABCMeta):
  """Metaclass for tf.ExtensionType types."""

  def __init__(cls, name, bases, namespace):
    # Don't transform base classes that are part of the framework -- only
    # transform user classes.  We identify classes that are part of the
    # framework by setting '_tf_extension_type_do_not_transform_this_class=True'
    # in the class definition.  (Note: we check for this in the class namespace,
    # so it is *not* ineherited.)
    if not namespace.get('_tf_extension_type_do_not_transform_this_class',
                         False):
      _check_field_annotations(cls)
      _add_extension_type_constructor(cls)
      _add_type_spec(cls)
    super(ExtensionTypeMetaclass, cls).__init__(name, bases, namespace)


# ==============================================================================
# Base class for user-defined types
# ==============================================================================
class ExtensionType(
    composite_tensor.CompositeTensor, metaclass=ExtensionTypeMetaclass):
  """Base class for TensorFlow `ExtensionType` classes.

  Tensorflow `ExtensionType` classes are specialized Python classes that can be
  used transparently with TensorFlow -- e.g., they can be used with ops
  such as `tf.cond` or `tf.while_loop` and used as inputs or outputs for
  `tf.function` and Keras layers.

  New `ExtensionType` classes are defined by creating a subclass of
  `tf.ExtensionType` that
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
  Composite Tensors    | `rt: tf.RaggdTensor`
  Extension Types      | `m: MyMaskedTensor`
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

  ExtensionType values are immutable -- i.e., once constructed, you can not
  modify or delete any of their instance members.

  ### Examples

  >>> class MaskedTensor(ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.TensorSpec(shape=None, dtype=tf.bool)

  >>> class Toy(ExtensionType):
  ...   name: str
  ...   price: ops.Tensor
  ...   features: typing.Mapping[str, ops.Tensor]

  >>> class ToyStore(ExtensionType):
  ...   name: str
  ...   toys: typing.Tuple[Toy, ...]
  """

  # Let the metaclass know that it should *not* transform this class (since
  # this class is part of the ExtensionType framework, and not a user class).
  _tf_extension_type_do_not_transform_this_class = True

  def __init__(self, *args, **kwargs):
    if type(self) is ExtensionType:  # pylint: disable=unidiomatic-typecheck
      raise AssertionError('ExtensionType is an abstract base class.')

  # This class variable is used to cache the return value for
  # _tf_extension_type_fields.
  _tf_extension_type_cached_fields = None

  @classmethod
  def _tf_extension_type_fields(cls):  # pylint: disable=no-self-argument
    """An ordered list describing the fields of this ExtensionType.

    Returns:
      A list of `ExtensionTypeField` objects.  Forward references are resolved
      if possible, or left unresolved otherwise.
    """
    if cls._tf_extension_type_cached_fields is not None:
      return cls._tf_extension_type_cached_fields

    try:
      type_hints = typing.get_type_hints(cls)
      ok_to_cache = True  # all forward references have been resolved.
    except (NameError, AttributeError):
      # Unresolved forward reference -- gather type hints manually.
      # * NameError comes from an annotation like `Foo` where class
      #   `Foo` hasn't been defined yet.
      # * AttributeError comes from an annotation like `foo.Bar`, where
      #   the module `foo` exists but `Bar` hasn't been defined yet.
      # Note: If a user attempts to instantiate a `ExtensionType` type that
      # still has unresolved forward references (e.g., because of a typo or a
      # missing import), then the constructor will raise an exception.
      type_hints = {}
      for base in reversed(cls.__mro__):
        type_hints.update(base.__dict__.get('__annotations__', {}))
      ok_to_cache = False

    fields = []
    for (name, value_type) in type_hints.items():
      default = getattr(cls, name,
                        extension_type_field.ExtensionTypeField.NO_DEFAULT)
      fields.append(
          extension_type_field.ExtensionTypeField(name, value_type, default))
    fields = tuple(fields)

    if ok_to_cache:
      cls._tf_extension_type_cached_fields = fields

    return fields

  @classmethod
  def _tf_extension_type_has_field(cls, name):
    return any(name == field.name for field in cls._tf_extension_type_fields())

  def _tf_extension_type_convert_fields(self):
    extension_type_field.convert_fields(self._tf_extension_type_fields(),
                                        self.__dict__)

  def __repr__(self):
    fields = ', '.join([
        f'{field.name}={getattr(self, field.name)!r}'
        for field in self._tf_extension_type_fields()
    ])
    return f'{type(self).__name__}({fields})'

  def __setattr__(self, name, value):
    if hasattr(self,
               _IN_CONSTRUCTOR) and self._tf_extension_type_has_field(name):
      self.__dict__[name] = value
    else:
      raise AttributeError('cannot assign to field %r' % name)

  def __delattr__(self, name):
    if hasattr(self,
               _IN_CONSTRUCTOR) and self._tf_extension_type_has_field(name):
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
  _tf_extension_type_cached_type_spec = None

  @property
  def _type_spec(self):  # CompositeTensor API.
    # Note: the TypeSpec contains all static (non-tensor) data from `self`.
    if self._tf_extension_type_cached_type_spec is None:
      self.__dict__[
          '_tf_extension_type_cached_type_spec'] = self.Spec.from_value(self)
    return self._tf_extension_type_cached_type_spec


# ==============================================================================
# Base class for the tf.ExtensionType TypeSpecs
# ==============================================================================
# TODO(b/184565242) Support custom TypeSpec constructors.
# TODO(b/184565242) Support custom TypeSpec methods & properties.
# TODO(b/184565242) Support custom TypeSpec validation.
# TODO(b/184565242) Support custom TypeSpec repr.
# TODO(b/184565242) Support customizing type relaxation for tracing.
# TODO(b/184565242) Support conversion to/from FullType


class ExtensionTypeSpec(type_spec.TypeSpec):
  """Base class for tf.ExtensionType TypeSpec."""

  def _serialize(self):  # TypeSpec API.
    return tuple(
        (f.name, _change_nested_mappings_to(self.__dict__[f.name], dict))
        for f in self._tf_extension_type_fields())

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
    # who defined the ExtensionType used a custom constructor.
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
    spec_fields.pop('_tf_extension_type_cached_type_spec', None)
    spec_fields.pop('_tf_extension_type_cached_fields', None)
    return _create_object_from_type_and_dict(cls, spec_fields)

  def __setattr__(self, name, value):
    if hasattr(self,
               _IN_CONSTRUCTOR) and self._tf_extension_type_has_field(name):
      self.__dict__[name] = value
    else:
      raise AttributeError('cannot assign to field %r' % name)

  def __delattr__(self, name):
    if hasattr(self,
               _IN_CONSTRUCTOR) and self._tf_extension_type_has_field(name):
      del self.__dict__[name]
    else:
      raise AttributeError('cannot delete field %r' % name)

  def __validate__(self):
    """Perform post-construction validation."""

  @classmethod
  def _tf_extension_type_fields(cls):
    return cls.value_type._tf_extension_type_fields()  # pylint: disable=protected-access

  @classmethod
  def _tf_extension_type_has_field(cls, name):
    return any(name == field.name for field in cls._tf_extension_type_fields())

  def _tf_extension_type_convert_fields(self):
    extension_type_field.convert_fields_for_spec(
        self._tf_extension_type_fields(), self.__dict__)

  def __repr__(self):
    fields = ', '.join([f'{k}={v!r}' for (k, v) in self._serialize()])
    return f'{type(self).__name__}({fields})'


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
# Helper methods for tf.ExtensionTypeMetaclass
# ==============================================================================


def _check_field_annotations(cls):
  """Validates the field annotations for tf.ExtensionType subclass `cls`."""
  # Check that no fields use reserved names.
  for name in cls.__dict__:
    if extension_type_field.ExtensionTypeField.is_reserved_name(name):
      raise ValueError(f"The field name '{name}' is reserved.")

  # Check that all fields have type annotaitons.
  annotations = getattr(cls, '__annotations__', {})
  for (key, value) in cls.__dict__.items():
    if not (key in annotations or callable(value) or key.startswith('_abc_') or
            key == '_tf_extension_type_fields' or key.startswith('__') and
            key.endswith('__') or isinstance(value, property)):
      raise ValueError('Field %s must have a type annotation' % key)


def _add_extension_type_constructor(cls):
  """Creates a constructor for a ExtensionType or ExtensionTypeSpec subclass."""
  if '__init__' in cls.__dict__:
    _wrap_user_constructor(cls)
  else:
    _build_extension_type_constructor(cls)


def _wrap_user_constructor(cls):
  """Wraps a user-defined constructor for tf.ExtensionType subclass `cls`."""
  user_constructor = cls.__init__

  def wrapped_init(self, *args, **kwargs):
    self.__dict__[_IN_CONSTRUCTOR] = True
    user_constructor(self, *args, **kwargs)
    del self.__dict__[_IN_CONSTRUCTOR]

    self._tf_extension_type_convert_fields()  # pylint: disable=protected-access
    self.__validate__()

  cls.__init__ = tf_decorator.make_decorator(user_constructor, wrapped_init)


# TODO(b/184565242) Consider using the templating system from autograph here.
def _build_extension_type_constructor(cls):
  """Builds a constructor for tf.ExtensionType subclass `cls`."""
  fields = cls._tf_extension_type_fields()  # pylint: disable=protected-access

  # Check that no-default fields don't follow default fields.  (Otherwise, we
  # can't build a well-formed constructor.)
  default_fields = []
  for field in fields:
    if field.default is not extension_type_field.ExtensionTypeField.NO_DEFAULT:
      default_fields.append(field.name)
    elif default_fields:
      raise ValueError(
          f'In definition for {cls.__name__}: Field without default '
          f'{field.name!r} follows field with default {default_fields[-1]!r}.  '
          f'Either add a default value for {field.name!r}, or move it before '
          f'{default_fields[0]!r} in the field annotations.')

  params = []
  kind = tf_inspect.Parameter.POSITIONAL_OR_KEYWORD
  for field in fields:
    if field.default is extension_type_field.ExtensionTypeField.NO_DEFAULT:
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
    self._tf_extension_type_convert_fields()  # pylint: disable=protected-access
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
  """Builds a constructor for ExtensionTypeSpec subclass `cls`."""
  params = []
  kind = tf_inspect.Parameter.POSITIONAL_OR_KEYWORD
  for field in cls._tf_extension_type_fields():  # pylint: disable=protected-access
    params.append(tf_inspect.Parameter(field.name, kind))

  signature = tf_inspect.Signature(params, return_annotation=cls.__name__)

  def __init__(self, *args, **kwargs):  # pylint: disable=invalid-name
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    self.__dict__.update(bound_args.arguments)
    self._tf_extension_type_convert_fields()  # pylint: disable=protected-access
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
  """Creates a nested TypeSpec class for tf.ExtensionType subclass `cls`."""
  # Build the TypeSpec class for this ExtensionType, and add it as a
  # nested class.
  spec_name = cls.__name__ + '.Spec'
  spec_dict = {'value_type': cls}
  spec = type(spec_name, (ExtensionTypeSpec,), spec_dict)
  setattr(cls, 'Spec', spec)

  # Build a constructor for the TypeSpec class.
  _build_spec_constructor(spec)

  cls.__abstractmethods__ -= {'_type_spec'}

  # If the user included an explicit `__name__` attribute, then use that to
  # register the TypeSpec (so it can be used in SavedModel signatures).
  if '__name__' in cls.__dict__:
    type_spec.register(cls.__dict__['__name__'] + '.Spec')(spec)


# ==============================================================================
# Anonymous ExtensionType
# ==============================================================================
class AnonymousExtensionType(ExtensionType):
  """Fallback used to decode `tf.ExtensionType` when the original type is unavailable.

  When a SavedModel is serialized, the signatures of any functions in the
  SavedModel can include `tf.ExtensionType` subclasses.  These subclasses are
  usually
  registered, so they can be restored when the SavedModel is loaded.  However,
  if a SavedModel is loaded without first registering the ExtensionType types in
  its
  signature, then the SavedModel will fall back to using the
  `AnonymousExtensionType`
  type instead.

  If necessary, `AnonymousExtensionType` objects can be converted to a concrete
  `tf.ExtensionType` subclass (and vice versa) using `reinterpret`.
  """

  # Let the metaclass know that it should *not* transform this class (since
  # this class is part of the ExtensionType framework, and not a user class).
  _tf_extension_type_do_not_transform_this_class = True

  def __init__(self, **fields):
    for name in fields:
      if (extension_type_field.ExtensionTypeField.is_reserved_name(name) or
          (name.startswith('__') and name.endswith('__'))):
        raise ValueError(f'The field name {name!r} is reserved.')
    fields = [(k, _convert_anonymous_fields(v)) for (k, v) in fields.items()]
    self.__dict__.update(fields)
    self._tf_extension_type_convert_fields()
    super().__init__()

  @classmethod
  def _tf_extension_type_fields(cls):
    return [
        extension_type_field.ExtensionTypeField(name, None)
        for name in cls.__dict__
        if not extension_type_field.ExtensionTypeField.is_reserved_name(name)
    ]

  def __setattr__(self, name, value):
    raise AttributeError('cannot assign to field %r' % name)

  def __delattr__(self, name):
    raise AttributeError('cannot delete field %r' % name)

  def _tf_extension_type_convert_fields(self):
    fields = [(k, _convert_anonymous_fields(v))
              for (k, v) in self.__dict__.items()
              if not extension_type_field.ExtensionTypeField.is_reserved_name(k)
             ]
    self.__dict__.update(fields)

  def __repr__(self):
    fields = [
        f'{k}={v!r}' for (k, v) in self.__dict__.items()
        if not extension_type_field.ExtensionTypeField.is_reserved_name(k)
    ]
    return f'AnonymousExtensionType({", ".join(fields)})'

  _tf_extension_type_cached_type_spec = None

  @property
  def _type_spec(self):  # CompositeTensor API.
    # Note: the TypeSpec contains all static (non-tensor) data from `self`.
    if self._tf_extension_type_cached_type_spec is None:
      spec = AnonymousExtensionTypeSpec.from_value(self)
      self.__dict__['_tf_extension_type_cached_type_spec'] = spec
    return self._tf_extension_type_cached_type_spec


@type_spec.register('tf.AnonymousExtensionType.Spec')
class AnonymousExtensionTypeSpec(ExtensionTypeSpec):
  """TypeSpec for AnonymousExtensionType."""

  def __init__(self, **fields):
    for name in fields:
      if (extension_type_field.ExtensionTypeField.is_reserved_name(name) or
          (name.startswith('__') and name.endswith('__'))):
        raise ValueError(f'The field name {name!r} is reserved.')
    fields = [(k, _convert_anonymous_fields(v, for_spec=True))
              for (k, v) in fields.items()]
    self.__dict__.update(fields)
    super().__init__()

  value_type = AnonymousExtensionType  # TypeSpec API.

  def _serialize(self):  # TypeSpec API.
    return tuple(
        (name, _change_nested_mappings_to(value, dict))
        for (name, value) in self.__dict__.items()
        if not extension_type_field.ExtensionTypeField.is_reserved_name(name))

  def __setattr__(self, name, value):
    raise AttributeError('cannot assign to field %r' % name)

  def __delattr__(self, name):
    raise AttributeError('cannot delete field %r' % name)


def _convert_anonymous_fields(value, for_spec=False):
  """Type-checks and converts `value` for inclusion in an AnonymousExtensionType."""
  if isinstance(value, (int, float, bool, str, bytes, type(None), dtypes.DType,
                        tensor_shape.TensorShape)):
    return value

  if isinstance(value, tuple):
    return tuple(_convert_anonymous_fields(v, for_spec) for v in value)

  if isinstance(value, typing.Mapping):
    return immutable_dict.ImmutableDict([
        (_convert_anonymous_fields(k, for_spec),
         _convert_anonymous_fields(v, for_spec)) for (k, v) in value.items()
    ])

  if (isinstance(value, (ops.Tensor, composite_tensor.CompositeTensor)) and
      not for_spec):
    return value

  if isinstance(value, type_spec.TypeSpec) and for_spec:
    return value

  raise ValueError(f'Unsupported field value: {value!r}')


# ==============================================================================
# reinterpret
# ==============================================================================
def reinterpret(value, new_type):
  """Converts a given `ExtensionType` to a new type with compatible fields.

  In particular, this can be used to convert a concrete subclass of
  `ExtensionType` to an `AnonymousExtensionType`, or vice versa.  When
  converting to a non-anonymous ExtensionType, field values are type-checked to
  ensure they are consistent with `new_type`'s type annotations, and validated
  with `new_type.__validate__`.

  Args:
    value: An instance of a subclass of `tf.ExtensionType`
    new_type: A subclass of `tf.ExtensionType`

  Returns:
    An instance of `new_type`, whose fields are copied from `value`.
  """
  if not isinstance(value, ExtensionType):
    raise ValueError(
        f'Expected `value` to be a tf.ExtensionType; got {value!r}')
  if not (isinstance(new_type, type) and issubclass(new_type, ExtensionType)):
    raise ValueError('Expected `new_type` to be a subclass of tf.ExtensionType;'
                     f' got {new_type!r}')

  fields = [
      item for item in value.__dict__.items()
      if not extension_type_field.ExtensionTypeField.is_reserved_name(item[0])
  ]
  new_value = _create_object_from_type_and_dict(new_type, fields)
  new_value._tf_extension_type_convert_fields()  # pylint: disable=protected-access
  new_value.__validate__()
  return new_value
