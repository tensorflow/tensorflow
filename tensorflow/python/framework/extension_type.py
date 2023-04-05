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
import warnings
import typing_extensions

from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

# Attribute used to keep track of when we're inside a user-defined constructor
# (in which case the fields of `self` may be modified).
_IN_CONSTRUCTOR = '_tf_extension_type_in_constructor'

_MUTABLE_KERAS_PROPERTIES = [
    # Keras uses _keras_mask property to pass the mask around
    '_keras_mask',
]


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
@tf_export('experimental.ExtensionType')
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

  Type                      | Example
  ------------------------- | --------------------------------------------
  Python integers           | `i: int`
  Python floats             | `f: float`
  Python strings            | `s: str`
  Python booleans           | `b: bool`
  Python None               | `n: None`
  Python tuple              | `params: tuple[int, float, int, int]`
  Python tuple w/ Ellipsis  | `lengths: tuple[int, ...]`
  Tensors                   | `t: tf.Tensor`
  Composite Tensors         | `rt: tf.RaggedTensor`
  Extension Types           | `m: MyMaskedTensor`
  Tensor shapes             | `shape: tf.TensorShape`
  Tensor dtypes             | `dtype: tf.DType`
  Type unions               | `length: typing.Union[int, float]`
  Tuples                    | `params: typing.Tuple[int, float, int, int]`
  Tuples w/ Ellipsis        | `lengths: typing.Tuple[int, ...]`
  Mappings                  | `tags: typing.Mapping[str, str]`

  Fields annotated with `typing.Mapping` will be stored using an immutable
  mapping type.

  ExtensionType values are immutable -- i.e., once constructed, you can not
  modify or delete any of their instance members.

  ### Examples

  >>> class MaskedTensor(ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor

  >>> class Toy(ExtensionType):
  ...   name: str
  ...   price: ops.Tensor
  ...   features: typing.Mapping[str, tf.Tensor]

  >>> class ToyStore(ExtensionType):
  ...   name: str
  ...   toys: typing.Tuple[Toy, ...]
  """

  # Let the metaclass know that it should *not* transform this class (since
  # this class is part of the ExtensionType framework, and not a user class).
  _tf_extension_type_do_not_transform_this_class = True

  def __init__(self, *args, **kwargs):
    if type(self) is ExtensionType:  # pylint: disable=unidiomatic-typecheck
      raise AssertionError('Cannot create an instance of ExtensionType '
                           'because ExtensionType is an abstract base class.')

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
    if '_tf_extension_type_cached_fields' in cls.__dict__:  # do not inherit.
      return cls._tf_extension_type_cached_fields

    try:
      # Using include_extras=False will replace all Annotated[T, ...] with T.
      # The typing_extensions module is used since this is only supported in
      # Python 3.9.
      type_hints = typing_extensions.get_type_hints(cls, include_extras=False)
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
    return f'{type(self).__qualname__}({fields})'

  def __setattr__(self, name, value):
    if (name in _MUTABLE_KERAS_PROPERTIES or
        (hasattr(self, _IN_CONSTRUCTOR) and
         self._tf_extension_type_has_field(name))):
      self.__dict__[name] = value
    else:
      raise AttributeError(f'Cannot mutate attribute `{name}` '
                           f'outside the custom constructor of ExtensionType.')

  def __delattr__(self, name):
    if (name in _MUTABLE_KERAS_PROPERTIES or
        (hasattr(self, _IN_CONSTRUCTOR) and
         self._tf_extension_type_has_field(name))):
      del self.__dict__[name]
    else:
      raise AttributeError(f'Cannot mutate attribute `{name}` '
                           f'outside the custom constructor of ExtensionType.')

  def __getattr__(self, name):
    if name in _MUTABLE_KERAS_PROPERTIES:
      return object.__getattribute__(self, name)
    if '_tf_extension_type_packed_variant' in self.__dict__:
      # Note: it's *not* ok to cache the results of unpack() here.  In
      # particular, it would be nice if we could do something like
      # `self.__dict__.update(unpack(self).__dict__)`, but that (potentially)
      # violates an invariant required by the `cond` operation.  E.g., if we had
      # `tf.cond(lambda: x.foo, lambda: x.bar)`, then tensor `x.bar` used in the
      # "else" branch would be created by an op in the "then" branch (when
      # looking up `x.foo`); and that's not allowed.
      return getattr(unpack(self), name)

    raise AttributeError(
        f'{type(self).__name__!r} object has no attribute {name!r}')

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
    return math_ops.reduce_all(array_ops_stack.stack(conditions))

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
      assert not is_packed(self)  # Packed version always caches TypeSpec.
      self.__dict__[
          '_tf_extension_type_cached_type_spec'] = self.Spec.from_value(self)
    return self._tf_extension_type_cached_type_spec


@tf_export('experimental.extension_type.as_dict')
def as_dict(value):
  """Extracts the attributes of `value` and their values to a dict format.

  Unlike `dataclasses.asdict()`, this function is not recursive and in case of
  nested `ExtensionType` objects, only the top level object is converted to a
  dict.

  Args:
    value: An `ExtensionType` object.

  Returns:
    A dict that contains the attributes of `value` and their values.
  """
  return {
      field.name: getattr(value, field.name)
      for field in value._tf_extension_type_fields()  # pylint: disable=protected-access
  }


def pack(value):
  """Returns a copy of `value` with fields packed in a single Variant.

  Args:
    value: An `ExtensionType` object.

  Returns:
    An `ExtensionType` object.
  """
  if is_packed(value):
    return value

  spec = value._type_spec._tf_extension_type_with_packed(True)  # pylint: disable=protected-access
  try:
    variant = composite_tensor_ops.composite_tensor_to_variants(value)
  except nested_structure_coder.NotEncodableError as e:
    # Note: the only time `_TypeSpecCodec.can_encode` returns False is if the
    # named type is not registered.  The default error message would simply
    # tell the user that there is no encoder for the object, so we provide
    # a more useful message letting them know how to register the type.
    raise ValueError('ExtensionTypes must have a __name__ field in order '
                     'to be packed.') from e

  return _create_object_from_type_and_dict(
      type(value), {
          '_tf_extension_type_cached_type_spec': spec,
          '_tf_extension_type_packed_variant': variant,
      })


def unpack(value):
  """Returns a copy of `value` with individual fields stored in __dict__.

  Args:
    value: An `ExtensionType` object.

  Returns:
    An `ExtensionType` object.
  """
  if not is_packed(value):
    return value

  # pylint: disable=protected-access
  variant = value._tf_extension_type_packed_variant
  spec = value._tf_extension_type_cached_type_spec
  spec = spec._tf_extension_type_with_packed(False)
  return composite_tensor_ops.composite_tensor_from_variant(variant, spec)


def is_packed(value):
  """Returns true if `value`'s fields are packed in a single Variant."""
  if not isinstance(value, ExtensionType):
    raise ValueError(f'Expected `value` to be an object of type ExtensionType,'
                     f'got an instance of {type(value)}.')
  return '_tf_extension_type_packed_variant' in value.__dict__


# ==============================================================================
# Base class for the tf.ExtensionType TypeSpecs
# ==============================================================================


class ExtensionTypeSpec(type_spec.TypeSpec):
  """Base class for tf.ExtensionType TypeSpec."""

  def _serialize(self):  # TypeSpec API.
    # Use a tuple of (name, value) pairs, to ensure we preserve field ordering.
    fields = [f.name for f in self._tf_extension_type_fields()]
    if self._tf_extension_type_is_packed:
      fields.append('_tf_extension_type_is_packed')
    return tuple(
        (f, _change_nested_mappings_to(self.__dict__[f], dict)) for f in fields)

  @classmethod
  def _deserialize(cls, state):  # TypeSpec API.
    state = _change_nested_mappings_to(state, immutable_dict.ImmutableDict)
    return _create_object_from_type_and_dict(cls, state)

  def __reduce__(self):
    # Use value_type instead of spec_type, as spec_type is a nested class.
    # Pickle support of nested class requries Pickle protocol version 4, which
    # is not enabled by default until py 3.8.
    #
    # https://www.python.org/dev/peps/pep-3154/#serializing-more-lookupable-objects
    # https://docs.python.org/3/library/pickle.html#pickle.DEFAULT_PROTOCOL
    return _deserialize_for_reduce, (self.value_type, self._serialize())

  def _to_components(self, value):  # TypeSpec API.
    if self._tf_extension_type_is_packed:
      return value._tf_extension_type_packed_variant  # pylint: disable=protected-access

    tensor_or_composite = (ops.Tensor, composite_tensor.CompositeTensor)
    # Retireve fields by the order of spec dict to preserve field ordering. This
    # is needed as nest.flatten would sort dictionary entries by key.
    value_tuple = tuple(value.__dict__[key] for key in self.__dict__)
    return tuple(
        x for x in nest.flatten(value_tuple)
        if isinstance(x, tensor_or_composite))

  def _from_components(self, components):  # TypeSpec API.
    if self._tf_extension_type_is_packed:
      return _create_object_from_type_and_dict(
          self.value_type, {
              '_tf_extension_type_cached_type_spec': self,
              '_tf_extension_type_packed_variant': components
          })

    spec_tuple = tuple(self.__dict__.values())
    components_iter = iter(components)
    flat = [
        next(components_iter) if isinstance(x, type_spec.TypeSpec) else x
        for x in nest.flatten(spec_tuple)
    ]
    if list(components_iter):
      raise ValueError(
          'Cannot build an ExtensionType instance from components '
          'because more components are provided than the number expected '
          'by the type spec.')
    value_tuple = nest.pack_sequence_as(spec_tuple, flat)
    fields = dict(zip(self.__dict__.keys(), value_tuple))

    # Build the new value.  Bypass the constructor (__init__), in case the user
    # who defined the ExtensionType used a custom constructor.
    return _create_object_from_type_and_dict(self.value_type, fields)

  @property
  def _component_specs(self):  # TypeSpec API.
    if self._tf_extension_type_is_packed:
      return tensor_spec.TensorSpec((), dtypes.variant)

    components = []

    def push_if_type_spec(x):
      if isinstance(x, type_spec.TypeSpec):
        components.append(x)

    nest.map_structure(push_if_type_spec, tuple(self.__dict__.values()))
    return tuple(components)

  @classmethod
  def from_value(cls, value):
    cached_spec = getattr(value, '_tf_extension_type_cached_type_spec', None)
    if cached_spec is not None:
      return cached_spec

    value_fields = value.__dict__
    spec_fields = nest.map_structure(_replace_tensor_with_spec, value_fields)
    spec_fields.pop('_tf_extension_type_cached_fields', None)
    return _create_object_from_type_and_dict(cls, spec_fields)

  def __setattr__(self, name, value):
    if (hasattr(self, _IN_CONSTRUCTOR) and
        self._tf_extension_type_has_field(name)):
      self.__dict__[name] = value
    else:
      raise AttributeError(
          f'Cannot mutate attribute `{name}` '
          f'outside the custom constructor of ExtensionTypeSpec.')

  def __delattr__(self, name):
    if (hasattr(self, _IN_CONSTRUCTOR) and
        self._tf_extension_type_has_field(name)):
      del self.__dict__[name]
    else:
      raise AttributeError(
          f'Cannot mutate attribute `{name}` '
          f'outside the custom constructor of ExtensionTypeSpec.')

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
    return f'{type(self).__qualname__}({fields})'

  _tf_extension_type_is_packed = False

  def _tf_extension_type_with_packed(self, value):
    """Returns a copy of this `TypeSpec` with `packed=value`.

    Args:
      value: A boolean value.

    Returns:
      A copy of `self` with `_tf_extension_type_is_packed=value`.
    """
    copy = _create_object_from_type_and_dict(type(self), self.__dict__)
    copy.__dict__['_tf_extension_type_is_packed'] = value
    return copy


class _ExtensionTypeSpecCodec:
  """Codec for `tf.ExtensionTypeSpec`."""

  def can_encode(self, pyobj):
    """Returns true if `pyobj` can be encoded as an ExtensionTypeSpec."""
    if isinstance(pyobj, ExtensionTypeSpec):
      try:
        type_spec_registry.get_name(type(pyobj))
        return True
      except ValueError:
        return False
    return False

  def do_encode(self, extension_type_spec_value, encode_fn):
    """Returns an encoded proto for the given `tf.ExtensionTypeSpec`."""
    type_spec_class_name = type_spec_registry.get_name(
        type(extension_type_spec_value))

    type_state = extension_type_spec_value._serialize()  # pylint: disable=protected-access
    num_flat_components = len(
        nest.flatten(
            extension_type_spec_value._component_specs, expand_composites=True))  # pylint: disable=protected-access
    encoded_type_spec = struct_pb2.StructuredValue()
    encoded_type_spec.type_spec_value.CopyFrom(
        struct_pb2.TypeSpecProto(
            type_spec_class=struct_pb2.TypeSpecProto.EXTENSION_TYPE_SPEC,
            type_state=encode_fn(type_state),
            type_spec_class_name=type_spec_class_name,
            num_flat_components=num_flat_components))
    return encoded_type_spec

  def can_decode(self, value):
    """Returns true if `value` can be decoded into a `tf.ExtensionTypeSpec`."""
    if value.HasField('type_spec_value'):
      type_spec_class_enum = value.type_spec_value.type_spec_class
      return (
          type_spec_class_enum == struct_pb2.TypeSpecProto.EXTENSION_TYPE_SPEC)
    return False

  def do_decode(self, value, decode_fn):
    """Returns the `tf.TypeSpec` encoded by the proto `value`."""
    type_spec_proto = value.type_spec_value
    class_name = type_spec_proto.type_spec_class_name

    try:
      type_spec_class = type_spec_registry.lookup(class_name)
    except ValueError:
      type_spec_class = AnonymousExtensionTypeSpec
      warnings.warn(
          f"The type '{class_name}' has not been registered. "
          'Falling back to using AnonymousExtensionTypeSpec '
          'instead.'
      )

    # pylint: disable=protected-access
    return type_spec_class._deserialize(decode_fn(type_spec_proto.type_state))


nested_structure_coder.register_codec(_ExtensionTypeSpecCodec())


@tf_export('experimental.ExtensionTypeBatchEncoder')
class ExtensionTypeBatchEncoder(type_spec.TypeSpecBatchEncoder):
  """Class used to encode and decode extension type values for batching.

  In order to be batched and unbatched by APIs such as `tf.data.Dataset`,
  `tf.keras`, and `tf.map_fn`, extension type values must be encoded as a list
  of `tf.Tensor`s, where stacking, unstacking, or concatenating these encoded
  tensors and then decoding the result must be equivalent to stacking,
  unstacking, or concatenating the original values. `ExtensionTypeBatchEncoder`s
  are responsible for implementing this encoding.

  The default `ExtensionTypeBatchEncoder` that is used by
  `BatchableExtensionType` assumes that extension type values can be stacked,
  unstacked, or concatenated by simply stacking, unstacking, or concatenating
  every nested `Tensor`, `ExtensionType`, `CompositeTensor`, and `TensorShape`
  field.

  Extension types where this is not the case will need to override
  `__batch_encoder__` with a custom encoder that overrides the `batch`,
  `unbatch`, `encode`, and `decode` methods. E.g.:

  >>> class CustomBatchEncoder(ExtensionTypeBatchEncoder):
  ...   pass # Override batch(), unbatch(), encode(), and decode().

  >>> class CustomType(BatchableExtensionType):
  ...   x: tf.Tensor
  ...   y: tf.Tensor
  ...   shape: tf.TensorShape
  ...   __batch_encoder__ = CustomBatchEncoder()

  For example, `tf.RaggedTensor` and `tf.SparseTensor` both use custom batch
  encodings which define ops to "box" and "unbox" individual values into
  `tf.variant` tensors.
  """

  def batch(self, spec, batch_size):
    """Returns the TypeSpec representing a batch of values described by `spec`.

    The default definition returns a `TypeSpec` that is equal to `spec`, except
    that an outer axis with size `batch_size` is added to every nested
    `TypeSpec` and `TensorShape` field.  Subclasses may override this default
    definition, when necessary.

    Args:
      spec: The `TypeSpec` for an individual value.
      batch_size: An `int` indicating the number of values that are batched
        together, or `None` if the batch size is not known.

    Returns:
      A `TypeSpec` for a batch of values.
    """

    def batch_field(f):
      if isinstance(f, type_spec.BatchableTypeSpec):
        return f.__batch_encoder__.batch(f, batch_size)
      elif isinstance(f, tensor_shape.TensorShape):
        return [batch_size] + f
      else:
        return f

    fields = tuple(spec.__dict__.items())
    batched_fields = nest.map_structure(batch_field, fields)
    return _create_object_from_type_and_dict(type(spec), batched_fields)

  def unbatch(self, spec):
    """Returns the TypeSpec for a single unbatched element in `spec`.

    The default definition returns a `TypeSpec` that is equal to `spec`, except
    that the outermost axis is removed from every nested `TypeSpec`, and
    `TensorShape` field.  Subclasses may override this default definition, when
    necessary.

    Args:
      spec: The `TypeSpec` for a batch of values.

    Returns:
      A `TypeSpec` for an individual value.
    """

    def unbatch_field(f):
      if isinstance(f, type_spec.BatchableTypeSpec):
        return f.__batch_encoder__.unbatch(f)
      elif isinstance(f, tensor_shape.TensorShape):
        return f[1:]
      else:
        return f

    fields = tuple(spec.__dict__.items())
    unbatched_fields = nest.map_structure(unbatch_field, fields)
    return _create_object_from_type_and_dict(type(spec), unbatched_fields)

  def encode(self, spec, value, minimum_rank=0):
    """Encodes `value` as a nest of batchable Tensors or CompositeTensors.

    The default definition returns a flat tuple of all the `Tensor`s,
    `CompositeTensor`s, and `ExtensionType`s from a depth-first traversal of
    `value`'s fields. Subclasses may override this default definition, when
    necessary.

    Args:
      spec: The TypeSpec of the value to encode.
      value: A value compatible with `spec`.
      minimum_rank: The minimum rank for the returned Tensors, CompositeTensors,
        and ExtensionType values.  This can be used to ensure that the encoded
        values can be unbatched this number of times.   If `minimum_rank>0`,
        then `t.shape[:minimum_rank]` must be compatible for all values `t`
        returned by `encode`.

    Returns:
      A nest (as defined by `tf.nest`) of `tf.Tensor`s, batchable
      `tf.CompositeTensor`s, or `tf.ExtensionType`s.  Stacking, unstacking, or
      concatenating these encoded values and then decoding the result must be
      equivalent to stacking, unstacking, or concatenating the original values.
    """
    return spec._to_components(value)  # pylint: disable=protected-access

  def decode(self, spec, encoded_value):
    """Decodes `value` from a batchable tensor encoding.

    See `encode` for a description of the default encoding.  Subclasses may
    override this default definition, when necessary.

    Args:
      spec: The TypeSpec for the result value.  If encoded values with spec `s`
        were batched, then `spec` should be `s.batch(batch_size)`; or if encoded
        values with spec `s` were unbatched, then `spec` should be
        `s.unbatch()`.
      encoded_value: A nest of values returned by `encode`; or a nest of
        values that was formed by stacking, unstacking, or concatenating the
        corresponding elements of values returned by `encode`.

    Returns:
      A value compatible with `type_spec`.
    """
    return spec._from_components(encoded_value)  # pylint: disable=protected-access

  def encoding_specs(self, spec):
    """Returns a list of `TensorSpec`(s) describing the encoding for `spec`.

    See `encode` for a description of the default encoding.  Subclasses may
    override this default definition, when necessary.

    Args:
      spec: The TypeSpec whose encoding should be described.

    Returns:
      A nest (as defined by `tf.nest) of `tf.TypeSpec`, describing the values
      that are returned by `self.encode(spec, ...)`.  All TypeSpecs in this
      nest must be batchable.
    """
    return spec._component_specs  # pylint: disable=protected-access


class BatchableExtensionTypeSpec(ExtensionTypeSpec,
                                 type_spec.BatchableTypeSpec):
  """Base class for TypeSpecs for BatchableExtensionTypes."""

  __batch_encoder__ = ExtensionTypeBatchEncoder()

  def _batch(self, batch_size):
    return self.__batch_encoder__.batch(self, batch_size)

  def _unbatch(self):
    return self.__batch_encoder__.unbatch(self)

  def _to_tensor_list(self, value):
    return type_spec.batchable_to_tensor_list(self, value)

  def _to_batched_tensor_list(self, value):
    return type_spec.batchable_to_tensor_list(self, value, minimum_rank=1)

  def _from_compatible_tensor_list(self, tensor_list):
    return type_spec.batchable_from_tensor_list(self, tensor_list)

  @property
  def _flat_tensor_specs(self):
    return type_spec.get_batchable_flat_tensor_specs(self)


@tf_export('experimental.BatchableExtensionType')
class BatchableExtensionType(ExtensionType):
  """An ExtensionType that can be batched and unbatched.

  `BatchableExtensionType`s can be used with APIs that require batching or
  unbatching, including `Keras`, `tf.data.Dataset`, and `tf.map_fn`.  E.g.:

  >>> class Vehicle(tf.experimental.BatchableExtensionType):
  ...   top_speed: tf.Tensor
  ...   mpg: tf.Tensor
  >>> batch = Vehicle([120, 150, 80], [30, 40, 12])
  >>> tf.map_fn(lambda vehicle: vehicle.top_speed * vehicle.mpg, batch,
  ...           fn_output_signature=tf.int32).numpy()
  array([3600, 6000,  960], dtype=int32)

  An `ExtensionTypeBatchEncoder` is used by these APIs to encode `ExtensionType`
  values. The default encoder assumes that values can be stacked, unstacked, or
  concatenated by simply stacking, unstacking, or concatenating every nested
  `Tensor`, `ExtensionType`, `CompositeTensor`, or `TensorShape` field.
  Extension types where this is not the case will need to override
  `__batch_encoder__` with a custom `ExtensionTypeBatchEncoder`.  See
  `tf.experimental.ExtensionTypeBatchEncoder` for more details.
  """
  # Let the metaclass know that it should *not* transform this class (since
  # this class is part of the ExtensionType framework, and not a user class).
  _tf_extension_type_do_not_transform_this_class = True


# For Pickle __reduce__ protocol:
def _deserialize_for_reduce(value_type, serialization):
  return value_type.Spec._deserialize(serialization)  # pylint: disable=protected-access


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
  annotations = getattr(cls, '__annotations__', {})

  # Check that no fields use reserved names.
  for name, value in cls.__dict__.items():
    if name == 'Spec':
      if not isinstance(value, type):
        raise ValueError(f'{cls.__qualname__}.Spec must be a nested class; '
                         f'got {value}.')
      if (value.__bases__ != (type_spec.TypeSpec,) and value.__bases__ !=
          (object,)):
        raise ValueError(f'{cls.__qualname__}.Spec must be directly subclassed '
                         'from tf.TypeSpec.')
    elif extension_type_field.ExtensionTypeField.is_reserved_name(name):
      raise ValueError(f'The field annotations for {cls.__name__} are '
                       f"invalid. Field '{name}' is reserved.")
  for name in annotations:
    if extension_type_field.ExtensionTypeField.is_reserved_name(name):
      raise ValueError(f'The field annotations for {cls.__name__} are '
                       f"invalid. Field '{name}' is reserved.")

  # Check that all fields have type annotaitons.
  for (key, value) in cls.__dict__.items():
    if not (key in annotations or callable(value) or key.startswith('_abc_') or
            key == '_tf_extension_type_fields' or
            key.startswith('__') and key.endswith('__') or
            isinstance(value, (property, classmethod, staticmethod))):
      raise ValueError(f'The field annotations for {cls.__name__} are '
                       f'invalid. Field {key} is missing a type annotation.')


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


_NO_DEFAULT = extension_type_field.ExtensionTypeField.NO_DEFAULT


def _build_extension_type_constructor(cls):
  """Builds a constructor for tf.ExtensionType subclass `cls`."""
  fields = cls._tf_extension_type_fields()  # pylint: disable=protected-access

  # Mark any no-default fields that follow default fields as keyword_only.
  got_default = False
  keyword_only_start = len(fields)
  for i in range(len(fields)):
    if got_default:
      if fields[i].default is _NO_DEFAULT:
        keyword_only_start = i
        break
    elif fields[i].default is not _NO_DEFAULT:
      got_default = True

  params = []
  for i, field in enumerate(fields):
    if i < keyword_only_start:
      kind = tf_inspect.Parameter.POSITIONAL_OR_KEYWORD
    else:
      kind = tf_inspect.Parameter.KEYWORD_ONLY
    if field.default is _NO_DEFAULT:
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
  spec_name = cls.__name__ + '.Spec'
  spec_qualname = cls.__qualname__ + '.Spec'

  # Set __module__ explicitly as a dynamic created class has module='abc'
  # by default.
  spec_dict = {'value_type': cls, '__module__': cls.__module__}

  # Copy user-supplied customizations into the TypeSpec.
  user_spec = cls.__dict__.get('Spec', None)
  if user_spec is not None:
    for (name, value) in user_spec.__dict__.items():
      if extension_type_field.ExtensionTypeField.is_reserved_name(name):
        raise ValueError(f'TypeSpec {spec_qualname} uses reserved '
                         f"name '{name}'.")
      if cls._tf_extension_type_has_field(name):  # pylint: disable=protected-access
        raise ValueError(f"TypeSpec {spec_qualname} defines a variable '{name}'"
                         f' which shadows a field in {cls.__qualname__}')
      if name in ('__module__', '__dict__', '__weakref__'):
        continue

      spec_dict[name] = value

  if issubclass(cls, BatchableExtensionType):
    type_spec_base = BatchableExtensionTypeSpec
    if hasattr(cls,
               '__batch_encoder__') and '__batch_encoder__' not in spec_dict:
      spec_dict['__batch_encoder__'] = cls.__batch_encoder__
  else:
    type_spec_base = ExtensionTypeSpec
    if hasattr(cls, '__batch_encoder__') or '__batch_encoder__' in spec_dict:
      raise ValueError('__batch_encoder__ should only be defined for '
                       'BatchableExtensionType classes.')

  # Build the TypeSpec and store it as a nested class inside `cls`.
  spec = type(spec_name, (type_spec_base,), spec_dict)
  spec.__qualname__ = spec_qualname
  setattr(cls, 'Spec', spec)

  # Build a constructor for the TypeSpec class.
  if '__init__' in spec.__dict__:
    _wrap_user_constructor(spec)
  else:
    _build_spec_constructor(spec)

  cls.__abstractmethods__ -= {'_type_spec'}

  # If the user included an explicit `__name__` attribute, then use that to
  # register the TypeSpec (so it can be used in SavedModel signatures).
  if '__name__' in cls.__dict__:
    type_spec_registry.register(cls.__dict__['__name__'] + '.Spec')(spec)


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
        raise ValueError(
            f'Reserved field name {name} was encountered '
            f'when trying to instantiate an AnonymousExtensionType.')
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
    raise AttributeError(f'Cannot set attribute `{name}`. '
                         f'AnonymousExtensionType instances are immutable.')

  def __delattr__(self, name):
    raise AttributeError(f'Cannot delete attribute `{name}`. '
                         f'AnonymousExtensionType instances are immutable.')

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


@type_spec_registry.register('tf.AnonymousExtensionType.Spec')
class AnonymousExtensionTypeSpec(ExtensionTypeSpec):
  """TypeSpec for AnonymousExtensionType."""

  def __init__(self, **fields):
    for name in fields:
      if (extension_type_field.ExtensionTypeField.is_reserved_name(name) or
          (name.startswith('__') and name.endswith('__'))):
        raise ValueError(
            f'Reserved field name {name} was encountered '
            f'when trying to instantiate an AnonymousExtensionTypeSpec.')
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
    raise AttributeError(f'Cannot set attribute `{name}`. '
                         f'AnonymousExtensionTypeSpec instances are immutable.')

  def __delattr__(self, name):
    raise AttributeError(f'Cannot delete attribute `{name}`. '
                         f'AnonymousExtensionTypeSpec instances are immutable.')


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

  raise ValueError(f'Cannot convert anonymous fields from '
                   f'an unsupported `value` argument: {value!r}.')


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
        f'reinterpret expects `value` to be a tf.ExtensionType instance; '
        f'got {value!r}')
  if not (isinstance(new_type, type) and issubclass(new_type, ExtensionType)):
    raise ValueError(
        f'reinterpret expects `new_type` to be a subclass of tf.ExtensionType; '
        f'got {new_type!r}')

  fields = [
      item for item in value.__dict__.items()
      if not extension_type_field.ExtensionTypeField.is_reserved_name(item[0])
  ]
  new_value = _create_object_from_type_and_dict(new_type, fields)
  new_value._tf_extension_type_convert_fields()  # pylint: disable=protected-access
  new_value.__validate__()
  return new_value
