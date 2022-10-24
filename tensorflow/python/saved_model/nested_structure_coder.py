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
"""Module that encodes (decodes) nested structures into (from) protos.

The intended use is to serialize everything needed to restore a `Function` that
was saved into a SavedModel. This may include concrete function inputs and
outputs, signatures, function specs, etc.

Example use:
# Encode into proto.
signature_proto = nested_structure_coder.encode_structure(
    function.input_signature)
# Decode into a Python object.
restored_signature = nested_structure_coder.decode_proto(signature_proto)
"""

import collections
import functools
import warnings

from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import values
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import row_partition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


class NotEncodableError(Exception):
  """Error raised when a coder cannot encode an object."""


def register_codec(x):
  """Registers a codec to use for encoding/decoding.

  Args:
    x: The codec object to register. The object must implement can_encode,
      do_encode, can_decode, and do_decode. See the various _*Codec classes for
      examples.
  """
  _codecs.append(x)


def _get_encoders():
  return [(c.can_encode, c.do_encode) for c in _codecs]


def _get_decoders():
  return [(c.can_decode, c.do_decode) for c in _codecs]


def _map_structure(pyobj, coders):
  for can, do in coders:
    if can(pyobj):
      recursion_fn = functools.partial(_map_structure, coders=coders)
      return do(pyobj, recursion_fn)
  raise NotEncodableError(
      f"No encoder for object {str(pyobj)} of type {type(pyobj)}.")


@tf_export("__internal__.saved_model.encode_structure", v1=[])
def encode_structure(nested_structure):
  """Encodes nested structures composed of encodable types into a proto.

  Args:
    nested_structure: Structure to encode.

  Returns:
    Encoded proto.

  Raises:
    NotEncodableError: For values for which there are no encoders.
  """
  return _map_structure(nested_structure, _get_encoders())


def can_encode(nested_structure):
  """Determines whether a nested structure can be encoded into a proto.

  Args:
    nested_structure: Structure to encode.

  Returns:
    True if the nested structured can be encoded.
  """
  try:
    encode_structure(nested_structure)
  except NotEncodableError:
    return False
  return True


@tf_export("__internal__.saved_model.decode_proto", v1=[])
def decode_proto(proto):
  """Decodes proto representing a nested structure.

  Args:
    proto: Proto to decode.

  Returns:
    Decoded structure.

  Raises:
    NotEncodableError: For values for which there are no encoders.
  """
  return _map_structure(proto, _get_decoders())


class _ListCodec:
  """Codec for lists."""

  def can_encode(self, pyobj):
    return isinstance(pyobj, list)

  def do_encode(self, list_value, encode_fn):
    encoded_list = struct_pb2.StructuredValue()
    encoded_list.list_value.CopyFrom(struct_pb2.ListValue())
    for element in list_value:
      encoded_list.list_value.values.add().CopyFrom(encode_fn(element))
    return encoded_list

  def can_decode(self, value):
    return value.HasField("list_value")

  def do_decode(self, value, decode_fn):
    return [decode_fn(element) for element in value.list_value.values]


def _is_tuple(obj):
  return not _is_named_tuple(obj) and isinstance(obj, tuple)


def _is_named_tuple(instance):
  """Returns True iff `instance` is a `namedtuple`.

  Args:
    instance: An instance of a Python object.

  Returns:
    True if `instance` is a `namedtuple`.
  """
  if not isinstance(instance, tuple):
    return False
  return (hasattr(instance, "_fields") and
          isinstance(instance._fields, collections_abc.Sequence) and
          all(isinstance(f, str) for f in instance._fields))


class _TupleCodec:
  """Codec for tuples."""

  def can_encode(self, pyobj):
    return _is_tuple(pyobj)

  def do_encode(self, tuple_value, encode_fn):
    encoded_tuple = struct_pb2.StructuredValue()
    encoded_tuple.tuple_value.CopyFrom(struct_pb2.TupleValue())
    for element in tuple_value:
      encoded_tuple.tuple_value.values.add().CopyFrom(encode_fn(element))
    return encoded_tuple

  def can_decode(self, value):
    return value.HasField("tuple_value")

  def do_decode(self, value, decode_fn):
    return tuple(decode_fn(element) for element in value.tuple_value.values)


class _DictCodec:
  """Codec for dicts."""

  def can_encode(self, pyobj):
    return isinstance(pyobj, dict)

  def do_encode(self, dict_value, encode_fn):
    encoded_dict = struct_pb2.StructuredValue()
    encoded_dict.dict_value.CopyFrom(struct_pb2.DictValue())
    for key, value in dict_value.items():
      encoded_dict.dict_value.fields[key].CopyFrom(encode_fn(value))
    return encoded_dict

  def can_decode(self, value):
    return value.HasField("dict_value")

  def do_decode(self, value, decode_fn):
    return {key: decode_fn(val) for key, val in value.dict_value.fields.items()}


class _NamedTupleCodec:
  """Codec for namedtuples.

  Encoding and decoding a namedtuple reconstructs a namedtuple with a different
  actual Python type, but with the same `typename` and `fields`.
  """

  def can_encode(self, pyobj):
    return _is_named_tuple(pyobj)

  def do_encode(self, named_tuple_value, encode_fn):
    encoded_named_tuple = struct_pb2.StructuredValue()
    encoded_named_tuple.named_tuple_value.CopyFrom(struct_pb2.NamedTupleValue())
    encoded_named_tuple.named_tuple_value.name = \
      named_tuple_value.__class__.__name__
    for key in named_tuple_value._fields:
      pair = encoded_named_tuple.named_tuple_value.values.add()
      pair.key = key
      pair.value.CopyFrom(encode_fn(named_tuple_value._asdict()[key]))
    return encoded_named_tuple

  def can_decode(self, value):
    return value.HasField("named_tuple_value")

  def do_decode(self, value, decode_fn):
    key_value_pairs = value.named_tuple_value.values
    items = [(pair.key, decode_fn(pair.value)) for pair in key_value_pairs]
    named_tuple_type = collections.namedtuple(value.named_tuple_value.name,
                                              [item[0] for item in items])
    return named_tuple_type(**dict(items))


class _Float64Codec:
  """Codec for floats."""

  def can_encode(self, pyobj):
    return isinstance(pyobj, float)

  def do_encode(self, float64_value, encode_fn):
    del encode_fn
    value = struct_pb2.StructuredValue()
    value.float64_value = float64_value
    return value

  def can_decode(self, value):
    return value.HasField("float64_value")

  def do_decode(self, value, decode_fn):
    del decode_fn
    return value.float64_value


class _Int64Codec:
  """Codec for Python integers (limited to 64 bit values)."""

  def can_encode(self, pyobj):
    return not isinstance(pyobj, bool) and isinstance(pyobj, int)

  def do_encode(self, int_value, encode_fn):
    del encode_fn
    value = struct_pb2.StructuredValue()
    value.int64_value = int_value
    return value

  def can_decode(self, value):
    return value.HasField("int64_value")

  def do_decode(self, value, decode_fn):
    del decode_fn
    return int(value.int64_value)


class _StringCodec:
  """Codec for strings.

  See StructuredValue.string_value in proto/struct.proto for more detailed
  explanation.
  """

  def can_encode(self, pyobj):
    return isinstance(pyobj, str)

  def do_encode(self, string_value, encode_fn):
    del encode_fn
    value = struct_pb2.StructuredValue()
    value.string_value = string_value
    return value

  def can_decode(self, value):
    return value.HasField("string_value")

  def do_decode(self, value, decode_fn):
    del decode_fn
    return compat.as_str(value.string_value)


class _NoneCodec:
  """Codec for None."""

  def can_encode(self, pyobj):
    return pyobj is None

  def do_encode(self, none_value, encode_fn):
    del encode_fn, none_value
    value = struct_pb2.StructuredValue()
    value.none_value.CopyFrom(struct_pb2.NoneValue())
    return value

  def can_decode(self, value):
    return value.HasField("none_value")

  def do_decode(self, value, decode_fn):
    del decode_fn, value
    return None


class _BoolCodec:
  """Codec for booleans."""

  def can_encode(self, pyobj):
    return isinstance(pyobj, bool)

  def do_encode(self, bool_value, encode_fn):
    del encode_fn
    value = struct_pb2.StructuredValue()
    value.bool_value = bool_value
    return value

  def can_decode(self, value):
    return value.HasField("bool_value")

  def do_decode(self, value, decode_fn):
    del decode_fn
    return value.bool_value


class _TensorShapeCodec:
  """Codec for `TensorShape`."""

  def can_encode(self, pyobj):
    return isinstance(pyobj, tensor_shape.TensorShape)

  def do_encode(self, tensor_shape_value, encode_fn):
    del encode_fn
    encoded_tensor_shape = struct_pb2.StructuredValue()
    encoded_tensor_shape.tensor_shape_value.CopyFrom(
        tensor_shape_value.as_proto())
    return encoded_tensor_shape

  def can_decode(self, value):
    return value.HasField("tensor_shape_value")

  def do_decode(self, value, decode_fn):
    del decode_fn
    return tensor_shape.TensorShape(value.tensor_shape_value)


class _TensorTypeCodec:
  """Codec for `TensorType`."""

  def can_encode(self, pyobj):
    return isinstance(pyobj, dtypes.DType)

  def do_encode(self, tensor_dtype_value, encode_fn):
    del encode_fn
    encoded_tensor_type = struct_pb2.StructuredValue()
    encoded_tensor_type.tensor_dtype_value = tensor_dtype_value.as_datatype_enum
    return encoded_tensor_type

  def can_decode(self, value):
    return value.HasField("tensor_dtype_value")

  def do_decode(self, value, decode_fn):
    del decode_fn
    return dtypes.DType(value.tensor_dtype_value)


class _TensorSpecCodec:
  """Codec for `TensorSpec`."""

  def can_encode(self, pyobj):
    # BoundedTensorSpec has its own decoder.
    return (isinstance(pyobj, tensor_spec.TensorSpec) and
            not isinstance(pyobj, tensor_spec.BoundedTensorSpec))

  def do_encode(self, tensor_spec_value, encode_fn):
    encoded_tensor_spec = struct_pb2.StructuredValue()
    encoded_tensor_spec.tensor_spec_value.CopyFrom(
        struct_pb2.TensorSpecProto(
            shape=encode_fn(tensor_spec_value.shape).tensor_shape_value,
            dtype=encode_fn(tensor_spec_value.dtype).tensor_dtype_value,
            name=tensor_spec_value.name))
    return encoded_tensor_spec

  def can_decode(self, value):
    return value.HasField("tensor_spec_value")

  def do_decode(self, value, decode_fn):
    name = value.tensor_spec_value.name
    return tensor_spec.TensorSpec(
        shape=decode_fn(
            struct_pb2.StructuredValue(
                tensor_shape_value=value.tensor_spec_value.shape)),
        dtype=decode_fn(
            struct_pb2.StructuredValue(
                tensor_dtype_value=value.tensor_spec_value.dtype)),
        name=(name if name else None))


class _BoundedTensorSpecCodec:
  """Codec for `BoundedTensorSpec`."""

  def can_encode(self, pyobj):
    return isinstance(pyobj, tensor_spec.BoundedTensorSpec)

  def do_encode(self, bounded_tensor_spec_value, encode_fn):
    """Returns an encoded proto for the given `tf.BoundedTensorSpec`."""
    encoded_bounded_tensor_spec = struct_pb2.StructuredValue()
    encoded_bounded_tensor_spec.bounded_tensor_spec_value.CopyFrom(
        struct_pb2.BoundedTensorSpecProto(
            shape=encode_fn(bounded_tensor_spec_value.shape).tensor_shape_value,
            dtype=encode_fn(bounded_tensor_spec_value.dtype).tensor_dtype_value,
            name=bounded_tensor_spec_value.name,
            minimum=tensor_util.make_tensor_proto(
                bounded_tensor_spec_value.minimum),
            maximum=tensor_util.make_tensor_proto(
                bounded_tensor_spec_value.maximum)))
    return encoded_bounded_tensor_spec

  def can_decode(self, value):
    return value.HasField("bounded_tensor_spec_value")

  def do_decode(self, value, decode_fn):
    btsv = value.bounded_tensor_spec_value
    name = btsv.name
    return tensor_spec.BoundedTensorSpec(
        shape=decode_fn(
            struct_pb2.StructuredValue(tensor_shape_value=btsv.shape)),
        dtype=decode_fn(
            struct_pb2.StructuredValue(tensor_dtype_value=btsv.dtype)),
        minimum=tensor_util.MakeNdarray(btsv.minimum),
        maximum=tensor_util.MakeNdarray(btsv.maximum),
        name=(name if name else None))


# TODO(b/238903802): Use TraceType serialization and specific protos.
class _TypeSpecCodec:
  """Codec for `tf.TypeSpec`."""

  # Mapping from enum value to type (TypeSpec subclass).
  TYPE_SPEC_CLASS_FROM_PROTO = {
      struct_pb2.TypeSpecProto.SPARSE_TENSOR_SPEC:
          sparse_tensor.SparseTensorSpec,
      struct_pb2.TypeSpecProto.INDEXED_SLICES_SPEC:
          indexed_slices.IndexedSlicesSpec,
      struct_pb2.TypeSpecProto.RAGGED_TENSOR_SPEC:
          ragged_tensor.RaggedTensorSpec,
      struct_pb2.TypeSpecProto.TENSOR_ARRAY_SPEC:
          tensor_array_ops.TensorArraySpec,
      struct_pb2.TypeSpecProto.DATA_DATASET_SPEC:
          dataset_ops.DatasetSpec,
      struct_pb2.TypeSpecProto.DATA_ITERATOR_SPEC:
          iterator_ops.IteratorSpec,
      struct_pb2.TypeSpecProto.OPTIONAL_SPEC:
          optional_ops.OptionalSpec,
      struct_pb2.TypeSpecProto.PER_REPLICA_SPEC:
          values.PerReplicaSpec,
      struct_pb2.TypeSpecProto.VARIABLE_SPEC:
          resource_variable_ops.VariableSpec,
      struct_pb2.TypeSpecProto.ROW_PARTITION_SPEC:
          row_partition.RowPartitionSpec,
  }

  # Mapping from type (TypeSpec subclass) to enum value.
  TYPE_SPEC_CLASS_TO_PROTO = dict(
      (cls, enum) for (enum, cls) in TYPE_SPEC_CLASS_FROM_PROTO.items())

  def can_encode(self, pyobj):
    """Returns true if `pyboj` can be encoded as a TypeSpec."""
    if type(pyobj) in self.TYPE_SPEC_CLASS_TO_PROTO:  # pylint: disable=unidiomatic-typecheck
      return True

    # Check if it's a registered type.
    if isinstance(pyobj, type_spec.TypeSpec):
      try:
        type_spec.get_name(type(pyobj))
        return True
      except ValueError:
        return False

    return False

  def do_encode(self, type_spec_value, encode_fn):
    """Returns an encoded proto for the given `tf.TypeSpec`."""
    type_spec_class = self.TYPE_SPEC_CLASS_TO_PROTO.get(type(type_spec_value))
    type_spec_class_name = type(type_spec_value).__name__

    if type_spec_class is None:
      type_spec_class_name = type_spec.get_name(type(type_spec_value))
      if isinstance(type_spec_value, extension_type.ExtensionTypeSpec):
        type_spec_class = struct_pb2.TypeSpecProto.EXTENSION_TYPE_SPEC
      else:
        type_spec_class = struct_pb2.TypeSpecProto.REGISTERED_TYPE_SPEC
        # Support for saving registered TypeSpecs is currently experimental.
        # Issue a warning to indicate the limitations.
        warnings.warn("Encoding a StructuredValue with type %s; loading this "
                      "StructuredValue will require that this type be "
                      "imported and registered." % type_spec_class_name)

    type_state = type_spec_value._serialize()  # pylint: disable=protected-access
    num_flat_components = len(
        nest.flatten(type_spec_value._component_specs, expand_composites=True))  # pylint: disable=protected-access
    encoded_type_spec = struct_pb2.StructuredValue()
    encoded_type_spec.type_spec_value.CopyFrom(
        struct_pb2.TypeSpecProto(
            type_spec_class=type_spec_class,
            type_state=encode_fn(type_state),
            type_spec_class_name=type_spec_class_name,
            num_flat_components=num_flat_components))
    return encoded_type_spec

  def can_decode(self, value):
    return value.HasField("type_spec_value")

  def do_decode(self, value, decode_fn):
    """Returns the `tf.TypeSpec` encoded by the proto `value`."""
    type_spec_proto = value.type_spec_value
    type_spec_class_enum = type_spec_proto.type_spec_class
    class_name = type_spec_proto.type_spec_class_name

    if type_spec_class_enum == struct_pb2.TypeSpecProto.REGISTERED_TYPE_SPEC:
      try:
        type_spec_class = type_spec.lookup(class_name)
      except ValueError as e:
        raise ValueError(
            f"The type '{class_name}' has not been registered.  It must be "
            "registered before you load this object (typically by importing "
            "its module).") from e
    elif type_spec_class_enum == struct_pb2.TypeSpecProto.EXTENSION_TYPE_SPEC:
      try:
        type_spec_class = type_spec.lookup(class_name)
      except ValueError:
        type_spec_class = extension_type.AnonymousExtensionTypeSpec
        warnings.warn("The type %r has not been registered.  Falling back to "
                      "using AnonymousExtensionTypeSpec instead.")
    else:
      if type_spec_class_enum not in self.TYPE_SPEC_CLASS_FROM_PROTO:
        raise ValueError(
            f"The type '{class_name}' is not supported by this version of "
            "TensorFlow. (The object you are loading must have been created "
            "with a newer version of TensorFlow.)")
      type_spec_class = self.TYPE_SPEC_CLASS_FROM_PROTO[type_spec_class_enum]

    # pylint: disable=protected-access
    return type_spec_class._deserialize(decode_fn(type_spec_proto.type_state))


_codecs = [
    _ListCodec(),
    _TupleCodec(),
    _NamedTupleCodec(),
    _StringCodec(),
    _Float64Codec(),
    _NoneCodec(),
    _Int64Codec(),
    _TensorShapeCodec(),
    _BoolCodec(),
    _BoundedTensorSpecCodec(),
    _TensorTypeCodec(),
    _DictCodec(),
    _TensorSpecCodec(),
    _TypeSpecCodec(),
]
