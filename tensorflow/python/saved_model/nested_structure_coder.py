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
coder = nested_structure_coder.StructureCoder()
# Encode into proto.
signature_proto = coder.encode_structure(function.input_signature)
# Decode into a Python object.
restored_signature = coder.decode_proto(signature_proto)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.saved_model import struct_pb2


class NotEncodableError(Exception):
  """Error raised when a coder cannot encode an object."""


class StructureCoder(object):
  """Encoder and decoder for nested structures into protos."""

  _codecs = []

  @classmethod
  def register_codec(cls, x):
    cls._codecs.append(x)

  @classmethod
  def _get_encoders(cls):
    return [(c.can_encode, c.do_encode) for c in cls._codecs]

  @classmethod
  def _get_decoders(cls):
    return [(c.can_decode, c.do_decode) for c in cls._codecs]

  def _map_structure(self, pyobj, coders):
    for can, do in coders:
      if can(pyobj):
        recursion_fn = functools.partial(self._map_structure, coders=coders)
        return do(pyobj, recursion_fn)
    raise NotEncodableError(
        "No encoder for object [%s] of type [%s]." % (str(pyobj), type(pyobj)))

  def encode_structure(self, nested_structure):
    """Encodes nested structures composed of encodable types into a proto.

    Args:
      nested_structure: Structure to encode.

    Returns:
      Encoded proto.

    Raises:
      NotEncodableError: For values for which there are no encoders.
    """
    return self._map_structure(nested_structure, self._get_encoders())


  def can_encode(self, nested_structure):
    """Determines whether a nested structure can be encoded into a proto.

    Args:
      nested_structure: Structure to encode.

    Returns:
      True if the nested structured can be encoded.
    """
    try:
      self.encode_structure(nested_structure)
    except NotEncodableError:
      return False
    return True

  def decode_proto(self, proto):
    """Decodes proto representing a nested structure.

    Args:
      proto: Proto to decode.

    Returns:
      Decoded structure.

    Raises:
      NotEncodableError: For values for which there are no encoders.
    """
    return self._map_structure(proto, self._get_decoders())


class _ListCodec(object):
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


StructureCoder.register_codec(_ListCodec())


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
          isinstance(instance._fields, collections.Sequence) and
          all(isinstance(f, six.string_types) for f in instance._fields))


class _TupleCodec(object):
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


StructureCoder.register_codec(_TupleCodec())


class _DictCodec(object):
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


StructureCoder.register_codec(_DictCodec())


class _NamedTupleCodec(object):
  """Codec for namedtuples.

  Encoding and decoding a namedtuple reconstructs a namedtuple with a different
  actual Python type, but with same `typename` and `fields`.
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


StructureCoder.register_codec(_NamedTupleCodec())


class _Float64Codec(object):
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


StructureCoder.register_codec(_Float64Codec())


class _Int64Codec(object):
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


StructureCoder.register_codec(_Int64Codec())


class _StringCodec(object):
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
    return value.string_value


StructureCoder.register_codec(_StringCodec())


class _NoneCodec(object):
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


StructureCoder.register_codec(_NoneCodec())


class _BoolCodec(object):
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


StructureCoder.register_codec(_BoolCodec())


class _TensorShapeCodec(object):
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


StructureCoder.register_codec(_TensorShapeCodec())


class _TensorTypeCodec(object):
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


StructureCoder.register_codec(_TensorTypeCodec())


class _TensorSpecCodec(object):
  """Codec for `TensorSpec`."""

  def can_encode(self, pyobj):
    return isinstance(pyobj, tensor_spec.TensorSpec)

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
    return tensor_spec.TensorSpec(
        shape=decode_fn(
            struct_pb2.StructuredValue(
                tensor_shape_value=value.tensor_spec_value.shape)),
        dtype=decode_fn(
            struct_pb2.StructuredValue(
                tensor_dtype_value=value.tensor_spec_value.dtype)),
        name=value.tensor_spec_value.name)


StructureCoder.register_codec(_TensorSpecCodec())
