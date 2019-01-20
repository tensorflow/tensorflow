# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ignite Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import socket
import ssl
import struct

import six

from tensorflow.contrib.ignite.python.ops import gen_dataset_ops
from tensorflow.contrib.ignite.python.ops import ignite_op_loader  # pylint: disable=unused-import
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import deprecation


@six.add_metaclass(abc.ABCMeta)
class Readable(object):
  """Abstract class that exposes methods to do reading-related operations."""

  @abc.abstractmethod
  def __init__(self):
    pass

  def read_byte(self):
    """Reads and returnes byte."""
    return self._read("b", 1)

  def read_short(self):
    """Reads and returns short (2 bytes, little-endian)."""
    return self._read("h", 2)

  def read_int(self):
    """Reads and returns int (4 bytes, little-endian)."""
    return self._read("i", 4)

  def read_long(self):
    """Reads and returns long (8 bytes, little-endian)."""
    return self._read("q", 8)

  def skip(self, length):
    """Skips the specified number of bytes."""
    self.read_data(length)

  @abc.abstractmethod
  def read_data(self, length):
    """Reads the specified number of bytes and returns them as a buffer."""
    return None

  def _read(self, data_type, length):
    """Reads, unpacks and returns specified type (little-endian)."""
    data_buffer = self.read_data(length)
    return struct.unpack("<" + data_type, data_buffer)[0]


class DataBuffer(Readable):
  """DataBuffer class that exposes methods to read data from a byte buffer."""

  def __init__(self, data_buffer):
    """Constructs a new instance based on the specified byte  buffer.

    Args:
      data_buffer: Buffer to be read.
    """
    Readable.__init__(self)
    self.buffer = data_buffer
    self.ptr = 0

  def read_data(self, length):
    """Reads the specified number of bytes and returns them as a buffer."""
    data_buffer = self.buffer[self.ptr:][:length]
    self.ptr += length
    return data_buffer


class TcpClient(Readable):
  """TcpClient class that exposes methods to read data from a socket."""

  def __init__(self, host, port, certfile=None, keyfile=None, password=None):
    """Constructs a new instance based on the specified host and port.

    Args:
      host: Host to be connected.
      port: Port to be connected.
      certfile: File in PEM format containing the certificate as well as any
        number of CA certificates needed to establish the certificate's
        authenticity.
      keyfile: File containing the private key (otherwise the private key will
        be taken from certfile as well).
      password: Password to be used if the private key is encrypted and a
        password is necessary.

    Raises:
      ValueError: If the wrong combination of arguments is provided.
    """
    Readable.__init__(self)
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if certfile is not None:
      context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
      context.load_cert_chain(certfile, keyfile, password)
      self.sock = context.wrap_socket(self.sock)
    else:
      if keyfile is not None:
        raise ValueError("SSL is disabled, keyfile must not be specified "
                         "(to enable SSL specify certfile)")
      if password is not None:
        raise ValueError("SSL is disabled, password must not be specified "
                         "(to enable SSL specify certfile)")

    self.host = host
    self.port = port

  def __enter__(self):
    """Connects to host and port specified in the constructor."""
    self.sock.connect((self.host, self.port))
    return self

  def __exit__(self, t, v, traceback):
    """Disconnects the socket."""
    self.sock.close()

  def write_byte(self, v):
    """Writes the specified byte."""
    self._write(v, "b")

  def write_short(self, v):
    """Writes the specified short (2 bytes, little-endian)."""
    self._write(v, "h")

  def write_int(self, v):
    """Writes the specified short (4 bytes, little-endian)."""
    self._write(v, "i")

  def write_long(self, v):
    """Writes the specified int (8 bytes, little-endian)."""
    self._write(v, "q")

  def write_string(self, v):
    """Writes the specified string."""
    self.sock.sendall(v.encode("UTF-8"))

  def read_data(self, length):
    """Reads the specified number of bytes and returns them as a buffer."""
    data_buffer = None
    rem = length
    while rem > 0:
      buf = self.sock.recv(rem)
      rem = rem - len(buf)
      if data_buffer is None:
        data_buffer = buf
      else:
        data_buffer += buf
    return data_buffer

  def _write(self, value, data_type):
    """Packs and writes data using the specified type (little-endian)."""
    data_buffer = struct.pack("<" + data_type, value)
    self.sock.sendall(data_buffer)


class BinaryType(object):
  """BinaryType class that encapsulated type id, type name and fields."""

  def __init__(self, type_id, type_name, fields):
    """Constructs a new instance of BinaryType."""
    self.type_id = type_id
    self.type_name = type_name
    self.fields = fields


class BinaryField(object):
  """BinaryField class that encapsulated field name, type id and field id."""

  def __init__(self, field_name, type_id, field_id):
    """Constructs a new instance of BinaryField."""
    self.field_name = field_name
    self.type_id = type_id
    self.field_id = field_id


# Binary types defined in Apache Ignite Thin client and supported by
# TensorFlow on Apache Ignite, see
# https://apacheignite.readme.io/v2.6/docs/binary-client-protocol.
# True means that type is a vector, False means type is scalar.
types = {
    1: (dtypes.uint8, False),
    2: (dtypes.int16, False),
    3: (dtypes.int32, False),
    4: (dtypes.int64, False),
    5: (dtypes.float32, False),
    6: (dtypes.float64, False),
    7: (dtypes.uint16, False),
    8: (dtypes.bool, False),
    9: (dtypes.string, False),
    12: (dtypes.uint8, True),
    13: (dtypes.int16, True),
    14: (dtypes.int32, True),
    15: (dtypes.int64, True),
    16: (dtypes.float32, True),
    17: (dtypes.float64, True),
    18: (dtypes.uint16, True),
    19: (dtypes.bool, True),
    20: (dtypes.string, True)
}


class TypeTreeNode(object):
  """TypeTreeNode class exposes methods to format object tree structure data."""

  def __init__(self, name, type_id, fields=None, permutation=None):
    """Constructs a new instance of TypeTreeNode.

    Args:
      name: Name of the object tree node.
      type_id: Type id of the object tree node.
      fields: List of fields (children of the object tree node).
      permutation: Permutation that should be applied to order object children.
    """
    self.name = name
    self.type_id = type_id
    self.fields = fields
    self.permutation = permutation

  def to_output_classes(self):
    """Formats the tree object as required by `Dataset.output_classes`."""
    if self.fields is None:
      return ops.Tensor
    output_classes = {}
    for field in self.fields:
      output_classes[field.name] = field.to_output_classes()
    return output_classes

  def to_output_shapes(self):
    """Formats the tree object as required by `Dataset.output_shapes`."""
    if self.fields is None:
      if self.type_id in types:
        object_type = types[self.type_id]
        is_array = object_type[1]
        if is_array:
          return tensor_shape.TensorShape([None])
        return tensor_shape.TensorShape([])
      raise ValueError("Unsupported type [type_id=%d]" % self.type_id)
    output_shapes = {}
    for field in self.fields:
      output_shapes[field.name] = field.to_output_shapes()
    return output_shapes

  def to_output_types(self):
    """Formats the tree object as required by `Dataset.output_types`."""
    if self.fields is None:
      if self.type_id in types:
        object_type = types[self.type_id]
        return object_type[0]
      raise ValueError("Unsupported type [type_id=%d]" % self.type_id)
    else:
      output_types = {}
      for field in self.fields:
        output_types[field.name] = field.to_output_types()
      return output_types

  def to_flat(self):
    """Returns a list of node types."""
    return self.to_flat_rec([])

  def to_permutation(self):
    """Returns a permutation that should be applied to order object leaves."""
    correct_order_dict = {}
    self.traversal_rec(correct_order_dict, 0)
    object_order = []
    self.traversal_permutation_rec(object_order)
    return [correct_order_dict[o] for o in object_order]

  def to_flat_rec(self, flat):
    """Formats a list of leaf node types in pre-order."""
    if self.fields is None:
      flat.append(self.type_id)
    else:
      for field in self.fields:
        field.to_flat_rec(flat)
    return flat

  def traversal_permutation_rec(self, permutation):
    """Collects nodes in accordance with permutation."""
    if self.fields is None:
      permutation.append(self)
    else:
      for idx in self.permutation:
        field = self.fields[idx]
        field.traversal_permutation_rec(permutation)

  def traversal_rec(self, d, i):
    """Collects nodes in pre-order traversal."""
    if self.fields is None:
      d[self] = i
      i += 1
    else:
      for field in self.fields:
        i = field.traversal_rec(d, i)
    return i


class IgniteClient(TcpClient):
  """IgniteClient enables working with Apache Ignite using a thin client.

  This client works with assumption that all object in the cache
  have the same structure (homogeneous objects) and the cache contains at
  least one object.
  """

  def __init__(self,
               host,
               port,
               username=None,
               password=None,
               certfile=None,
               keyfile=None,
               cert_password=None):
    """Constructs a new instance of IgniteClient.

    Args:
      host: Apache Ignite Thin client host to be connected.
      port: Apache Ignite Thin client port to be connected.
      username: Apache Ignite Thin Client authentication username.
      password: Apache Ignite Thin Client authentication password.
      certfile: File in PEM format containing the certificate as well as any
        number of CA certificates needed to establish the certificate's
        authenticity.
      keyfile: File containing the private key (otherwise the private key will
        be taken from certfile as well).
      cert_password: Password to be used if the private key is encrypted and a
        password is necessary.
    """
    TcpClient.__init__(self, host, port, certfile, keyfile, cert_password)
    self.username = username
    self.password = password

  def handshake(self):
    """Makes a handshake  after connect and before any other calls."""
    msg_len = 8

    if self.username is None:
      msg_len += 1
    else:
      msg_len += 5 + len(self.username)

    if self.password is None:
      msg_len += 1
    else:
      msg_len += 5 + len(self.password)

    self.write_int(msg_len)  # Message length
    self.write_byte(1)  # Handshake operation
    self.write_short(1)  # Version (1.1.0)
    self.write_short(1)
    self.write_short(0)
    self.write_byte(2)  # Thin client

    if self.username is None:  # Username
      self.write_byte(101)
    else:
      self.write_byte(9)
      self.write_int(len(self.username))
      self.write_string(self.username)

    if self.password is None:  # Password
      self.write_byte(101)
    else:
      self.write_byte(9)
      self.write_int(len(self.password))
      self.write_string(self.password)

    self.read_int()  # Result length
    res = self.read_byte()

    if res != 1:
      serv_ver_major = self.read_short()
      serv_ver_minor = self.read_short()
      serv_ver_patch = self.read_short()
      err_msg = self._parse_string()
      if err_msg is None:
        raise RuntimeError(
            "Handshake Error [result=%d, version=%d.%d.%d]" %
            (res, serv_ver_major, serv_ver_minor, serv_ver_patch))
      else:
        raise RuntimeError(
            "Handshake Error [result=%d, version=%d.%d.%d, message='%s']" %
            (res, serv_ver_major, serv_ver_minor, serv_ver_patch, err_msg))

  def get_cache_type(self, cache_name):
    """Collects type information about objects stored in the specified cache."""
    cache_name_hash = self._java_hash_code(cache_name)
    self.write_int(25)  # Message length
    self.write_short(2000)  # Operation code
    self.write_long(0)  # Request ID
    self.write_int(cache_name_hash)  # Cache name
    self.write_byte(0)  # Flags
    self.write_byte(101)  # Filter (NULL)
    self.write_int(1)  # Cursor page size
    self.write_int(-1)  # Partition to query
    self.write_byte(0)  # Local flag

    result_length = self.read_int()
    self.read_long()  # Request id
    status = self.read_int()

    if status != 0:
      err_msg = self._parse_string()
      if err_msg is None:
        raise RuntimeError("Scan Query Error [status=%s]" % status)
      else:
        raise RuntimeError(
            "Scan Query Error [status=%s, message='%s']" % (status, err_msg))

    self.read_long()  # Cursor id
    row_count = self.read_int()

    if row_count == 0:
      raise RuntimeError("Scan Query returned empty result, so it's "
                         "impossible to derive the cache type")

    payload = DataBuffer(self.read_data(result_length - 25))

    self.read_byte()  # Next page

    res = TypeTreeNode("root", 0, [
        self._collect_types("key", payload),
        self._collect_types("val", payload)
    ], [0, 1])

    return res

  def _java_hash_code(self, s):
    """Computes hash code of the specified string using Java code."""
    h = 0
    for c in s:
      h = (31 * h + ord(c)) & 0xFFFFFFFF
    return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000

  def _collect_types(self, field_name, data):
    """Extracts type information from the specified object."""
    type_id = data.read_byte()

    # Byte scalar.
    if type_id == 1:
      data.skip(1)
      return TypeTreeNode(field_name, type_id)

    # Short scalar.
    if type_id == 2:
      data.skip(2)
      return TypeTreeNode(field_name, type_id)

    # Integer scalar.
    if type_id == 3:
      data.skip(4)
      return TypeTreeNode(field_name, type_id)

    # Long scalar.
    if type_id == 4:
      data.skip(8)
      return TypeTreeNode(field_name, type_id)

    # Float scalar.
    if type_id == 5:
      data.skip(4)
      return TypeTreeNode(field_name, type_id)

    # Double scalar.
    if type_id == 6:
      data.skip(8)
      return TypeTreeNode(field_name, type_id)

    # Char scalar.
    if type_id == 7:
      data.skip(2)
      return TypeTreeNode(field_name, type_id)

    # Bool scalar.
    if type_id == 8:
      data.skip(1)
      return TypeTreeNode(field_name, type_id)

    # String scalar.
    if type_id == 9:
      length = data.read_int()
      data.skip(length)
      return TypeTreeNode(field_name, type_id)

    # UUID scalar.
    if type_id == 10:
      data.skip(16)
      return TypeTreeNode(field_name, type_id)

    # Date scalar.
    if type_id == 11:
      data.skip(8)
      return TypeTreeNode(field_name, type_id)

    # Byte array.
    if type_id == 12:
      length = data.read_int()
      data.skip(length)
      return TypeTreeNode(field_name, type_id)

    # Short array.
    if type_id == 13:
      length = data.read_int()
      data.skip(length * 2)
      return TypeTreeNode(field_name, type_id)

    # Integer array.
    if type_id == 14:
      length = data.read_int()
      data.skip(length * 4)
      return TypeTreeNode(field_name, type_id)

    # Long array.
    if type_id == 15:
      length = data.read_int()
      data.skip(length * 8)
      return TypeTreeNode(field_name, type_id)

    # Float array.
    if type_id == 16:
      length = data.read_int()
      data.skip(length * 4)
      return TypeTreeNode(field_name, type_id)

    # Double array.
    if type_id == 17:
      length = data.read_int()
      data.skip(length * 8)
      return TypeTreeNode(field_name, type_id)

    # Char array.
    if type_id == 18:
      length = data.read_int()
      data.skip(length * 2)
      return TypeTreeNode(field_name, type_id)

    # Bool array.
    if type_id == 19:
      length = data.read_int()
      data.skip(length)
      return TypeTreeNode(field_name, type_id)

    # String array.
    if type_id == 20:
      length = data.read_int()
      for _ in range(length):
        header = data.read_byte()
        if header == 9:
          str_length = data.read_int()
          data.skip(str_length)
        elif header == 101:
          pass
        else:
          raise RuntimeError(
              "Unknown binary type when expected string [type_id=%d]" % header)
      return TypeTreeNode(field_name, type_id)

    # UUID array.
    if type_id == 21:
      length = data.read_int()
      data.skip(length * 16)  # TODO(dmitrievanthony): support NULL values.
      return TypeTreeNode(field_name, type_id)

    # Date array.
    if type_id == 22:
      length = data.read_int()
      data.skip(length * 8)
      return TypeTreeNode(field_name, type_id)

    # Wrapped Binary Object.
    if type_id == 27:
      length = data.read_int()
      inner_data = data.read_data(length)
      data.read_int()  # Offset
      return self._collect_types(field_name, DataBuffer(inner_data))

    # Complex Object.
    if type_id == 103:
      data.read_byte()  # Object version
      data.read_short()  # Object flags
      obj_type_id = data.read_int()
      data.read_int()  # Object hash code
      obj_length = data.read_int()
      data.read_int()  # Object schema id
      obj_schema_offset = data.read_int()

      obj_type = self._get_type(obj_type_id)
      children = []

      for obj_field in obj_type.fields:
        child = self._collect_types(obj_field.field_name, data)
        children.append(child)

      children_sorted = sorted(children, key=lambda child: child.name)
      permutation = [children_sorted.index(child) for child in children]
      children = children_sorted

      data.skip(obj_length - obj_schema_offset)

      return TypeTreeNode(field_name, type_id, children, permutation)

    raise RuntimeError("Unknown binary type [type_id=%d]" % type_id)

  def _get_type(self, type_id):
    """Queries Apache Ignite information about type by type id."""
    self.write_int(14)  # Message length
    self.write_short(3002)  # Operation code
    self.write_long(0)  # Request ID
    self.write_int(type_id)  # Type ID

    self.read_int()  # Result length
    self.read_long()  # Request id
    status = self.read_int()

    if status != 0:
      err_msg = self._parse_string()
      if err_msg is None:
        raise RuntimeError("Get Binary Type Error [status=%d, message='%s']" %
                           (status, err_msg))
      else:
        raise RuntimeError("Get Binary Type Error [status=%d]" % status)

    binary_type_exists = self.read_byte()

    if binary_type_exists == 0:
      raise RuntimeError("Binary type not found [type_id=%d] " % type_id)

    binary_type_id = self.read_int()
    binary_type_name = self._parse_string()
    self._parse_string()  # Affinity field name

    fields = []
    for _ in range(self.read_int()):
      field_name = self._parse_string()
      field_type_id = self.read_int()
      field_id = self.read_int()

      field = BinaryField(field_name, field_type_id, field_id)
      fields.append(field)

    is_enum = self.read_byte()
    if is_enum == 1:
      raise RuntimeError("Enum fields are not supported yet")

    schema_cnt = self.read_int()
    for _ in range(schema_cnt):
      self.read_int()  # Schema id
      field_cnt = self.read_int()
      self.skip(field_cnt * 4)

    return BinaryType(binary_type_id, binary_type_name, fields)

  def _parse_string(self):
    """Parses string."""
    header = self.read_byte()
    if header == 9:
      length = self.read_int()
      return self.read_data(length).decode("utf-8")
    if header == 101:
      return None
    raise RuntimeError(
        "Unknown binary type when expected string [type_id=%d]" % header)


class IgniteDataset(dataset_ops.DatasetSource):
  """Apache Ignite is a memory-centric distributed database.

     It acts as a caching and processing platform for transactional, analytical,
     and streaming workloads, delivering in-memory speeds at petabyte scale.
     This contrib package contains an integration between Apache Ignite and
     TensorFlow. The integration is based on tf.data from TensorFlow side and
     Binary Client Protocol from Apache Ignite side. It allows to use Apache
     Ignite as a datasource for neural network training, inference and all other
     computations supported by TensorFlow. Ignite Dataset is based on Apache
     Ignite Binary Client Protocol.
  """

  @deprecation.deprecated(
      None,
      "tf.contrib.ignite will be removed in 2.0, the support for Apache Ignite "
      "will continue to be provided through the tensorflow/io GitHub project.")
  def __init__(self,
               cache_name,
               host="localhost",
               port=10800,
               local=False,
               part=-1,
               page_size=100,
               username=None,
               password=None,
               certfile=None,
               keyfile=None,
               cert_password=None):
    """Create a IgniteDataset.

    Args:
      cache_name: Cache name to be used as datasource.
      host: Apache Ignite Thin Client host to be connected.
      port: Apache Ignite Thin Client port to be connected.
      local: Local flag that defines to query only local data.
      part: Number of partitions to be queried.
      page_size: Apache Ignite Thin Client page size.
      username: Apache Ignite Thin Client authentication username.
      password: Apache Ignite Thin Client authentication password.
      certfile: File in PEM format containing the certificate as well as any
        number of CA certificates needed to establish the certificate's
        authenticity.
      keyfile: File containing the private key (otherwise the private key will
        be taken from certfile as well).
      cert_password: Password to be used if the private key is encrypted and a
        password is necessary.
    """
    super(IgniteDataset, self).__init__()

    with IgniteClient(host, port, username, password, certfile, keyfile,
                      cert_password) as client:
      client.handshake()
      self.cache_type = client.get_cache_type(cache_name)

    self.cache_name = ops.convert_to_tensor(
        cache_name, dtype=dtypes.string, name="cache_name")
    self.host = ops.convert_to_tensor(host, dtype=dtypes.string, name="host")
    self.port = ops.convert_to_tensor(port, dtype=dtypes.int32, name="port")
    self.local = ops.convert_to_tensor(local, dtype=dtypes.bool, name="local")
    self.part = ops.convert_to_tensor(part, dtype=dtypes.int32, name="part")
    self.page_size = ops.convert_to_tensor(
        page_size, dtype=dtypes.int32, name="page_size")
    self.schema = ops.convert_to_tensor(
        self.cache_type.to_flat(), dtype=dtypes.int32, name="schema")
    self.permutation = ops.convert_to_tensor(
        self.cache_type.to_permutation(),
        dtype=dtypes.int32,
        name="permutation")
    self._structure = structure.convert_legacy_structure(
        self.cache_type.to_output_types(), self.cache_type.to_output_shapes(),
        self.cache_type.to_output_classes())

  def _as_variant_tensor(self):
    return gen_dataset_ops.ignite_dataset(self.cache_name, self.host, self.port,
                                          self.local, self.part, self.page_size,
                                          self.schema, self.permutation)

  @property
  def _element_structure(self):
    return self._structure
