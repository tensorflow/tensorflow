# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Kafka Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kafka.python.ops import kafka_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.kafka.python.ops import gen_dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

import socket
import struct

class Readable(object):
    """Readable abstract class that exposes methods to do reading-related operations."""
    
    def read_byte(self): 
        """Reads and returnes byte."""
        return self.__read("b", 1)

    def read_short(self):
        """Reads and returns short (2 bytes, little-endian)."""
        return self.__read("h", 2)

    def read_int(self):
        """Reads and returns int (4 bytes, little-endian)."""
        return self.__read("i", 4)

    def read_long(self):
        """Reads and returns long (8 bytes, little-endian)."""
        return self.__read("q", 8)
    
    def skip(self, length):
        """Skips the specified number of bytes."""
        self.read_data(length)
    
    def read_data(self, length):
        """Reads the specified number of bytes and returns them as a buffer."""
        pass
    
    def __read(self, data_type, length):
        """Reads, unpacks and returns specified type (little-endian)."""
        buffer = self.read_data(length)
        return struct.unpack("<" + data_type, buffer)[0]

class DataBuffer(Readable):
    """DataBuffer class that exposes methods to read data from a byte buffer."""
    
    def __init__(self, buffer):
        """Constructs a new instance of DataBuffer based on the specified byte buffer.
        
        Args:
            buffer: Buffer to be read.
        """
        Readable.__init__(self)
        self.buffer = buffer
        self.ptr = 0

    def read_data(self, length):
        """Reads the specified number of bytes and returns them as a buffer."""
        data_buffer = self.buffer[self.ptr:][:length]
        self.ptr += length
        return data_buffer

class TcpClient(Readable):
    """TcpClient class that exposes methods to read data from a socket."""
    
    def __init__(self, host, port):
        """Constructs a new instance of TcpClient based on the specified host and port.
        
        Args:
            host: Host to be connected.
            port: Port to be connected.
        """
        Readable.__init__(self)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
        self.__write(v, "b")
        
    def write_short(self, v):
        """Writes the specified short (2 bytes, little-endian)."""
        self.__write(v, "h")
    
    def write_int(self, v):
        """Writes the specified short (4 bytes, little-endian)."""
        self.__write(v, "i")
        
    def write_long(self, v):
        """Writes the specified int (8 bytes, little-endian)."""
        self.__write(v, "q")
    
    def read_data(self, length):
        """Reads the specified number of bytes and returns them as a buffer."""
        data_buffer = self.sock.recv(length)
        return data_buffer
        
    def __write(self, value, data_type):
        """Packs and writes data using the specified type (little-endian)."""
        data_buffer = struct.pack("<" + data_type, value)
        self.sock.sendall(data_buffer)

class BinaryType():
    """BinaryType class that encapsulated type id, type name and fields."""
    
    def __init__(self, type_id, type_name, fields):
        """Constructs a new instance of BinaryType."""
        self.type_id = type_id
        self.type_name = type_name
        self.fields = fields

class BinaryField():
    """BinaryField class that encapsulated field name, type id and field id."""
    
    def __init__(self, field_name, type_id, field_id):
        """Constructs a new instance of BinaryField."""
        self.field_name = field_name
        self.type_id = type_id
        self.field_id = field_id        
        
# Binary types defined in Apache Ignite Thin client and supported by TensorFlow on Apache Ignite,
# see https://cwiki.apache.org/confluence/display/IGNITE/IEP-9+Thin+Client+Protocol.
types = {
    1 :  (dtypes.uint8,   False),
    2 :  (dtypes.int16,   False),
    3 :  (dtypes.int32,   False),
    4 :  (dtypes.int64,   False),
    5 :  (dtypes.float32, False),
    6 :  (dtypes.float64, False),
    7 :  (dtypes.uint16,  False),
    8 :  (dtypes.bool,    False),
    9 :  (dtypes.string,  False),
    12 : (dtypes.uint8,   True),
    13 : (dtypes.int16,   True),
    14 : (dtypes.int32,   True),
    15 : (dtypes.int64,   True),
    16 : (dtypes.float32, True),
    17 : (dtypes.float64, True),
    18 : (dtypes.uint16,  True),
    19 : (dtypes.bool,    True),
    20 : (dtypes.string,  True)
}        
        
class TypeTreeNode():
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
        """Formats the tree object the way required in 'output_classes' property of dataset. """
        if self.fields is None:
            return ops.Tensor
        else:
            output_classes = {}
            for field in self.fields:
                output_classes[field.name] = field.to_output_classes()
            return output_classes

    def to_output_shapes(self):
        """Formats the tree object the way required in 'output_shapes' property of dataset. """
        if self.fields is None:
            object_type = types[self.type_id]
            if object_type is not None:
                is_array = object_type[1]
                if is_array:
                    return tensor_shape.TensorShape([None])
                else:
                    return tensor_shape.TensorShape([])
            else:
                raise Exception("Unsupported type [type_id=%d]" % self.type_id)
        else:
            output_shapes = {}
            for field in self.fields:
                output_shapes[field.name] = field.to_output_shapes()
            return output_shapes

    def to_output_types(self):
        """Formats the tree object the way required in 'output_types' property of dataset. """
        if self.fields is None:
            object_type = types[self.type_id]
            if object_type is not None:
                return object_type[0]
            else:
                raise Exception("Unsupported type [type_id=%d]" % self.type_id)
        else:
            output_types = {}
            for field in self.fields:
                output_types[field.name] = field.to_output_types()
            return output_types  
        
    def to_flat(self):
        """Returns a list of leaf node types."""
        return self.__to_flat([])
    
    def to_permutation(self):
        """Returns a permutation that should be applied to order object leafs."""
        correct_order_dict = {}
        self.__traversal(correct_order_dict, 0)
        object_order = []
        self.__traversal_permutation(object_order)
        return [correct_order_dict[o] for o in object_order]

    def __to_flat(self, flat):
        """Formats a list of leaf node types."""
        flat.append(self.type_id)
        if self.fields is not None:
            for field in self.fields:
                field.__to_flat(flat)
        return flat

    def __traversal_permutation(self, permutation):
        """Collects nodes in accordance with permutation."""
        if self.fields is None:
            permutation.append(self)
        else:
            for idx in self.permutation:
                field = self.fields[idx]
                field.__traversal_permutation(permutation)
        
    def __traversal(self, d, i):
        """Collects nodes in pre-order traversal."""
        if self.fields is None:
            d[self] = i
            i += 1
        else:
            for field in self.fields:
                i = field.__traversal(d, i)
        return i

class IgniteClient(TcpClient):
    """IgniteClient class exposes methods to work with Apache Ignite using Thin client."""
    
    def __init__(self, host, port):
        """Constructs a new instance of IgniteClient.
        
        Args:
            host: Apache Ignite Thin client host to be connected.
            port: Apache Ignite Thin client port to be connected.
        """
        TcpClient.__init__(self, host, port)
        
    def handshake(self):
        """Makes a handshake required to be made after connect before any other calls."""
        self.write_int(8)    # Message length
        self.write_byte(1)   # Handshake operation
        self.write_short(1); # Version (1.0.0)
        self.write_short(0);
        self.write_short(0);
        self.write_byte(2);  # Thin client

        res_len = self.read_int()
        res = self.read_byte()

        if res != 1:
            raise Exception("Handshake failed [result=%d]" % res)
            
    def get_cache_type(self, cache_name):
        """Collects type information about objects stored in the specified cache."""
        cache_name_hash = self.__java_hash_code(cache_name)
        self.write_int(25)              # Message length
        self.write_short(2000)          # Operation code
        self.write_long(0)              # Request ID
        self.write_int(cache_name_hash) # Cache name
        self.write_byte(0)              # Flags
        self.write_byte(101)            # Filter (NULL)
        self.write_int(1)               # Cursor page size
        self.write_int(-1)              # Partition to query
        self.write_byte(0)              # Local flag

        result_length = self.read_int()
        request_id = self.read_long()
        status = self.read_int()
        
        if status != 0:
            raise Exception("Scan query completed with error [status=%s]" % status)

        cursor_id = self.read_long()
        row_count = self.read_int()
        
        if row_count == 0:
            raise Exception("Scan query returned empty result")

        payload = DataBuffer(self.read_data(result_length - 25))

        next_page = self.read_byte()
        
        res = TypeTreeNode("root", 0, [
            self.__collect_types("key", payload), 
            self.__collect_types("val", payload)
        ], [0, 1])
        
        return res        
            
    def __java_hash_code(self, s):
        """Computes hash code of the specified string using Java code."""
        h = 0
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000
    
    def __collect_types(self, field_name, data):
        """Extracts type information from the specified object."""
        type_id = data.read_byte()

        # Byte scalar.
        if type_id == 1:
            data.skip(1)
            return TypeTreeNode(field_name, type_id) 
        
        # Short scalar.
        elif type_id == 2:
            data.skip(2)
            return TypeTreeNode(field_name, type_id)
        
        # Integer scalar.
        elif type_id == 3:
            data.skip(4)
            return TypeTreeNode(field_name, type_id)

        # Long scalar.
        elif type_id == 4:
            data.skip(8)
            return TypeTreeNode(field_name, type_id)
        
        # Float scalar.
        elif type_id == 5:
            data.skip(4)
            return TypeTreeNode(field_name, type_id)
        
        # Double scalar.
        elif type_id == 6:
            data.skip(8)
            return TypeTreeNode(field_name, type_id)
        
        # Char scalar.
        elif type_id == 7:
            data.skip(2)
            return TypeTreeNode(field_name, type_id)
        
        # Bool scalar.
        elif type_id == 8:
            data.skip(1)
            return TypeTreeNode(field_name, type_id)
        
        # String scalar.
        elif type_id == 9:
            length = data.read_int()
            data.skip(length)
            return TypeTreeNode(field_name, type_id)
        
        # Byte array.
        elif type_id == 12:
            length = data.read_int()
            data.skip(length)
            return TypeTreeNode(field_name, type_id)
        
        # Short array.
        elif type_id == 13:
            length = data.read_int()
            data.skip(length * 2)
            return TypeTreeNode(field_name, type_id)
        
        # Integer array.
        elif type_id == 14:
            length = data.read_int()
            data.skip(length * 4)
            return TypeTreeNode(field_name, type_id)
        
        # Long array.
        elif type_id == 15:
            length = data.read_int()
            data.skip(length * 8)
            return TypeTreeNode(field_name, type_id)
        
        # Float array.
        elif type_id == 16:
            length = data.read_int()
            data.skip(length * 4)
            return TypeTreeNode(field_name, type_id)
        
        # Double array.
        elif type_id == 17:
            length = data.read_int()
            data.skip(length * 8)
            return TypeTreeNode(field_name, type_id)
        
        # Char array.
        elif type_id == 18:
            length = data.read_int()
            data.skip(length * 2)
            return TypeTreeNode(field_name, type_id)
        
        # Bool array.
        elif type_id == 19:
            length = data.read_int()
            data.skip(length)
            return TypeTreeNode(field_name, type_id)
        
        # String array.
        elif type_id == 20:
            length = data.read_int()
            for i in range(length):
                header = data.read_byte()
                if header == 9:
                    str_length = data.read_int()
                    data.skip(str_length)
                elif header == 101:
                    pass
                else:
                     raise Exception("Unknown binary type when expected string [type_id=%d]" % header)
            return TypeTreeNode(field_name, type_id)

        # Wrapped Binary Object.
        elif type_id == 27:
            length = data.read_int()
            inner_data = data.read_data(length)
            offset = data.read_int()
            return self.__collect_types(field_name, DataBuffer(inner_data))

        # Complex Object.
        elif type_id == 103:
            obj_version = data.read_byte()
            obj_flags = data.read_short()
            obj_type_id = data.read_int()
            obj_hash_code = data.read_int()
            obj_length = data.read_int()
            obj_schema_id = data.read_int()
            obj_schema_offset = data.read_int()

            obj_type = self.__get_type(obj_type_id)
            children = []
            
            for obj_field in obj_type.fields:
                child = self.__collect_types(obj_field.field_name, data)
                children.append(child)

            children_sorted = sorted(children, key=lambda child: child.name)
            permutation = [children_sorted.index(child) for child in children]
            children = children_sorted

            data.skip(obj_length - obj_schema_offset)

            return TypeTreeNode(field_name, type_id, children, permutation)
        
        else:
            raise Exception("Unknown binary type [type_id=%d]" % type_id)
        
    def __get_type(self, type_id):
        """Queries Apache Ignite information about type by type id."""
        self.write_int(14)      # Message length
        self.write_short(3002)  # Operation code
        self.write_long(0)      # Request ID
        self.write_int(type_id) # Type ID

        result_length = self.read_int()
        request_id = self.read_long()
        status = self.read_int()
        
        if status != 0:
            raise Exception("Get binary type completed with error [status=%d]" % status)

        binary_type_exists = self.read_byte()
        
        if binary_type_exists == 0:
            raise Exception("Binary type not found [type_id=%d] " % type_id)
        
        binary_type_id = self.read_int()
        binary_type_name = self.__parse_string()
        affinity_field_name = self.__parse_string()

        fields = []
        for i in range(self.read_int()):
            field_name = self.__parse_string();
            field_type_id = self.read_int()
            field_id = self.read_int()
        
            field = BinaryField(field_name, field_type_id, field_id)
            fields.append(field)

        is_enum = self.read_byte()
        if is_enum == 1:
            raise Exception("Enum fields are not supported yet")
            
        schema_cnt = self.read_int()
        for i in range(schema_cnt):
            schema_id = self.read_int()
            field_cnt = self.read_int()
            self.skip(field_cnt * 4)
            
        return BinaryType(binary_type_id, binary_type_name, fields)
    
    def __parse_string(self):
        """Parses string."""
        header = self.read_byte()
        if header == 9:
            length = self.read_int()
            return self.read_data(length).decode("utf-8")
        elif header == 101:
            return None
        else:
            raise Exception("Unknown binary type when expected string [type_id=%d]" % header)

class KafkaDataset(Dataset):
  """A Kafka Dataset that consumes the message.
  """

  def __init__(self, cache_name, host="localhost", port=10800, local=False, part=-1, partitioned=False, page_size=100):
    """Create a KafkaReader.

    Args:
      cache_name: Cache Name.
      host: Host.
      port: Port.
      local: Local.
      part: Part.
      page_size: Page size.
    """
    super(KafkaDataset, self).__init__()

    with IgniteClient(host, port) as client:
      client.handshake()
      self.cache_type = client.get_cache_type(cache_name)

    self.cache_name = ops.convert_to_tensor(cache_name, dtype=dtypes.string, name="cache_name")
    self.host = ops.convert_to_tensor(host, dtype=dtypes.string, name="host")
    self.port = ops.convert_to_tensor(port, dtype=dtypes.int32, name="port")
    self.local = ops.convert_to_tensor(local, dtype=dtypes.bool, name="local")
    self.part = ops.convert_to_tensor(part, dtype=dtypes.int32, name="part")
    self.partitioned = ops.convert_to_tensor(partitioned, dtype=dtypes.bool, name="partitioned")
    self.page_size = ops.convert_to_tensor(page_size, dtype=dtypes.int32, name="page_size")
    self.schema = ops.convert_to_tensor(self.cache_type.to_flat(), dtype=dtypes.int32, name="schema")
    self.permutation = ops.convert_to_tensor(self.cache_type.to_permutation(), dtype=dtypes.int32, name="permutation")

  def _as_variant_tensor(self):
    return gen_dataset_ops.kafka_dataset(self.cache_name, self.host, self.port, self.local, self.part, self.partitioned, self.page_size, self.schema, self.permutation)

  @property
  def output_classes(self):
    return self.cache_type.to_output_classes()

  @property
  def output_shapes(self):
    return self.cache_type.to_output_shapes()

  @property
  def output_types(self):
    return self.cache_type.to_output_types()
