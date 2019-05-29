# =============================================================================
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
# =============================================================================
"""Tests for proto ops reading descriptors from other sources."""
# Python3 preparedness imports.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from google.protobuf.descriptor_pb2 import FieldDescriptorProto
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from tensorflow.python.framework import dtypes
from tensorflow.python.kernel_tests.proto import proto_op_test_base as test_base
from tensorflow.python.platform import test


class DescriptorSourceTestBase(test.TestCase):
  """Base class for testing descriptor sources."""

  def __init__(self, decode_module, encode_module, methodName='runTest'):  # pylint: disable=invalid-name
    """DescriptorSourceTestBase initializer.

    Args:
      decode_module: a module containing the `decode_proto_op` method
      encode_module: a module containing the `encode_proto_op` method
      methodName: the name of the test method (same as for test.TestCase)
    """

    super(DescriptorSourceTestBase, self).__init__(methodName)
    self._decode_module = decode_module
    self._encode_module = encode_module

  # NOTE: We generate the descriptor programmatically instead of via a compiler
  # because of differences between different versions of the compiler.
  #
  # The generated descriptor should capture the subset of `test_example.proto`
  # used in `test_base.simple_test_case()`.
  def _createDescriptorFile(self):
    set_proto = FileDescriptorSet()

    file_proto = set_proto.file.add(
        name='types.proto',
        package='tensorflow',
        syntax='proto3')
    enum_proto = file_proto.enum_type.add(name='DataType')
    enum_proto.value.add(name='DT_DOUBLE', number=0)
    enum_proto.value.add(name='DT_BOOL', number=1)

    file_proto = set_proto.file.add(
        name='test_example.proto',
        package='tensorflow.contrib.proto',
        dependency=['types.proto'])
    message_proto = file_proto.message_type.add(name='TestCase')
    message_proto.field.add(
        name='values',
        number=1,
        type=FieldDescriptorProto.TYPE_MESSAGE,
        type_name='.tensorflow.contrib.proto.TestValue',
        label=FieldDescriptorProto.LABEL_REPEATED)
    message_proto.field.add(
        name='shapes',
        number=2,
        type=FieldDescriptorProto.TYPE_INT32,
        label=FieldDescriptorProto.LABEL_REPEATED)
    message_proto.field.add(
        name='sizes',
        number=3,
        type=FieldDescriptorProto.TYPE_INT32,
        label=FieldDescriptorProto.LABEL_REPEATED)
    message_proto.field.add(
        name='fields',
        number=4,
        type=FieldDescriptorProto.TYPE_MESSAGE,
        type_name='.tensorflow.contrib.proto.FieldSpec',
        label=FieldDescriptorProto.LABEL_REPEATED)

    message_proto = file_proto.message_type.add(
        name='TestValue')
    message_proto.field.add(
        name='double_value',
        number=1,
        type=FieldDescriptorProto.TYPE_DOUBLE,
        label=FieldDescriptorProto.LABEL_REPEATED)
    message_proto.field.add(
        name='bool_value',
        number=2,
        type=FieldDescriptorProto.TYPE_BOOL,
        label=FieldDescriptorProto.LABEL_REPEATED)

    message_proto = file_proto.message_type.add(
        name='FieldSpec')
    message_proto.field.add(
        name='name',
        number=1,
        type=FieldDescriptorProto.TYPE_STRING,
        label=FieldDescriptorProto.LABEL_OPTIONAL)
    message_proto.field.add(
        name='dtype',
        number=2,
        type=FieldDescriptorProto.TYPE_ENUM,
        type_name='.tensorflow.DataType',
        label=FieldDescriptorProto.LABEL_OPTIONAL)
    message_proto.field.add(
        name='value',
        number=3,
        type=FieldDescriptorProto.TYPE_MESSAGE,
        type_name='.tensorflow.contrib.proto.TestValue',
        label=FieldDescriptorProto.LABEL_OPTIONAL)

    fn = os.path.join(self.get_temp_dir(), 'descriptor.pb')
    with open(fn, 'wb') as f:
      f.write(set_proto.SerializeToString())
    return fn

  def _testRoundtrip(self, descriptor_source):
    # Numpy silently truncates the strings if you don't specify dtype=object.
    in_bufs = np.array(
        [test_base.ProtoOpTestBase.simple_test_case().SerializeToString()],
        dtype=object)
    message_type = 'tensorflow.contrib.proto.TestCase'
    field_names = ['values', 'shapes', 'sizes', 'fields']
    tensor_types = [dtypes.string, dtypes.int32, dtypes.int32, dtypes.string]

    with self.cached_session() as sess:
      sizes, field_tensors = self._decode_module.decode_proto(
          in_bufs,
          message_type=message_type,
          field_names=field_names,
          output_types=tensor_types,
          descriptor_source=descriptor_source)

      out_tensors = self._encode_module.encode_proto(
          sizes,
          field_tensors,
          message_type=message_type,
          field_names=field_names,
          descriptor_source=descriptor_source)

      out_bufs, = sess.run([out_tensors])

      # Check that the re-encoded tensor has the same shape.
      self.assertEqual(in_bufs.shape, out_bufs.shape)

      # Compare the input and output.
      for in_buf, out_buf in zip(in_bufs.flat, out_bufs.flat):
        # Check that the input and output serialized messages are identical.
        # If we fail here, there is a difference in the serialized
        # representation but the new serialization still parses. This could
        # be harmless (a change in map ordering?) or it could be bad (e.g.
        # loss of packing in the encoding).
        self.assertEqual(in_buf, out_buf)

  def testWithFileDescriptorSet(self):
    # First try parsing with a local proto db, which should fail.
    with self.assertRaisesOpError('No descriptor found for message type'):
      self._testRoundtrip('local://')

    # Now try parsing with a FileDescriptorSet which contains the test proto.
    descriptor_file = self._createDescriptorFile()
    self._testRoundtrip(descriptor_file)
