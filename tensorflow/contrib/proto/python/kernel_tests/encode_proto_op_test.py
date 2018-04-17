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
"""Table-driven test for encode_proto op.

This test is run once with each of the *.TestCase.pbtxt files
in the test directory.

It tests that encode_proto is a lossless inverse of decode_proto
(for the specified fields).
"""
# Python3 readiness boilerplate
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from google.protobuf import text_format

from tensorflow.contrib.proto.python.kernel_tests import test_case
from tensorflow.contrib.proto.python.kernel_tests import test_example_pb2
from tensorflow.contrib.proto.python.ops import decode_proto_op
from tensorflow.contrib.proto.python.ops import encode_proto_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import flags
from tensorflow.python.platform import test

FLAGS = flags.FLAGS

flags.DEFINE_string('message_text_file', None,
                    'A file containing a text serialized TestCase protobuf.')


class EncodeProtoOpTest(test_case.ProtoOpTestCase):

  def testBadInputs(self):
    # Invalid field name
    with self.test_session():
      with self.assertRaisesOpError('Unknown field: non_existent_field'):
        encode_proto_op.encode_proto(
            sizes=[[1]],
            values=[np.array([[0.0]], dtype=np.int32)],
            message_type='tensorflow.contrib.proto.RepeatedPrimitiveValue',
            field_names=['non_existent_field']).eval()

    # Incorrect types.
    with self.test_session():
      with self.assertRaisesOpError(
          'Incompatible type for field double_value.'):
        encode_proto_op.encode_proto(
            sizes=[[1]],
            values=[np.array([[0.0]], dtype=np.int32)],
            message_type='tensorflow.contrib.proto.RepeatedPrimitiveValue',
            field_names=['double_value']).eval()

    # Incorrect shapes of sizes.
    with self.test_session():
      with self.assertRaisesOpError(
          r'sizes should be batch_size \+ \[len\(field_names\)\]'):
        sizes = array_ops.placeholder(dtypes.int32)
        values = array_ops.placeholder(dtypes.float64)
        encode_proto_op.encode_proto(
            sizes=sizes,
            values=[values],
            message_type='tensorflow.contrib.proto.RepeatedPrimitiveValue',
            field_names=['double_value']).eval(feed_dict={
                sizes: [[[0, 0]]],
                values: [[0.0]]
            })

    # Inconsistent shapes of values.
    with self.test_session():
      with self.assertRaisesOpError(
          'Values must match up to the last dimension'):
        sizes = array_ops.placeholder(dtypes.int32)
        values1 = array_ops.placeholder(dtypes.float64)
        values2 = array_ops.placeholder(dtypes.int32)
        (encode_proto_op.encode_proto(
            sizes=[[1, 1]],
            values=[values1, values2],
            message_type='tensorflow.contrib.proto.RepeatedPrimitiveValue',
            field_names=['double_value', 'int32_value']).eval(feed_dict={
                values1: [[0.0]],
                values2: [[0], [0]]
            }))

  def _testRoundtrip(self, in_bufs, message_type, fields):

    field_names = [f.name for f in fields]
    out_types = [f.dtype for f in fields]

    with self.test_session() as sess:
      sizes, field_tensors = decode_proto_op.decode_proto(
          in_bufs,
          message_type=message_type,
          field_names=field_names,
          output_types=out_types)

      out_tensors = encode_proto_op.encode_proto(
          sizes,
          field_tensors,
          message_type=message_type,
          field_names=field_names)

      out_bufs, = sess.run([out_tensors])

      # Check that the re-encoded tensor has the same shape.
      self.assertEqual(in_bufs.shape, out_bufs.shape)

      # Compare the input and output.
      for in_buf, out_buf in zip(in_bufs.flat, out_bufs.flat):
        in_obj = test_example_pb2.RepeatedPrimitiveValue()
        in_obj.ParseFromString(in_buf)

        out_obj = test_example_pb2.RepeatedPrimitiveValue()
        out_obj.ParseFromString(out_buf)

        # Check that the deserialized objects are identical.
        self.assertEqual(in_obj, out_obj)

        # Check that the input and output serialized messages are identical.
        # If we fail here, there is a difference in the serialized
        # representation but the new serialization still parses. This could
        # be harmless (a change in map ordering?) or it could be bad (e.g.
        # loss of packing in the encoding).
        self.assertEqual(in_buf, out_buf)

  def testRoundtrip(self):
    with open(FLAGS.message_text_file, 'r') as fp:
      case = text_format.Parse(fp.read(), test_example_pb2.TestCase())

    in_bufs = [primitive.SerializeToString() for primitive in case.primitive]

    # np.array silently truncates strings if you don't specify dtype=object.
    in_bufs = np.reshape(np.array(in_bufs, dtype=object), list(case.shape))
    return self._testRoundtrip(
        in_bufs, 'tensorflow.contrib.proto.RepeatedPrimitiveValue', case.field)

  def testRoundtripPacked(self):
    with open(FLAGS.message_text_file, 'r') as fp:
      case = text_format.Parse(fp.read(), test_example_pb2.TestCase())

    # Now try with the packed serialization.
    # We test the packed representations by loading the same test cases
    # using PackedPrimitiveValue instead of RepeatedPrimitiveValue.
    # To do this we rely on the text format being the same for packed and
    # unpacked fields, and reparse the test message using the packed version
    # of the proto.
    in_bufs = [
        # Note: float_format='.17g' is necessary to ensure preservation of
        # doubles and floats in text format.
        text_format.Parse(
            text_format.MessageToString(
                primitive, float_format='.17g'),
            test_example_pb2.PackedPrimitiveValue()).SerializeToString()
        for primitive in case.primitive
    ]

    # np.array silently truncates strings if you don't specify dtype=object.
    in_bufs = np.reshape(np.array(in_bufs, dtype=object), list(case.shape))
    return self._testRoundtrip(
        in_bufs, 'tensorflow.contrib.proto.PackedPrimitiveValue', case.field)


if __name__ == '__main__':
  test.main()
