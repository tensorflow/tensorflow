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
"""Table-driven test for decode_proto op.

This test is run once with each of the *.TestCase.pbtxt files
in the test directory.
"""
# Python3 preparedness imports.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from google.protobuf import text_format

from tensorflow.contrib.proto.python.kernel_tests import test_case
from tensorflow.contrib.proto.python.kernel_tests import test_example_pb2
from tensorflow.contrib.proto.python.ops import decode_proto_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import flags
from tensorflow.python.platform import test

FLAGS = flags.FLAGS

flags.DEFINE_string('message_text_file', None,
                    'A file containing a text serialized TestCase protobuf.')


class DecodeProtoOpTest(test_case.ProtoOpTestCase):

  def _compareValues(self, fd, vs, evs):
    """Compare lists/arrays of field values."""

    if len(vs) != len(evs):
      self.fail('Field %s decoded %d outputs, expected %d' %
                (fd.name, len(vs), len(evs)))
    for i, ev in enumerate(evs):
      # Special case fuzzy match for float32. TensorFlow seems to mess with
      # MAX_FLT slightly and the test doesn't work otherwise.
      # TODO(nix): ask on TF list about why MAX_FLT doesn't pass through.
      if fd.cpp_type == fd.CPPTYPE_FLOAT:
        # Numpy isclose() is better than assertIsClose() which uses an absolute
        # value comparison.
        self.assertTrue(
            np.isclose(vs[i], ev), 'expected %r, actual %r' % (ev, vs[i]))
      elif fd.cpp_type == fd.CPPTYPE_STRING:
        # In Python3 string tensor values will be represented as bytes, so we
        # reencode the proto values to match that.
        self.assertEqual(vs[i], ev.encode('ascii'))
      else:
        # Doubles and other types pass through unscathed.
        self.assertEqual(vs[i], ev)

  def _compareRepeatedPrimitiveValue(self, batch_shape, sizes, fields,
                                     field_dict):
    """Compare protos of type RepeatedPrimitiveValue.

    Args:
      batch_shape: the shape of the input tensor of serialized messages.
      sizes: int matrix of repeat counts returned by decode_proto
      fields: list of test_example_pb2.FieldSpec (types and expected values)
      field_dict: map from field names to decoded numpy tensors of values
    """

    # Check that expected values match.
    for field in fields:
      values = field_dict[field.name]
      self.assertEqual(dtypes.as_dtype(values.dtype), field.dtype)

      fd = field.expected.DESCRIPTOR.fields_by_name[field.name]

      # Values has the same shape as the input plus an extra
      # dimension for repeats.
      self.assertEqual(list(values.shape)[:-1], batch_shape)

      # Nested messages are represented as TF strings, requiring
      # some special handling.
      if field.name == 'message_value':
        vs = []
        for buf in values.flat:
          msg = test_example_pb2.PrimitiveValue()
          msg.ParseFromString(buf)
          vs.append(msg)
        evs = getattr(field.expected, field.name)
        if len(vs) != len(evs):
          self.fail('Field %s decoded %d outputs, expected %d' %
                    (fd.name, len(vs), len(evs)))
        for v, ev in zip(vs, evs):
          self.assertEqual(v, ev)
        continue

      # This can be a little confusing. For testing we are using
      # RepeatedPrimitiveValue in two ways: it's the proto that we
      # decode for testing, and it's used in the expected value as a
      # union type. The two cases are slightly different: this is the
      # second case.
      # We may be fetching the uint64_value from the test proto, but
      # in the expected proto we store it in the int64_value field
      # because TensorFlow doesn't support unsigned int64.
      tf_type_to_primitive_value_field = {
          dtypes.float32:
              'float_value',
          dtypes.float64:
              'double_value',
          dtypes.int32:
              'int32_value',
          dtypes.uint8:
              'uint8_value',
          dtypes.int8:
              'int8_value',
          dtypes.string:
              'string_value',
          dtypes.int64:
              'int64_value',
          dtypes.bool:
              'bool_value',
          # Unhandled TensorFlow types:
          # DT_INT16 DT_COMPLEX64 DT_QINT8 DT_QUINT8 DT_QINT32
          # DT_BFLOAT16 DT_QINT16 DT_QUINT16 DT_UINT16
      }
      tf_field_name = tf_type_to_primitive_value_field.get(field.dtype)
      if tf_field_name is None:
        self.fail('Unhandled tensorflow type %d' % field.dtype)

      self._compareValues(fd, values.flat,
                          getattr(field.expected, tf_field_name))

  def _runDecodeProtoTests(self, fields, case_sizes, batch_shape, batch,
                           message_type, message_format, sanitize,
                           force_disordered=False):
    """Run decode tests on a batch of messages.

    Args:
      fields: list of test_example_pb2.FieldSpec (types and expected values)
      case_sizes: expected sizes array
      batch_shape: the shape of the input tensor of serialized messages
      batch: list of serialized messages
      message_type: descriptor name for messages
      message_format: format of messages, 'text' or 'binary'
      sanitize: whether to sanitize binary protobuf inputs
      force_disordered: whether to force fields encoded out of order.
    """

    if force_disordered:
      # Exercise code path that handles out-of-order fields by prepending extra
      # fields with tag numbers higher than any real field. Note that this won't
      # work with sanitization because that forces reserialization using a
      # trusted decoder and encoder.
      assert not sanitize
      extra_fields = test_example_pb2.ExtraFields()
      extra_fields.string_value = 'IGNORE ME'
      extra_fields.bool_value = False
      extra_msg = extra_fields.SerializeToString()
      batch = [extra_msg + msg for msg in batch]

    # Numpy silently truncates the strings if you don't specify dtype=object.
    batch = np.array(batch, dtype=object)
    batch = np.reshape(batch, batch_shape)

    field_names = [f.name for f in fields]
    output_types = [f.dtype for f in fields]

    with self.test_session() as sess:
      sizes, vtensor = decode_proto_op.decode_proto(
          batch,
          message_type=message_type,
          field_names=field_names,
          output_types=output_types,
          message_format=message_format,
          sanitize=sanitize)

      vlist = sess.run([sizes] + vtensor)
      sizes = vlist[0]
      # Values is a list of tensors, one for each field.
      value_tensors = vlist[1:]

      # Check that the repeat sizes are correct.
      self.assertTrue(
          np.all(np.array(sizes.shape) == batch_shape + [len(field_names)]))

      # Check that the decoded sizes match the expected sizes.
      self.assertEqual(len(sizes.flat), len(case_sizes))
      self.assertTrue(
          np.all(sizes.flat == np.array(
              case_sizes, dtype=np.int32)))

      field_dict = dict(zip(field_names, value_tensors))

      self._compareRepeatedPrimitiveValue(batch_shape, sizes, fields,
                                          field_dict)

  def testBinary(self):
    with open(FLAGS.message_text_file, 'r') as fp:
      case = text_format.Parse(fp.read(), test_example_pb2.TestCase())

    batch = [primitive.SerializeToString() for primitive in case.primitive]
    self._runDecodeProtoTests(
        case.field,
        case.sizes,
        list(case.shape),
        batch,
        'tensorflow.contrib.proto.RepeatedPrimitiveValue',
        'binary',
        sanitize=False)

  def testBinaryDisordered(self):
    with open(FLAGS.message_text_file, 'r') as fp:
      case = text_format.Parse(fp.read(), test_example_pb2.TestCase())

    batch = [primitive.SerializeToString() for primitive in case.primitive]
    self._runDecodeProtoTests(
        case.field,
        case.sizes,
        list(case.shape),
        batch,
        'tensorflow.contrib.proto.RepeatedPrimitiveValue',
        'binary',
        sanitize=False,
        force_disordered=True)

  def testPacked(self):
    with open(FLAGS.message_text_file, 'r') as fp:
      case = text_format.Parse(fp.read(), test_example_pb2.TestCase())

    # Now try with the packed serialization.
    # We test the packed representations by loading the same test cases
    # using PackedPrimitiveValue instead of RepeatedPrimitiveValue.
    # To do this we rely on the text format being the same for packed and
    # unpacked fields, and reparse the test message using the packed version
    # of the proto.
    packed_batch = [
        # Note: float_format='.17g' is necessary to ensure preservation of
        # doubles and floats in text format.
        text_format.Parse(
            text_format.MessageToString(
                primitive, float_format='.17g'),
            test_example_pb2.PackedPrimitiveValue()).SerializeToString()
        for primitive in case.primitive
    ]

    self._runDecodeProtoTests(
        case.field,
        case.sizes,
        list(case.shape),
        packed_batch,
        'tensorflow.contrib.proto.PackedPrimitiveValue',
        'binary',
        sanitize=False)

  def testText(self):
    with open(FLAGS.message_text_file, 'r') as fp:
      case = text_format.Parse(fp.read(), test_example_pb2.TestCase())

    # Note: float_format='.17g' is necessary to ensure preservation of
    # doubles and floats in text format.
    text_batch = [
        text_format.MessageToString(
            primitive, float_format='.17g') for primitive in case.primitive
    ]

    self._runDecodeProtoTests(
        case.field,
        case.sizes,
        list(case.shape),
        text_batch,
        'tensorflow.contrib.proto.RepeatedPrimitiveValue',
        'text',
        sanitize=False)

  def testSanitizerGood(self):
    with open(FLAGS.message_text_file, 'r') as fp:
      case = text_format.Parse(fp.read(), test_example_pb2.TestCase())

    batch = [primitive.SerializeToString() for primitive in case.primitive]
    self._runDecodeProtoTests(
        case.field,
        case.sizes,
        list(case.shape),
        batch,
        'tensorflow.contrib.proto.RepeatedPrimitiveValue',
        'binary',
        sanitize=True)


if __name__ == '__main__':
  test.main()
