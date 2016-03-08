# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for tensorflow.python.framework.importer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class TypesTest(test_util.TensorFlowTestCase):

  def testAllTypesConstructible(self):
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      self.assertEqual(
          datatype_enum, tf.DType(datatype_enum).as_datatype_enum)

  def testAllTypesConvertibleToDType(self):
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      self.assertEqual(
          datatype_enum, tf.as_dtype(datatype_enum).as_datatype_enum)

  def testAllTypesConvertibleToNumpyDtype(self):
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      dtype = tf.as_dtype(datatype_enum)
      numpy_dtype = dtype.as_numpy_dtype
      _ = np.empty((1, 1, 1, 1), dtype=numpy_dtype)
      if dtype.base_dtype != tf.bfloat16:
        # NOTE(touts): Intentionally no way to feed a DT_BFLOAT16.
        self.assertEqual(tf.as_dtype(datatype_enum).base_dtype,
                         tf.as_dtype(numpy_dtype))

  def testInvalid(self):
    with self.assertRaises(TypeError):
      tf.DType(types_pb2.DT_INVALID)
    with self.assertRaises(TypeError):
      tf.as_dtype(types_pb2.DT_INVALID)

  def testNumpyConversion(self):
    self.assertIs(tf.float32, tf.as_dtype(np.float32))
    self.assertIs(tf.float64, tf.as_dtype(np.float64))
    self.assertIs(tf.int32, tf.as_dtype(np.int32))
    self.assertIs(tf.int64, tf.as_dtype(np.int64))
    self.assertIs(tf.uint8, tf.as_dtype(np.uint8))
    self.assertIs(tf.uint16, tf.as_dtype(np.uint16))
    self.assertIs(tf.int16, tf.as_dtype(np.int16))
    self.assertIs(tf.int8, tf.as_dtype(np.int8))
    self.assertIs(tf.complex64, tf.as_dtype(np.complex64))
    self.assertIs(tf.complex128, tf.as_dtype(np.complex128))
    self.assertIs(tf.string, tf.as_dtype(np.object))
    self.assertIs(tf.string, tf.as_dtype(np.array(["foo", "bar"]).dtype))
    self.assertIs(tf.bool, tf.as_dtype(np.bool))
    with self.assertRaises(TypeError):
      tf.as_dtype(np.dtype([("f1", np.uint), ("f2", np.int32)]))

  def testRealDtype(self):
    for dtype in [tf.float32, tf.float64, tf.bool, tf.uint8, tf.int8, tf.int16,
                  tf.int32, tf.int64]:
      self.assertIs(dtype.real_dtype, dtype)
    self.assertIs(tf.complex64.real_dtype, tf.float32)
    self.assertIs(tf.complex128.real_dtype, tf.float64)

  def testStringConversion(self):
    self.assertIs(tf.float32, tf.as_dtype("float32"))
    self.assertIs(tf.float64, tf.as_dtype("float64"))
    self.assertIs(tf.int32, tf.as_dtype("int32"))
    self.assertIs(tf.uint8, tf.as_dtype("uint8"))
    self.assertIs(tf.uint16, tf.as_dtype("uint16"))
    self.assertIs(tf.int16, tf.as_dtype("int16"))
    self.assertIs(tf.int8, tf.as_dtype("int8"))
    self.assertIs(tf.string, tf.as_dtype("string"))
    self.assertIs(tf.complex64, tf.as_dtype("complex64"))
    self.assertIs(tf.complex128, tf.as_dtype("complex128"))
    self.assertIs(tf.int64, tf.as_dtype("int64"))
    self.assertIs(tf.bool, tf.as_dtype("bool"))
    self.assertIs(tf.qint8, tf.as_dtype("qint8"))
    self.assertIs(tf.quint8, tf.as_dtype("quint8"))
    self.assertIs(tf.qint32, tf.as_dtype("qint32"))
    self.assertIs(tf.bfloat16, tf.as_dtype("bfloat16"))
    self.assertIs(tf.float32_ref, tf.as_dtype("float32_ref"))
    self.assertIs(tf.float64_ref, tf.as_dtype("float64_ref"))
    self.assertIs(tf.int32_ref, tf.as_dtype("int32_ref"))
    self.assertIs(tf.uint8_ref, tf.as_dtype("uint8_ref"))
    self.assertIs(tf.int16_ref, tf.as_dtype("int16_ref"))
    self.assertIs(tf.int8_ref, tf.as_dtype("int8_ref"))
    self.assertIs(tf.string_ref, tf.as_dtype("string_ref"))
    self.assertIs(tf.complex64_ref, tf.as_dtype("complex64_ref"))
    self.assertIs(tf.complex128_ref, tf.as_dtype("complex128_ref"))
    self.assertIs(tf.int64_ref, tf.as_dtype("int64_ref"))
    self.assertIs(tf.bool_ref, tf.as_dtype("bool_ref"))
    self.assertIs(tf.qint8_ref, tf.as_dtype("qint8_ref"))
    self.assertIs(tf.quint8_ref, tf.as_dtype("quint8_ref"))
    self.assertIs(tf.qint32_ref, tf.as_dtype("qint32_ref"))
    self.assertIs(tf.bfloat16_ref, tf.as_dtype("bfloat16_ref"))
    with self.assertRaises(TypeError):
      tf.as_dtype("not_a_type")

  def testDTypesHaveUniqueNames(self):
    dtypes = []
    names = set()
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      dtype = tf.as_dtype(datatype_enum)
      dtypes.append(dtype)
      names.add(dtype.name)
    self.assertEqual(len(dtypes), len(names))

  def testIsInteger(self):
    self.assertEqual(tf.as_dtype("int8").is_integer, True)
    self.assertEqual(tf.as_dtype("int16").is_integer, True)
    self.assertEqual(tf.as_dtype("int32").is_integer, True)
    self.assertEqual(tf.as_dtype("int64").is_integer, True)
    self.assertEqual(tf.as_dtype("uint8").is_integer, True)
    self.assertEqual(tf.as_dtype("uint16").is_integer, True)
    self.assertEqual(tf.as_dtype("complex64").is_integer, False)
    self.assertEqual(tf.as_dtype("complex128").is_integer, False)
    self.assertEqual(tf.as_dtype("float").is_integer, False)
    self.assertEqual(tf.as_dtype("double").is_integer, False)
    self.assertEqual(tf.as_dtype("string").is_integer, False)
    self.assertEqual(tf.as_dtype("bool").is_integer, False)

  def testIsFloating(self):
    self.assertEqual(tf.as_dtype("int8").is_floating, False)
    self.assertEqual(tf.as_dtype("int16").is_floating, False)
    self.assertEqual(tf.as_dtype("int32").is_floating, False)
    self.assertEqual(tf.as_dtype("int64").is_floating, False)
    self.assertEqual(tf.as_dtype("uint8").is_floating, False)
    self.assertEqual(tf.as_dtype("uint16").is_floating, False)
    self.assertEqual(tf.as_dtype("complex64").is_floating, False)
    self.assertEqual(tf.as_dtype("complex128").is_floating, False)
    self.assertEqual(tf.as_dtype("float32").is_floating, True)
    self.assertEqual(tf.as_dtype("float64").is_floating, True)
    self.assertEqual(tf.as_dtype("string").is_floating, False)
    self.assertEqual(tf.as_dtype("bool").is_floating, False)

  def testIsComplex(self):
    self.assertEqual(tf.as_dtype("int8").is_complex, False)
    self.assertEqual(tf.as_dtype("int16").is_complex, False)
    self.assertEqual(tf.as_dtype("int32").is_complex, False)
    self.assertEqual(tf.as_dtype("int64").is_complex, False)
    self.assertEqual(tf.as_dtype("uint8").is_complex, False)
    self.assertEqual(tf.as_dtype("uint16").is_complex, False)
    self.assertEqual(tf.as_dtype("complex64").is_complex, True)
    self.assertEqual(tf.as_dtype("complex128").is_complex, True)
    self.assertEqual(tf.as_dtype("float32").is_complex, False)
    self.assertEqual(tf.as_dtype("float64").is_complex, False)
    self.assertEqual(tf.as_dtype("string").is_complex, False)
    self.assertEqual(tf.as_dtype("bool").is_complex, False)

  def testIsUnsigned(self):
    self.assertEqual(tf.as_dtype("int8").is_unsigned, False)
    self.assertEqual(tf.as_dtype("int16").is_unsigned, False)
    self.assertEqual(tf.as_dtype("int32").is_unsigned, False)
    self.assertEqual(tf.as_dtype("int64").is_unsigned, False)
    self.assertEqual(tf.as_dtype("uint8").is_unsigned, True)
    self.assertEqual(tf.as_dtype("uint16").is_unsigned, True)
    self.assertEqual(tf.as_dtype("float32").is_unsigned, False)
    self.assertEqual(tf.as_dtype("float64").is_unsigned, False)
    self.assertEqual(tf.as_dtype("bool").is_unsigned, False)
    self.assertEqual(tf.as_dtype("string").is_unsigned, False)
    self.assertEqual(tf.as_dtype("complex64").is_unsigned, False)
    self.assertEqual(tf.as_dtype("complex128").is_unsigned, False)

  def testMinMax(self):
    # make sure min/max evaluates for all data types that have min/max
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      dtype = tf.as_dtype(datatype_enum)
      numpy_dtype = dtype.as_numpy_dtype

      # ignore types for which there are no minimum/maximum (or we cannot
      # compute it, such as for the q* types)
      if (dtype.is_quantized or
          dtype.base_dtype == tf.bool or
          dtype.base_dtype == tf.string or
          dtype.base_dtype == tf.complex64 or
          dtype.base_dtype == tf.complex128):
        continue

      print("%s: %s - %s" % (dtype, dtype.min, dtype.max))

      # check some values that are known
      if numpy_dtype == np.bool_:
        self.assertEquals(dtype.min, 0)
        self.assertEquals(dtype.max, 1)
      if numpy_dtype == np.int8:
        self.assertEquals(dtype.min, -128)
        self.assertEquals(dtype.max, 127)
      if numpy_dtype == np.int16:
        self.assertEquals(dtype.min, -32768)
        self.assertEquals(dtype.max, 32767)
      if numpy_dtype == np.int32:
        self.assertEquals(dtype.min, -2147483648)
        self.assertEquals(dtype.max, 2147483647)
      if numpy_dtype == np.int64:
        self.assertEquals(dtype.min, -9223372036854775808)
        self.assertEquals(dtype.max, 9223372036854775807)
      if numpy_dtype == np.uint8:
        self.assertEquals(dtype.min, 0)
        self.assertEquals(dtype.max, 255)
      if numpy_dtype == np.uint16:
        if dtype == tf.uint16:
          self.assertEquals(dtype.min, 0)
          self.assertEquals(dtype.max, 65535)
        elif dtype == tf.bfloat16:
          self.assertEquals(dtype.min, 0)
          self.assertEquals(dtype.max, 4294967295)
      if numpy_dtype == np.uint32:
        self.assertEquals(dtype.min, 0)
        self.assertEquals(dtype.max, 18446744073709551615)
      if numpy_dtype in (np.float16, np.float32, np.float64):
        self.assertEquals(dtype.min, np.finfo(numpy_dtype).min)
        self.assertEquals(dtype.max, np.finfo(numpy_dtype).max)

  def testRepr(self):
    for enum, name in dtypes._TYPE_TO_STRING.items():
      dtype = tf.DType(enum)
      self.assertEquals(repr(dtype), 'tf.' + name)
      dtype2 = eval(repr(dtype))
      self.assertEquals(type(dtype2), tf.DType)
      self.assertEquals(dtype, dtype2)


if __name__ == "__main__":
  googletest.main()
