# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.dtypes."""

import numpy as np

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import _dtypes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


def _is_numeric_dtype_enum(datatype_enum):
  non_numeric_dtypes = [types_pb2.DT_VARIANT,
                        types_pb2.DT_VARIANT_REF,
                        types_pb2.DT_INVALID,
                        types_pb2.DT_RESOURCE,
                        types_pb2.DT_RESOURCE_REF]
  return datatype_enum not in non_numeric_dtypes


class TypesTest(test_util.TensorFlowTestCase):

  def testAllTypesConstructible(self):
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      self.assertEqual(datatype_enum,
                       dtypes.DType(datatype_enum).as_datatype_enum)

  def testAllTypesConvertibleToDType(self):
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      dt = dtypes.as_dtype(datatype_enum)
      self.assertEqual(datatype_enum, dt.as_datatype_enum)

  def testAllTypesConvertibleToNumpyDtype(self):
    for datatype_enum in types_pb2.DataType.values():
      if not _is_numeric_dtype_enum(datatype_enum):
        continue
      dtype = dtypes.as_dtype(datatype_enum)
      numpy_dtype = dtype.as_numpy_dtype
      _ = np.empty((1, 1, 1, 1), dtype=numpy_dtype)
      if dtype.base_dtype != dtypes.bfloat16:
        # NOTE(touts): Intentionally no way to feed a DT_BFLOAT16.
        self.assertEqual(
            dtypes.as_dtype(datatype_enum).base_dtype,
            dtypes.as_dtype(numpy_dtype))

  def testAllPybind11DTypeConvertibleToDType(self):
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      dtype = _dtypes.DType(datatype_enum)
      self.assertEqual(dtypes.as_dtype(datatype_enum), dtype)

  def testInvalid(self):
    with self.assertRaises(TypeError):
      dtypes.DType(types_pb2.DT_INVALID)
    with self.assertRaises(TypeError):
      dtypes.as_dtype(types_pb2.DT_INVALID)

  def testNumpyConversion(self):
    self.assertIs(dtypes.float32, dtypes.as_dtype(np.float32))
    self.assertIs(dtypes.float64, dtypes.as_dtype(np.float64))
    self.assertIs(dtypes.int32, dtypes.as_dtype(np.int32))
    self.assertIs(dtypes.int64, dtypes.as_dtype(np.int64))
    self.assertIs(dtypes.uint8, dtypes.as_dtype(np.uint8))
    self.assertIs(dtypes.uint16, dtypes.as_dtype(np.uint16))
    self.assertIs(dtypes.int16, dtypes.as_dtype(np.int16))
    self.assertIs(dtypes.int8, dtypes.as_dtype(np.int8))
    self.assertIs(dtypes.complex64, dtypes.as_dtype(np.complex64))
    self.assertIs(dtypes.complex128, dtypes.as_dtype(np.complex128))
    self.assertIs(dtypes.string, dtypes.as_dtype(np.object_))
    self.assertIs(dtypes.string,
                  dtypes.as_dtype(np.array(["foo", "bar"]).dtype))
    self.assertIs(dtypes.bool, dtypes.as_dtype(np.bool_))
    with self.assertRaises(TypeError):
      dtypes.as_dtype(np.dtype([("f1", np.uint), ("f2", np.int32)]))

    class AnObject(object):
      dtype = "f4"

    self.assertIs(dtypes.float32, dtypes.as_dtype(AnObject))

    class AnotherObject(object):
      dtype = np.dtype(np.complex64)

    self.assertIs(dtypes.complex64, dtypes.as_dtype(AnotherObject))

  def testRealDtype(self):
    for dtype in [
        dtypes.float32, dtypes.float64, dtypes.bool, dtypes.uint8, dtypes.int8,
        dtypes.int16, dtypes.int32, dtypes.int64
    ]:
      self.assertIs(dtype.real_dtype, dtype)
    self.assertIs(dtypes.complex64.real_dtype, dtypes.float32)
    self.assertIs(dtypes.complex128.real_dtype, dtypes.float64)

  def testStringConversion(self):
    self.assertIs(dtypes.float32, dtypes.as_dtype("float32"))
    self.assertIs(dtypes.float64, dtypes.as_dtype("float64"))
    self.assertIs(dtypes.int32, dtypes.as_dtype("int32"))
    self.assertIs(dtypes.uint8, dtypes.as_dtype("uint8"))
    self.assertIs(dtypes.uint16, dtypes.as_dtype("uint16"))
    self.assertIs(dtypes.int16, dtypes.as_dtype("int16"))
    self.assertIs(dtypes.int8, dtypes.as_dtype("int8"))
    self.assertIs(dtypes.string, dtypes.as_dtype("string"))
    self.assertIs(dtypes.complex64, dtypes.as_dtype("complex64"))
    self.assertIs(dtypes.complex128, dtypes.as_dtype("complex128"))
    self.assertIs(dtypes.int64, dtypes.as_dtype("int64"))
    self.assertIs(dtypes.bool, dtypes.as_dtype("bool"))
    self.assertIs(dtypes.qint8, dtypes.as_dtype("qint8"))
    self.assertIs(dtypes.quint8, dtypes.as_dtype("quint8"))
    self.assertIs(dtypes.qint32, dtypes.as_dtype("qint32"))
    self.assertIs(dtypes.bfloat16, dtypes.as_dtype("bfloat16"))
    self.assertIs(dtypes.float32_ref, dtypes.as_dtype("float32_ref"))
    self.assertIs(dtypes.float64_ref, dtypes.as_dtype("float64_ref"))
    self.assertIs(dtypes.int32_ref, dtypes.as_dtype("int32_ref"))
    self.assertIs(dtypes.uint8_ref, dtypes.as_dtype("uint8_ref"))
    self.assertIs(dtypes.int16_ref, dtypes.as_dtype("int16_ref"))
    self.assertIs(dtypes.int8_ref, dtypes.as_dtype("int8_ref"))
    self.assertIs(dtypes.string_ref, dtypes.as_dtype("string_ref"))
    self.assertIs(dtypes.complex64_ref, dtypes.as_dtype("complex64_ref"))
    self.assertIs(dtypes.complex128_ref, dtypes.as_dtype("complex128_ref"))
    self.assertIs(dtypes.int64_ref, dtypes.as_dtype("int64_ref"))
    self.assertIs(dtypes.bool_ref, dtypes.as_dtype("bool_ref"))
    self.assertIs(dtypes.qint8_ref, dtypes.as_dtype("qint8_ref"))
    self.assertIs(dtypes.quint8_ref, dtypes.as_dtype("quint8_ref"))
    self.assertIs(dtypes.qint32_ref, dtypes.as_dtype("qint32_ref"))
    self.assertIs(dtypes.bfloat16_ref, dtypes.as_dtype("bfloat16_ref"))
    with self.assertRaises(TypeError):
      dtypes.as_dtype("not_a_type")

  def testDTypesHaveUniqueNames(self):
    dtypez = []
    names = set()
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      dtype = dtypes.as_dtype(datatype_enum)
      dtypez.append(dtype)
      names.add(dtype.name)
    self.assertEqual(len(dtypez), len(names))

  def testIsInteger(self):
    self.assertEqual(dtypes.as_dtype("int8").is_integer, True)
    self.assertEqual(dtypes.as_dtype("int16").is_integer, True)
    self.assertEqual(dtypes.as_dtype("int32").is_integer, True)
    self.assertEqual(dtypes.as_dtype("int64").is_integer, True)
    self.assertEqual(dtypes.as_dtype("uint8").is_integer, True)
    self.assertEqual(dtypes.as_dtype("uint16").is_integer, True)
    self.assertEqual(dtypes.as_dtype("complex64").is_integer, False)
    self.assertEqual(dtypes.as_dtype("complex128").is_integer, False)
    self.assertEqual(dtypes.as_dtype("float").is_integer, False)
    self.assertEqual(dtypes.as_dtype("double").is_integer, False)
    self.assertEqual(dtypes.as_dtype("string").is_integer, False)
    self.assertEqual(dtypes.as_dtype("bool").is_integer, False)
    self.assertEqual(dtypes.as_dtype("bfloat16").is_integer, False)
    self.assertEqual(dtypes.as_dtype("qint8").is_integer, False)
    self.assertEqual(dtypes.as_dtype("qint16").is_integer, False)
    self.assertEqual(dtypes.as_dtype("qint32").is_integer, False)
    self.assertEqual(dtypes.as_dtype("quint8").is_integer, False)
    self.assertEqual(dtypes.as_dtype("quint16").is_integer, False)

  def testIsFloating(self):
    self.assertEqual(dtypes.as_dtype("int8").is_floating, False)
    self.assertEqual(dtypes.as_dtype("int16").is_floating, False)
    self.assertEqual(dtypes.as_dtype("int32").is_floating, False)
    self.assertEqual(dtypes.as_dtype("int64").is_floating, False)
    self.assertEqual(dtypes.as_dtype("uint8").is_floating, False)
    self.assertEqual(dtypes.as_dtype("uint16").is_floating, False)
    self.assertEqual(dtypes.as_dtype("complex64").is_floating, False)
    self.assertEqual(dtypes.as_dtype("complex128").is_floating, False)
    self.assertEqual(dtypes.as_dtype("float32").is_floating, True)
    self.assertEqual(dtypes.as_dtype("float64").is_floating, True)
    self.assertEqual(dtypes.as_dtype("string").is_floating, False)
    self.assertEqual(dtypes.as_dtype("bool").is_floating, False)
    self.assertEqual(dtypes.as_dtype("bfloat16").is_floating, True)
    self.assertEqual(dtypes.as_dtype("qint8").is_floating, False)
    self.assertEqual(dtypes.as_dtype("qint16").is_floating, False)
    self.assertEqual(dtypes.as_dtype("qint32").is_floating, False)
    self.assertEqual(dtypes.as_dtype("quint8").is_floating, False)
    self.assertEqual(dtypes.as_dtype("quint16").is_floating, False)

  def testIsComplex(self):
    self.assertEqual(dtypes.as_dtype("int8").is_complex, False)
    self.assertEqual(dtypes.as_dtype("int16").is_complex, False)
    self.assertEqual(dtypes.as_dtype("int32").is_complex, False)
    self.assertEqual(dtypes.as_dtype("int64").is_complex, False)
    self.assertEqual(dtypes.as_dtype("uint8").is_complex, False)
    self.assertEqual(dtypes.as_dtype("uint16").is_complex, False)
    self.assertEqual(dtypes.as_dtype("complex64").is_complex, True)
    self.assertEqual(dtypes.as_dtype("complex128").is_complex, True)
    self.assertEqual(dtypes.as_dtype("float32").is_complex, False)
    self.assertEqual(dtypes.as_dtype("float64").is_complex, False)
    self.assertEqual(dtypes.as_dtype("string").is_complex, False)
    self.assertEqual(dtypes.as_dtype("bool").is_complex, False)
    self.assertEqual(dtypes.as_dtype("bfloat16").is_complex, False)
    self.assertEqual(dtypes.as_dtype("qint8").is_complex, False)
    self.assertEqual(dtypes.as_dtype("qint16").is_complex, False)
    self.assertEqual(dtypes.as_dtype("qint32").is_complex, False)
    self.assertEqual(dtypes.as_dtype("quint8").is_complex, False)
    self.assertEqual(dtypes.as_dtype("quint16").is_complex, False)

  def testIsUnsigned(self):
    self.assertEqual(dtypes.as_dtype("int8").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("int16").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("int32").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("int64").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("uint8").is_unsigned, True)
    self.assertEqual(dtypes.as_dtype("uint16").is_unsigned, True)
    self.assertEqual(dtypes.as_dtype("float32").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("float64").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("bool").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("string").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("complex64").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("complex128").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("bfloat16").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("qint8").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("qint16").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("qint32").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("quint8").is_unsigned, False)
    self.assertEqual(dtypes.as_dtype("quint16").is_unsigned, False)

  def testMinMax(self):
    # make sure min/max evaluates for all data types that have min/max
    for datatype_enum in types_pb2.DataType.values():
      if not _is_numeric_dtype_enum(datatype_enum):
        continue
      dtype = dtypes.as_dtype(datatype_enum)
      numpy_dtype = dtype.as_numpy_dtype

      # ignore types for which there are no minimum/maximum (or we cannot
      # compute it, such as for the q* types)
      if (dtype.is_quantized or dtype.base_dtype == dtypes.bool or
          dtype.base_dtype == dtypes.string or
          dtype.base_dtype == dtypes.complex64 or
          dtype.base_dtype == dtypes.complex128):
        continue

      print("%s: %s - %s" % (dtype, dtype.min, dtype.max))

      # check some values that are known
      if numpy_dtype == np.bool_:
        self.assertEqual(dtype.min, 0)
        self.assertEqual(dtype.max, 1)
      if numpy_dtype == np.int8:
        self.assertEqual(dtype.min, -128)
        self.assertEqual(dtype.max, 127)
      if numpy_dtype == np.int16:
        self.assertEqual(dtype.min, -32768)
        self.assertEqual(dtype.max, 32767)
      if numpy_dtype == np.int32:
        self.assertEqual(dtype.min, -2147483648)
        self.assertEqual(dtype.max, 2147483647)
      if numpy_dtype == np.int64:
        self.assertEqual(dtype.min, -9223372036854775808)
        self.assertEqual(dtype.max, 9223372036854775807)
      if numpy_dtype == np.uint8:
        self.assertEqual(dtype.min, 0)
        self.assertEqual(dtype.max, 255)
      if numpy_dtype == np.uint16:
        if dtype == dtypes.uint16:
          self.assertEqual(dtype.min, 0)
          self.assertEqual(dtype.max, 65535)
        elif dtype == dtypes.bfloat16:
          self.assertEqual(dtype.min, 0)
          self.assertEqual(dtype.max, 4294967295)
      if numpy_dtype == np.uint32:
        self.assertEqual(dtype.min, 0)
        self.assertEqual(dtype.max, 4294967295)
      if numpy_dtype == np.uint64:
        self.assertEqual(dtype.min, 0)
        self.assertEqual(dtype.max, 18446744073709551615)
      if numpy_dtype in (np.float16, np.float32, np.float64):
        self.assertEqual(dtype.min, np.finfo(numpy_dtype).min)
        self.assertEqual(dtype.max, np.finfo(numpy_dtype).max)
      if numpy_dtype == dtypes.bfloat16.as_numpy_dtype:
        self.assertEqual(dtype.min, float.fromhex("-0x1.FEp127"))
        self.assertEqual(dtype.max, float.fromhex("0x1.FEp127"))

  def testRepr(self):
    self.skipTest("b/142725777")
    for enum, name in dtypes._TYPE_TO_STRING.items():
      if enum > 100:
        continue
      dtype = dtypes.DType(enum)
      self.assertEqual(repr(dtype), "tf." + name)
      import tensorflow as tf
      dtype2 = eval(repr(dtype))
      self.assertEqual(type(dtype2), dtypes.DType)
      self.assertEqual(dtype, dtype2)

  def testEqWithNonTFTypes(self):
    self.assertNotEqual(dtypes.int32, int)
    self.assertNotEqual(dtypes.float64, 2.1)

  def testPythonLongConversion(self):
    self.assertIs(dtypes.int64, dtypes.as_dtype(np.array(2**32).dtype))

  def testPythonTypesConversion(self):
    self.assertIs(dtypes.float32, dtypes.as_dtype(float))
    self.assertIs(dtypes.bool, dtypes.as_dtype(bool))

  def testReduce(self):
    for enum in dtypes._TYPE_TO_STRING:
      dtype = dtypes.DType(enum)
      ctor, args = dtype.__reduce__()
      self.assertEqual(ctor, dtypes.as_dtype)
      self.assertEqual(args, (dtype.name,))
      reconstructed = ctor(*args)
      self.assertEqual(reconstructed, dtype)

  def testAsDtypeInvalidArgument(self):
    with self.assertRaises(TypeError):
      dtypes.as_dtype((dtypes.int32, dtypes.float32))

  def testAsDtypeReturnsInternedVersion(self):
    dt = dtypes.DType(types_pb2.DT_VARIANT)
    self.assertIs(dtypes.as_dtype(dt), dtypes.variant)


if __name__ == "__main__":
  googletest.main()
