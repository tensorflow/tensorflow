"""Tests for tensorflow.python.framework.importer."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.framework import types
from tensorflow.python.platform import googletest


class TypesTest(test_util.TensorFlowTestCase):

  def testAllTypesConstructible(self):
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      self.assertEqual(
          datatype_enum, types.DType(datatype_enum).as_datatype_enum)

  def testAllTypesConvertibleToDType(self):
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      self.assertEqual(
          datatype_enum, types.as_dtype(datatype_enum).as_datatype_enum)

  def testAllTypesConvertibleToNumpyDtype(self):
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      dtype = types.as_dtype(datatype_enum)
      numpy_dtype = dtype.as_numpy_dtype
      _ = np.empty((1, 1, 1, 1), dtype=numpy_dtype)
      if dtype.base_dtype != types.bfloat16:
        # NOTE(touts): Intentionally no way to feed a DT_BFLOAT16.
        self.assertEqual(
            types.as_dtype(datatype_enum).base_dtype, types.as_dtype(numpy_dtype))

  def testInvalid(self):
    with self.assertRaises(TypeError):
      types.DType(types_pb2.DT_INVALID)
    with self.assertRaises(TypeError):
      types.as_dtype(types_pb2.DT_INVALID)

  def testNumpyConversion(self):
    self.assertIs(types.float32, types.as_dtype(np.float32))
    self.assertIs(types.float64, types.as_dtype(np.float64))
    self.assertIs(types.int32, types.as_dtype(np.int32))
    self.assertIs(types.int64, types.as_dtype(np.int64))
    self.assertIs(types.uint8, types.as_dtype(np.uint8))
    self.assertIs(types.int16, types.as_dtype(np.int16))
    self.assertIs(types.int8, types.as_dtype(np.int8))
    self.assertIs(types.complex64, types.as_dtype(np.complex64))
    self.assertIs(types.string, types.as_dtype(np.object))
    self.assertIs(types.string, types.as_dtype(np.array(["foo", "bar"]).dtype))
    self.assertIs(types.bool, types.as_dtype(np.bool))
    with self.assertRaises(TypeError):
      types.as_dtype(np.dtype([("f1", np.uint), ("f2", np.int32)]))

  def testStringConversion(self):
    self.assertIs(types.float32, types.as_dtype("float32"))
    self.assertIs(types.float64, types.as_dtype("float64"))
    self.assertIs(types.int32, types.as_dtype("int32"))
    self.assertIs(types.uint8, types.as_dtype("uint8"))
    self.assertIs(types.int16, types.as_dtype("int16"))
    self.assertIs(types.int8, types.as_dtype("int8"))
    self.assertIs(types.string, types.as_dtype("string"))
    self.assertIs(types.complex64, types.as_dtype("complex64"))
    self.assertIs(types.int64, types.as_dtype("int64"))
    self.assertIs(types.bool, types.as_dtype("bool"))
    self.assertIs(types.qint8, types.as_dtype("qint8"))
    self.assertIs(types.quint8, types.as_dtype("quint8"))
    self.assertIs(types.qint32, types.as_dtype("qint32"))
    self.assertIs(types.bfloat16, types.as_dtype("bfloat16"))
    self.assertIs(types.float32_ref, types.as_dtype("float32_ref"))
    self.assertIs(types.float64_ref, types.as_dtype("float64_ref"))
    self.assertIs(types.int32_ref, types.as_dtype("int32_ref"))
    self.assertIs(types.uint8_ref, types.as_dtype("uint8_ref"))
    self.assertIs(types.int16_ref, types.as_dtype("int16_ref"))
    self.assertIs(types.int8_ref, types.as_dtype("int8_ref"))
    self.assertIs(types.string_ref, types.as_dtype("string_ref"))
    self.assertIs(types.complex64_ref, types.as_dtype("complex64_ref"))
    self.assertIs(types.int64_ref, types.as_dtype("int64_ref"))
    self.assertIs(types.bool_ref, types.as_dtype("bool_ref"))
    self.assertIs(types.qint8_ref, types.as_dtype("qint8_ref"))
    self.assertIs(types.quint8_ref, types.as_dtype("quint8_ref"))
    self.assertIs(types.qint32_ref, types.as_dtype("qint32_ref"))
    self.assertIs(types.bfloat16_ref, types.as_dtype("bfloat16_ref"))
    with self.assertRaises(TypeError):
      types.as_dtype("not_a_type")

  def testDTypesHaveUniqueNames(self):
    dtypes = []
    names = set()
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      dtype = types.as_dtype(datatype_enum)
      dtypes.append(dtype)
      names.add(dtype.name)
    self.assertEqual(len(dtypes), len(names))

  def testIsInteger(self):
    self.assertEqual(types.as_dtype("int8").is_integer, True)
    self.assertEqual(types.as_dtype("int16").is_integer, True)
    self.assertEqual(types.as_dtype("int32").is_integer, True)
    self.assertEqual(types.as_dtype("int64").is_integer, True)
    self.assertEqual(types.as_dtype("uint8").is_integer, True)
    self.assertEqual(types.as_dtype("complex64").is_integer, False)
    self.assertEqual(types.as_dtype("float").is_integer, False)
    self.assertEqual(types.as_dtype("double").is_integer, False)
    self.assertEqual(types.as_dtype("string").is_integer, False)
    self.assertEqual(types.as_dtype("bool").is_integer, False)

  def testMinMax(self):
    # make sure min/max evaluates for all data types that have min/max
    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      dtype = types.as_dtype(datatype_enum)
      numpy_dtype = dtype.as_numpy_dtype

      # ignore types for which there are no minimum/maximum (or we cannot
      # compute it, such as for the q* types)
      if (dtype.is_quantized or
          dtype.base_dtype == types.bool or
          dtype.base_dtype == types.string or
          dtype.base_dtype == types.complex64):
        continue

      print "%s: %s - %s" % (dtype, dtype.min, dtype.max)

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
        self.assertEquals(dtype.min, 0)
        self.assertEquals(dtype.max, 4294967295)
      if numpy_dtype == np.uint32:
        self.assertEquals(dtype.min, 0)
        self.assertEquals(dtype.max, 18446744073709551615)
      if numpy_dtype in (np.float16, np.float32, np.float64):
        self.assertEquals(dtype.min, np.finfo(numpy_dtype).min)
        self.assertEquals(dtype.max, np.finfo(numpy_dtype).max)

  def testRepr(self):
    for enum, name in types._TYPE_TO_STRING.iteritems():
      dtype = types.DType(enum)
      self.assertEquals(repr(dtype), 'tf.' + name)
      dtype2 = eval(repr(dtype))
      self.assertEquals(type(dtype2), types.DType)
      self.assertEquals(dtype, dtype2)


if __name__ == "__main__":
  googletest.main()
