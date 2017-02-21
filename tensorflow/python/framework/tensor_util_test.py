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
"""Functional tests for tensor_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class TensorUtilTest(test.TestCase):

  def testFloat(self):
    value = 10.0
    t = tensor_util.make_tensor_proto(value)
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape {}
      float_val: %.1f
      """ % value, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array(value, dtype=np.float32), a)

  def testFloatN(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0])
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_FLOAT  
        tensor_shape { dim { size: 3 } }  
        tensor_content: "A \000\000A\240\000\000A\360\000\000"  
        """, t)  
    else:  
      self.assertProtoEquals("""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTyped(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], dtype=dtypes.float32)
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_FLOAT  
        tensor_shape { dim { size: 3 } }  
        tensor_content: "A \000\000A\240\000\000A\360\000\000"  
        """, t)  
    else:
      self.assertProtoEquals("""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTypeCoerce(self):
    t = tensor_util.make_tensor_proto([10, 20, 30], dtype=dtypes.float32)
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_FLOAT  
        tensor_shape { dim { size: 3 } }  
        tensor_content: "A \000\000A\240\000\000A\360\000\000"  
        """, t)  
    else:  
      self.assertProtoEquals("""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTypeCoerceNdarray(self):
    arr = np.asarray([10, 20, 30], dtype="int")
    t = tensor_util.make_tensor_proto(arr, dtype=dtypes.float32)
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_FLOAT  
        tensor_shape { dim { size: 3 } }  
        tensor_content: "A \000\000A\240\000\000A\360\000\000"  
        """, t)  
    else: 
      self.assertProtoEquals("""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatSizes(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], shape=[1, 3])
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_FLOAT  
        tensor_shape { dim { size: 1 } dim { size: 3 } }  
        tensor_content: "A \000\000A\240\000\000A\360\000\000"  
        """, t)  
    else:  
      self.assertProtoEquals("""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([[10.0, 20.0, 30.0]], dtype=np.float32), a)

  def testFloatSizes2(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], shape=[3, 1])
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_FLOAT  
        tensor_shape { dim { size: 3 } dim { size: 1 } }  
        tensor_content: "A \000\000A\240\000\000A\360\000\000"  
        """, t)  
    else:  
      self.assertProtoEquals("""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } dim { size: 1 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([[10.0], [20.0], [30.0]], dtype=np.float32), a)

  def testFloatSizesLessValues(self):
    t = tensor_util.make_tensor_proto(10.0, shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      float_val: 10.0
      """, t)
    # No conversion to Ndarray for this one: not enough values.

  def testFloatNpArrayFloat64(self):
    t = tensor_util.make_tensor_proto(
        np.array([[10.0, 20.0, 30.0]], dtype=np.float64))
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_DOUBLE  
        tensor_shape { dim { size: 1 } dim { size: 3 } }  
        tensor_content: "@$\000\000\000\000\000\000@4\000\000\000\000\000\000@>\000\000\000\000\000\000"  
        """, t)  
    else:  
      self.assertProtoEquals("""
        dtype: DT_DOUBLE
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\000\000\000\000\000\000$@\000\000\000\000\000\0004@\000\000\000\000\000\000>@"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float64, a.dtype)
    self.assertAllClose(
        np.array([[10.0, 20.0, 30.0]], dtype=np.float64),
        tensor_util.MakeNdarray(t))

  def testFloatTypesWithImplicitRepeat(self):
    for dtype, nptype in [(dtypes.float32, np.float32),
                          (dtypes.float64, np.float64)]:
      t = tensor_util.make_tensor_proto([10.0], shape=[3, 4], dtype=dtype)
      a = tensor_util.MakeNdarray(t)
      self.assertAllClose(
          np.array(
              [[10.0, 10.0, 10.0, 10.0],
               [10.0, 10.0, 10.0, 10.0],
               [10.0, 10.0, 10.0, 10.0]],
              dtype=nptype),
          a)

  def testHalf(self):
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=np.float16))
    self.assertProtoEquals("""
      dtype: DT_HALF
      tensor_shape {
        dim {
          size: 2
        }
      }
      half_val: 18688
      half_val: 19712
      """, t)

    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float16, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0], dtype=np.float16), a)

  def testInt(self):
    t = tensor_util.make_tensor_proto(10)
    self.assertProtoEquals("""
      dtype: DT_INT32
      tensor_shape {}
      int_val: 10
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int32, a.dtype)
    self.assertAllClose(np.array(10, dtype=np.int32), a)

  def testLargeInt(self):
    value = np.iinfo(np.int64).max
    t = tensor_util.make_tensor_proto(value)
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape {}
      int64_val: %d
      """ % value, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int64, a.dtype)
    self.assertAllClose(np.array(value, dtype=np.int64), a)

  def testLargeNegativeInt(self):
    # We don't use the min np.int64 value here
    # because it breaks np.abs().
    #
    # np.iinfo(np.int64).min = -9223372036854775808
    # np.iinfo(np.int64).max = 9223372036854775807
    # np.abs(-9223372036854775808) = -9223372036854775808
    value = np.iinfo(np.int64).min + 1
    t = tensor_util.make_tensor_proto(value)
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape {}
      int64_val: %d
      """ % value, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int64, a.dtype)
    self.assertAllClose(np.array(value, dtype=np.int64), a)

  def testIntNDefaultType(self):
    t = tensor_util.make_tensor_proto([10, 20, 30, 40], shape=[2, 2])
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_INT32  
        tensor_shape { dim { size: 2 } dim { size: 2 } }  
        tensor_content: "\000\000\000\\n\000\000\000\024\000\000\000\036\000\000\000("  
        """, t)  
    else:  
      self.assertProtoEquals("""
        dtype: DT_INT32
        tensor_shape { dim { size: 2 } dim { size: 2 } }
        tensor_content: "\\n\000\000\000\024\000\000\000\036\000\000\000(\000\000\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int32, a.dtype)
    self.assertAllClose(np.array([[10, 20], [30, 40]], dtype=np.int32), a)

  def testIntTypes(self):
    for dtype, nptype in [(dtypes.int32, np.int32),
                          (dtypes.uint8, np.uint8),
                          (dtypes.uint16, np.uint16),
                          (dtypes.int16, np.int16),
                          (dtypes.int8, np.int8)]:
      # Test with array.
      t = tensor_util.make_tensor_proto([10, 20, 30], dtype=dtype)
      self.assertEquals(dtype, t.dtype)
      self.assertProtoEquals("dim { size: 3 }", t.tensor_shape)
      a = tensor_util.MakeNdarray(t)
      self.assertEquals(nptype, a.dtype)
      self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)
      # Test with ndarray.
      t = tensor_util.make_tensor_proto(np.array([10, 20, 30], dtype=nptype))
      self.assertEquals(dtype, t.dtype)
      self.assertProtoEquals("dim { size: 3 }", t.tensor_shape)
      a = tensor_util.MakeNdarray(t)
      self.assertEquals(nptype, a.dtype)
      self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)

  def testIntTypesWithImplicitRepeat(self):
    for dtype, nptype in [(dtypes.int64, np.int64),
                          (dtypes.int32, np.int32),
                          (dtypes.uint8, np.uint8),
                          (dtypes.uint16, np.uint16),
                          (dtypes.int16, np.int16),
                          (dtypes.int8, np.int8)]:
      self.assertAllEqual(
          np.array(
              [[10, 10, 10, 10],
               [10, 10, 10, 10],
               [10, 10, 10, 10]],
              dtype=nptype),
          tensor_util.MakeNdarray(
              tensor_util.make_tensor_proto(
                  [10],
                  shape=[3, 4],
                  dtype=dtype)))

  def testLong(self):
    t = tensor_util.make_tensor_proto(10, dtype=dtypes.int64)
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape {}
      int64_val: 10
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int64, a.dtype)
    self.assertAllClose(np.array(10, dtype=np.int64), a)

  def testLongN(self):
    t = tensor_util.make_tensor_proto(
        [10, 20, 30], shape=[1, 3], dtype=dtypes.int64)
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_INT64  
        tensor_shape { dim { size: 1 } dim { size: 3 } }  
        tensor_content: "\000\000\000\000\000\000\000\\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036"  
        """, t)  
    else: 
      self.assertProtoEquals("""
        dtype: DT_INT64
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int64, a.dtype)
    self.assertAllClose(np.array([[10, 20, 30]], dtype=np.int64), a)

  def testLongNpArray(self):
    t = tensor_util.make_tensor_proto(np.array([10, 20, 30]))
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_INT64  
        tensor_shape { dim { size: 3 } }  
        tensor_content: "\000\000\000\000\000\000\000\\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036"  
        """, t)  
    else:  
      self.assertProtoEquals("""
        dtype: DT_INT64
        tensor_shape { dim { size: 3 } }
        tensor_content: "\\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int64, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=np.int64), a)

  def testQuantizedTypes(self):
    # Test with array.
    data = [(21,), (22,), (23,)]

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.qint32)
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_QINT32  
        tensor_shape { dim { size: 3 } }  
        tensor_content: "\000\000\000\025\000\000\000\026\000\000\000\027"  
        """, t)  
    else:  
      self.assertProtoEquals("""
        dtype: DT_QINT32
        tensor_shape { dim { size: 3 } }
        tensor_content: "\025\000\000\000\026\000\000\000\027\000\000\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(dtypes.qint32.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.quint8)
    self.assertProtoEquals("""
      dtype: DT_QUINT8
      tensor_shape { dim { size: 3 } }
      tensor_content: "\025\026\027"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(dtypes.quint8.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.qint8)
    self.assertProtoEquals("""
      dtype: DT_QINT8
      tensor_shape { dim { size: 3 } }
      tensor_content: "\025\026\027"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(dtypes.qint8.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.quint16)
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_QUINT16  
        tensor_shape { dim { size: 3 } }  
        tensor_content: "\000\025\000\026\000\027"  
        """, t)  
    else:  
      self.assertProtoEquals("""
        dtype: DT_QUINT16
        tensor_shape { dim { size: 3 } }
        tensor_content: "\025\000\026\000\027\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(dtypes.quint16.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.qint16)
    if sys.byteorder == "big":  
      self.assertProtoEquals("""  
        dtype: DT_QINT16  
        tensor_shape { dim { size: 3 } }  
        tensor_content: "\000\025\000\026\000\027"  
        """, t)  
    else: 
      self.assertProtoEquals("""
        dtype: DT_QINT16
        tensor_shape { dim { size: 3 } }
        tensor_content: "\025\000\026\000\027\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(dtypes.qint16.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

  def testString(self):
    t = tensor_util.make_tensor_proto("foo")
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape {}
      string_val: "foo"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.object, a.dtype)
    self.assertEquals([b"foo"], a)

  def testStringWithImplicitRepeat(self):
    t = tensor_util.make_tensor_proto("f", shape=[3, 4])
    a = tensor_util.MakeNdarray(t)
    self.assertAllEqual(np.array([[b"f"] * 4] * 3, dtype=np.object), a)

  def testStringN(self):
    t = tensor_util.make_tensor_proto([b"foo", b"bar", b"baz"], shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      string_val: "foo"
      string_val: "bar"
      string_val: "baz"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.object, a.dtype)
    self.assertAllEqual(np.array([[b"foo", b"bar", b"baz"]]), a)

  def testStringNpArray(self):
    t = tensor_util.make_tensor_proto(
        np.array([[b"a", b"ab"], [b"abc", b"abcd"]]))
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      string_val: "a"
      string_val: "ab"
      string_val: "abc"
      string_val: "abcd"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.object, a.dtype)
    self.assertAllEqual(np.array([[b"a", b"ab"], [b"abc", b"abcd"]]), a)

  def testStringTuple(self):
    t = tensor_util.make_tensor_proto((b"a", b"ab", b"abc", b"abcd"))
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 4 } }
      string_val: "a"
      string_val: "ab"
      string_val: "abc"
      string_val: "abcd"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.object, a.dtype)
    self.assertAllEqual(np.array((b"a", b"ab", b"abc", b"abcd")), a)

  def testStringNestedTuple(self):
    t = tensor_util.make_tensor_proto(((b"a", b"ab"), (b"abc", b"abcd")))
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      string_val: "a"
      string_val: "ab"
      string_val: "abc"
      string_val: "abcd"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.object, a.dtype)
    self.assertAllEqual(np.array(((b"a", b"ab"), (b"abc", b"abcd"))), a)

  def testComplex64(self):
    t = tensor_util.make_tensor_proto((1 + 2j), dtype=dtypes.complex64)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX64
      tensor_shape {}
      scomplex_val: 1
      scomplex_val: 2
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex64, a.dtype)
    self.assertAllEqual(np.array(1 + 2j), a)

  def testComplex128(self):
    t = tensor_util.make_tensor_proto((1 + 2j), dtype=dtypes.complex128)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX128
      tensor_shape {}
      dcomplex_val: 1
      dcomplex_val: 2
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex128, a.dtype)
    self.assertAllEqual(np.array(1 + 2j), a)

  def testComplexWithImplicitRepeat(self):
    for dtype, np_dtype in [(dtypes.complex64, np.complex64),
                            (dtypes.complex128, np.complex128)]:
      t = tensor_util.make_tensor_proto((1 + 1j), shape=[3, 4], dtype=dtype)
      a = tensor_util.MakeNdarray(t)
      self.assertAllClose(
          np.array(
              [[(1 + 1j), (1 + 1j), (1 + 1j), (1 + 1j)],
               [(1 + 1j), (1 + 1j), (1 + 1j), (1 + 1j)],
               [(1 + 1j), (1 + 1j), (1 + 1j), (1 + 1j)]],
              dtype=np_dtype),
          a)

  def testComplex64N(self):
    t = tensor_util.make_tensor_proto(
        [(1 + 2j), (3 + 4j), (5 + 6j)], shape=[1, 3], dtype=dtypes.complex64)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX64
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      scomplex_val: 1
      scomplex_val: 2
      scomplex_val: 3
      scomplex_val: 4
      scomplex_val: 5
      scomplex_val: 6
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex64, a.dtype)
    self.assertAllEqual(np.array([[(1 + 2j), (3 + 4j), (5 + 6j)]]), a)

  def testComplex128N(self):
    t = tensor_util.make_tensor_proto(
        [(1 + 2j), (3 + 4j), (5 + 6j)], shape=[1, 3], dtype=dtypes.complex128)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX128
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      dcomplex_val: 1
      dcomplex_val: 2
      dcomplex_val: 3
      dcomplex_val: 4
      dcomplex_val: 5
      dcomplex_val: 6
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex128, a.dtype)
    self.assertAllEqual(np.array([[(1 + 2j), (3 + 4j), (5 + 6j)]]), a)

  def testComplex64NpArray(self):
    t = tensor_util.make_tensor_proto(
        np.array([[(1 + 2j), (3 + 4j)], [(5 + 6j), (7 + 8j)]]),
        dtype=dtypes.complex64)
    # scomplex_val are real_0, imag_0, real_1, imag_1, ...
    self.assertProtoEquals("""
      dtype: DT_COMPLEX64
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      scomplex_val: 1
      scomplex_val: 2
      scomplex_val: 3
      scomplex_val: 4
      scomplex_val: 5
      scomplex_val: 6
      scomplex_val: 7
      scomplex_val: 8
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex64, a.dtype)
    self.assertAllEqual(
        np.array([[(1 + 2j), (3 + 4j)], [(5 + 6j), (7 + 8j)]]), a)

  def testComplex128NpArray(self):
    t = tensor_util.make_tensor_proto(
        np.array([[(1 + 2j), (3 + 4j)], [(5 + 6j), (7 + 8j)]]),
        dtype=dtypes.complex128)
    # scomplex_val are real_0, imag_0, real_1, imag_1, ...
    self.assertProtoEquals("""
      dtype: DT_COMPLEX128
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      dcomplex_val: 1
      dcomplex_val: 2
      dcomplex_val: 3
      dcomplex_val: 4
      dcomplex_val: 5
      dcomplex_val: 6
      dcomplex_val: 7
      dcomplex_val: 8
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex128, a.dtype)
    self.assertAllEqual(
        np.array([[(1 + 2j), (3 + 4j)], [(5 + 6j), (7 + 8j)]]), a)

  def testUnsupportedDTypes(self):
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto(np.array([1]), 0)
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto(3, dtype=dtypes.qint8)
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto([3], dtype=dtypes.qint8)

  def testTensorShapeVerification(self):
    array = np.array([[1], [2]])
    correct_shape = (2, 1)
    incorrect_shape = (1, 2)
    tensor_util.make_tensor_proto(array, shape=correct_shape, verify_shape=True)
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto(
          array, shape=incorrect_shape, verify_shape=True)

  def testShapeTooLarge(self):
    with self.assertRaises(ValueError):
      tensor_util.make_tensor_proto(np.array([1, 2]), shape=[1])

  def testLowRankSupported(self):
    t = tensor_util.make_tensor_proto(np.array(7))
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape {}
      int64_val: 7
      """, t)

  def testShapeEquals(self):
    t = tensor_util.make_tensor_proto([10, 20, 30, 40], shape=[2, 2])
    self.assertTrue(tensor_util.ShapeEquals(t, [2, 2]))
    self.assertTrue(tensor_util.ShapeEquals(t, (2, 2)))
    self.assertTrue(
        tensor_util.ShapeEquals(t, tensor_shape.as_shape([2, 2]).as_proto()))
    self.assertFalse(tensor_util.ShapeEquals(t, [5, 3]))
    self.assertFalse(tensor_util.ShapeEquals(t, [1, 4]))
    self.assertFalse(tensor_util.ShapeEquals(t, [4]))


class ConstantValueTest(test.TestCase):

  def testConstant(self):
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = constant_op.constant(np_val)
    self.assertAllClose(np_val, tensor_util.constant_value(tf_val))

    np_val = np.random.rand(3, 0, 7).astype(np.float32)
    tf_val = constant_op.constant(np_val)
    self.assertAllClose(np_val, tensor_util.constant_value(tf_val))

  def testUnknown(self):
    tf_val = gen_state_ops._variable(
        shape=[3, 4, 7],
        dtype=dtypes.float32,
        name="tf_val",
        container="",
        shared_name="")
    self.assertIs(None, tensor_util.constant_value(tf_val))

  def testShape(self):
    np_val = np.array([1, 2, 3], dtype=np.int32)
    tf_val = array_ops.shape(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(np_val, c_val)
    self.assertEqual(np.int32, c_val.dtype)

  def testSize(self):
    tf_val = array_ops.size(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value(tf_val)
    self.assertEqual(6, c_val)

  def testSizeOfScalar(self):
    tf_val = array_ops.size(constant_op.constant(0.0))
    c_val = tensor_util.constant_value(tf_val)
    self.assertEqual(1, c_val)
    self.assertEqual(np.ndarray, type(c_val))

  def testRank(self):
    tf_val = array_ops.rank(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value(tf_val)

    self.assertEqual(np.ndarray, type(c_val))
    self.assertEqual((), c_val.shape)
    self.assertEqual(3, c_val)

    # Repeat test using array_ops.rank_internal to avoid the optimization that
    # happens in the rank function.
    tf_val = array_ops.rank_internal(
        constant_op.constant(
            0.0, shape=[1, 2, 3]), optimize=False)
    c_val = tensor_util.constant_value(tf_val)

    self.assertEqual(np.ndarray, type(c_val))
    self.assertEqual((), c_val.shape)
    self.assertEqual(3, c_val)
    self.assertEqual([3], c_val)

  def testCast(self):
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = math_ops.cast(constant_op.constant(np_val), dtypes.float64)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllClose(np_val.astype(np.float64), c_val)

    np_val = np.random.rand(3, 0, 7).astype(np.float32)
    tf_val = math_ops.cast(constant_op.constant(np_val), dtypes.float64)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllClose(np_val.astype(np.float64), c_val)

  def testConcat(self):
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = array_ops.concat(
        [np_val[0:1, :, :], np_val[1:2, :, :], np_val[2:3, :, :]], 0)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllClose(np_val, c_val)

    tf_val = array_ops.concat(
        [np_val[0, :, :], np_val[1, :, :], np_val[2, :, :]],
        array_ops.placeholder(dtypes.int32))
    c_val = tensor_util.constant_value(tf_val)
    self.assertIs(None, c_val)

    tf_val = array_ops.concat([
        np_val[0, :, :], array_ops.placeholder(dtypes.float32), np_val[2, :, :]
    ], 1)
    c_val = tensor_util.constant_value(tf_val)
    self.assertIs(None, c_val)

  def testPack(self):
    inputs = [np.random.rand(4, 7) for _ in range(3)]
    np_val = np.array(inputs)
    tf_val = array_ops.stack(inputs)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllClose(np_val, c_val)

    tf_val = array_ops.stack(
        [inputs[0], array_ops.placeholder(dtypes.float32), inputs[2]])
    c_val = tensor_util.constant_value(tf_val)
    self.assertIs(None, c_val)


class ConstantValueAsShapeTest(test.TestCase):

  def testConstant(self):
    np_val = np.random.rand(3).astype(np.int32)
    tf_val = constant_op.constant(np_val)
    self.assertEqual(
        tensor_shape.TensorShape(np_val),
        tensor_util.constant_value_as_shape(tf_val))

    tf_val = constant_op.constant([], dtype=dtypes.int32)
    self.assertEqual(
        tensor_shape.TensorShape([]),
        tensor_util.constant_value_as_shape(tf_val))

  def testShape(self):
    tf_val = array_ops.shape(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual(tensor_shape.TensorShape([1, 2, 3]), c_val)

  def testPack(self):
    tf_val = array_ops.stack(
        [constant_op.constant(16), 37, array_ops.placeholder(dtypes.int32)])
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([16, 37, None], c_val.as_list())

  def testConcat(self):
    tf_val = array_ops.concat(
        [[16, 37], array_ops.placeholder(
            dtypes.int32, shape=(2,))], 0)
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([16, 37, None, None], c_val.as_list())

    tf_val = array_ops.concat(
        [[16, 37], array_ops.placeholder(
            dtypes.int32, shape=(1,)), [48]], 0)
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([16, 37, None, 48], c_val.as_list())


if __name__ == "__main__":
  test.main()
