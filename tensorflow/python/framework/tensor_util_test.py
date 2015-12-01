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

"""Functional tests for tensor_util."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import googletest


class TensorUtilTest(test_util.TensorFlowTestCase):

  def testFloat(self):
    t = tensor_util.make_tensor_proto(10.0)
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape {}
      float_val: 10.0
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array(10.0, dtype=np.float32), a)

  def testFloatN(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0])
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
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape { dim { size: 3 } dim { size: 1 } }
      tensor_content: "\000\000 A\000\000\240A\000\000\360A"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([[10.0], [20.0], [30.0]], dtype=np.float32),
                        a)

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
    self.assertProtoEquals("""
      dtype: DT_DOUBLE
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      tensor_content: "\000\000\000\000\000\000$@\000\000\000\000\000\0004@\000\000\000\000\000\000>@"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float64, a.dtype)
    self.assertAllClose(np.array([[10.0, 20.0, 30.0]], dtype=np.float64),
                        tensor_util.MakeNdarray(t))

  def testFloatTypesWithImplicitRepeat(self):
    for dtype, nptype in [
        (dtypes.float32, np.float32), (dtypes.float64, np.float64)]:
      t = tensor_util.make_tensor_proto([10.0], shape=[3, 4], dtype=dtype)
      a = tensor_util.MakeNdarray(t)
      self.assertAllClose(np.array([[10.0, 10.0, 10.0, 10.0],
                                    [10.0, 10.0, 10.0, 10.0],
                                    [10.0, 10.0, 10.0, 10.0]], dtype=nptype), a)

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

  def testIntNDefaultType(self):
    t = tensor_util.make_tensor_proto([10, 20, 30, 40], shape=[2, 2])
    self.assertProtoEquals("""
      dtype: DT_INT32
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      tensor_content: "\\n\000\000\000\024\000\000\000\036\000\000\000(\000\000\000"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int32, a.dtype)
    self.assertAllClose(np.array([[10, 20], [30, 40]], dtype=np.int32), a)

  def testIntTypes(self):
    for dtype, nptype in [
        (dtypes.int32, np.int32),
        (dtypes.uint8, np.uint8),
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
    for dtype, nptype in [
        (dtypes.int64, np.int64),
        (dtypes.int32, np.int32),
        (dtypes.uint8, np.uint8),
        (dtypes.int16, np.int16),
        (dtypes.int8, np.int8)]:
      t = tensor_util.make_tensor_proto([10], shape=[3, 4], dtype=dtype)
      a = tensor_util.MakeNdarray(t)
      self.assertAllEqual(np.array([[10, 10, 10, 10],
                                    [10, 10, 10, 10],
                                    [10, 10, 10, 10]], dtype=nptype), a)

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
    t = tensor_util.make_tensor_proto([10, 20, 30], shape=[1, 3],
                                    dtype=dtypes.int64)
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
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape { dim { size: 3 } }
      tensor_content: "\\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int64, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=np.int64), a)

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
    t = tensor_util.make_tensor_proto(np.array([[b"a", b"ab"],
                                                [b"abc", b"abcd"]]))
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

  def testComplex(self):
    t = tensor_util.make_tensor_proto((1+2j), dtype=dtypes.complex64)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX64
      tensor_shape {}
      scomplex_val: 1
      scomplex_val: 2
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex64, a.dtype)
    self.assertAllEqual(np.array(1 + 2j), a)

  def testComplexWithImplicitRepeat(self):
    t = tensor_util.make_tensor_proto((1+1j), shape=[3, 4],
                                      dtype=dtypes.complex64)
    a = tensor_util.MakeNdarray(t)
    self.assertAllClose(np.array([[(1+1j), (1+1j), (1+1j), (1+1j)],
                                  [(1+1j), (1+1j), (1+1j), (1+1j)],
                                  [(1+1j), (1+1j), (1+1j), (1+1j)]],
                                 dtype=np.complex64), a)

  def testComplexN(self):
    t = tensor_util.make_tensor_proto([(1+2j), (3+4j), (5+6j)], shape=[1, 3],
                                      dtype=dtypes.complex64)
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
    self.assertAllEqual(np.array([[(1+2j), (3+4j), (5+6j)]]), a)

  def testComplexNpArray(self):
    t = tensor_util.make_tensor_proto(
        np.array([[(1+2j), (3+4j)], [(5+6j), (7+8j)]]), dtype=dtypes.complex64)
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
    self.assertAllEqual(np.array([[(1+2j), (3+4j)], [(5+6j), (7+8j)]]), a)

  def testUnsupportedDType(self):
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto(np.array([1]), 0)

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
        tensor_util.ShapeEquals(t, tensor_util.MakeTensorShapeProto([2, 2])))
    self.assertFalse(tensor_util.ShapeEquals(t, [5, 3]))
    self.assertFalse(tensor_util.ShapeEquals(t, [1, 4]))
    self.assertFalse(tensor_util.ShapeEquals(t, [4]))


class ConstantValueTest(test_util.TensorFlowTestCase):

  def testConstant(self):
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = constant_op.constant(np_val)
    self.assertAllClose(np_val, tensor_util.ConstantValue(tf_val))

    np_val = np.random.rand(3, 0, 7).astype(np.float32)
    tf_val = constant_op.constant(np_val)
    self.assertAllClose(np_val, tensor_util.ConstantValue(tf_val))

  def testUnknown(self):
    tf_val = state_ops.variable_op(shape=[3, 4, 7], dtype=dtypes.float32)
    self.assertIs(None, tensor_util.ConstantValue(tf_val))

  def testShape(self):
    np_val = np.array([1, 2, 3], dtype=np.int32)
    tf_val = array_ops.shape(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.ConstantValue(tf_val)
    self.assertAllEqual(np_val, c_val)
    self.assertEqual(np.int32, c_val.dtype)

  def testSize(self):
    np_val = np.array([6], dtype=np.int32)
    tf_val = array_ops.size(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.ConstantValue(tf_val)
    self.assertAllEqual(np_val, c_val)
    self.assertEqual(np.int32, c_val.dtype)

  def testRank(self):
    np_val = np.array([3], dtype=np.int32)
    tf_val = array_ops.rank(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.ConstantValue(tf_val)
    self.assertAllEqual(np_val, c_val)
    self.assertEqual(np.int32, c_val.dtype)


if __name__ == "__main__":
  googletest.main()
