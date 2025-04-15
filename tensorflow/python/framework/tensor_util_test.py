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

import contextlib
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class TensorUtilTest(test.TestCase, parameterized.TestCase):

  def testFloat(self):
    value = 10.0
    t = tensor_util.make_tensor_proto(value)
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape {}
      float_val: %.1f
      """ % value, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array(value, dtype=np.float32), a)

  def testFloatN(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0])
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTyped(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], dtype=dtypes.float32)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTypeCoerce(self):
    t = tensor_util.make_tensor_proto([10, 20, 30], dtype=dtypes.float32)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTypeCoerceNdarray(self):
    arr = np.asarray([10, 20, 30], dtype="int")
    t = tensor_util.make_tensor_proto(arr, dtype=dtypes.float32)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatSizes(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], shape=[1, 3])
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([[10.0, 20.0, 30.0]], dtype=np.float32), a)

  def testFloatSizes2(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], shape=[3, 1])
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } dim { size: 1 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } dim { size: 1 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
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
      self.assertProtoEquals(r"""
        dtype: DT_DOUBLE
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "@$\000\000\000\000\000\000@4\000\000\000\000\000\000@>\000\000\000\000\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_DOUBLE
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\000\000\000\000\000\000$@\000\000\000\000\000\0004@\000\000\000\000\000\000>@"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float64, a.dtype)
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

  def testFloatMutateArray(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], dtype=dtypes.float32)
    a = tensor_util.MakeNdarray(t)
    a[0] = 5.0
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([5.0, 20.0, 30.0], dtype=np.float32), a)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "A \000\000A\240\000\000A\360\000\000"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_FLOAT
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000 A\000\000\240A\000\000\360A"
        """, t)

  def testHalf(self):
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=np.float16))
    if sys.byteorder == "big":
      self.assertProtoEquals(
          """
        dtype: DT_HALF
        tensor_shape { dim { size: 2 } }
        tensor_content: "I\000M\000"
        """,
          t,
      )
    else:
      self.assertProtoEquals(
          """
        dtype: DT_HALF
        tensor_shape { dim { size: 2 } }
        tensor_content: "\000I\000M"
        """,
          t,
      )

    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float16, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0], dtype=np.float16), a)

  def testBfloat16(self):
    test_type = dtypes.bfloat16.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    if sys.byteorder == "big":
      self.assertProtoEquals(
          """
        dtype: DT_BFLOAT16
        tensor_shape {
          dim {
            size: 2
          }
      }
      tensor_content: "\x41\x20\x41\x5C\x32\x34\x30"
      """,
          t,
      )
    else:
      self.assertProtoEquals(
          """
        dtype: DT_BFLOAT16
        tensor_shape {
          dim {
            size: 2
          }
        }
      tensor_content: "\x20\x41\x5C\x32\x34\x30\x41"
      """,
          t,
      )

    a = tensor_util.MakeNdarray(t)
    self.assertEqual(test_type, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0], dtype=test_type), a)

  def testFloat8e5m2(self):
    test_type = dtypes.float8_e5m2.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    # 10.0: "I" = 73 = 10010 01: 2^(18 - 15) * (1 + 1/4)
    # 20.0: "M" = 77 = 10011 01: 2^(19 - 15) * (1 + 1/4)
    self.assertProtoEquals(
        """
      dtype: DT_FLOAT8_E5M2
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "IM"
      """, t)

    a = tensor_util.MakeNdarray(t)
    self.assertEqual(test_type, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0], dtype=test_type), a)

  def testFloat8e4m3fn(self):
    test_type = dtypes.float8_e4m3fn.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    # 10.0: "R" = 82 = 1010 010: 2^(10 - 7) * (1 + 1/4)
    # 20.0: "Z" = 90 = 1011 010: 2^(11 - 7) * (1 + 1/4)
    self.assertProtoEquals(
        """
      dtype: DT_FLOAT8_E4M3FN
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "RZ"
      """, t)

  def testFloat8e4m3fnuz(self):
    test_type = dtypes.float8_e4m3fnuz.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    # 10.0: "Z" = 90 = 1010 010: 2^(10 - 7) * (1 + 1/4) + 8
    # 20.0: "b" = 98 = 1011 010: 2^(11 - 7) * (1 + 1/4) + 8
    self.assertProtoEquals(
        """
      dtype: DT_FLOAT8_E4M3FNUZ
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "Zb"
      """,
        t,
    )

  def testFloat8e4m3b11fnuz(self):
    test_type = dtypes.float8_e4m3b11fnuz.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    # 10.0: "r" = 114 = 1010 010: 2^(10 - 7) * (1 + 1/4) + 36
    # 20.0: "z" = 126 = 1011 010: 2^(11 - 7) * (1 + 1/4) + 36
    self.assertProtoEquals(
        """
      dtype: DT_FLOAT8_E4M3B11FNUZ
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "rz"
      """,
        t,
    )

  def testFloat8e5m2fnuz(self):
    test_type = dtypes.float8_e5m2fnuz.as_numpy_dtype
    t = tensor_util.make_tensor_proto(np.array([10.0, 20.0], dtype=test_type))
    # 10.0: "M" = 77 = 1010 010: 2^(10 - 7) * (1 + 1/4) - 3
    # 20.0: "Q" = 87 = 1011 010: 2^(11 - 7) * (1 + 1/4) - 3
    self.assertProtoEquals(
        """
      dtype: DT_FLOAT8_E5M2FNUZ
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "MQ"
      """,
        t,
    )

  def testInt(self):
    t = tensor_util.make_tensor_proto(10)
    self.assertProtoEquals("""
      dtype: DT_INT32
      tensor_shape {}
      int_val: 10
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.int32, a.dtype)
    self.assertAllClose(np.array(10, dtype=np.int32), a)

  def testInt4(self):
    test_type = dtypes.int4.as_numpy_dtype
    t = tensor_util.make_tensor_proto(
        np.array(
            [-8, -1, 0, 1, 7],
            dtype=test_type,
        )
    )
    #
    self.assertProtoEquals(
        """
      dtype: DT_INT4
      tensor_shape {
        dim {
          size: 5
        }
      }
      int_val: -8
      int_val: -1
      int_val: 0
      int_val: 1
      int_val: 7
      """,
        t,
    )

  def testUInt4(self):
    test_type = dtypes.uint4.as_numpy_dtype
    t = tensor_util.make_tensor_proto(
        np.array(
            [0, 1, 7, 8, 15],
            dtype=test_type,
        )
    )
    self.assertProtoEquals(
        """
      dtype: DT_UINT4
      tensor_shape {
        dim {
          size: 5
        }
      }
      int_val: 0
      int_val: 1
      int_val: 7
      int_val: 8
      int_val: 15
      """,
        t,
    )

  def testLargeInt(self):
    value = np.iinfo(np.int64).max
    t = tensor_util.make_tensor_proto(value)
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape {}
      int64_val: %d
      """ % value, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.int64, a.dtype)
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
    self.assertEqual(np.int64, a.dtype)
    self.assertAllClose(np.array(value, dtype=np.int64), a)

  def testIntNDefaultType(self):
    t = tensor_util.make_tensor_proto([10, 20, 30, 40], shape=[2, 2])
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_INT32
        tensor_shape { dim { size: 2 } dim { size: 2 } }
        tensor_content: "\000\000\000\n\000\000\000\024\000\000\000\036\000\000\000("
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_INT32
        tensor_shape { dim { size: 2 } dim { size: 2 } }
        tensor_content: "\n\000\000\000\024\000\000\000\036\000\000\000(\000\000\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.int32, a.dtype)
    self.assertAllClose(np.array([[10, 20], [30, 40]], dtype=np.int32), a)

  @parameterized.named_parameters(
      ("_int8", dtypes.int8, np.int8), ("_int16", dtypes.int16, np.int16),
      ("_int32", dtypes.int32, np.int32), ("_int64", dtypes.int64, np.int64),
      ("_uint8", dtypes.uint8, np.uint8), ("_uint16", dtypes.uint16, np.uint16),
      ("_uint32", dtypes.uint32, np.uint32),
      ("_uint64", dtypes.uint64, np.uint64))
  def testIntTypes(self, dtype, nptype):
    # Test with array.
    t = tensor_util.make_tensor_proto([10, 20, 30], dtype=dtype)
    self.assertEqual(dtype, t.dtype)
    self.assertProtoEquals("dim { size: 3 }", t.tensor_shape)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)
    # Test with ndarray.
    t = tensor_util.make_tensor_proto(np.array([10, 20, 30], dtype=nptype))
    self.assertEqual(dtype, t.dtype)
    self.assertProtoEquals("dim { size: 3 }", t.tensor_shape)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)

  @parameterized.named_parameters(
      ("_int8", dtypes.int8, np.int8), ("_int16", dtypes.int16, np.int16),
      ("_int32", dtypes.int32, np.int32), ("_int64", dtypes.int64, np.int64),
      ("_uint8", dtypes.uint8, np.uint8), ("_uint16", dtypes.uint16, np.uint16),
      ("_uint32", dtypes.uint32, np.uint32),
      ("_uint64", dtypes.uint64, np.uint64))
  def testIntTypesWithImplicitRepeat(self, dtype, nptype):
    self.assertAllEqual(
        np.array([[10, 11, 12, 12], [12, 12, 12, 12], [12, 12, 12, 12]],
                 dtype=nptype),
        tensor_util.MakeNdarray(
            tensor_util.make_tensor_proto([10, 11, 12],
                                          shape=[3, 4],
                                          dtype=dtype)))

  def testIntMixedWithDimension(self):
    # Github issue: 11974
    dtype = dtypes.int32
    nptype = np.int32
    t = tensor_util.make_tensor_proto(
        [10, tensor_shape.Dimension(20), 30], dtype=dtype)
    self.assertEqual(dtype, t.dtype)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)

  @parameterized.named_parameters(
      ("_int64", dtypes.int64, np.int64, "DT_INT64", "int64_val"),
      ("_uint64", dtypes.uint64, np.uint64, "DT_UINT64", "uint64_val"))
  def testLong(self, dtype, nptype, proto_dtype, proto_value_name):
    t = tensor_util.make_tensor_proto(10, dtype=dtype)
    self.assertProtoEquals(
        """
      dtype: %s
      tensor_shape {}
      %s: 10
    """ % (proto_dtype, proto_value_name), t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array(10, dtype=nptype), a)

  @parameterized.named_parameters(
      ("_int64", dtypes.int64, np.int64, "DT_INT64"),
      ("_uint64", dtypes.uint64, np.uint64, "DT_UINT64"))
  def testLongN(self, dtype, nptype, proto_dtype):
    t = tensor_util.make_tensor_proto([10, 20, 30], shape=[1, 3], dtype=dtype)
    if sys.byteorder == "big":
      # pylint: disable=line-too-long
      self.assertProtoEquals(
          r"""
        dtype: %s
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\000\000\000\000\000\000\000\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036"
        """ % proto_dtype, t)
      # pylint: enable=line-too-long
    else:
      # pylint: disable=line-too-long
      self.assertProtoEquals(
          r"""
        dtype: %s
        tensor_shape { dim { size: 1 } dim { size: 3 } }
        tensor_content: "\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000"
        """ % proto_dtype, t)
      # pylint: enable=line-too-long
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array([[10, 20, 30]], dtype=nptype), a)

  @parameterized.named_parameters(("_int64", np.int64, "DT_INT64"),
                                  ("_uint64", np.uint64, "DT_UINT64"))
  def testLongNpArray(self, nptype, proto_dtype):
    t = tensor_util.make_tensor_proto(np.array([10, 20, 30], dtype=nptype))
    if sys.byteorder == "big":
      # pylint: disable=line-too-long
      self.assertProtoEquals(
          r"""
        dtype: %s
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000\000\000\000\000\000\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036"
        """ % proto_dtype, t)
      # pylint: enable=line-too-long
    else:
      # pylint: disable=line-too-long
      self.assertProtoEquals(
          r"""
        dtype: %s
        tensor_shape { dim { size: 3 } }
        tensor_content: "\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000"
        """ % proto_dtype, t)
      # pylint: enable=line-too-long
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(nptype, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)

  def testQuantizedTypes(self):
    # Test with array.
    data = [(21,), (22,), (23,)]

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.qint32)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_QINT32
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\000\000\025\000\000\000\026\000\000\000\027"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_QINT32
        tensor_shape { dim { size: 3 } }
        tensor_content: "\025\000\000\000\026\000\000\000\027\000\000\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(dtypes.qint32.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.quint8)
    self.assertProtoEquals(r"""
      dtype: DT_QUINT8
      tensor_shape { dim { size: 3 } }
      tensor_content: "\025\026\027"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(dtypes.quint8.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.qint8)
    self.assertProtoEquals(r"""
      dtype: DT_QINT8
      tensor_shape { dim { size: 3 } }
      tensor_content: "\025\026\027"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(dtypes.qint8.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.quint16)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_QUINT16
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\025\000\026\000\027"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_QUINT16
        tensor_shape { dim { size: 3 } }
        tensor_content: "\025\000\026\000\027\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(dtypes.quint16.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

    t = tensor_util.make_tensor_proto(data, dtype=dtypes.qint16)
    if sys.byteorder == "big":
      self.assertProtoEquals(r"""
        dtype: DT_QINT16
        tensor_shape { dim { size: 3 } }
        tensor_content: "\000\025\000\026\000\027"
        """, t)
    else:
      self.assertProtoEquals(r"""
        dtype: DT_QINT16
        tensor_shape { dim { size: 3 } }
        tensor_content: "\025\000\026\000\027\000"
        """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(dtypes.qint16.as_numpy_dtype, a.dtype)
    self.assertAllEqual(np.array(data, dtype=a.dtype), a)

  def testString(self):
    t = tensor_util.make_tensor_proto("foo")
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape {}
      string_val: "foo"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.object_, a.dtype)
    self.assertEqual([b"foo"], a)

  def testStringWithImplicitRepeat(self):
    t = tensor_util.make_tensor_proto(["f", "g"], shape=[3, 4])
    a = tensor_util.MakeNdarray(t)
    self.assertAllEqual(
        np.array([[b"f", b"g", b"g", b"g"], [b"g", b"g", b"g", b"g"],
                  [b"g", b"g", b"g", b"g"]],
                 dtype=np.object_), a)

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
    self.assertEqual(np.object_, a.dtype)
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
    self.assertEqual(np.object_, a.dtype)
    self.assertAllEqual(np.array([[b"a", b"ab"], [b"abc", b"abcd"]]), a)

  def testArrayMethod(self):

    class Wrapper(object):

      def __array__(self, dtype=None):
        del dtype
        return np.array([b"foo", b"bar", b"baz"])

    t = tensor_util.make_tensor_proto(Wrapper(), shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      string_val: "foo"
      string_val: "bar"
      string_val: "baz"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.object_, a.dtype)
    self.assertAllEqual(np.array([[b"foo", b"bar", b"baz"]]), a)

  def testArrayInterface(self):

    class Wrapper(object):

      def __init__(self):
        self.a = np.array([b"foo", b"bar", b"baz"])

      @property
      def __array_interface__(self):
        return self.a.__array_interface__

    t = tensor_util.make_tensor_proto(Wrapper(), shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      string_val: "foo"
      string_val: "bar"
      string_val: "baz"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.object_, a.dtype)
    self.assertAllEqual(np.array([[b"foo", b"bar", b"baz"]]), a)

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
    self.assertEqual(np.object_, a.dtype)
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
    self.assertEqual(np.object_, a.dtype)
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
    self.assertEqual(np.complex64, a.dtype)
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
    self.assertEqual(np.complex128, a.dtype)
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
    self.assertEqual(np.complex64, a.dtype)
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
    self.assertEqual(np.complex128, a.dtype)
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
    self.assertEqual(np.complex64, a.dtype)
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
    self.assertEqual(np.complex128, a.dtype)
    self.assertAllEqual(
        np.array([[(1 + 2j), (3 + 4j)], [(5 + 6j), (7 + 8j)]]), a)

  def testNestedNumpyArrayWithoutDType(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, np.array(30.0)])
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testNestedNumpyArrayWithDType(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, np.array(30.0)],
                                      dtype=dtypes.float32)
    a = tensor_util.MakeNdarray(t)
    self.assertEqual(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testUnsupportedDTypes(self):
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto(np.array([1]), 0)
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto(3, dtype=dtypes.qint8)
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto([3], dtype=dtypes.qint8)

    # Validate the helpful error message when trying to convert an
    # unconvertible list as strings.
    with self.assertRaisesRegex(TypeError, "Failed to convert elements"):
      tensor_util.make_tensor_proto([tensor_shape.Dimension(1)])

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


@test_util.run_all_in_graph_and_eager_modes
class IsTensorTest(test.TestCase):

  def testConstantTensor(self):
    np_val = np.random.rand(3).astype(np.int32)
    tf_val = constant_op.constant(np_val)
    self.assertFalse(tensor_util.is_tf_type(np_val))
    self.assertTrue(tensor_util.is_tf_type(tf_val))

  def testRaggedTensor(self):
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    rt_value = self.evaluate(rt)
    self.assertTrue(tensor_util.is_tf_type(rt))
    self.assertFalse(tensor_util.is_tf_type(rt_value))

  def testSparseTensor(self):
    st = sparse_tensor.SparseTensor([[1, 2]], [3], [10, 10])
    st_value = self.evaluate(st)
    self.assertTrue(tensor_util.is_tf_type(st))
    self.assertFalse(tensor_util.is_tf_type(st_value))

  def testIndexedSlices(self):
    x = indexed_slices.IndexedSlices(
        constant_op.constant([1, 2, 3]), constant_op.constant([10, 20, 30]))
    x_value = indexed_slices.IndexedSlicesValue(
        np.array([1, 2, 3]), np.array([10, 20, 30]), np.array([100]))
    self.assertTrue(tensor_util.is_tf_type(x))
    self.assertFalse(tensor_util.is_tf_type(x_value))

  def testVariable(self):
    v = variables.Variable([1, 2, 3])
    self.assertTrue(tensor_util.is_tf_type(v))


class ConstantValueTest(test.TestCase):

  def testConstant(self):
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = constant_op.constant(np_val)
    self.assertAllClose(np_val, tensor_util.constant_value(tf_val))

    np_val = np.random.rand(3, 0, 7).astype(np.float32)
    tf_val = constant_op.constant(np_val)
    self.assertAllClose(np_val, tensor_util.constant_value(tf_val))

  def testUnknown(self):
    with ops.Graph().as_default():
      tf_val = gen_state_ops.variable(
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

  def testFill(self):
    np_val = np.array([-1, -1, -1], dtype=np.float32)
    tf_val = array_ops.fill([3], constant_op.constant(-1.0))
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(np_val, c_val)
    self.assertEqual(np.float32, c_val.dtype)

  def testSize(self):
    tf_val = array_ops.size(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value(tf_val)
    self.assertEqual(6, c_val)

  def testSizeOfScalar(self):
    tf_val = array_ops.size(constant_op.constant(0.0))
    c_val = tensor_util.constant_value(tf_val)
    self.assertEqual(1, c_val)
    self.assertIn(type(c_val), [np.ndarray, np.int32])

  def testRank(self):
    tf_val = array_ops.rank(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value(tf_val)

    self.assertIn(type(c_val), [np.ndarray, np.int32])
    self.assertEqual((), c_val.shape)
    self.assertEqual(3, c_val)

    # Repeat test using array_ops.rank_internal to avoid the optimization that
    # happens in the rank function.
    tf_val = array_ops.rank_internal(
        constant_op.constant(
            0.0, shape=[1, 2, 3]), optimize=False)
    c_val = tensor_util.constant_value(tf_val)

    self.assertIn(type(c_val), [np.ndarray, np.int32])
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

    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops.concat(
          [np_val[0, :, :], np_val[1, :, :], np_val[2, :, :]],
          array_ops.placeholder(dtypes.int32))
      c_val = tensor_util.constant_value(tf_val)
      self.assertIs(None, c_val)

      tf_val = array_ops.concat([
          np_val[0, :, :],
          array_ops.placeholder(dtypes.float32), np_val[2, :, :]
      ], 1)
      c_val = tensor_util.constant_value(tf_val)
      self.assertIs(None, c_val)

  def testPack_Axis0(self):
    inputs = [np.random.rand(4, 7) for _ in range(3)]
    np_val = np.array(inputs)
    tf_val = array_ops_stack.stack(inputs)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllClose(np_val, c_val)

    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops_stack.stack(
          [inputs[0],
           array_ops.placeholder(dtypes.float32), inputs[2]])
      c_val = tensor_util.constant_value(tf_val)
      self.assertIs(None, c_val)

  def testPack_Axis1(self):
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      inputs = [np.random.rand(4, 7) for _ in range(3)]
      tf_val = array_ops_stack.stack(inputs, axis=1)
      c_val = tensor_util.constant_value(tf_val)
      self.assertIsNone(c_val)

      tf_val = array_ops_stack.stack(
          [inputs[0],
           array_ops.placeholder(dtypes.float32), inputs[2]], axis=1)
      c_val = tensor_util.constant_value(tf_val)
      self.assertIs(None, c_val)

  def testPack_Partial_Axis0(self):
    input_ = np.random.rand(4, 7)
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops_stack.stack(
          [input_, array_ops.placeholder(dtypes.float32)])
      c_val = tensor_util.constant_value(tf_val, partial=True)
      self.assertAllClose(input_, c_val[0])
      self.assertIsNone(c_val[1])

  def testPack_Partial_Axis1(self):
    input_ = np.random.rand(4, 7)
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops_stack.stack(
          [input_, array_ops.placeholder(dtypes.float32)], axis=1)
      c_val = tensor_util.constant_value(tf_val, partial=True)
      self.assertIsNone(c_val)

  def testUnpack_Axis0(self):
    inputs = np.random.rand(3, 4, 7)
    tf_vals = array_ops_stack.unstack(inputs)
    c_vals = [tensor_util.constant_value(x) for x in tf_vals]
    self.assertAllClose(inputs, c_vals)

  def testUnpack_Partial_Axis0(self):
    input_ = np.random.rand(4, 7)
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      packed = array_ops_stack.stack(
          [input_, array_ops.placeholder(dtypes.float32)])
      tf_vals = array_ops_stack.unstack(packed)
      c_vals = [tensor_util.constant_value(x, partial=True) for x in tf_vals]
      self.assertAllClose(input_, c_vals[0])
      self.assertIsNone(c_vals[1])

  def testSplit_Axis0(self):
    inputs = np.random.rand(6, 5, 7)
    tf_vals = array_ops.split(inputs, 3)
    c_vals = [tensor_util.constant_value(x) for x in tf_vals]
    self.assertAllClose(np.split(inputs, 3), c_vals)

  def testSplit_Partial_Axis0(self):
    input_ = np.random.rand(4, 7)
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      placeholder = array_ops.placeholder(dtypes.float32, shape=(4, 7))
      # it'd be better to use concat here, but concat doesn't support partial
      packed = array_ops_stack.stack([input_, placeholder])
      tf_vals = array_ops.split(packed, 2)
      c_vals = [tensor_util.constant_value(x, partial=True) for x in tf_vals]
      self.assertAllClose(input_, c_vals[0][0])
      self.assertIsNone(c_vals[1][0])

  def testEqual(self):
    # Scalar inputs.
    tf_val = math_ops.equal(constant_op.constant(1), constant_op.constant(1))
    self.assertEqual(tensor_util.constant_value(tf_val), True)

    tf_val = math_ops.equal(constant_op.constant(1), constant_op.constant(0))
    self.assertEqual(tensor_util.constant_value(tf_val), False)

    # Shaped inputs with broadcast semantics.
    tf_val = math_ops.equal(constant_op.constant([[0, 1]]),
                            constant_op.constant([[0], [1]]))
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(c_val, [[True, False], [False, True]])

  def testNotEqual(self):
    # Scalar inputs.
    tf_val = math_ops.not_equal(constant_op.constant(1),
                                constant_op.constant(1))
    self.assertEqual(tensor_util.constant_value(tf_val), False)

    tf_val = math_ops.not_equal(constant_op.constant(1),
                                constant_op.constant(0))
    self.assertEqual(tensor_util.constant_value(tf_val), True)

    # Shaped inputs with broadcast semantics.
    tf_val = math_ops.not_equal(constant_op.constant([[0, 1]]),
                                constant_op.constant([[0], [1]]))
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(c_val, [[False, True], [True, False]])

  def testStopGradient(self):
    input_ = np.random.rand(4, 7)
    tf_val = array_ops.stop_gradient(input_)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(input_, c_val)

  def testIdentity(self):
    input_ = np.random.rand(4, 7)
    tf_val = array_ops.identity(input_)
    c_val = tensor_util.constant_value(tf_val)
    self.assertAllEqual(input_, c_val)

  def testLiteral(self):
    x = "hi"
    self.assertIs(x, tensor_util.constant_value(x))

  def testNumpyNdarray(self):
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    self.assertIs(np_val, tensor_util.constant_value(np_val))

  def testVariable(self):
    var = variables.Variable(1.0, name="variable_node")
    self.assertIsNone(tensor_util.constant_value(var))

  def testVariableV1(self):
    var = variable_v1.VariableV1(1.0, name="variable_node")
    self.assertIsNone(tensor_util.constant_value(var))


class ConstantValueAsShapeTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
  def testCast(self):
    tf_val = math_ops.cast(
        array_ops.shape(constant_op.constant(0.0, shape=[1, 2, 3])),
        dtypes.int64)
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual(tensor_shape.TensorShape([1, 2, 3]), c_val)

  @test_util.run_in_graph_and_eager_modes
  def testCastWithUnknown(self):
    tf_val = math_ops.cast(constant_op.constant([-1, 1, -1]), dtypes.int64)
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([None, 1, None], c_val.as_list())

  @test_util.run_in_graph_and_eager_modes
  def testShape(self):
    tf_val = array_ops.shape(constant_op.constant(0.0, shape=[1, 2, 3]))
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual(tensor_shape.TensorShape([1, 2, 3]), c_val)

  @test_util.run_in_graph_and_eager_modes
  def testMinusOneBecomesNone(self):
    tf_val = constant_op.constant([-1, 1, -1], shape=[3])
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([None, 1, None], c_val.as_list())

  def testPack(self):
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops_stack.stack(
          [constant_op.constant(16), 37,
           array_ops.placeholder(dtypes.int32)])
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([16, 37, None], c_val.as_list())

  def testConcat(self):
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops.concat(
          [[16, 37], array_ops.placeholder(dtypes.int32, shape=(2,))], 0)
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([16, 37, None, None], c_val.as_list())

      tf_val = array_ops.concat(
          [[16, 37],
           array_ops.placeholder(dtypes.int32, shape=(1,)), [48]], 0)
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([16, 37, None, 48], c_val.as_list())

  def testSlice(self):
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val = array_ops.placeholder(dtypes.int32, shape=(4,))[0:2]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([None, None], c_val.as_list())

    # begin:end
    tf_val = constant_op.constant([10, 20, 30])[1:3]
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([20, 30], c_val.as_list())

    # begin:end:stride
    tf_val = array_ops.strided_slice(
        constant_op.constant([10, 20, 30]), [1], [3], strides=[2])
    c_val = tensor_util.constant_value_as_shape(tf_val)
    self.assertEqual([20], c_val.as_list())

    # [1, 2, 16, 37, None, 48]
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      tf_val_orig = array_ops.concat(
          [[1, 2, 16, 37],
           array_ops.placeholder(dtypes.int32, shape=(1,)), [48]], 0)

      # begin: no end
      tf_val = tf_val_orig[2:]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([16, 37, None, 48], c_val.as_list())

      # begin::negative slice
      tf_val = tf_val_orig[2::-1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([16, 2, 1], c_val.as_list())

      # :end:negative slice
      tf_val = tf_val_orig[:1:-2]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([48, 37], c_val.as_list())

      # begin:end:negative slice
      tf_val = tf_val_orig[3:1:-1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([37, 16], c_val.as_list())

      # begin:negative end:slice
      tf_val = tf_val_orig[1:-3:1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([2, 16], c_val.as_list())

      # negative begin::slice
      tf_val = tf_val_orig[-3::1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([37, None, 48], c_val.as_list())

      # negative begin::negative slice
      tf_val = tf_val_orig[-3::-1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([37, 16, 2, 1], c_val.as_list())

      # negative begin:negative end:negative slice
      tf_val = tf_val_orig[-3:-5:-1]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([37, 16], c_val.as_list())

      # Do not support shape inference for additional arguments
      tf_val = constant_op.constant([10, 20, 30])[...]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual([None, None, None], c_val.as_list())

      # Do not support shape inference for tensor slices.
      tf_val = constant_op.constant(
          [10, 20, 30])[array_ops.placeholder(dtypes.int32, shape=()):]
      c_val = tensor_util.constant_value_as_shape(tf_val)
      self.assertEqual(tensor_shape.unknown_shape(), c_val)

      # Do not support shape inference for higher rank
      with self.assertRaises(ValueError):
        tf_val = constant_op.constant([[10], [20], [30]])[:, 0:]
        c_val = tensor_util.constant_value_as_shape(tf_val)


class MaybeSetStaticShapeTest(test.TestCase):

  @contextlib.contextmanager
  def disableSetStaticShape(self):
    flag_old = shape_util._ENABLE_MAYBE_SET_STATIC_SHAPE
    shape_util._ENABLE_MAYBE_SET_STATIC_SHAPE = False
    try:
      yield
    finally:
      shape_util._ENABLE_MAYBE_SET_STATIC_SHAPE = flag_old

  def testMaybeSetStaticShape(self):
    shape = constant_op.constant([2, 5], dtype=dtypes.int32)

    def reshape():
      v = array_ops.zeros([10])
      return array_ops.reshape(v, shape)
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      with self.disableSetStaticShape():
        graph_without_shape_propagation = func_graph.func_graph_from_py_func(
            "without_shape_propagation", reshape, [], {})
      graph_with_shape_propagation = func_graph.func_graph_from_py_func(
          "with_shape_propagation", reshape, [], {})
      self.assertCountEqual(
          [op.type for op in graph_without_shape_propagation.get_operations()],
          [op.type for op in graph_with_shape_propagation.get_operations()])

  def testMaybeSetStaticShapeScalarShape(self):

    def reshape():
      v = array_ops.placeholder(dtypes.float32)
      t = array_ops.reshape(v, [-1])
      return t

    with self.disableSetStaticShape():
      graph_without_shape_propagation = func_graph.func_graph_from_py_func(
          "without_shape_propagation", reshape, [], {})
    graph_with_shape_propagation = func_graph.func_graph_from_py_func(
        "with_shape_propagation", reshape, [], {})
    self.assertCountEqual(
        [op.type for op in graph_without_shape_propagation.get_operations()],
        [op.type for op in graph_with_shape_propagation.get_operations()])


class ShapeTensorTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testConversion(self):
    """Make sure fully known TensorShape objects convert to Tensors."""
    shape = tensor_shape.TensorShape([1, tensor_shape.Dimension(2)])
    shape_tensor = shape_util.shape_tensor(shape)
    self.assertAllEqual((1, 2), shape_tensor)


if __name__ == "__main__":
  test.main()
