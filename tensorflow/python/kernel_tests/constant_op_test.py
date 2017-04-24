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
"""Tests for ConstantOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ConstantTest(test.TestCase):

  def _testCpu(self, x):
    np_ans = np.array(x)
    with self.test_session(use_gpu=False):
      tf_ans = ops.convert_to_tensor(x).eval()
    if np_ans.dtype in [np.float32, np.float64, np.complex64, np.complex128]:
      self.assertAllClose(np_ans, tf_ans)
    else:
      self.assertAllEqual(np_ans, tf_ans)

  def _testGpu(self, x):
    np_ans = np.array(x)
    with self.test_session(use_gpu=True):
      tf_ans = ops.convert_to_tensor(x).eval()
    if np_ans.dtype in [np.float32, np.float64, np.complex64, np.complex128]:
      self.assertAllClose(np_ans, tf_ans)
    else:
      self.assertAllEqual(np_ans, tf_ans)

  def _testAll(self, x):
    self._testCpu(x)
    self._testGpu(x)

  def testFloat(self):
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32))
    self._testAll(
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.float32))
    self._testAll(np.empty((2, 0, 5)).astype(np.float32))

  def testDouble(self):
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float64))
    self._testAll(
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.float64))
    self._testAll(np.empty((2, 0, 5)).astype(np.float64))

  def testInt32(self):
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int32))
    self._testAll((100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(
        np.int32))
    self._testAll(np.empty((2, 0, 5)).astype(np.int32))

  def testInt64(self):
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int64))
    self._testAll((100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(
        np.int64))
    self._testAll(np.empty((2, 0, 5)).astype(np.int64))

  def testComplex64(self):
    self._testAll(
        np.complex(1, 2) *
        np.arange(-15, 15).reshape([2, 3, 5]).astype(np.complex64))
    self._testAll(
        np.complex(1, 2) *
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.complex64))
    self._testAll(np.empty((2, 0, 5)).astype(np.complex64))

  def testComplex128(self):
    self._testAll(
        np.complex(1, 2) *
        np.arange(-15, 15).reshape([2, 3, 5]).astype(np.complex128))
    self._testAll(
        np.complex(1, 2) *
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.complex128))
    self._testAll(np.empty((2, 0, 5)).astype(np.complex128))

  def testString(self):
    self._testCpu(
        np.array([compat.as_bytes(str(x)) for x in np.arange(-15, 15)]).reshape(
            [2, 3, 5]))
    self._testCpu(np.empty((2, 0, 5)).astype(np.str_))

  def testStringWithNulls(self):
    with self.test_session():
      val = ops.convert_to_tensor(b"\0\0\0\0").eval()
    self.assertEqual(len(val), 4)
    self.assertEqual(val, b"\0\0\0\0")

    with self.test_session():
      val = ops.convert_to_tensor(b"xx\0xx").eval()
    self.assertEqual(len(val), 5)
    self.assertAllEqual(val, b"xx\0xx")
    nested = [[b"\0\0\0\0", b"xx\0xx"], [b"\0_\0_\0_\0", b"\0"]]

    with self.test_session():
      val = ops.convert_to_tensor(nested).eval()
    # NOTE(mrry): Do not use assertAllEqual, because it converts nested to a
    #   numpy array, which loses the null terminators.
    self.assertEqual(val.tolist(), nested)

  def testExplicitShapeNumPy(self):
    with ops.Graph().as_default():
      c = constant_op.constant(
          np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32),
          shape=[2, 3, 5])
    self.assertEqual(c.get_shape(), [2, 3, 5])

  def testImplicitShapeNumPy(self):
    with ops.Graph().as_default():
      c = constant_op.constant(
          np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32))
    self.assertEqual(c.get_shape(), [2, 3, 5])

  def testExplicitShapeList(self):
    with ops.Graph().as_default():
      c = constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[7])
    self.assertEqual(c.get_shape(), [7])

  def testImplicitShapeList(self):
    with ops.Graph().as_default():
      c = constant_op.constant([1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(c.get_shape(), [7])

  def testExplicitShapeNumber(self):
    with ops.Graph().as_default():
      c = constant_op.constant(1, shape=[1])
    self.assertEqual(c.get_shape(), [1])

  def testImplicitShapeNumber(self):
    with ops.Graph().as_default():
      c = constant_op.constant(1)
    self.assertEqual(c.get_shape(), [])

  def testShapeInconsistent(self):
    with ops.Graph().as_default():
      c = constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[10])
    self.assertEqual(c.get_shape(), [10])

  # pylint: disable=g-long-lambda
  def testShapeWrong(self):
    with ops.Graph().as_default():
      with self.assertRaisesWithPredicateMatch(
          ValueError,
          lambda e: ("Too many elements provided. Needed at most 5, "
                     "but received 7" == str(e))):
        constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[5])

  # pylint: enable=g-long-lambda
  # TODO(b/35396543): Temporarily disable: suspicion that
  # this is causing test timeouts.
  def _testTooLargeConstant(self):
    with ops.Graph().as_default():
      large_array = np.zeros((512, 1024, 1024), dtype=np.float32)
      with self.assertRaisesRegexp(
          ValueError,
          "Cannot create a tensor proto whose content is larger than 2GB."):
        c = constant_op.constant(large_array)

  # TODO(b/35396543): Temporarily disable: suspicion that
  # this is causing test timeouts.
  def _testTooLargeGraph(self):
    with ops.Graph().as_default() as g:
      large_array = np.zeros((256, 1024, 1024), dtype=np.float32)
      c = constant_op.constant(large_array)
      d = constant_op.constant(large_array)
      with self.assertRaisesRegexp(ValueError,
                                   "GraphDef cannot be larger than 2GB."):
        g.as_graph_def()

  def testSparseValuesRaiseErrors(self):
    with self.assertRaisesRegexp(ValueError,
                                 "setting an array element with a sequence"):
      c = constant_op.constant([[1, 2], [3]], dtype=dtypes_lib.int32)

    with self.assertRaisesRegexp(ValueError, "must be a dense"):
      c = constant_op.constant([[1, 2], [3]])

    with self.assertRaisesRegexp(ValueError, "must be a dense"):
      c = constant_op.constant([[1, 2], [3], [4, 5]])


class AsTensorTest(test.TestCase):

  def testAsTensorForTensorInput(self):
    with ops.Graph().as_default():
      t = constant_op.constant(10.0)
      x = ops.convert_to_tensor(t)
    self.assertIs(t, x)

  def testAsTensorForNonTensorInput(self):
    with ops.Graph().as_default():
      x = ops.convert_to_tensor(10.0)
    self.assertTrue(isinstance(x, ops.Tensor))

  def testAsTensorForShapeInput(self):
    with self.test_session():
      x = ops.convert_to_tensor(tensor_shape.TensorShape([]))
      self.assertEqual(dtypes_lib.int32, x.dtype)
      self.assertAllEqual([], x.eval())

      x = ops.convert_to_tensor(tensor_shape.TensorShape([1, 2, 3]))
      self.assertEqual(dtypes_lib.int32, x.dtype)
      self.assertAllEqual([1, 2, 3], x.eval())

      x = ops.convert_to_tensor(tensor_shape.TensorShape([2**31-1, 2, 3]))
      self.assertEqual(dtypes_lib.int32, x.dtype)
      self.assertAllEqual([2**31-1, 2, 3], x.eval())

      x = ops.convert_to_tensor(tensor_shape.TensorShape([2**31-1, 2, 3]),
                                dtype=dtypes_lib.int32)
      self.assertEqual(dtypes_lib.int32, x.dtype)
      self.assertAllEqual([2**31-1, 2, 3], x.eval())

      x = ops.convert_to_tensor(tensor_shape.TensorShape([2**31, 2, 3]))
      self.assertEqual(dtypes_lib.int64, x.dtype)
      self.assertAllEqual([2**31, 2, 3], x.eval())

      x = ops.convert_to_tensor(tensor_shape.TensorShape([2**31, 2, 3]),
                                dtype=dtypes_lib.int64)
      self.assertEqual(dtypes_lib.int64, x.dtype)
      self.assertAllEqual([2**31, 2, 3], x.eval())

      with self.assertRaisesRegexp(
          ValueError, "a dimension is too large .2147483648."):
        x = ops.convert_to_tensor(tensor_shape.TensorShape([2**31, 2, 3]),
                                  dtype=dtypes_lib.int32)

      x = ops.convert_to_tensor(
          tensor_shape.TensorShape([1, 2, 3]), dtype=dtypes_lib.int64)
      self.assertEqual(dtypes_lib.int64, x.dtype)
      self.assertAllEqual([1, 2, 3], x.eval())

      x = array_ops.reshape(
          array_ops.zeros([6]), tensor_shape.TensorShape([2, 3]))
      self.assertAllEqual([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], x.eval())

    with self.assertRaisesRegexp(ValueError, "partially known"):
      ops.convert_to_tensor(tensor_shape.TensorShape(None))

    with self.assertRaisesRegexp(ValueError, "partially known"):
      ops.convert_to_tensor(tensor_shape.TensorShape([1, None, 64]))

    with self.assertRaises(TypeError):
      ops.convert_to_tensor(
          tensor_shape.TensorShape([1, 2, 3]), dtype=dtypes_lib.float32)

  def testAsTensorForDimensionInput(self):
    with self.test_session():
      x = ops.convert_to_tensor(tensor_shape.TensorShape([1, 2, 3])[1])
      self.assertEqual(dtypes_lib.int32, x.dtype)
      self.assertAllEqual(2, x.eval())

      x = ops.convert_to_tensor(
          tensor_shape.TensorShape([1, 2, 3])[1], dtype=dtypes_lib.int64)
      self.assertEqual(dtypes_lib.int64, x.dtype)
      self.assertAllEqual(2, x.eval())

    with self.assertRaisesRegexp(ValueError, "unknown Dimension"):
      ops.convert_to_tensor(tensor_shape.TensorShape(None)[1])

    with self.assertRaisesRegexp(ValueError, "unknown Dimension"):
      ops.convert_to_tensor(tensor_shape.TensorShape([1, None, 64])[1])

    with self.assertRaises(TypeError):
      ops.convert_to_tensor(
          tensor_shape.TensorShape([1, 2, 3])[1], dtype=dtypes_lib.float32)


class IdentityOpTest(test.TestCase):

  def testIdTensor(self):
    with ops.Graph().as_default():
      x = constant_op.constant(2.0, shape=[6], name="input")
      id_op = array_ops.identity(x, name="id")
    self.assertTrue(isinstance(id_op.op.inputs[0], ops.Tensor))
    self.assertProtoEquals("name: 'id' op: 'Identity' input: 'input' "
                           "attr { key: 'T' value { type: DT_FLOAT } }",
                           id_op.op.node_def)


class ZerosTest(test.TestCase):

  def _Zeros(self, shape):
    with self.test_session():
      ret = array_ops.zeros(shape)
      self.assertEqual(shape, ret.get_shape())
      return ret.eval()

  def testConst(self):
    self.assertTrue(
        np.array_equal(self._Zeros([2, 3]), np.array([[0] * 3] * 2)))

  def testScalar(self):
    self.assertEqual(0, self._Zeros([]))
    self.assertEqual(0, self._Zeros(()))
    with self.test_session():
      scalar = array_ops.zeros(constant_op.constant([], dtype=dtypes_lib.int32))
      self.assertEqual(0, scalar.eval())

  def testDynamicSizes(self):
    np_ans = np.array([[0] * 3] * 2)
    with self.test_session():
      # Creates a tensor of 2 x 3.
      d = array_ops.fill([2, 3], 12., name="fill")
      # Constructs a tensor of zeros of the same dimensions as "d".
      z = array_ops.zeros(array_ops.shape(d))
      out = z.eval()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, d)
    self.assertShapeEqual(np_ans, z)

  def testDtype(self):
    with self.test_session():
      d = array_ops.fill([2, 3], 12., name="fill")
      self.assertEqual(d.get_shape(), [2, 3])
      # Test default type for both constant size and dynamic size
      z = array_ops.zeros([2, 3])
      self.assertEqual(z.dtype, dtypes_lib.float32)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z.eval(), np.zeros([2, 3]))
      z = array_ops.zeros(array_ops.shape(d))
      self.assertEqual(z.dtype, dtypes_lib.float32)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z.eval(), np.zeros([2, 3]))
      # Test explicit type control
      for dtype in [
          dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
          dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
          dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64,
          dtypes_lib.bool, dtypes_lib.string
      ]:
        z = array_ops.zeros([2, 3], dtype=dtype)
        self.assertEqual(z.dtype, dtype)
        self.assertEqual([2, 3], z.get_shape())
        z_value = z.eval()
        self.assertFalse(np.any(z_value))
        self.assertEqual((2, 3), z_value.shape)
        z = array_ops.zeros(array_ops.shape(d), dtype=dtype)
        self.assertEqual(z.dtype, dtype)
        self.assertEqual([2, 3], z.get_shape())
        z_value = z.eval()
        self.assertFalse(np.any(z_value))
        self.assertEqual((2, 3), z_value.shape)


class ZerosLikeTest(test.TestCase):

  def _compareZeros(self, dtype, fully_defined_shape, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      # Creates a tensor of non-zero values with shape 2 x 3.
      # NOTE(kearnes): The default numpy dtype associated with tf.string is
      # np.object (and can't be changed without breaking a lot things), which
      # causes a TypeError in constant_op.constant below. Here we catch the
      # special case of tf.string and set the numpy dtype appropriately.
      if dtype == dtypes_lib.string:
        numpy_dtype = np.string_
      else:
        numpy_dtype = dtype.as_numpy_dtype
      if fully_defined_shape:
        d = constant_op.constant(
            np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
      else:
        d = array_ops.placeholder(dtype=dtype)
      # Constructs a tensor of zeros of the same dimensions and type as "d".
      z_var = array_ops.zeros_like(d)
      # Test that the type is correct
      self.assertEqual(z_var.dtype, dtype)
      # Test that the shape is correct
      if fully_defined_shape:
        self.assertEqual([2, 3], z_var.get_shape())

      # Test that the value is correct
      feed_dict = {}
      if not fully_defined_shape:
        feed_dict[d] = np.ones((2, 3), dtype=numpy_dtype)
      z_value = z_var.eval(feed_dict=feed_dict)
      self.assertFalse(np.any(z_value))
      self.assertEqual((2, 3), z_value.shape)

  def testZerosLikeCPU(self):
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64,
        dtypes_lib.string
    ]:
      self._compareZeros(dtype, fully_defined_shape=False, use_gpu=False)
      self._compareZeros(dtype, fully_defined_shape=True, use_gpu=False)

  def testZerosLikeGPU(self):
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.bool, dtypes_lib.int64, dtypes_lib.string
    ]:
      self._compareZeros(dtype, fully_defined_shape=False, use_gpu=True)
      self._compareZeros(dtype, fully_defined_shape=True, use_gpu=True)

  def testZerosLikePartialShape(self):
    d = array_ops.placeholder(dtypes_lib.float32, shape=[None, 4, None])
    z = array_ops.zeros_like(d)
    self.assertEqual(d.get_shape().as_list(), z.get_shape().as_list())

  def testZerosLikeDtype(self):
    # Make sure zeros_like works even for dtypes that cannot be cast between
    with self.test_session():
      shape = (3, 5)
      dtypes = np.float32, np.complex64
      for in_type in dtypes:
        x = np.arange(15).astype(in_type).reshape(*shape)
        for out_type in dtypes:
          y = array_ops.zeros_like(x, dtype=out_type).eval()
          self.assertEqual(y.dtype, out_type)
          self.assertEqual(y.shape, shape)
          self.assertAllEqual(y, np.zeros(shape, dtype=out_type))


class OnesTest(test.TestCase):

  def _Ones(self, shape):
    with self.test_session():
      ret = array_ops.ones(shape)
      self.assertEqual(shape, ret.get_shape())
      return ret.eval()

  def testConst(self):
    self.assertTrue(np.array_equal(self._Ones([2, 3]), np.array([[1] * 3] * 2)))

  def testScalar(self):
    self.assertEqual(1, self._Ones([]))
    self.assertEqual(1, self._Ones(()))
    with self.test_session():
      scalar = array_ops.ones(constant_op.constant([], dtype=dtypes_lib.int32))
      self.assertEqual(1, scalar.eval())

  def testDynamicSizes(self):
    np_ans = np.array([[1] * 3] * 2)
    with self.test_session():
      # Creates a tensor of 2 x 3.
      d = array_ops.fill([2, 3], 12., name="fill")
      # Constructs a tensor of ones of the same dimensions as "d".
      z = array_ops.ones(array_ops.shape(d))
      out = z.eval()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, d)
    self.assertShapeEqual(np_ans, z)

  def testAutoPack(self):
    with self.test_session():
      h = array_ops.placeholder(dtypes_lib.int32, shape=[])
      w = array_ops.placeholder(dtypes_lib.int32, shape=[])
      z = array_ops.ones([h, w])
      out = z.eval(feed_dict={h: 4, w: 16})
    self.assertAllEqual(out, np.array([[1] * 16] * 4))

  def testDtype(self):
    with self.test_session():
      d = array_ops.fill([2, 3], 12., name="fill")
      self.assertEqual(d.get_shape(), [2, 3])
      # Test default type for both constant size and dynamic size
      z = array_ops.ones([2, 3])
      self.assertEqual(z.dtype, dtypes_lib.float32)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z.eval(), np.ones([2, 3]))
      z = array_ops.ones(array_ops.shape(d))
      self.assertEqual(z.dtype, dtypes_lib.float32)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z.eval(), np.ones([2, 3]))
      # Test explicit type control
      for dtype in (dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
                    dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
                    dtypes_lib.complex64, dtypes_lib.complex128,
                    dtypes_lib.int64, dtypes_lib.bool):
        z = array_ops.ones([2, 3], dtype=dtype)
        self.assertEqual(z.dtype, dtype)
        self.assertEqual([2, 3], z.get_shape())
        self.assertAllEqual(z.eval(), np.ones([2, 3]))
        z = array_ops.ones(array_ops.shape(d), dtype=dtype)
        self.assertEqual(z.dtype, dtype)
        self.assertEqual([2, 3], z.get_shape())
        self.assertAllEqual(z.eval(), np.ones([2, 3]))


class OnesLikeTest(test.TestCase):

  def testOnesLike(self):
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64
    ]:
      numpy_dtype = dtype.as_numpy_dtype
      with self.test_session():
        # Creates a tensor of non-zero values with shape 2 x 3.
        d = constant_op.constant(
            np.ones(
                (2, 3), dtype=numpy_dtype), dtype=dtype)
        # Constructs a tensor of zeros of the same dimensions and type as "d".
        z_var = array_ops.ones_like(d)
        # Test that the type is correct
        self.assertEqual(z_var.dtype, dtype)
        z_value = z_var.eval()

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[1] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())

  def testOnesLikePartialShape(self):
    d = array_ops.placeholder(dtypes_lib.float32, shape=[None, 4, None])
    z = array_ops.ones_like(d)
    self.assertEqual(d.get_shape().as_list(), z.get_shape().as_list())


class FillTest(test.TestCase):

  def _compare(self, dims, val, np_ans, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      tf_ans = array_ops.fill(dims, val, name="fill")
      out = tf_ans.eval()
    self.assertAllClose(np_ans, out)
    # Fill does not set the shape.
    # self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, dims, val, np_ans):
    self._compare(dims, val, np_ans, False)
    self._compare(dims, val, np_ans, True)

  def testFillFloat(self):
    np_ans = np.array([[3.1415] * 3] * 2).astype(np.float32)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillDouble(self):
    np_ans = np.array([[3.1415] * 3] * 2).astype(np.float64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillInt32(self):
    np_ans = np.array([[42] * 3] * 2).astype(np.int32)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillInt64(self):
    np_ans = np.array([[-42] * 3] * 2).astype(np.int64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillComplex64(self):
    np_ans = np.array([[0.15] * 3] * 2).astype(np.complex64)
    self._compare([2, 3], np_ans[0][0], np_ans, use_gpu=False)

  def testFillComplex128(self):
    np_ans = np.array([[0.15] * 3] * 2).astype(np.complex128)
    self._compare([2, 3], np_ans[0][0], np_ans, use_gpu=False)

  def testFillString(self):
    np_ans = np.array([[b"yolo"] * 3] * 2)
    with self.test_session(use_gpu=False):
      tf_ans = array_ops.fill([2, 3], np_ans[0][0], name="fill").eval()
    self.assertAllEqual(np_ans, tf_ans)

  def testFillNegative(self):
    with self.test_session():
      for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
        with self.assertRaises(ValueError):
          array_ops.fill(shape, 7)

      # Using a placeholder so this won't be caught in static analysis.
      dims = array_ops.placeholder(dtypes_lib.int32)
      fill_t = array_ops.fill(dims, 3.0)
      for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
        with self.assertRaises(errors_impl.InvalidArgumentError):
          fill_t.eval({dims: shape})

  def testShapeFunctionEdgeCases(self):
    # Non-vector dimensions.
    with self.assertRaises(ValueError):
      array_ops.fill([[0, 1], [2, 3]], 1.0)

    # Non-scalar value.
    with self.assertRaises(ValueError):
      array_ops.fill([3, 2], [1.0, 2.0])

    # Partial dimension information.
    f = array_ops.fill(array_ops.placeholder(dtypes_lib.int32, shape=(4,)), 3.0)
    self.assertEqual([None, None, None, None], f.get_shape().as_list())

    f = array_ops.fill(
        [array_ops.placeholder(
            dtypes_lib.int32, shape=()), 17], 1.0)
    self.assertEqual([None, 17], f.get_shape().as_list())

  def testGradient(self):
    with self.test_session():
      in_v = constant_op.constant(5.0)
      out_shape = [3, 2]
      out_filled = array_ops.fill(out_shape, in_v)
      err = gradient_checker.compute_gradient_error(in_v, [], out_filled,
                                                    out_shape)
    self.assertLess(err, 1e-3)


class PlaceholderTest(test.TestCase):

  def testDtype(self):
    with self.test_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=(10, 10), name="p")
      p_identity = array_ops.identity(p)
      feed_array = np.random.rand(10, 10)
      self.assertAllClose(
          p_identity.eval(feed_dict={p: feed_array}), feed_array)

      with self.assertRaisesOpError(
          "must feed a value for placeholder tensor 'p' with dtype float"):
        p_identity.eval()

  def testShape(self):
    with self.test_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=(10, 10), name="p")
      p_identity = array_ops.identity(p)
      feed_array = np.random.rand(10, 10)
      self.assertAllClose(
          p_identity.eval(feed_dict={p: feed_array}), feed_array)

      with self.assertRaisesOpError(
          "must feed a value for placeholder tensor 'p' with dtype float and "
          r"shape \[10,10\]"):
        p_identity.eval()

      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Cannot feed value of shape" in str(e)):
        p_identity.eval(feed_dict={p: feed_array[:5, :5]})

  def testUnknownShape(self):
    with self.test_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=None, name="p")
      p_identity = array_ops.identity(p)
      # can feed anything
      feed_array = np.random.rand(10, 3)
      self.assertAllClose(
          p_identity.eval(feed_dict={p: feed_array}), feed_array)
      feed_array = np.random.rand(4, 2, 5)
      self.assertAllClose(
          p_identity.eval(feed_dict={p: feed_array}), feed_array)

  def testScalarShape(self):
    with self.test_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=[], name="p")
      p_identity = array_ops.identity(p)
      self.assertAllClose(p_identity.eval(feed_dict={p: 5}), 5)

  def testPartialShape(self):
    with self.test_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=[None, 3], name="p")
      p_identity = array_ops.identity(p)
      feed_array = np.random.rand(10, 3)
      self.assertAllClose(
          p_identity.eval(feed_dict={p: feed_array}), feed_array)

      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Cannot feed value of shape" in str(e)):
        p_identity.eval(feed_dict={p: feed_array[:5, :2]})

  def testControlDependency(self):
    with self.test_session():
      p = array_ops.placeholder(dtypes_lib.int32, shape=[], name="p")
      with ops.control_dependencies([p]):
        c = constant_op.constant(5, dtypes_lib.int32)
      d = math_ops.multiply(p, c)
      val = np.array(2).astype(np.int)
      self.assertEqual(10, d.eval(feed_dict={p: val}))

  def testBadShape(self):
    with self.assertRaises(ValueError):
      array_ops.placeholder(dtypes_lib.float32, shape=(-1, 10))

  def testTensorStr(self):
    a = array_ops.placeholder(dtypes_lib.float32, shape=None, name="a")
    self.assertEqual("<tf.Tensor 'a:0' shape=<unknown> dtype=float32>", repr(a))

    b = array_ops.placeholder(dtypes_lib.int32, shape=(32, 40), name="b")
    self.assertEqual("<tf.Tensor 'b:0' shape=(32, 40) dtype=int32>", repr(b))

    c = array_ops.placeholder(dtypes_lib.qint32, shape=(32, None, 2), name="c")
    self.assertEqual("<tf.Tensor 'c:0' shape=(32, ?, 2) dtype=qint32>", repr(c))

  def testOldGraph(self):
    # Load graph generated from earlier version of TF where
    # placeholder shape was not set.
    #
    # a = tf.placeholder(tf.float32)
    # b = a + 1.0
    #
    # Older graph's default shape is 'shape {}', not 'shape {
    # unknown_rank: true }'
    graph = """
node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "add"
  op: "Add"
  input: "Placeholder"
  input: "add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
versions {
  producer: 21
}
"""
    gdef = graph_pb2.GraphDef()
    text_format.Merge(graph, gdef)
    with self.test_session():
      p, ret = importer.import_graph_def(
          gdef, return_elements=["Placeholder:0", "add:0"])

      # Feed in a vector of two elements.  Since the producer version
      # of 21, a shape of {} is interpreted as "any shape".  If
      # producer version were 22, then we'd get a shape mismatch
      # error.
      self.assertAllEqual([2.0, 3.0], ret.eval(feed_dict={p: [1.0, 2.0]}))


class PlaceholderWithDefaultTest(test.TestCase):

  def testFullShape(self):
    with self.test_session():
      p = array_ops.placeholder_with_default([[2, 2], [2, 2]], shape=[2, 2])
      a = array_ops.identity(p)
      self.assertAllEqual([[2, 2], [2, 2]], a.eval())
      self.assertAllEqual(
          [[3, 3], [3, 3]], a.eval(feed_dict={p: [[3, 3], [3, 3]]}))

      with self.assertRaises(ValueError):
        a.eval(feed_dict={p: [[6, 6, 6], [6, 6, 6]]})

  def testPartialShape(self):
    with self.test_session():
      p = array_ops.placeholder_with_default([1, 2, 3], shape=[None])
      a = array_ops.identity(p)
      self.assertAllEqual([1, 2, 3], a.eval())
      self.assertAllEqual([3, 37], a.eval(feed_dict={p: [3, 37]}))

      with self.assertRaises(ValueError):
        a.eval(feed_dict={p: [[2, 2], [2, 2]]})

  def testNoShape(self):
    with self.test_session():
      p = array_ops.placeholder_with_default([17], shape=None)
      a = array_ops.identity(p)
      self.assertAllEqual([17], a.eval())
      self.assertAllEqual([3, 37], a.eval(feed_dict={p: [3, 37]}))
      self.assertAllEqual(
          [[3, 3], [3, 3]], a.eval(feed_dict={p: [[3, 3], [3, 3]]}))

  def testGradient(self):
    with self.test_session():
      x = array_ops.placeholder(dtypes_lib.float32, [5, 7])
      y = array_ops.placeholder_with_default(x, None)
      err = gradient_checker.compute_gradient_error(x, [5, 7], y, [5, 7])
      self.assertLess(err, 1e-3)

if __name__ == "__main__":
  test.main()
