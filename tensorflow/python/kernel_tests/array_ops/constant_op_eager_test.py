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

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.util import compat


# TODO(josh11b): add tests with lists/tuples, Shape.
# TODO(ashankar): Collapse with tests in constant_op_test.py and use something
# like the test_util.run_in_graph_and_eager_modes decorator to confirm
# equivalence between graph and eager execution.
class ConstantTest(test.TestCase):

  def _testCpu(self, x):
    np_ans = np.array(x)
    with context.device("/device:CPU:0"):
      tf_ans = ops.convert_to_tensor(x).numpy()
    if np_ans.dtype in [np.float32, np.float64, np.complex64, np.complex128]:
      self.assertAllClose(np_ans, tf_ans)
    else:
      self.assertAllEqual(np_ans, tf_ans)

  def _testGpu(self, x):
    device = test_util.gpu_device_name()
    if device:
      np_ans = np.array(x)
      with context.device(device):
        tf_ans = ops.convert_to_tensor(x).numpy()
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

    orig = [-1.0, 2.0, 0.0]
    tf_ans = constant_op.constant(orig)
    self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

    # Mix floats and ints
    orig = [-1.5, 2, 0]
    tf_ans = constant_op.constant(orig)
    self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

    orig = [-5, 2.5, 0]
    tf_ans = constant_op.constant(orig)
    self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

    # Mix floats and ints that don't fit in int32
    orig = [1, 2**42, 0.5]
    tf_ans = constant_op.constant(orig)
    self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

  def testDouble(self):
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float64))
    self._testAll(
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.float64))
    self._testAll(np.empty((2, 0, 5)).astype(np.float64))

    orig = [-5, 2.5, 0]
    tf_ans = constant_op.constant(orig, dtypes_lib.float64)
    self.assertEqual(dtypes_lib.float64, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

    # This integer is not exactly representable as a double, gets rounded.
    tf_ans = constant_op.constant(2**54 + 1, dtypes_lib.float64)
    self.assertEqual(2**54, tf_ans.numpy())

    # This integer is larger than all non-infinite numbers representable
    # by a double, raises an exception.
    with self.assertRaisesRegex(ValueError, "out-of-range integer"):
      constant_op.constant(10**310, dtypes_lib.float64)

  def testInt32(self):
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int32))
    self._testAll(
        (100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(np.int32))
    self._testAll(np.empty((2, 0, 5)).astype(np.int32))
    self._testAll([-1, 2])

  def testInt64(self):
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int64))
    self._testAll(
        (100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(np.int64))
    self._testAll(np.empty((2, 0, 5)).astype(np.int64))
    # Should detect out of range for int32 and use int64 instead.
    orig = [2, 2**48, -2**48]
    tf_ans = constant_op.constant(orig)
    self.assertEqual(dtypes_lib.int64, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

    # Out of range for an int64
    with self.assertRaisesRegex(ValueError, "out-of-range integer"):
      constant_op.constant([2**72])

  def testComplex64(self):
    self._testAll(
        (1 + 2j) * np.arange(-15, 15).reshape([2, 3, 5]).astype(np.complex64))
    self._testAll(
        (1 + 2j) *
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.complex64))
    self._testAll(np.empty((2, 0, 5)).astype(np.complex64))

  def testComplex128(self):
    self._testAll(
        (1 + 2j) * np.arange(-15, 15).reshape([2, 3, 5]).astype(np.complex128))
    self._testAll(
        (1 + 2j) *
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.complex128))
    self._testAll(np.empty((2, 0, 5)).astype(np.complex128))

  @test_util.disable_tfrt("support creating string tensors from empty "
                          "numpy arrays.")
  def testString(self):
    val = [compat.as_bytes(str(x)) for x in np.arange(-15, 15)]
    self._testCpu(np.array(val).reshape([2, 3, 5]))
    self._testCpu(np.empty((2, 0, 5)).astype(np.str_))

  def testStringWithNulls(self):
    val = ops.convert_to_tensor(b"\0\0\0\0").numpy()
    self.assertEqual(len(val), 4)
    self.assertEqual(val, b"\0\0\0\0")

    val = ops.convert_to_tensor(b"xx\0xx").numpy()
    self.assertEqual(len(val), 5)
    self.assertAllEqual(val, b"xx\0xx")

    nested = [[b"\0\0\0\0", b"xx\0xx"], [b"\0_\0_\0_\0", b"\0"]]
    val = ops.convert_to_tensor(nested).numpy()
    # NOTE(mrry): Do not use assertAllEqual, because it converts nested to a
    #   numpy array, which loses the null terminators.
    self.assertEqual(val.tolist(), nested)

  def testStringConstantOp(self):
    s = constant_op.constant("uiuc")
    self.assertEqual(s.numpy().decode("utf-8"), "uiuc")
    s_array = constant_op.constant(["mit", "stanford"])
    self.assertAllEqual(s_array.numpy(), ["mit", "stanford"])

    with ops.device("/cpu:0"):
      s = constant_op.constant("cmu")
      self.assertEqual(s.numpy().decode("utf-8"), "cmu")

      s_array = constant_op.constant(["berkeley", "ucla"])
      self.assertAllEqual(s_array.numpy(), ["berkeley", "ucla"])

  def testExplicitShapeNumPy(self):
    c = constant_op.constant(
        np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32),
        shape=[2, 3, 5])
    self.assertEqual(c.get_shape(), [2, 3, 5])

  def testImplicitShapeNumPy(self):
    c = constant_op.constant(
        np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32))
    self.assertEqual(c.get_shape(), [2, 3, 5])

  def testExplicitShapeList(self):
    c = constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[7])
    self.assertEqual(c.get_shape(), [7])

  def testExplicitShapeFill(self):
    c = constant_op.constant(12, shape=[7])
    self.assertEqual(c.get_shape(), [7])
    self.assertAllEqual([12, 12, 12, 12, 12, 12, 12], c.numpy())

  def testExplicitShapeReshape(self):
    c = constant_op.constant(
        np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32),
        shape=[5, 2, 3])
    self.assertEqual(c.get_shape(), [5, 2, 3])

  def testImplicitShapeList(self):
    c = constant_op.constant([1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(c.get_shape(), [7])

  def testExplicitShapeNumber(self):
    c = constant_op.constant(1, shape=[1])
    self.assertEqual(c.get_shape(), [1])

  def testImplicitShapeNumber(self):
    c = constant_op.constant(1)
    self.assertEqual(c.get_shape(), [])

  def testShapeTooBig(self):
    with self.assertRaises(TypeError):
      constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[10])

  def testShapeTooSmall(self):
    with self.assertRaises(TypeError):
      constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[5])

  def testShapeWrong(self):
    with self.assertRaisesRegex(TypeError, None):
      constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[5])

  def testShape(self):
    self._testAll(constant_op.constant([1]).get_shape())

  def testDimension(self):
    x = constant_op.constant([1]).shape[0]
    self._testAll(x)

  def testDimensionList(self):
    x = [constant_op.constant([1]).shape[0]]
    self._testAll(x)

    # Mixing with regular integers is fine too
    self._testAll([1] + x)
    self._testAll(x + [1])

  def testDimensionTuple(self):
    x = constant_op.constant([1]).shape[0]
    self._testAll((x,))
    self._testAll((1, x))
    self._testAll((x, 1))

  def testInvalidLength(self):

    class BadList(list):

      def __init__(self):
        super(BadList, self).__init__([1, 2, 3])  # pylint: disable=invalid-length-returned

      def __len__(self):  # pylint: disable=invalid-length-returned
        return -1

    with self.assertRaisesRegex(ValueError, "should return >= 0"):
      constant_op.constant([BadList()])
    with self.assertRaisesRegex(ValueError, "mixed types"):
      constant_op.constant([1, 2, BadList()])
    with self.assertRaisesRegex(ValueError, "should return >= 0"):
      constant_op.constant(BadList())
    with self.assertRaisesRegex(ValueError, "should return >= 0"):
      constant_op.constant([[BadList(), 2], 3])
    with self.assertRaisesRegex(ValueError, "should return >= 0"):
      constant_op.constant([BadList(), [1, 2, 3]])
    with self.assertRaisesRegex(ValueError, "should return >= 0"):
      constant_op.constant([BadList(), []])

    # TODO(allenl, josh11b): These cases should return exceptions rather than
    # working (currently shape checking only checks the first element of each
    # sequence recursively). Maybe the first one is fine, but the second one
    # silently truncating is rather bad.

    # with self.assertRaisesRegex(ValueError, "should return >= 0"):
    #   constant_op.constant([[3, 2, 1], BadList()])
    # with self.assertRaisesRegex(ValueError, "should return >= 0"):
    #   constant_op.constant([[], BadList()])

  def testSparseValuesRaiseErrors(self):
    with self.assertRaisesRegex(ValueError, "non-rectangular Python sequence"):
      constant_op.constant([[1, 2], [3]], dtype=dtypes_lib.int32)

    with self.assertRaisesRegex(ValueError, None):
      constant_op.constant([[1, 2], [3]])

    with self.assertRaisesRegex(ValueError, None):
      constant_op.constant([[1, 2], [3], [4, 5]])

  # TODO(ashankar): This test fails with graph construction since
  # tensor_util.make_tensor_proto (invoked from constant_op.constant)
  # does not handle iterables (it relies on numpy conversion).
  # For consistency, should graph construction handle Python objects
  # that implement the sequence protocol (but not numpy conversion),
  # or should eager execution fail on such sequences?
  def testCustomSequence(self):

    # This is inspired by how many objects in pandas are implemented:
    # - They implement the Python sequence protocol
    # - But may raise a KeyError on __getitem__(self, 0)
    # See https://github.com/tensorflow/tensorflow/issues/20347
    class MySeq(object):

      def __getitem__(self, key):
        if key != 1 and key != 3:
          raise KeyError(key)
        return key

      def __len__(self):
        return 2

      def __iter__(self):
        l = list([1, 3])
        return l.__iter__()

    self.assertAllEqual([1, 3], self.evaluate(constant_op.constant(MySeq())))


class AsTensorTest(test.TestCase):

  def testAsTensorForTensorInput(self):
    t = constant_op.constant(10.0)
    x = ops.convert_to_tensor(t)
    self.assertIs(t, x)

  def testAsTensorForNonTensorInput(self):
    x = ops.convert_to_tensor(10.0)
    self.assertTrue(isinstance(x, ops.EagerTensor))


class ZerosTest(test.TestCase):

  def _Zeros(self, shape):
    ret = array_ops.zeros(shape)
    self.assertEqual(shape, ret.get_shape())
    return ret.numpy()

  def testConst(self):
    self.assertTrue(
        np.array_equal(self._Zeros([2, 3]), np.array([[0] * 3] * 2)))

  def testScalar(self):
    self.assertEqual(0, self._Zeros([]))
    self.assertEqual(0, self._Zeros(()))
    scalar = array_ops.zeros(constant_op.constant([], dtype=dtypes_lib.int32))
    self.assertEqual(0, scalar.numpy())

  def testDynamicSizes(self):
    np_ans = np.array([[0] * 3] * 2)
    # Creates a tensor of 2 x 3.
    d = array_ops.fill([2, 3], 12., name="fill")
    # Constructs a tensor of zeros of the same dimensions as "d".
    z = array_ops.zeros(array_ops.shape(d))
    out = z.numpy()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, d)
    self.assertShapeEqual(np_ans, z)

  def testDtype(self):
    d = array_ops.fill([2, 3], 12., name="fill")
    self.assertEqual(d.get_shape(), [2, 3])
    # Test default type for both constant size and dynamic size
    z = array_ops.zeros([2, 3])
    self.assertEqual(z.dtype, dtypes_lib.float32)
    self.assertEqual([2, 3], z.get_shape())
    self.assertAllEqual(z.numpy(), np.zeros([2, 3]))
    z = array_ops.zeros(array_ops.shape(d))
    self.assertEqual(z.dtype, dtypes_lib.float32)
    self.assertEqual([2, 3], z.get_shape())
    self.assertAllEqual(z.numpy(), np.zeros([2, 3]))
    # Test explicit type control
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64,
        dtypes_lib.bool,
        # TODO(josh11b): Support string type here.
        # dtypes_lib.string
    ]:
      z = array_ops.zeros([2, 3], dtype=dtype)
      self.assertEqual(z.dtype, dtype)
      self.assertEqual([2, 3], z.get_shape())
      z_value = z.numpy()
      self.assertFalse(np.any(z_value))
      self.assertEqual((2, 3), z_value.shape)
      z = array_ops.zeros(array_ops.shape(d), dtype=dtype)
      self.assertEqual(z.dtype, dtype)
      self.assertEqual([2, 3], z.get_shape())
      z_value = z.numpy()
      self.assertFalse(np.any(z_value))
      self.assertEqual((2, 3), z_value.shape)


class ZerosLikeTest(test.TestCase):

  def _compareZeros(self, dtype, use_gpu):
    # Creates a tensor of non-zero values with shape 2 x 3.
    # NOTE(kearnes): The default numpy dtype associated with tf.string is
    # np.object_ (and can't be changed without breaking a lot things), which
    # causes a TypeError in constant_op.constant below. Here we catch the
    # special case of tf.string and set the numpy dtype appropriately.
    if dtype == dtypes_lib.string:
      numpy_dtype = np.bytes_
    else:
      numpy_dtype = dtype.as_numpy_dtype
    d = constant_op.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
    # Constructs a tensor of zeros of the same dimensions and type as "d".
    z_var = array_ops.zeros_like(d)
    # Test that the type is correct
    self.assertEqual(z_var.dtype, dtype)
    # Test that the shape is correct
    self.assertEqual([2, 3], z_var.get_shape())

    # Test that the value is correct
    z_value = z_var.numpy()
    self.assertFalse(np.any(z_value))
    self.assertEqual((2, 3), z_value.shape)

  @test_util.disable_tfrt("b/169112823: unsupported dtype for Op:ZerosLike.")
  def testZerosLikeCPU(self):
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64,
        # TODO(josh11b): Support string type here.
        # dtypes_lib.string
    ]:
      self._compareZeros(dtype, use_gpu=False)

  @test_util.disable_tfrt("b/169112823: unsupported dtype for Op:ZerosLike.")
  def testZerosLikeGPU(self):
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.bool, dtypes_lib.int64,
        # TODO(josh11b): Support string type here.
        # dtypes_lib.string
    ]:
      self._compareZeros(dtype, use_gpu=True)

  @test_util.disable_tfrt("b/169112823: unsupported dtype for Op:ZerosLike.")
  def testZerosLikeDtype(self):
    # Make sure zeros_like works even for dtypes that cannot be cast between
    shape = (3, 5)
    dtypes = np.float32, np.complex64
    for in_type in dtypes:
      x = np.arange(15).astype(in_type).reshape(*shape)
      for out_type in dtypes:
        y = array_ops.zeros_like(x, dtype=out_type).numpy()
        self.assertEqual(y.dtype, out_type)
        self.assertEqual(y.shape, shape)
        self.assertAllEqual(y, np.zeros(shape, dtype=out_type))


class OnesTest(test.TestCase):

  def _Ones(self, shape):
    ret = array_ops.ones(shape)
    self.assertEqual(shape, ret.get_shape())
    return ret.numpy()

  def testConst(self):
    self.assertTrue(np.array_equal(self._Ones([2, 3]), np.array([[1] * 3] * 2)))

  def testScalar(self):
    self.assertEqual(1, self._Ones([]))
    self.assertEqual(1, self._Ones(()))
    scalar = array_ops.ones(constant_op.constant([], dtype=dtypes_lib.int32))
    self.assertEqual(1, scalar.numpy())

  def testDynamicSizes(self):
    np_ans = np.array([[1] * 3] * 2)
    # Creates a tensor of 2 x 3.
    d = array_ops.fill([2, 3], 12., name="fill")
    # Constructs a tensor of ones of the same dimensions as "d".
    z = array_ops.ones(array_ops.shape(d))
    out = z.numpy()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, d)
    self.assertShapeEqual(np_ans, z)

  def testDtype(self):
    d = array_ops.fill([2, 3], 12., name="fill")
    self.assertEqual(d.get_shape(), [2, 3])
    # Test default type for both constant size and dynamic size
    z = array_ops.ones([2, 3])
    self.assertEqual(z.dtype, dtypes_lib.float32)
    self.assertEqual([2, 3], z.get_shape())
    self.assertAllEqual(z.numpy(), np.ones([2, 3]))
    z = array_ops.ones(array_ops.shape(d))
    self.assertEqual(z.dtype, dtypes_lib.float32)
    self.assertEqual([2, 3], z.get_shape())
    self.assertAllEqual(z.numpy(), np.ones([2, 3]))
    # Test explicit type control
    for dtype in (dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
                  dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
                  dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64,
                  dtypes_lib.bool):
      z = array_ops.ones([2, 3], dtype=dtype)
      self.assertEqual(z.dtype, dtype)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z.numpy(), np.ones([2, 3]))
      z = array_ops.ones(array_ops.shape(d), dtype=dtype)
      self.assertEqual(z.dtype, dtype)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z.numpy(), np.ones([2, 3]))


class OnesLikeTest(test.TestCase):

  def testOnesLike(self):
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64
    ]:
      numpy_dtype = dtype.as_numpy_dtype
      # Creates a tensor of non-zero values with shape 2 x 3.
      d = constant_op.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
      # Constructs a tensor of zeros of the same dimensions and type as "d".
      z_var = array_ops.ones_like(d)
      # Test that the type is correct
      self.assertEqual(z_var.dtype, dtype)
      z_value = z_var.numpy()

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[1] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())


class FillTest(test.TestCase):

  def _compare(self, dims, val, np_ans, use_gpu):
    ctx = context.context()
    device = "GPU:0" if (use_gpu and ctx.num_gpus()) else "CPU:0"
    with ops.device(device):
      tf_ans = array_ops.fill(dims, val, name="fill")
      out = tf_ans.numpy()
    self.assertAllClose(np_ans, out)

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
    tf_ans = array_ops.fill([2, 3], np_ans[0][0], name="fill").numpy()
    self.assertAllEqual(np_ans, tf_ans)

  def testFillNegative(self):
    for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
      with self.assertRaises(errors_impl.InvalidArgumentError):
        array_ops.fill(shape, 7)

  def testShapeFunctionEdgeCases(self):
    # Non-vector dimensions.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      array_ops.fill([[0, 1], [2, 3]], 1.0)

    # Non-scalar value.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      array_ops.fill([3, 2], [1.0, 2.0])


if __name__ == "__main__":
  test.main()
