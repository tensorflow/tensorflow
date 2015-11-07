"""Tests for ConstantOp."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gen_array_ops


class ConstantTest(tf.test.TestCase):

  def _testCpu(self, x):
    np_ans = np.array(x)
    with self.test_session(use_gpu=False):
      tf_ans = tf.convert_to_tensor(x).eval()
    if np_ans.dtype in [np.float32, np.float64, np.complex64]:
      self.assertAllClose(np_ans, tf_ans)
    else:
      self.assertAllEqual(np_ans, tf_ans)

  def _testGpu(self, x):
    np_ans = np.array(x)
    with self.test_session(use_gpu=True):
      tf_ans = tf.convert_to_tensor(x).eval()
    if np_ans.dtype in [np.float32, np.float64, np.complex64]:
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
    self._testAll(
        (100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(np.int32))
    self._testAll(np.empty((2, 0, 5)).astype(np.int32))

  def testInt64(self):
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int64))
    self._testAll(
        (100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(np.int64))
    self._testAll(np.empty((2, 0, 5)).astype(np.int64))

  def testSComplex(self):
    self._testAll(
        np.complex(1, 2) * np.arange(-15, 15).reshape([2, 3, 5]).astype(
            np.complex64))
    self._testAll(np.complex(
        1, 2) * np.random.normal(size=30).reshape([2, 3, 5]).astype(
            np.complex64))
    self._testAll(np.empty((2, 0, 5)).astype(np.complex64))

  def testString(self):
    self._testCpu(np.array([str(x) for x in np.arange(-15, 15)]).reshape(
        [2, 3, 5]))
    self._testCpu(np.empty((2, 0, 5)).astype(np.str_))

  def testStringWithNulls(self):
    with self.test_session():
      val = tf.convert_to_tensor("\0\0\0\0").eval()
    self.assertEqual(len(val), 4)
    self.assertEqual(val, "\0\0\0\0")

    with self.test_session():
      val = tf.convert_to_tensor("xx\0xx").eval()
    self.assertEqual(len(val), 5)
    self.assertAllEqual(val, "xx\0xx")
    nested = [["\0\0\0\0", "xx\0xx"], ["\0_\0_\0_\0", "\0"]]

    with self.test_session():
      val = tf.convert_to_tensor(nested).eval()
    # NOTE(mrry): Do not use assertAllEqual, because it converts nested to a
    #   numpy array, which loses the null terminators.
    self.assertEqual(val.tolist(), nested)

  def testExplicitShapeNumPy(self):
    with tf.Graph().as_default():
      c = tf.constant(
          np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32),
          shape=[2, 3, 5])
    self.assertEqual(c.get_shape(), [2, 3, 5])

  def testImplicitShapeNumPy(self):
    with tf.Graph().as_default():
      c = tf.constant(
          np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32))
    self.assertEqual(c.get_shape(), [2, 3, 5])

  def testExplicitShapeList(self):
    with tf.Graph().as_default():
      c = tf.constant([1, 2, 3, 4, 5, 6, 7], shape=[7])
    self.assertEqual(c.get_shape(), [7])

  def testImplicitShapeList(self):
    with tf.Graph().as_default():
      c = tf.constant([1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(c.get_shape(), [7])

  def testExplicitShapeNumber(self):
    with tf.Graph().as_default():
      c = tf.constant(1, shape=[1])
    self.assertEqual(c.get_shape(), [1])

  def testImplicitShapeNumber(self):
    with tf.Graph().as_default():
      c = tf.constant(1)
    self.assertEqual(c.get_shape(), [])

  def testShapeInconsistent(self):
    with tf.Graph().as_default():
      c = tf.constant([1, 2, 3, 4, 5, 6, 7], shape=[10])
    self.assertEqual(c.get_shape(), [10])

  # pylint: disable=g-long-lambda
  def testShapeWrong(self):
    with tf.Graph().as_default():
      with self.assertRaisesWithPredicateMatch(
          ValueError,
          lambda e: ("Too many elements provided. Needed at most 5, "
                     "but received 7" == str(e))):
        tf.constant([1, 2, 3, 4, 5, 6, 7], shape=[5])
  # pylint: enable=g-long-lambda

  def testTooLargeConstant(self):
    with tf.Graph().as_default():
      large_array = np.zeros((512, 1024, 1024), dtype=np.float32)
      with self.assertRaisesRegexp(
          ValueError,
          "Cannot create an Operation with a NodeDef larger than 2GB."):
        c = tf.constant(large_array)

  def testTooLargeGraph(self):
    with tf.Graph().as_default() as g:
      large_array = np.zeros((256, 1024, 1024), dtype=np.float32)
      c = tf.constant(large_array)
      d = tf.constant(large_array)
      with self.assertRaisesRegexp(
          ValueError, "GraphDef cannot be larger than 2GB."):
        g.as_graph_def()

  def testSparseValuesRaiseErrors(self):
    with self.assertRaisesRegexp(ValueError,
                                 "setting an array element with a sequence"):
      c = tf.constant([[1, 2], [3]], dtype=tf.int32)

    with self.assertRaisesRegexp(ValueError, "must be a dense"):
      c = tf.constant([[1, 2], [3]])

    with self.assertRaisesRegexp(ValueError, "must be a dense"):
      c = tf.constant([[1, 2], [3], [4, 5]])


class AsTensorTest(tf.test.TestCase):

  def testAsTensorForTensorInput(self):
    with tf.Graph().as_default():
      t = tf.constant(10.0)
      x = tf.convert_to_tensor(t)
    self.assertIs(t, x)

  def testAsTensorForNonTensorInput(self):
    with tf.Graph().as_default():
      x = tf.convert_to_tensor(10.0)
    self.assertTrue(isinstance(x, tf.Tensor))

  def testAsTensorForShapeInput(self):
    with self.test_session():
      x = tf.convert_to_tensor(tf.TensorShape([]))
      self.assertEqual(tf.int32, x.dtype)
      self.assertAllEqual([], x.eval())

      x = tf.convert_to_tensor(tf.TensorShape([1, 2, 3]))
      self.assertEqual(tf.int32, x.dtype)
      self.assertAllEqual([1, 2, 3], x.eval())

      x = tf.convert_to_tensor(tf.TensorShape([1, 2, 3]), dtype=tf.int64)
      self.assertEqual(tf.int64, x.dtype)
      self.assertAllEqual([1, 2, 3], x.eval())

      x = tf.reshape(tf.zeros([6]), tf.TensorShape([2, 3]))
      self.assertAllEqual([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], x.eval())

    with self.assertRaisesRegexp(ValueError, "partially known"):
      tf.convert_to_tensor(tf.TensorShape(None))

    with self.assertRaisesRegexp(ValueError, "partially known"):
      tf.convert_to_tensor(tf.TensorShape([1, None, 64]))

    with self.assertRaises(TypeError):
      tf.convert_to_tensor(tf.TensorShape([1, 2, 3]), dtype=tf.float32)

  def testAsTensorForDimensionInput(self):
    with self.test_session():
      x = tf.convert_to_tensor(tf.TensorShape([1, 2, 3])[1])
      self.assertEqual(tf.int32, x.dtype)
      self.assertAllEqual(2, x.eval())

      x = tf.convert_to_tensor(tf.TensorShape([1, 2, 3])[1], dtype=tf.int64)
      self.assertEqual(tf.int64, x.dtype)
      self.assertAllEqual(2, x.eval())

    with self.assertRaisesRegexp(ValueError, "unknown Dimension"):
      tf.convert_to_tensor(tf.TensorShape(None)[1])

    with self.assertRaisesRegexp(ValueError, "unknown Dimension"):
      tf.convert_to_tensor(tf.TensorShape([1, None, 64])[1])

    with self.assertRaises(TypeError):
      tf.convert_to_tensor(tf.TensorShape([1, 2, 3])[1], dtype=tf.float32)


class IdentityOpTest(tf.test.TestCase):

  def testIdTensor(self):
    with tf.Graph().as_default():
      x = tf.constant(2.0, shape=[6], name="input")
      id_op = tf.identity(x, name="id")
    self.assertTrue(isinstance(id_op.op.inputs[0], tf.Tensor))
    self.assertProtoEquals(
        "name: 'id' op: 'Identity' input: 'input' "
        "attr { key: 'T' value { type: DT_FLOAT } }", id_op.op.node_def)


class ZerosTest(tf.test.TestCase):

  def _Zeros(self, shape):
    with self.test_session():
      ret = tf.zeros(shape)
      self.assertEqual(shape, ret.get_shape())
      return ret.eval()

  def testConst(self):
    self.assertTrue(np.array_equal(self._Zeros([2, 3]), np.array([[0] * 3] *
                                                                 2)))

  def testDynamicSizes(self):
    np_ans = np.array([[0] * 3] * 2)
    with self.test_session():
      # Creates a tensor of 2 x 3.
      d = tf.fill([2, 3], 12., name="fill")
      # Constructs a tensor of zeros of the same dimensions as "d".
      z = tf.zeros(tf.shape(d))
      out = z.eval()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, d)
    self.assertShapeEqual(np_ans, z)

  def testDtype(self):
    with self.test_session():
      d = tf.fill([2, 3], 12., name="fill")
      self.assertEqual(d.get_shape(), [2, 3])
      # Test default type for both constant size and dynamic size
      z = tf.zeros([2, 3])
      self.assertEquals(z.dtype, tf.float32)
      self.assertEqual([2, 3], z.get_shape())
      z = tf.zeros(tf.shape(d))
      self.assertEquals(z.dtype, tf.float32)
      self.assertEqual([2, 3], z.get_shape())
      # Test explicit type control
      for dtype in [tf.float32, tf.float64, tf.int32,
                    tf.uint8, tf.int16, tf.int8,
                    tf.complex64, tf.int64]:
        z = tf.zeros([2, 3], dtype=dtype)
        self.assertEquals(z.dtype, dtype)
        self.assertEquals([2, 3], z.get_shape())
        z = tf.zeros(tf.shape(d), dtype=dtype)
        self.assertEquals(z.dtype, dtype)
        self.assertEquals([2, 3], z.get_shape())


class ZerosLikeTest(tf.test.TestCase):

  def testZerosLike(self):
    for dtype in [tf.float32, tf.float64, tf.int32,
                  tf.uint8, tf.int16, tf.int8,
                  tf.complex64, tf.int64]:
      numpy_dtype = dtype.as_numpy_dtype
      with self.test_session():
        # Creates a tensor of non-zero values with shape 2 x 3.
        d = tf.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
        # Constructs a tensor of zeros of the same dimensions and type as "d".
        z_var = tf.zeros_like(d)
        # Test that the type is correct
        self.assertEquals(z_var.dtype, dtype)
        z_value = z_var.eval()

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[0] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())

  def testGenZerosLike(self):
    for dtype in [tf.float32, tf.float64, tf.int32,
                  tf.uint8, tf.int16, tf.int8,
                  tf.complex64, tf.int64]:
      numpy_dtype = dtype.as_numpy_dtype
      with self.test_session():
        # Creates a tensor of non-zero values with shape 2 x 3.
        d = tf.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
        # Constructs a tensor of zeros of the same dimensions and type as "d".
        z_var = gen_array_ops._zeros_like(d)
        # Test that the type is correct
        self.assertEquals(z_var.dtype, dtype)
        z_value = z_var.eval()

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[0] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())


class OnesTest(tf.test.TestCase):

  def _Ones(self, shape):
    with self.test_session():
      ret = tf.ones(shape)
      self.assertEqual(shape, ret.get_shape())
      return ret.eval()

  def testConst(self):
    self.assertTrue(np.array_equal(self._Ones([2, 3]), np.array([[1] * 3] * 2)))

  def testDynamicSizes(self):
    np_ans = np.array([[1] * 3] * 2)
    with self.test_session():
      # Creates a tensor of 2 x 3.
      d = tf.fill([2, 3], 12., name="fill")
      # Constructs a tensor of ones of the same dimensions as "d".
      z = tf.ones(tf.shape(d))
      out = z.eval()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, d)
    self.assertShapeEqual(np_ans, z)

  def testDtype(self):
    with self.test_session():
      d = tf.fill([2, 3], 12., name="fill")
      self.assertEqual(d.get_shape(), [2, 3])
      # Test default type for both constant size and dynamic size
      z = tf.ones([2, 3])
      self.assertEquals(z.dtype, tf.float32)
      self.assertEqual([2, 3], z.get_shape())
      z = tf.ones(tf.shape(d))
      self.assertEquals(z.dtype, tf.float32)
      self.assertEqual([2, 3], z.get_shape())
      # Test explicit type control
      for dtype in [tf.float32, tf.float64, tf.int32,
                    tf.uint8, tf.int16, tf.int8,
                    tf.complex64, tf.int64]:
        z = tf.ones([2, 3], dtype=dtype)
        self.assertEquals(z.dtype, dtype)
        self.assertEqual([2, 3], z.get_shape())
        z = tf.ones(tf.shape(d), dtype=dtype)
        self.assertEquals(z.dtype, dtype)
        self.assertEqual([2, 3], z.get_shape())


class OnesLikeTest(tf.test.TestCase):

  def testOnesLike(self):
    for dtype in [tf.float32, tf.float64, tf.int32,
                  tf.uint8, tf.int16, tf.int8,
                  tf.complex64, tf.int64]:
      numpy_dtype = dtype.as_numpy_dtype
      with self.test_session():
        # Creates a tensor of non-zero values with shape 2 x 3.
        d = tf.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
        # Constructs a tensor of zeros of the same dimensions and type as "d".
        z_var = tf.ones_like(d)
        # Test that the type is correct
        self.assertEquals(z_var.dtype, dtype)
        z_value = z_var.eval()

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[1] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())

  def testGenOnesLike(self):
    for dtype in [tf.float32, tf.float64, tf.int32,
                  tf.uint8, tf.int16, tf.int8,
                  tf.complex64, tf.int64]:
      numpy_dtype = dtype.as_numpy_dtype
      with self.test_session():
        # Creates a tensor of non-zero values with shape 2 x 3.
        d = tf.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
        # Constructs a tensor of zeros of the same dimensions and type as "d".
        z_var = tf.ones_like(d)
        # Test that the type is correct
        self.assertEquals(z_var.dtype, dtype)
        z_value = z_var.eval()

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[1] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())


class FillTest(tf.test.TestCase):

  def _compare(self, dims, val, np_ans, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      tf_ans = tf.fill(dims, val, name="fill")
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

  def testFillComplex(self):
    np_ans = np.array([[0.15] * 3] * 2).astype(np.complex64)
    self._compare([2, 3], np_ans[0][0], np_ans, use_gpu=False)

  def testFillString(self):
    np_ans = np.array([["yolo"] * 3] * 2)
    with self.test_session(use_gpu=False):
      tf_ans = tf.fill([2, 3], np_ans[0][0], name="fill").eval()
    self.assertAllEqual(np_ans, tf_ans)

  def testShapeFunctionEdgeCases(self):
    # Non-vector dimensions.
    with self.assertRaises(ValueError):
      tf.fill([[0, 1], [2, 3]], 1.0)

    # Non-scalar value.
    with self.assertRaises(ValueError):
      tf.fill([3, 2], [1.0, 2.0])

    # Partial dimension information.
    f = tf.fill(
        tf.placeholder(tf.int32, shape=(4,)), 3.0)
    self.assertEqual([None, None, None, None], f.get_shape().as_list())


class PlaceholderTest(tf.test.TestCase):

  def testDtype(self):
    with self.test_session():
      p = tf.placeholder(tf.float32, name="p")
      p_identity = tf.identity(p)
      feed_array = np.random.rand(10, 10)
      self.assertAllClose(p_identity.eval(feed_dict={p: feed_array}),
                          feed_array)

      with self.assertRaisesOpError(
          "must feed a value for placeholder tensor 'p' with dtype float"):
        p_identity.eval()

  def testShape(self):
    with self.test_session():
      p = tf.placeholder(tf.float32, shape=(10, 10), name="p")
      p_identity = tf.identity(p)
      feed_array = np.random.rand(10, 10)
      self.assertAllClose(p_identity.eval(feed_dict={p: feed_array}),
                          feed_array)

      with self.assertRaisesOpError(
          "must feed a value for placeholder tensor 'p' with dtype float and "
          "shape dim { size: 10 } dim { size: 10 }"):
        p_identity.eval()

      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Cannot feed value of shape" in e.message):
        p_identity.eval(feed_dict={p: feed_array[:5, :5]})

  def testPartialShape(self):
    with self.test_session():
      p = tf.placeholder(tf.float32, shape=[None, 3], name="p")
      p_identity = tf.identity(p)
      feed_array = np.random.rand(10, 3)
      self.assertAllClose(p_identity.eval(feed_dict={p: feed_array}),
                          feed_array)

      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Cannot feed value of shape" in e.message):
        p_identity.eval(feed_dict={p: feed_array[:5, :2]})

  def testControlDependency(self):
    with self.test_session():
      p = tf.placeholder(tf.int32, shape=[], name="p")
      with tf.control_dependencies([p]):
        c = tf.constant(5, tf.int32)
      d = tf.mul(p, c)
      self.assertEqual(10, d.eval(feed_dict={p: 2}))

  def testFillNegative(self):
    with self.test_session():
      for shape in (-1,), (2, -1), (-1, 2):
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                     " must be nonnegative"):
          tf.fill(shape, 7).eval()


if __name__ == "__main__":
  tf.test.main()
