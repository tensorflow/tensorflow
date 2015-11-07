"""Tests for tensorflow.kernels.sparse_op."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


def _SparseToDense(sparse_indices, output_size, sparse_values,
                   default_value):
  return tf.sparse_to_dense(sparse_indices, output_size,
                            sparse_values, default_value)


class SparseToDenseTest(tf.test.TestCase):

  def testInt(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], 1, 0).eval()
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testFloat(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], 1.0, 0.0).eval()
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.float32)
    self.assertAllClose(np_ans, tf_ans)

  def testString(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], "a", "b").eval()
    np_ans = np.array(["b", "a", "b", "a", "b"]).astype(np.string_)
    self.assertAllEqual(np_ans, tf_ans)

  def testSetValue(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], [1, 2], -1).eval()
    np_ans = np.array([-1, 1, -1, 2, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testSetSingleValue(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([1, 3], [5], 1, -1).eval()
    np_ans = np.array([-1, 1, -1, 1, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def test2d(self):
    # pylint: disable=bad-whitespace
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([[1, 3], [2, 0]], [3, 4], 1, -1).eval()
    np_ans = np.array([[-1, -1, -1, -1],
                       [-1, -1, -1,  1],
                       [ 1, -1, -1, -1]]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def test3d(self):
    with self.test_session(use_gpu=False):
      tf_ans = _SparseToDense([[1, 3, 0], [2, 0, 1]], [3, 4, 2], 1, -1).eval()
    np_ans = np.ones((3, 4, 2), dtype=np.int32) * -1
    np_ans[1, 3, 0] = 1
    np_ans[2, 0, 1] = 1
    self.assertAllClose(np_ans, tf_ans)

  def testBadShape(self):
    with self.test_session():
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: ("Input shape should be a vector" == str(e))):
        _SparseToDense([1, 3], [[5], [3]], 1, -1)

  def testBadValue(self):
    with self.test_session():
      dense = _SparseToDense([1, 3], [5], [[5], [3]], -1)
      with self.assertRaisesOpError(
          r"sparse_values has incorrect shape \[2,1\], "
          r"should be \[\] or \[2\]"):
        dense.eval()

  def testBadNumValues(self):
    with self.test_session():
      dense = _SparseToDense([1, 3], [5], [1, 2, 3], -1)
      with self.assertRaisesOpError(
          r"sparse_values has incorrect shape \[3\], should be \[\] or \[2\]"):
        dense.eval()

  def testBadDefault(self):
    with self.test_session():
      dense = _SparseToDense([1, 3], [5], [1, 2], [1, 2])
      with self.assertRaisesOpError("default_value should be a scalar"):
        dense.eval()

  def testShapeInferenceKnownShape(self):
    with self.test_session(use_gpu=False):
      indices = tf.placeholder(tf.int64)

      shape = [4, 5, 6]
      output = tf.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape(), [4, 5, 6])

      shape = tf.placeholder(tf.int64, shape=(3,))
      output = tf.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape().as_list(), [None, None, None])

  def testShapeInferenceUnknownShape(self):
    with self.test_session(use_gpu=False):
      indices = tf.placeholder(tf.int64)
      shape = tf.placeholder(tf.int64)
      output = tf.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape().ndims, None)


if __name__ == "__main__":
  tf.test.main()
