"""Tests for IdentityOp."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gen_array_ops


class IdentityOpTest(tf.test.TestCase):

  def testInt32_6(self):
    with self.test_session():
      value = tf.identity([1, 2, 3, 4, 5, 6]).eval()
    self.assertAllEqual(np.array([1, 2, 3, 4, 5, 6]), value)

  def testInt32_2_3(self):
    with self.test_session():
      inp = tf.constant([10, 20, 30, 40, 50, 60], shape=[2, 3])
      value = tf.identity(inp).eval()
    self.assertAllEqual(np.array([[10, 20, 30], [40, 50, 60]]), value)

  def testString(self):
    with self.test_session():
      value = tf.identity(["A", "b", "C", "d", "E", "f"]).eval()
    self.assertAllEqual(["A", "b", "C", "d", "E", "f"], value)

  def testIdentityShape(self):
    with self.test_session():
      shape = [2, 3]
      array_2x3 = [[1, 2, 3], [6, 5, 4]]
      tensor = tf.constant(array_2x3)
      self.assertEquals(shape, tensor.get_shape())
      self.assertEquals(shape, tf.identity(tensor).get_shape())
      self.assertEquals(shape, tf.identity(array_2x3).get_shape())
      self.assertEquals(shape, tf.identity(np.array(array_2x3)).get_shape())

  def testRefIdentityShape(self):
    with self.test_session():
      shape = [2, 3]
      tensor = tf.Variable(tf.constant([[1, 2, 3], [6, 5, 4]], dtype=tf.int32))
      self.assertEquals(shape, tensor.get_shape())
      self.assertEquals(shape, gen_array_ops._ref_identity(tensor).get_shape())


if __name__ == "__main__":
  tf.test.main()
