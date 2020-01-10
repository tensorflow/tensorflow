"""Tests for tf.py."""
import operator

import tensorflow.python.platform

import numpy as np

import tensorflow.python.platform

import tensorflow as tf
from tensorflow.python.ops import random_ops


class VariablesTestCase(tf.test.TestCase):

  def testInitialization(self):
    with self.test_session():
      var0 = tf.Variable(0.0)
      self.assertEqual("Variable:0", var0.name)
      self.assertEqual([], var0.get_shape())
      self.assertEqual([], var0.get_shape())

      var1 = tf.Variable(1.1)
      self.assertEqual("Variable_1:0", var1.name)
      self.assertEqual([], var1.get_shape())
      self.assertEqual([], var1.get_shape())

      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        var0.eval()

      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        var1.eval()

      tf.initialize_all_variables().run()

      self.assertAllClose(0.0, var0.eval())
      self.assertAllClose(1.1, var1.eval())

  def testInitializationOrder(self):
    with self.test_session():
      rnd = tf.Variable(random_ops.random_uniform([3, 6]), name="rnd")
      self.assertEqual("rnd:0", rnd.name)
      self.assertEqual([3, 6], rnd.get_shape())
      self.assertEqual([3, 6], rnd.get_shape())

      dep = tf.Variable(rnd.initialized_value(), name="dep")
      self.assertEqual("dep:0", dep.name)
      self.assertEqual([3, 6], dep.get_shape())
      self.assertEqual([3, 6], dep.get_shape())

      # Currently have to set the shape manually for Add.
      added_val = rnd.initialized_value() + dep.initialized_value() + 2.0
      added_val.set_shape(rnd.get_shape())

      depdep = tf.Variable(added_val, name="depdep")
      self.assertEqual("depdep:0", depdep.name)
      self.assertEqual([3, 6], depdep.get_shape())
      self.assertEqual([3, 6], depdep.get_shape())

      tf.initialize_all_variables().run()

      self.assertAllClose(rnd.eval(), dep.eval())
      self.assertAllClose(rnd.eval() + dep.eval() + 2.0,
                          depdep.eval())

  def testAssignments(self):
    with self.test_session():
      var = tf.Variable(0.0)
      plus_one = var.assign_add(1.0)
      minus_one = var.assign_sub(2.0)
      four = var.assign(4.0)
      tf.initialize_all_variables().run()
      self.assertAllClose(0.0, var.eval())

      self.assertAllClose(1.0, plus_one.eval())
      self.assertAllClose(1.0, var.eval())

      self.assertAllClose(-1.0, minus_one.eval())
      self.assertAllClose(-1.0, var.eval())

      self.assertAllClose(4.0, four.eval())
      self.assertAllClose(4.0, var.eval())

  def _countUpToTest(self, dtype):
    with self.test_session():
      zero = tf.constant(0, dtype=dtype)
      var = tf.Variable(zero)
      count_up_to = var.count_up_to(3)

      tf.initialize_all_variables().run()
      self.assertEqual(0, var.eval())

      self.assertEqual(0, count_up_to.eval())
      self.assertEqual(1, var.eval())

      self.assertEqual(1, count_up_to.eval())
      self.assertEqual(2, var.eval())

      self.assertEqual(2, count_up_to.eval())
      self.assertEqual(3, var.eval())

      with self.assertRaisesOpError("Reached limit of 3"):
        count_up_to.eval()
      self.assertEqual(3, var.eval())

      with self.assertRaisesOpError("Reached limit of 3"):
        count_up_to.eval()
      self.assertEqual(3, var.eval())

  def testCountUpToInt32(self):
    self._countUpToTest(tf.int32)

  def testCountUpToInt64(self):
    self._countUpToTest(tf.int64)

  def testUseVariableAsTensor(self):
    with self.test_session():
      var_x = tf.Variable(2.0)
      var_y = tf.Variable(3.0)
      tf.initialize_all_variables().run()
      self.assertAllClose(2.0, var_x.eval())
      self.assertAllClose(3.0, var_y.eval())
      self.assertAllClose(5.0, tf.add(var_x, var_y).eval())

  def testCollections(self):
    with self.test_session():
      var_x = tf.Variable(2.0)
      var_y = tf.Variable(2.0, trainable=False)
      var_z = tf.Variable(2.0, trainable=True)
      var_t = tf.Variable(
          2.0, trainable=True,
          collections=[tf.GraphKeys.TRAINABLE_VARIABLES,
                       tf.GraphKeys.VARIABLES])
      self.assertEqual([var_x, var_y, var_z, var_t], tf.all_variables())
      self.assertEqual([var_x, var_z, var_t], tf.trainable_variables())

  def testOperators(self):
    with self.test_session():
      var_f = tf.Variable([2.0])
      add = var_f + 0.0
      radd = 1.0 + var_f
      sub = var_f - 1.0
      rsub = 1.0 - var_f
      mul = var_f * 10.0
      rmul = 10.0 * var_f
      div = var_f / 10.0
      rdiv = 10.0 / var_f
      lt = var_f < 3.0
      rlt = 3.0 < var_f
      le = var_f <= 2.0
      rle = 2.0 <= var_f
      gt = var_f > 3.0
      rgt = 3.0 > var_f
      ge = var_f >= 2.0
      rge = 2.0 >= var_f
      neg = -var_f
      abs_v = abs(var_f)

      var_i = tf.Variable([20])
      mod = var_i % 7
      rmod = 103 % var_i

      var_b = tf.Variable([True, False])
      and_v = operator.and_(var_b, [True, True])
      or_v = operator.or_(var_b, [False, True])
      xor_v = operator.xor(var_b, [False, False])
      invert_v = ~var_b

      rnd = np.random.rand(4, 4).astype("f")
      var_t = tf.Variable(rnd)
      slice_v = var_t[2, 0:0]

      tf.initialize_all_variables().run()
      self.assertAllClose([2.0], add.eval())
      self.assertAllClose([3.0], radd.eval())
      self.assertAllClose([1.0], sub.eval())
      self.assertAllClose([-1.0], rsub.eval())
      self.assertAllClose([20.0], mul.eval())
      self.assertAllClose([20.0], rmul.eval())
      self.assertAllClose([0.2], div.eval())
      self.assertAllClose([5.0], rdiv.eval())
      self.assertAllClose([-2.0], neg.eval())
      self.assertAllClose([2.0], abs_v.eval())
      self.assertAllClose([True], lt.eval())
      self.assertAllClose([False], rlt.eval())
      self.assertAllClose([True], le.eval())
      self.assertAllClose([True], rle.eval())
      self.assertAllClose([False], gt.eval())
      self.assertAllClose([True], rgt.eval())
      self.assertAllClose([True], ge.eval())
      self.assertAllClose([True], rge.eval())

      self.assertAllClose([6], mod.eval())
      self.assertAllClose([3], rmod.eval())

      self.assertAllClose([True, False], and_v.eval())
      self.assertAllClose([True, True], or_v.eval())
      self.assertAllClose([True, False], xor_v.eval())
      self.assertAllClose([False, True], invert_v.eval())

      self.assertAllClose(rnd[2, 0:0], slice_v.eval())

  def testSession(self):
    with self.test_session() as sess:
      var = tf.Variable([1, 12])
      tf.initialize_all_variables().run()
      self.assertAllClose([1, 12], sess.run(var))


class IsInitializedTest(tf.test.TestCase):

  def testNoVars(self):
    with tf.Graph().as_default():
      self.assertEqual(None, tf.assert_variables_initialized())

  def testVariables(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      v = tf.Variable([1, 2])
      w = tf.Variable([3, 4])
      _ = v, w
      inited = tf.assert_variables_initialized()
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        sess.run(inited)
      tf.initialize_all_variables().run()
      sess.run(inited)

  def testVariableList(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      v = tf.Variable([1, 2])
      w = tf.Variable([3, 4])
      inited = tf.assert_variables_initialized([v])
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        inited.op.run()
      sess.run(w.initializer)
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        inited.op.run()
      v.initializer.run()
      inited.op.run()


if __name__ == "__main__":
  tf.test.main()
