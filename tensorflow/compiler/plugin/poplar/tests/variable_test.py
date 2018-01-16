# Copyright 2017 Graphcore Ltd
#

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util

from tensorflow.python.ops import resource_variable_ops

class IpuXlaVariableTest(test_util.TensorFlowTestCase):

  def testInitializeSimpleVariables(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:

        x = resource_variable_ops.ResourceVariable(
          tf.random_normal([5,5], stddev=0.1), name="x")
        y = resource_variable_ops.ResourceVariable(
          tf.random_normal([1], stddev=0.1), name="y")

        sess.run(tf.global_variables_initializer())

        r1, r2 = sess.run([x,y])

        self.assertAllClose(r1, np.zeros([5,5]), atol=1.0)
        self.assertAllClose(r2, [0.0], atol=1.0)

  def testInitializeSharedVariables(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:
        with tf.variable_scope("vs", use_resource=True):
          x = tf.get_variable("x", shape=[], dtype=tf.float32,
                              initializer=tf.constant_initializer(1))

          y = tf.get_variable("y", shape=[], dtype=tf.float32,
                               initializer=tf.constant_initializer(2))

        sess.run(tf.global_variables_initializer())

        r1, r2 = sess.run([x,y])

        self.assertAllClose(r1, 1)
        self.assertAllClose(r2, 2)

  def testRead(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:
        with tf.variable_scope("vs", use_resource=True):
          z = tf.get_variable("z", shape=[], dtype=tf.float32,
                              initializer=tf.constant_initializer(3))

        sess.run(tf.global_variables_initializer())

        r = sess.run(z.read_value())

        self.assertAllClose(r, 3)

  def testAssign(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:
        with tf.variable_scope("vs", use_resource=True):
          z = tf.get_variable("z", shape=[], dtype=tf.float32,
                              initializer=tf.constant_initializer(0))

        sess.run(tf.global_variables_initializer())

        sess.run(tf.assign(z, 2))
        r = sess.run(z)
        self.assertAllClose(r, 2)

        sess.run(tf.assign_add(z, 6))
        r = sess.run(z)
        self.assertAllClose(r, 8)

  def testGradientDescent(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.variable_scope("vs", use_resource=True):

        w = tf.get_variable("w", shape=[4, 2], dtype=tf.float32,
          initializer=tf.constant_initializer(
            np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)))
        b = tf.get_variable("b", shape=[2], dtype=tf.float32,
          initializer=tf.constant_initializer(
            np.array([2, 3], dtype=np.float32)))

      x = tf.placeholder(tf.float32, shape=[1, 4])
      y = tf.matmul(x, w) + b

      loss = tf.reduce_sum(y)
      optimizer = tf.train.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(loss)

    with tf.Session() as sess:

      sess.run(tf.global_variables_initializer())
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      vw, vb = sess.run([w, b])

      self.assertAllClose(
        np.array([[0.3, 1.3], [2.7, 3.7], [4.5, 5.5], [6.1, 7.1]],
                 dtype=np.float32),
        vw, rtol=1e-4)

      self.assertAllClose(np.array([1.9, 2.9], dtype=np.float32), vb, rtol=1e-4)

  def testRepeatedGradientDescent(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.variable_scope("vs", use_resource=True):

        w = tf.get_variable("w", shape=[4, 2], dtype=tf.float32,
                            initializer=tf.constant_initializer(
                              np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)))
        b = tf.get_variable("b", shape=[2], dtype=tf.float32,
                            initializer=tf.constant_initializer(
                              np.array([2, 3], dtype=np.float32)))

      x = tf.placeholder(tf.float32, shape=[1, 4])
      y = tf.matmul(x, w) + b

      loss = tf.reduce_sum(y)
      optimizer = tf.train.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(loss)

    with tf.Session() as sess:

      sess.run(tf.global_variables_initializer())
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run(train, {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run(train, {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      vw, vb = sess.run([w, b])

      self.assertAllClose(
        np.array([[-1.3, -0.3], [1.7, 2.7], [2.9, 3.9], [3.5, 4.5]],
                 dtype=np.float32),
        vw, rtol=1e-4)

      self.assertAllClose(np.array([1.5, 2.5], dtype=np.float32), vb, rtol=1e-4)


  def testMultipleUpdate(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:
        with tf.variable_scope("vs", use_resource=True):
          z = tf.get_variable("z", shape=[], dtype=tf.float32,
                              initializer=tf.constant_initializer(0))

        updater = tf.assign_add(z, 1.0)

        sess.run(tf.global_variables_initializer())

        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)

        r = sess.run(z)
        self.assertAllClose(r, 10.0)

  def testRandomNormalInitalizer(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:
        with tf.variable_scope("vs", use_resource=True):
          i = tf.random_normal_initializer(mean=2.0, stddev=0.01)
          z = tf.get_variable("z1", shape=[], dtype=tf.float32, initializer=i)

        sess.run(tf.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 2.0, 0.2, 0.2)

  def testDefaultRandomNormalInitalizer(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:
        with tf.variable_scope("vs", use_resource=True):
          i = tf.random_normal_initializer()
          z = tf.get_variable("z1", shape=[], dtype=tf.float32, initializer=i)

        sess.run(tf.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 0.0, 1.0, 3.0)

  def testTruncatedNormalInitalizer(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:
        with tf.variable_scope("vs", use_resource=True):
          i = tf.truncated_normal_initializer(mean=1.0, stddev=0.01)
          z = tf.get_variable("z1", shape=[], dtype=tf.float32, initializer=i)

        sess.run(tf.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 1.0, 0.2, 0.2)

  def testDefaultTruncatedNormalInitalizer(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:
        with tf.variable_scope("vs", use_resource=True):
          i = tf.truncated_normal_initializer()
          z = tf.get_variable("z1", shape=[], dtype=tf.float32, initializer=i)

        sess.run(tf.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 1.0, 2.0, 2.0)

  def testUniformRandomInitalizer(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:
        with tf.variable_scope("vs", use_resource=True):
          i = tf.random_uniform_initializer(minval=-2.0, maxval=2.0)
          z = tf.get_variable("z1", shape=[], dtype=tf.float32, initializer=i)

        sess.run(tf.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 0.0, 2.0, 2.0)

  def testDefaultUniformRandomInitalizer(self):
    with tf.device("/device:XLA_IPU:0"):
      with tf.Session() as sess:
        with tf.variable_scope("vs", use_resource=True):
          i = tf.random_uniform_initializer()
          z = tf.get_variable("z1", shape=[], dtype=tf.float32, initializer=i)

        sess.run(tf.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 0.5, 0.5, 0.5)


if __name__ == "__main__":
    googletest.main()
