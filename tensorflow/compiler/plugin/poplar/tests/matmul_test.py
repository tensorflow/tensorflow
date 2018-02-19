# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops


class IpuXlaMatMulTest(test_util.TensorFlowTestCase):
  def testMatMul2x2(self):
    with tf.device("/device:IPU:0"):
      with tf.Session() as sess:
        pa = tf.placeholder(tf.float32, [2, 2], name="a")
        pb = tf.placeholder(tf.float32, [2, 2], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {pa: [[1., 0.], [0., 2.]], pb: [[0., 1.], [4., 3.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[0., 1.], [8., 6.]])

  def testMatMul1x1(self):
    with tf.device("/device:IPU:0"):
      with tf.Session() as sess:
        pa = tf.placeholder(tf.float32, [1, 1], name="a")
        pb = tf.placeholder(tf.float32, [1, 1], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {pa: [[1.]], pb: [[4.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[4.]])

  def testMatMulVec1(self):
    with tf.device("/device:IPU:0"):
      with tf.Session() as sess:
        pa = tf.placeholder(tf.float32, [1, 3], name="a")
        pb = tf.placeholder(tf.float32, [3, 1], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {pa: [[1., 2., 3.]], pb: [[4.], [5.], [6.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[32.]])

  def testMatMulVec2(self):
    with tf.device("/device:IPU:0"):
      with tf.Session() as sess:
        pa = tf.placeholder(tf.float32, [3, 1], name="a")
        pb = tf.placeholder(tf.float32, [1, 3], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {pa: [[1.], [2.], [3.]], pb: [[4., 5., 6.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            [[4., 5., 6.], [8., 10., 12.], [12., 15., 18.]])

  def testMatMulEinsumDot(self):
    with tf.device("/device:IPU:0"):
      with tf.Session() as sess:
        pa = tf.placeholder(tf.float32, [3], name="a")
        pb = tf.placeholder(tf.float32, [3], name="b")
        output = special_math_ops.einsum('i,i->', pa, pb)

        fd = {pa: [1., 2., 3.], pb: [4., 5., 6.]}
        result = sess.run(output, fd)
        self.assertAllClose(result, 32.)

  def testMatMul3x1x2(self):
    with tf.device("/device:IPU:0"):
      with tf.Session() as sess:
        pa = tf.placeholder(tf.float32, [1, 3], name="a")
        pb = tf.placeholder(tf.float32, [3, 2], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {pa: [[100, 10, 0.5]], pb: [[1, 3], [2, 5], [6, 8]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[123, 354]])

  def testMatMulBatch(self):
    with tf.device("/device:IPU:0"):
      with tf.Session() as sess:
        pa = tf.placeholder(tf.float32, [2, 2, 2, 2], name="a")
        pb = tf.placeholder(tf.float32, [2, 2, 2, 2], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {
          pa: [[[[1000, 100], [10, 1]],
                [[2000, 200], [20, 2]]],
               [[[3000, 300], [30, 3]],
                [[4000, 400], [40, 4]]]],
          pb: [[[[1, 2], [3, 4]],
                [[5, 6], [7, 8]]],
               [[[11, 22], [33, 44]],
                [[55, 66], [77, 88]]]]
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            [[[[1300, 2400], [13, 24]],
                              [[11400, 13600], [114, 136]]],
                             [[[42900, 79200], [429, 792]],
                              [[250800, 299200], [2508, 2992]]]])

  def testMatMulBatch2(self):
    with tf.device("/device:IPU:0"):
      with tf.Session() as sess:
        pa = tf.placeholder(tf.float32, [6, 2, 2], name="a")
        pb = tf.placeholder(tf.float32, [6, 2, 2], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {
          pa: [[[1, 0], [0, 1]],
               [[2, 0], [0, 2]],
               [[3, 0], [0, 3]],
               [[4, 0], [0, 4]],
               [[5, 0], [0, 5]],
               [[6, 0], [0, 6]]],
          pb: [[[1, 1], [1, 1]],
               [[2, 2], [2, 2]],
               [[3, 3], [3, 3]],
               [[0, 2], [0, 2]],
               [[0, 1], [0, 1]],
               [[1, 0], [1, 0]]],
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            [[[1, 1], [1, 1]],
                             [[4, 4], [4, 4]],
                             [[9, 9], [9, 9]],
                             [[0, 8], [0, 8]],
                             [[0, 5], [0, 5]],
                             [[6, 0], [6, 0]]])

  def testMatMulFwdBackwd(self):
    with tf.device("/device:IPU:0"):
      with tf.variable_scope("vs", use_resource=True):
        w1 = tf.get_variable("w1", shape=[4, 3], dtype=tf.float32,
                            initializer=tf.constant_initializer(
                              np.array([[1,2,1], [1,3,4], [1,5,6], [1,7,8]],
                                       dtype=np.float32)))
        b1 = tf.get_variable("b1", shape=[3], dtype=tf.float32,
                            initializer=tf.constant_initializer(
                              np.array([2, 1, 1], dtype=np.float32)))
        w2 = tf.get_variable("w2", shape=[3, 2], dtype=tf.float32,
                             initializer=tf.constant_initializer(
                               np.array([[3, 4], [5, 6], [7, 8]],
                                        dtype=np.float32)))
        b2 = tf.get_variable("b2", shape=[2], dtype=tf.float32,
                             initializer=tf.constant_initializer(
                               np.array([2, 1], dtype=np.float32)))

      x = tf.placeholder(tf.float32, shape=[3, 4])
      y = tf.matmul(x, w1) + b1
      y = tf.matmul(y, w2) + b2

      expected = tf.placeholder(tf.float32, shape=[3, 2])
      xent = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=expected)

      optimizer = tf.train.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(xent)

    with tf.Session() as sess:
      fd = {x: np.array([[7, 3, 5, 9],
                         [1,2,3,4],
                         [5,6,7,8]], dtype=np.float32),
            expected: [[1, 2],[3,4],[5,6]]}

      sess.run(tf.global_variables_initializer())
      sess.run(train, feed_dict=fd)

  def testMatMulFwdBackwdLeftHandWeights(self):
    with tf.device("/device:IPU:0"):
      with tf.variable_scope("vs", use_resource=True):
        w1 = tf.get_variable("w1", shape=[3, 4], dtype=tf.float32,
                             initializer=tf.constant_initializer(
                               np.array([[1,2,1,1],[3,4,1,5],[6,1,7,8]],
                                        dtype=np.float32)))
        b1 = tf.get_variable("b1", shape=[3], dtype=tf.float32,
                             initializer=tf.constant_initializer(
                               np.array([2, 1, 1], dtype=np.float32)))
        w2 = tf.get_variable("w2", shape=[2, 3], dtype=tf.float32,
                             initializer=tf.constant_initializer(
                               np.array([[3,4,5],[6,7,8]],
                                        dtype=np.float32)))
        b2 = tf.get_variable("b2", shape=[3], dtype=tf.float32,
                             initializer=tf.constant_initializer(
                               np.array([2, 1, 1], dtype=np.float32)))

      x = tf.placeholder(tf.float32, shape=[4, 3])
      y = tf.matmul(w1, x) + b1
      y = tf.matmul(w2, y) + b2

      expected = tf.placeholder(tf.float32, shape=[2, 3])
      xent = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=expected)

      optimizer = tf.train.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(xent)

    with tf.Session() as sess:
      fd = {x: np.array([[7,3,5],
                         [1,2,3],
                         [5,6,7],
                         [3,5,2]], dtype=np.float32),
            expected: [[1,2,1],[3,4,3]]}

      sess.run(tf.global_variables_initializer())
      sess.run(train, feed_dict=fd)

if __name__ == "__main__":
  googletest.main()
