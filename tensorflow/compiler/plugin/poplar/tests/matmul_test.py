# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import googletest
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent


class IpuXlaMatMulTest(test_util.TensorFlowTestCase):
  def testMatMul2x2(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {pa: [[1., 0.], [0., 2.]], pb: [[0., 1.], [4., 3.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[0., 1.], [8., 6.]])

  def testMatMul1x1(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [1, 1], name="a")
        pb = array_ops.placeholder(np.float32, [1, 1], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {pa: [[1.]], pb: [[4.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[4.]])

  def testMatMulVec1(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [1, 3], name="a")
        pb = array_ops.placeholder(np.float32, [3, 1], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {pa: [[1., 2., 3.]], pb: [[4.], [5.], [6.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[32.]])

  def testMatMulVec2(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [3, 1], name="a")
        pb = array_ops.placeholder(np.float32, [1, 3], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {pa: [[1.], [2.], [3.]], pb: [[4., 5., 6.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            [[4., 5., 6.], [8., 10., 12.], [12., 15., 18.]])

  def testMatMulEinsumDot(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [3], name="a")
        pb = array_ops.placeholder(np.float32, [3], name="b")
        output = special_math_ops.einsum('i,i->', pa, pb)

        fd = {pa: [1., 2., 3.], pb: [4., 5., 6.]}
        result = sess.run(output, fd)
        self.assertAllClose(result, 32.)

  def testMatMul3x1x2(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [1, 3], name="a")
        pb = array_ops.placeholder(np.float32, [3, 2], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {pa: [[100, 10, 0.5]], pb: [[1, 3], [2, 5], [6, 8]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[123, 354]])

  def testMatMulBatch(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 2, 2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2, 2, 2], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {
            pa: [[[[1000, 100], [10, 1]], [[2000, 200], [20, 2]]],
                 [[[3000, 300], [30, 3]], [[4000, 400], [40, 4]]]],
            pb: [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                 [[[11, 22], [33, 44]], [[55, 66], [77, 88]]]]
        }
        result = sess.run(output, fd)
        self.assertAllClose(
            result,
            [[[[1300, 2400], [13, 24]], [[11400, 13600], [114, 136]]],
             [[[42900, 79200], [429, 792]], [[250800, 299200], [2508, 2992]]]])

  def testMatMulBatch2(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [6, 2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [6, 2, 2], name="b")
        output = math_ops.matmul(pa, pb)

        fd = {
            pa: [[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]],
                 [[4, 0], [0, 4]], [[5, 0], [0, 5]], [[6, 0], [0, 6]]],
            pb: [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]],
                 [[0, 2], [0, 2]], [[0, 1], [0, 1]], [[1, 0], [1, 0]]],
        }
        result = sess.run(output, fd)
        self.assertAllClose(
            result, [[[1, 1], [1, 1]], [[4, 4], [4, 4]], [[9, 9], [9, 9]],
                     [[0, 8], [0, 8]], [[0, 5], [0, 5]], [[6, 0], [6, 0]]])

  def testMatMulFwdBackwd(self):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        w1 = variable_scope.get_variable(
            "w1",
            shape=[4, 3],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(
                np.array([[1, 2, 1], [1, 3, 4], [1, 5, 6], [1, 7, 8]],
                         dtype=np.float32)))
        b1 = variable_scope.get_variable(
            "b1",
            shape=[3],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(
                np.array([2, 1, 1], dtype=np.float32)))
        w2 = variable_scope.get_variable(
            "w2",
            shape=[3, 2],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(
                np.array([[3, 4], [5, 6], [7, 8]], dtype=np.float32)))
        b2 = variable_scope.get_variable(
            "b2",
            shape=[2],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(
                np.array([2, 1], dtype=np.float32)))

      x = array_ops.placeholder(np.float32, shape=[3, 4])
      y = math_ops.matmul(x, w1) + b1
      y = math_ops.matmul(y, w2) + b2

      expected = array_ops.placeholder(np.float32, shape=[3, 2])
      xent = nn.softmax_cross_entropy_with_logits(logits=y, labels=expected)

      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(xent)

    with session_lib.Session() as sess:
      fd = {
          x:
          np.array([[7, 3, 5, 9], [1, 2, 3, 4], [5, 6, 7, 8]],
                   dtype=np.float32),
          expected: [[1, 2], [3, 4], [5, 6]]
      }

      sess.run(variables.global_variables_initializer())
      sess.run(train, feed_dict=fd)

  def testMatMulFwdBackwdLeftHandWeights(self):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        w1 = variable_scope.get_variable(
            "w1",
            shape=[3, 4],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(
                np.array([[1, 2, 1, 1], [3, 4, 1, 5], [6, 1, 7, 8]],
                         dtype=np.float32)))
        b1 = variable_scope.get_variable(
            "b1",
            shape=[3],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(
                np.array([2, 1, 1], dtype=np.float32)))
        w2 = variable_scope.get_variable(
            "w2",
            shape=[2, 3],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(
                np.array([[3, 4, 5], [6, 7, 8]], dtype=np.float32)))
        b2 = variable_scope.get_variable(
            "b2",
            shape=[3],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(
                np.array([2, 1, 1], dtype=np.float32)))

      x = array_ops.placeholder(np.float32, shape=[4, 3])
      y = math_ops.matmul(w1, x) + b1
      y = math_ops.matmul(w2, y) + b2

      expected = array_ops.placeholder(np.float32, shape=[2, 3])
      xent = nn.softmax_cross_entropy_with_logits(logits=y, labels=expected)

      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(xent)

    with session_lib.Session() as sess:
      fd = {
          x:
          np.array([[7, 3, 5], [1, 2, 3], [5, 6, 7], [3, 5, 2]],
                   dtype=np.float32),
          expected: [[1, 2, 1], [3, 4, 3]]
      }

      sess.run(variables.global_variables_initializer())
      sess.run(train, feed_dict=fd)

  def testMatMul1x2_2x3(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 1], name="a")
        pb = array_ops.placeholder(np.float32, [2, 3], name="b")
        output = math_ops.matmul(array_ops.transpose(pa), pb)

        fd = {pa: [[1.], [2.]], pb: [[1., 2., 3.], [4., 5., 6.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[9, 12., 15.]])


if __name__ == "__main__":
  googletest.main()
