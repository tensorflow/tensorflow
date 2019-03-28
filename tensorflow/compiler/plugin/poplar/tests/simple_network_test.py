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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables


class IpuXlaSimpleNetworkTest(test_util.TensorFlowTestCase):
  def testAdd(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        output = pa + pb

        fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[1., 2.], [6., 8.]])

        fd = {pa: [[0., 0.], [1., 1.]], pb: [[2., 1.], [4., 5.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[2., 1.], [5., 6.]])

  def testTransposeNegate(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 2, 3], name="a")
        a = array_ops.transpose(pa, [2, 1, 0])
        b = math_ops.negative(a)

        sess.run(variables.global_variables_initializer())

        fd = {pa: [[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]]}
        result = sess.run(b, fd)
        self.assertAllClose(
            result,
            [[[-1, -5], [-3, -7]], [[-2, -6], [-4, -8]], [[-3, -7], [-5, -9]]])

  def testTransposeNegate2(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 2, 3], name="a")
        a = array_ops.transpose(pa, [1, 2, 0])
        b = math_ops.negative(a)

        sess.run(variables.global_variables_initializer())

        fd = {pa: [[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]]}
        result = sess.run(b, fd)
        self.assertAllClose(
            result,
            [[[-1, -5], [-2, -6], [-3, -7]], [[-3, -7], [-4, -8], [-5, -9]]])

  def testReshape(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 1, 3], name="a")
        a = array_ops.reshape(pa, [1, 3, 2])

        sess.run(variables.global_variables_initializer())

        fd = {pa: [[[1, 2, 3]], [[5, 6, 7]]]}
        result = sess.run(a, fd)
        self.assertAllClose(result, [[[1, 2], [3, 5], [6, 7]]])

  def testAlgebraicSimplificationWithBroadcastIssue(self):
    # XLA re-arranges (a/b) / (c/d) -> (a*d) / (b*c)
    # however the re-arranged graph doesn't take into account the
    # broadcast:
    #
    # f32[32]{0} multiply(f32[] %convert.1, f32[] %reduce.1)
    #
    # It has an output which doesn't match it's inputs.  (and it doesn't
    # have any meta-information either)
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        a = array_ops.placeholder(np.float32, [])
        b = array_ops.placeholder(np.float32, [2])
        c = array_ops.placeholder(np.float32, [2])
        d = array_ops.placeholder(np.float32, [])
        e = (a / b) / (c / d)

        fd = {
            a: 4.0,
            b: [1.0, 1.0],
            c: [4.0, 4.0],
            d: 1.0,
        }
        result = sess.run(e, fd)
        self.assertAllClose(result, [1.0, 1.0])

  def testDropout(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        output = nn.dropout(pa, 0.5)

        result = sess.run(output, {pa: [[1., 1.], [2., 3.]]})

  def testControlDependencies(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        a = array_ops.placeholder(np.float32, [1])
        b = array_ops.placeholder(np.float32, [1])
        c = array_ops.placeholder(np.float32, [1])
        d = array_ops.placeholder(np.float32, [1])

        e = a + b
        f = c * d
        g = a - c

        with ops.control_dependencies([e, f, g]):
          h = e + f
          i = h + g

        result = sess.run(i, {a: [1], b: [2], c: [3], d: [4]})

  def testSigmoid(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        a = array_ops.placeholder(np.float32, [2, 2])
        b = math_ops.sigmoid(a)

        result = sess.run(b, {a: [[0, 0], [0, 0]]})
        self.assertAllClose(result, [[0.5, 0.5], [0.5, 0.5]])


if __name__ == "__main__":
  googletest.main()
