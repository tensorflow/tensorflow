# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

class IpuFuseOpsTest(test_util.TensorFlowTestCase):

  def testSigmoid(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [3], name="a")
      c = tf.sigmoid(pa)

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [-6.0, 0.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.002473, 0.5, 0.997527])

      result = sess.run(report)
      self.assertTrue(len(result) == 1)
      cs_list = tu.get_compute_sets_from_report(result[0])

      ok = ['Copy_arg0_to_call',
            'call/Nonlinearity']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testSigmoidGrad(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [3], name="grad")
      pb = tf.placeholder(tf.float32, [3], name="in")
      c = gen_math_ops.sigmoid_grad(pa, pb)

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [2.0, 0.5, 1.0],
        pb: [-1.0, 1.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [2.0, 0.25, 0.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 1)
      cs_list = tu.get_compute_sets_from_report(result[0])

      ok = ['call/NonLinearityGrad']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testRelu(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [3], name="a")
      c = nn_ops.relu(pa)

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [-6.0, 0.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.0, 6.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 1)
      cs_list = tu.get_compute_sets_from_report(result[0])

      ok = ['Copy_arg0_to_call',
            'call/Nonlinearity']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testReluGrad(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [3], name="grad")
      pb = tf.placeholder(tf.float32, [3], name="in")
      c = gen_nn_ops.relu_grad(pa, pb)

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [2.0, 0.5, 1.0],
        pb: [-1.0, 1.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.5, 1.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 1)
      cs_list = tu.get_compute_sets_from_report(result[0])

      ok = ['call/NonLinearityGrad']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

if __name__ == "__main__":
    googletest.main()
