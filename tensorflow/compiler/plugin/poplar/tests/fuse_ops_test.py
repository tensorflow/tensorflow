# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

import numpy as np
import contextlib
import re

@contextlib.contextmanager
def ipu_session():
  opts = config_pb2.IPUOptions()
  dev = opts.device_config.add()
  dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
  dev.enable_profile = True
  with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:
    yield sess

def get_compute_sets_from_report(report):
  lines = report.split('\n')
  return [x for x in lines if re.search('(\d+ execution.?)', x)]

def check_all_compute_sets_in_list(cs_list, whitelist):
  for cs in cs_list:
    if len([x for x in whitelist if cs.startswith(x)]) == 0:
      return False
  return True

class IpuFuseOpsTest(test_util.TensorFlowTestCase):

  def testSigmoid(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [3], name="a")
      c = tf.sigmoid(pa)

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_summary()

    with ipu_session() as sess:
      fd = {
        pa: [-6.0, 0.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.002473, 0.5, 0.997527])

      result = sess.run(report)
      self.assertTrue(len(result) == 1)
      cs_list = get_compute_sets_from_report(result[0])

      ok = ['Copy_arg0_to_call',
            'call/Nonlinearity']
      self.assertTrue(check_all_compute_sets_in_list(cs_list, ok))

  def testRelu(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [3], name="a")
      c = nn_ops.relu(pa)

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_summary()

    with ipu_session() as sess:
      fd = {
        pa: [-6.0, 0.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.0, 6.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 1)
      cs_list = get_compute_sets_from_report(result[0])

      ok = ['Copy_arg0_to_call',
            'call/Nonlinearity']
      self.assertTrue(check_all_compute_sets_in_list(cs_list, ok))

  def testReluGrad(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [3], name="grad")
      pb = tf.placeholder(tf.float32, [3], name="in")
      c = gen_nn_ops._relu_grad(pa, pb)

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_summary()

    with ipu_session() as sess:
      fd = {
        pa: [2.0, 0.5, 1.0],
        pb: [-1.0, 1.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.5, 1.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 1)
      cs_list = get_compute_sets_from_report(result[0])

      ok = ['call/NonLinearityGrad']
      self.assertTrue(check_all_compute_sets_in_list(cs_list, ok))

gen_nn_ops.bias_add_grad
if __name__ == "__main__":
    googletest.main()
