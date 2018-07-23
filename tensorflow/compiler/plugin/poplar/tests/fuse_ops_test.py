# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

class IpuFuseOpsTest(test_util.TensorFlowTestCase):

  def testSigmoid(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3], name="a")
      c = math_ops.sigmoid(pa)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [-6.0, 0.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.002473, 0.5, 0.997527])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Sigmoid/call/Nonlinearity']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testSigmoidGrad(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3], name="grad")
      pb = array_ops.placeholder(np.float32, [3], name="in")
      c = gen_math_ops.sigmoid_grad(pa, pb)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [2.0, 0.5, 1.0],
        pb: [-1.0, 1.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [2.0, 0.25, 0.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['SigmoidGrad/call/NonLinearityGrad']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testRelu(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3], name="a")
      c = nn_ops.relu(pa)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [-6.0, 0.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.0, 6.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Relu/call/Nonlinearity']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testReluGrad(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3], name="grad")
      pb = array_ops.placeholder(np.float32, [3], name="in")
      c = gen_nn_ops.relu_grad(pa, pb)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [2.0, 0.5, 1.0],
        pb: [-1.0, 1.0, 6.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.5, 1.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['ReluGrad/call/NonLinearityGrad']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

if __name__ == "__main__":
    googletest.main()
