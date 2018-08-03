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
from tensorflow.python.ops import nn
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

      ok = ['host-exchange-local-copy-',
            'Sigmoid/call/Nonlinearity']
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

      ok = ['host-exchange-local-copy-',
            'SigmoidGrad/call/NonLinearityGrad']
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

      ok = ['host-exchange-local-copy-',
            'Relu/call/Nonlinearity']
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

      ok = ['host-exchange-local-copy-',
            'ReluGrad/call/NonLinearityGrad']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testMaxPool(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,1,10,10], name="a")
      c = nn.max_pool(pa, ksize=[1,1,5,5], strides=[1,1,2,2],
                      data_format='NCHW', padding='SAME', name="max")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: np.ones([1,1,10,10]),
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, np.ones([1,1,5,5]))

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['host-exchange-local-copy-',
            'max/call/maxPool']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testFwdAndBwdMaxPool(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,1,2,2], name="a")
      pb = array_ops.placeholder(np.float32, [1,1,2,2], name="a")
      c = nn.max_pool(pa, ksize=[1,1,2,2], strides=[1,1,1,1],
                      data_format='NCHW', padding='SAME')
      d = gen_nn_ops.max_pool_grad(pa, pb, c, ksize=[1,1,2,2],
              strides=[1,1,1,1], data_format='NCHW', padding='SAME')

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fe = {
        pa: np.ones([1,1,2,2]),
        pb: np.zeros([1,1,2,2]),
      }
      result = sess.run(d, fe)
      self.assertAllClose(result, [[[[1, 2], [ 2, 4]]]])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['host-exchange-local-copy-',
            'MaxPool/call/maxPool',
            '/maxPoolBwd']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testNoCastsF32ToF16ToF32(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3])
      b = math_ops.cast(pa, np.float16)
      c = math_ops.cast(b, np.float32)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [2.0, 0.5, 1.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [2.0, 0.5, 1.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['host-exchange-local-copy-']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testNoCastsF16ToF32ToF16(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float16, [3])
      b = math_ops.cast(pa, np.float32)
      c = math_ops.cast(b, np.float16)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [2.0, 0.5, 1.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [2.0, 0.5, 1.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['host-exchange-local-copy-']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testDontRemoveCastsIfUsed(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float16, [3])
      b = math_ops.cast(pa, np.float32)
      const = array_ops.constant(1.0, np.float32)
      b = b + const
      c = math_ops.cast(b, np.float16)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [2.0, 0.5, 1.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [3.0, 1.5, 2.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['host-exchange-local-copy-',
            'Cast/convert.*/Cast',
            'add/add.*/Op/Add',
            'Cast_1/convert.*/Cast']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

if __name__ == "__main__":
    googletest.main()
