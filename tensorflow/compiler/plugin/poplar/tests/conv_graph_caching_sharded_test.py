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
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class ConvGraphCachingTest(test_util.TensorFlowTestCase):
  def testConvolutionsDontMatchDifferentDevices(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        with tu.ipu_shard(0):
          y = convolutional.conv2d(
              x,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
        with tu.ipu_shard(1):
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True, sharded=True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      # Note how there are two convolutions
      ok = [
          '__seed*', 'progIdCopy/GlobalPre', '/OnTileCopy',
          'vs/conv2d/Conv2D/convolution.*',
          'Copy_vs/conv2d/Conv2D/convolution.*',
          'vs/conv2d_1/Conv2D/convolution.*'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConvolutionsMatchShardingSameDevice(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        with tu.ipu_shard(0):
          y = convolutional.conv2d(
              x,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
        with tu.ipu_shard(0):
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True, sharded=True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Would fail if there were two convolutions in the graph as they would be
      # called conv2d and conv2d_1
      ok = [
          '__seed*', 'progIdCopy/GlobalPre', '/OnTileCopy',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


if __name__ == "__main__":
  googletest.main()
