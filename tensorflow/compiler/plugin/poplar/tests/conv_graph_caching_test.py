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
  def testConvolutionsMatch(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer())
        y = convolutional.conv2d(
            y,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer())

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

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
          '__seed*', 'host-exchange-local-copy-',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1', 'Copy_'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConvolutionsDontMatchDifferentTypes(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer())
        y = math_ops.cast(y, np.float16)
        y = convolutional.conv2d(
            y,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer())

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_*weightsRearranged', 'host-exchange-local-copy-',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1', 'vs/Cast/convert.*/Cast',
          'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConvolutionsDontMatchDifferentShapes(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer())
        y = array_ops.reshape(y, [1, 2, 8, 2])
        y = convolutional.conv2d(
            y,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer())

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_*weightsRearranged', 'host-exchange-local-copy-',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConvolutionsDontMatchDifferentConvParams(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer())
        y = convolutional.conv2d(
            y,
            2,
            1,
            use_bias=False,
            strides=(2, 1),
            kernel_initializer=init_ops.ones_initializer())

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_*weightsRearranged', 'host-exchange-local-copy-',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConvolutionsMatchFwdBwdWu(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer(),
            name='conv1')
        y = convolutional.conv2d(
            y,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer(),
            name='conv2')
        y = convolutional.conv2d(
            y,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer(),
            name='conv3')

      loss = math_ops.reduce_sum(y)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(loss)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      # Fwd and BackpropInput should be shared
      # Weight transpose for BackpropInput should be present
      # Both BackpropFilter should be shared
      ok = [
          '__seed*', 'host-exchange-local-copy-', 'Copy_',
          'vs/conv1/Conv2D/convolution.*/Conv_1x1',
          'Sum/reduce.*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
          'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
          'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropInput/fusion.*/WeightTranspose',
          'gradients/vs/conv2/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Conv_4x4',
          'gradients/vs/conv2/Conv2D_grad/Conv2DBackpropFilter/fusion.*/DeltasPartialTranspose',
          'gradients/vs/conv2/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo'
      ]

  def testConvolutionsMatchFwdBwdWuVariableLR(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      lr = array_ops.placeholder(np.float32, shape=[])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer(),
            name='conv1')
        y = convolutional.conv2d(
            y,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer(),
            name='conv2')
        y = convolutional.conv2d(
            y,
            2,
            1,
            use_bias=False,
            kernel_initializer=init_ops.ones_initializer(),
            name='conv3')

      loss = math_ops.reduce_sum(y)
      optimizer = gradient_descent.GradientDescentOptimizer(lr)
      train = optimizer.minimize(loss)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2]), lr: 0.1})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      # Fwd and BackpropInput should be shared
      # Weight transpose for BackpropInput should be present
      # Both BackpropFilter should be shared
      ok = [
          '__seed*', 'host-exchange-local-copy-', 'Copy_',
          'vs/conv1/Conv2D/convolution.*/Conv_1x1',
          'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
          'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Conv_4x4',
          'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo'
      ]

      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


if __name__ == "__main__":
  googletest.main()
