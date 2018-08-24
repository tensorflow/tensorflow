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
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class IpuXlaCacheConvTest(test_util.TensorFlowTestCase):

  def testConvolutionsMatch(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1,4,4,2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Would fail if there were two convolutions in the graph as they would be
      # called conv2d and conv2d_1
      ok = ['progIdCopy',
            'host-exchange-local-copy-',
            'Copy_XLA_Args/arg1.*_weights_to_weights/OnTileCopy',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
            'Copy_{XLA_Args/arg2.*_weights,partialReduceOut}_to_{XLA_Args/arg0.*_input,weights}']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testConvolutionsDontMatchDifferentTypes(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = math_ops.cast(y, np.float16)
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1,4,4,2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Matches two convolutions
      ok = ['progIdCopy',
            'host-exchange-local-copy-',
            'Copy_XLA_Args/arg1.*_weights_to_weights/OnTileCopy',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
            'vs/Cast/convert.*/Cast',
            'Copy_cast_to_cast[[]cloned[]]/OnTileCopy',
            'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testConvolutionsDontMatchDifferentShapes(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = array_ops.reshape(y, [1, 2, 8, 2])
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1,4,4,2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Matches two convolutions
      ok = ['progIdCopy',
            'host-exchange-local-copy-',
            'Copy_XLA_Args/arg1.*_weights_to_weights/OnTileCopy',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
            'Copy_{<const>,XLA_Args/arg2.*_weights}_to_weightsRearranged/OnTileCopy',
            'Copy_partials_to_partials[[]cloned[]]/OnTileCopy',
            'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testConvolutionsDontMatchDifferentConvParams(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = convolutional.conv2d(y, 2, 1, use_bias=False, strides=(2, 1),
                                 kernel_initializer=init_ops.ones_initializer())

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1,4,4,2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Matches two convolutions
      ok = ['progIdCopy',
            'host-exchange-local-copy-',
            'Copy_XLA_Args/arg1.*_weights_to_weights/OnTileCopy',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
            'Copy_{<const>,XLA_Args/arg2.*_weights}_to_weightsRearranged/OnTileCopy',
            'Copy_partials_to_partials[[]cloned[]]/OnTileCopy',
            'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testConvolutionsMatchFwdBwd(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())

      loss = math_ops.reduce_sum(y)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(loss)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run([train,loss], {x: np.zeros([1,4,4,2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Matches two convolutions
      ok = ['progIdCopy',
            'host-exchange-local-copy-',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
            'Copy_{XLA_Args/arg0.*_input,XLA_Args/arg1.*_weights}_to_{in,partialTranspose}/OnTileCopy',
            'Copy_partialReduceOut_to_partialReduceOut[[]cloned[]]/OnTileCopy',
            'Copy_{XLA_Args/arg2.*_weights,partialReduceOut[[]cloned[]]}_to_{in,partialTranspose}',
            'Sum/reduce.*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
            'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
            'gradients/vs/conv2d_1/Conv2D_grad/Conv2DBackpropInput/convolution.*.clone/WeightTranspose',
            'Copy_<const>_to_in',
            'Copy_{<const>,XLA_Args/arg0.*_input,partialReduceOut}_to_{inRearranged,weightsRearranged}',
            'gradients/vs/conv2d/Conv2D_grad/Conv2DBackpropFilter/convolution.*/Conv_4x4',
            'Copy_partialReduceOut_to_partialReduceOut[[]cloned[]]/OnTileCopy',
            'GradientDescent/update_vs/conv2d/kernel/ResourceApplyGradientDescent/call*/AddTo',
            'Copy_{<const>,partialReduceOut[[]cloned[]]}_to_{XLA_Args/arg0.*_input,partialReduceOut}',
            'Copy_partialReduceOut_to_partialReduceOut[[]cloned[]]/OnTileCopy',
            'GradientDescent/update_vs/conv2d_1/kernel/ResourceApplyGradientDescent/call*/AddTo',]
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

if __name__ == "__main__":
    googletest.main()
