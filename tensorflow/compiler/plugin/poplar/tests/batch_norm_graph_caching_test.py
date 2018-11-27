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
from tensorflow.python.layers import normalization as layers_norm


class BatchNormGraphCachingTest(test_util.TensorFlowTestCase):

  def testBatchNormalizeInference(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = layers_norm.batch_normalization(y, fused=True)
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = layers_norm.batch_normalization(y, fused=True)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1,4,4,2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      # Would fail if there were two batch norms in the graph
      ok = ['progIdCopy',
            'host-exchange-local-copy',
            'Copy_',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-inference.*/']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testBatchNormalizeInferenceDontMatchDifferentTypes(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = layers_norm.batch_normalization(y, fused=True)
        y = math_ops.cast(y, np.float16)
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = layers_norm.batch_normalization(y, fused=True)

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
            'Copy_',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-inference.*/',
            'vs/Cast/convert.*/Cast',
            'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1',
            'vs/batch_normalization_1/FusedBatchNormV2/batch-norm-inference.*/']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testBatchNormsDontMatchDifferentShapes(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = layers_norm.batch_normalization(y, fused=True)
        y = array_ops.reshape(y, [1, 2, 8, 2])
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = layers_norm.batch_normalization(y, fused=True)

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
            'Copy_',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-inference.*/',
            'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1',
            'vs/batch_normalization_1/FusedBatchNorm/batch-norm-inference.*/']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testBatchNormsMatchFwdBwd(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer(),
                                 name='conv1')
        y = layers_norm.batch_normalization(y, fused=True, training=True)
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer(),
                                 name='conv2')
        y = layers_norm.batch_normalization(y, fused=True, training=True)
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer(),
                                 name='conv3')
        y = layers_norm.batch_normalization(y, fused=True, training=True)

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

      # One BN for forwards and one BN for grad
      # (note that we don't cache gradient application)
      ok = ['progIdCopy',
            'host-exchange-local-copy-',
            'Copy_',
            'vs/conv1/Conv2D/convolution.*/Conv_1x1',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-training.*/',
            'Sum/reduce.*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
            'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
            'gradients/vs/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad/batch-norm-grad.*/',
            'GradientDescent/update_vs/batch_normalization/',
            'GradientDescent/update_vs/batch_normalization_1/',
            'GradientDescent/update_vs/batch_normalization_2/',
            'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropInput/convolution.*.clone/WeightTranspose',
            'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropFilter/convolution.*.clone/Conv_4x4',
            'GradientDescent/update_vs/conv3/kernel/ResourceApplyGradientDescent/subtract.*.clone/AddTo']

      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

if __name__ == "__main__":
    googletest.main()
