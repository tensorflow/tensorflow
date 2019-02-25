# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
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
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.layers import normalization as layers_norm


class NormGraphCachingTest(test_util.TensorFlowTestCase):

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
      ok = ['host-exchange-local-copy',
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
      ok = ['host-exchange-local-copy-',
            'Copy_',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-inference.*/',
            'vs/Cast/convert.*/Cast',
            'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1',
            'vs/batch_normalization_1/FusedBatchNormV2/batch-norm-inference.*/']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testBatchNormalizeInferenceDontMatchDifferentDevices(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        with tu.ipu_shard(0):
          a = convolutional.conv2d(x, 2, 1, use_bias=False,
                                   kernel_initializer=init_ops.ones_initializer())
          b = layers_norm.batch_normalization(a, fused=True)

        with tu.ipu_shard(1):
          c = convolutional.conv2d(b, 2, 1, use_bias=False,
                                   kernel_initializer=init_ops.ones_initializer())
          d = layers_norm.batch_normalization(c, fused=True)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True, sharded=True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(d, {x: np.zeros([1,4,4,2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      # Would fail if there weren't two batch norms in the graph
      ok = ['progIdCopy/GlobalPre',
            '/OnTileCopy',
            'Copy_',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
            'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1/Convolve',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-inference.*/',
            'vs/batch_normalization_1/FusedBatchNorm/batch-norm-inference.*/']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testBatchNormalizeInferenceMatchWithSharding(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        with tu.ipu_shard(0):
          a = convolutional.conv2d(x, 2, 1, use_bias=False,
                                   kernel_initializer=init_ops.ones_initializer())
          b = layers_norm.batch_normalization(a, fused=True)

        with tu.ipu_shard(0):
          c = convolutional.conv2d(b, 2, 1, use_bias=False,
                                   kernel_initializer=init_ops.ones_initializer())
          d = layers_norm.batch_normalization(c, fused=True)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True, sharded=True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(d, {x: np.zeros([1,4,4,2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      # Would fail if there were two batch norms in the graph
      ok = ['progIdCopy/GlobalPre',
            '/OnTileCopy',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-inference.*/']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

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
      ok = ['host-exchange-local-copy',
            'Copy_',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-inference.*/']
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
      ok = ['host-exchange-local-copy-',
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
      ok = ['host-exchange-local-copy-',
            'Copy_',
            'Copy.',
            'vs/conv1/Conv2D/convolution.*/Conv_1x1',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-training.*/',
            'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
            'gradients/vs/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad/batch-norm-grad.*/',
            'GradientDescent/update_vs/batch_normalization/',
            'GradientDescent/update_vs/batch_normalization_1/',
            'GradientDescent/update_vs/batch_normalization_2/',
            'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Conv_4x4/Convolve',
            'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo',
            ]

      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testBatchNormsMatchFwdBwdSomeOnShard0SomeOnShard1(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        with tu.ipu_shard(0):
          y = convolutional.conv2d(x, 2, 1, use_bias=False,
                  kernel_initializer=init_ops.ones_initializer(),
                  name='conv1')
          y = layers_norm.batch_normalization(y, fused=True, training=True)
          y = convolutional.conv2d(y, 2, 1, use_bias=False,
                  kernel_initializer=init_ops.ones_initializer(),
                  name='conv2')
          y = layers_norm.batch_normalization(y, fused=True, training=True)

        with tu.ipu_shard(1):
          y = convolutional.conv2d(y, 2, 1, use_bias=False,
                  kernel_initializer=init_ops.ones_initializer(),
                  name='conv3')
          y = layers_norm.batch_normalization(y, fused=True, training=True)

      loss = math_ops.reduce_sum(y)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(loss)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True, sharded=True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run([train,loss], {x: np.zeros([1,4,4,2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Two BN for forwards (on shards 0 and 1) and two BN for grad
      # (note that we don't cache gradient application)
      ok = ['progIdCopy/GlobalPre',
            '/OnTileCopy',
            'Copy_',
            'vs/conv1/Conv2D/convolution.*/Conv_1x1',
            'vs/conv3/Conv2D/convolution.*/Conv_1x1',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-training.*/',
            'vs/batch_normalization_2/FusedBatchNorm/batch-norm-training.*/',
            'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
            'gradients/vs/batch_normalization_2/FusedBatchNorm_grad/FusedBatchNormGrad/batch-norm-grad.*/',
            'gradients/vs/batch_normalization_1/FusedBatchNorm_grad/FusedBatchNormGrad/batch-norm-grad.*/',
            'GradientDescent/update_vs/batch_normalization/',
            'GradientDescent/update_vs/batch_normalization_1/',
            'GradientDescent/update_vs/batch_normalization_2/',
            'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Conv_4x4',
            'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo',
            'gradients/vs/conv2/Conv2D_grad/Conv2DBackpropFilter/fusion*/Conv_4x4',
            'gradients/vs/conv2/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo']

      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testGroupNormalizeInference(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        gamma = constant_op.constant([0.5, 0.5], np.float32)
        beta = constant_op.constant([0.5, 0.5], np.float32)
        mean = constant_op.constant([0.5, 0.5], np.float32)
        inv_std_dev = constant_op.constant([0.5, 0.5], np.float32)
        y = gen_popnn_ops.popnn_group_norm_inference(inputs=y,
                                                     gamma=gamma,
                                                     beta=beta,
                                                     mean=mean,
                                                     inv_std_dev=inv_std_dev,
                                                     data_format="NHWC",
                                                     epsilon=0.0015,
                                                     num_groups=2)
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        y = gen_popnn_ops.popnn_group_norm_inference(inputs=y,
                                                     gamma=gamma,
                                                     beta=beta,
                                                     mean=mean,
                                                     inv_std_dev=inv_std_dev,
                                                     data_format="NHWC",
                                                     epsilon=0.0015,
                                                     num_groups=2)

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
      ok = ['host-exchange-local-copy',
            'Copy_',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
            'vs/PopnnGroupNormInference/custom-call.*/']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testGroupNormalizeInferenceAndStatistics(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        gamma = constant_op.constant([0.5, 0.5], np.float32)
        beta = constant_op.constant([0.5, 0.5], np.float32)
        mean, inv_std_dev = gen_popnn_ops.popnn_group_norm_statistics(inputs=y,
                                                                   data_format="NHWC",
                                                                   epsilon=0.0015,
                                                                   num_groups=2)
        y = gen_popnn_ops.popnn_group_norm_inference(inputs=y,
                                                     gamma=gamma,
                                                     beta=beta,
                                                     mean=mean,
                                                     inv_std_dev=inv_std_dev,
                                                     data_format="NHWC",
                                                     epsilon=0.0015,
                                                     num_groups=2)
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        mean, inv_std_dev = gen_popnn_ops.popnn_group_norm_statistics(inputs=y,
                                                                   data_format="NHWC",
                                                                   epsilon=0.0015,
                                                                   num_groups=2)
        y = gen_popnn_ops.popnn_group_norm_inference(inputs=y,
                                                     gamma=gamma,
                                                     beta=beta,
                                                     mean=mean,
                                                     inv_std_dev=inv_std_dev,
                                                     data_format="NHWC",
                                                     epsilon=0.0015,
                                                     num_groups=2)

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
      ok = ['host-exchange-local-copy',
            'Copy_',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
            'vs/PopnnGroupNormStatistics/custom-call.*/',
            'vs/PopnnGroupNormInference/custom-call.*/']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testBatchNormAndGroupNormalizeMixedInference(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer())
        gamma = constant_op.constant([0.5, 0.5], np.float32)
        beta = constant_op.constant([0.5, 0.5], np.float32)
        mean = constant_op.constant([0.5, 0.5], np.float32)
        inv_std_dev = constant_op.constant([0.5, 0.5], np.float32)
        y = gen_popnn_ops.popnn_group_norm_inference(inputs=y,
                                                     gamma=gamma,
                                                     beta=beta,
                                                     mean=mean,
                                                     inv_std_dev=inv_std_dev,
                                                     data_format="NHWC",
                                                     epsilon=0.0015,
                                                     num_groups=2)
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
      ok = ['host-exchange-local-copy',
            'Copy_',
            'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
            'vs/PopnnGroupNormInference/custom-call.*/',
            'vs/batch_normalization/FusedBatchNorm/batch-norm-inference.*/']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testGroupNormsMatchFwdBwd(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(x, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer(),
                                 name='conv1')
        gamma = constant_op.constant([0.5, 0.5], np.float32)
        beta = constant_op.constant([0.5, 0.5], np.float32)
        y, _, _ = gen_popnn_ops.popnn_group_norm_training(inputs=y,
                                                        gamma=gamma,
                                                        beta=beta,
                                                        data_format="NHWC",
                                                        epsilon=0.0015,
                                                        num_groups=2)
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer(),
                                 name='conv2')
        y, _, _ = gen_popnn_ops.popnn_group_norm_training(inputs=y,
                                                        gamma=gamma,
                                                        beta=beta,
                                                        data_format="NHWC",
                                                        epsilon=0.0015,
                                                        num_groups=2)
        y = convolutional.conv2d(y, 2, 1, use_bias=False,
                                 kernel_initializer=init_ops.ones_initializer(),
                                 name='conv3')
        y, _, _ = gen_popnn_ops.popnn_group_norm_training(inputs=y,
                                                        gamma=gamma,
                                                        beta=beta,
                                                        data_format="NHWC",
                                                        epsilon=0.0015,
                                                        num_groups=2)

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

      # One GN for forwards and one GN for grad
      ok =  ['host-exchange-local-copy-',
             'Copy_',
             'vs/conv1/Conv2D/convolution*/Conv_1x1/Convolve',
             'vs/PopnnGroupNormTraining/custom-call*/Norm',
             'vs/PopnnGroupNormTraining/custom-call*/iStdDev',
             'vs/PopnnGroupNormTraining/custom-call*/Whiten',
             'Sum/reduce.*/*/Reduce',
             'gradients/vs/PopnnGroupNormTraining_2_grad/PopnnGroupNormGrad/custom-call*/',
             'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropFilter/fusion.*',
            ]

      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

if __name__ == "__main__":
    googletest.main()
