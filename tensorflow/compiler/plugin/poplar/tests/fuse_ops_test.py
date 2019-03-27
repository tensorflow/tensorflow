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
from tensorflow.python.layers import normalization as layers_norm
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


class IpuFuseOpsTest(test_util.TensorFlowTestCase):
  def testSigmoid(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3], name="a")
      c = math_ops.sigmoid(pa)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {pa: [-6.0, 0.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.002473, 0.5, 0.997527])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'Sigmoid/fusion/Nonlinearity']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testSigmoidNotInplace(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3], name="a")
      c = math_ops.sigmoid(pa) + pa

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {pa: [-6.0, 0.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [-5.997527, 0.5, 6.997527])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = [
          '__seed*', 'Sigmoid/fusion/Nonlinearity',
          'Copy_XLA_Args/arg0.*_to_Sigmoid/fusion.clone/OnTileCopy-0',
          'add/add.*/AddTo'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testSigmoidGrad(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3], name="grad")
      pb = array_ops.placeholder(np.float32, [3], name="in")
      c = gen_math_ops.sigmoid_grad(pa, pb)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {pa: [2.0, 0.5, 1.0], pb: [-1.0, 1.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [2.0, 0.25, 0.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'SigmoidGrad/fusion/NonLinearityGrad']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testRelu(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3], name="a")
      c = nn_ops.relu(pa)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      fd = {pa: [-6.0, 0.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.0, 6.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'Relu/fusion/Nonlinearity']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testReluNotInPlace(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3], name="a")
      c = nn_ops.relu(pa) + pa

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      fd = {pa: [1, -2, 1]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [2, -2, 2])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = [
          '__seed*', 'Relu/fusion/Nonlinearity',
          'Copy_XLA_Args/arg0.*_to_Relu/fusion.clone/OnTileCopy-0',
          'add/add.*/AddTo'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testReluGrad(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3], name="grad")
      pb = array_ops.placeholder(np.float32, [3], name="in")
      c = gen_nn_ops.relu_grad(pa, pb)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {pa: [2.0, 0.5, 1.0], pb: [-1.0, 1.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.5, 1.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'ReluGrad/fusion/NonLinearityGrad']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testMaxPool(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
      c = nn.max_pool(
          pa,
          ksize=[1, 1, 5, 5],
          strides=[1, 1, 2, 2],
          data_format='NCHW',
          padding='SAME',
          name="max")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
          pa: np.ones([1, 1, 10, 10]),
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, np.ones([1, 1, 5, 5]))

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'max/custom-call.*/maxPool5x5']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testFwdAndBwdMaxPool(self):
    input = np.arange(16).reshape(1, 4, 4, 1)
    output_grad = np.full((1, 2, 2, 1), 0.1)

    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1, 4, 4, 1], name="a")
      pb = array_ops.placeholder(np.float32, [1, 2, 2, 1], name="b")
      c = nn.max_pool(
          pa,
          ksize=[1, 2, 2, 1],
          strides=[1, 2, 2, 1],
          data_format='NCHW',
          padding='SAME')
      d = gen_nn_ops.max_pool_grad(
          pa,
          c,
          pb,
          ksize=[1, 2, 2, 1],
          strides=[1, 2, 2, 1],
          data_format='NCHW',
          padding='SAME')

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)
      fe = {
          pa: input,
          pb: output_grad,
      }
      output, input_grad = sess.run((c, d), fe)
      self.assertAllClose(output, [[[[5.], [7.]], [[13.], [15.]]]])
      self.assertAllClose(
          input_grad, [[[[0.], [0.], [0.], [0.]], [[0.], [0.1], [0.], [0.1]],
                        [[0.], [0.], [0.], [0.]], [[0.], [0.1], [0.], [0.1]]]])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = [
          '__seed*', 'Copy_*', 'MaxPool/custom-call.*/maxPool2x2/',
          'MaxPoolGrad/custom-call.*/maxPool2x2'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testScaledAddTo(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float16, [3])
      pb = array_ops.placeholder(np.float16, [3])
      const = array_ops.constant(2.0, np.float16)
      c = pa + pb * const

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [4.0, 4.5, 7.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'host-exchange-local-copy-', 'add/fusion/AddTo']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testScaledSubtractFrom(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float16, [3])
      pb = array_ops.placeholder(np.float16, [3])
      const = array_ops.constant(2.0, np.float16)
      # note how const operand index varies compared to testScaledAddTo
      # still should match as it will be reordered
      c = pa - const * pb

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, -3.5, -5.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'host-exchange-local-copy-', 'sub/fusion/AddTo']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testScaledAddToVariable(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float16, [3])
      pb = array_ops.placeholder(np.float16, [3])
      pc = array_ops.placeholder(np.float16, [1])
      c = pa + pb * pc

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0], pc: [2.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [4.0, 4.5, 7.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'host-exchange-local-copy-', 'add/fusion/AddTo']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testScaledSubtractFromVariable(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float16, [3])
      pb = array_ops.placeholder(np.float16, [3])
      pc = array_ops.placeholder(np.float16, [1])
      c = pa - pc * pb

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0], pc: [2.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, -3.5, -5.0])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'host-exchange-local-copy-', 'sub/fusion/AddTo']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConvolutionBiasApply(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())
        y = convolutional.conv2d(
            y,
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())

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
      self.assertEqual(len(result),
                       6)  # 2xcompile, 1xupload, 1xload, 1xdownload, 1xexecute

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = [
          '__seed*',
          'GradientDescent/update_vs/conv2d/bias/ResourceApplyGradientDescent/fusion.*/Reduce'
      ]
      self.assertTrue(tu.check_compute_sets_in_whitelist_entries(cs_list, ok))

  def testConvolutionBiasApplyVariableLR(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      lr = array_ops.placeholder(np.float32, shape=[])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())
        y = convolutional.conv2d(
            y,
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())

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
      self.assertEqual(len(result),
                       6)  # 2xcompile, 1xupload, 1xload, 1xdownload, 1xexecute

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      ok = [
          '__seed*', 'Copy_', 'host-exchange-local-copy-',
          'vs/conv2d/Conv2D/convolution*/Conv_1x1/Convolve',
          'vs/conv2d/BiasAdd/fusion*/addToChannel',
          'gradients/vs/conv2d_1/Conv2D_grad/Conv2DBackpropFilter/fusion*/Conv_4x4',
          'gradients/vs/conv2d_1/Conv2D_grad/Conv2DBackpropFilter/fusion*/AddTo',
          'GradientDescent/update_vs/conv2d/bias/ResourceApplyGradientDescent/fusion*/ReduceFinalStage/IntermediateToOutput/Reduce',
          'GradientDescent/update_vs/conv2d/bias/ResourceApplyGradientDescent/fusion*/AddTo',
          'GradientDescent/update_vs/conv2d_1/bias/ResourceApplyGradientDescent/multiply*/Op/Multiply',
          'GradientDescent/update_vs/conv2d_1/bias/ResourceApplyGradientDescent/subtract*/AddTo',
          'vs/conv2d_1/BiasAdd/fusion*/addToChannel',
          'Sum/reduce*/ReduceFinalStage/IntermediateToOutput/Reduce'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testAvgPoolValid(self):
    np.random.seed(0)
    shape = [1, 10, 10, 1]
    data = np.random.uniform(0, 1, shape)
    # The expected answer was generated using TF on the cpu
    expected = [[[[0.47279388]]]]

    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, shape, name="a")
      output = nn.avg_pool(
          pa,
          ksize=[1, 10, 10, 1],
          strides=[1, 1, 1, 1],
          data_format='NHWC',
          padding='VALID',
          name="avg")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      fd = {pa: data}
      result = sess.run(output, fd)
      self.assertAllClose(result, expected)

      result = sess.run(report)
      self.assertEqual(len(result), 4)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'avg/custom-call.*/avgPool10x10']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testAvgPoolValidWithBroadcast(self):
    np.random.seed(0)
    shape = [1, 10, 10, 1]
    data = np.random.uniform(0, 1, shape)
    # The expected answer was generated using TF on the cpu
    expected = [[[[0.52647954], [0.44196457], [0.49284577]],
                 [[0.44039682], [0.44067329], [0.44934618]],
                 [[0.46444583], [0.45419583], [0.38236427]]]]

    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, shape, name="a")
      output = nn.avg_pool(
          pa,
          ksize=[1, 5, 5, 1],
          strides=[1, 2, 2, 1],
          data_format='NHWC',
          padding='VALID',
          name="avg")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      fd = {pa: data}
      result = sess.run(output, fd)
      self.assertAllClose(result, expected)

      result = sess.run(report)
      self.assertEqual(len(result), 4)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['__seed*', 'avg/custom-call.*/avgPool5x5']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testAvgPoolSameWithReshape(self):
    np.random.seed(0)
    shape = [1, 10, 10, 1]
    data = np.random.uniform(0, 1, shape)
    # The expected answer was generated using TF on the cpu
    expected = [[[[0.64431685], [0.51738459], [0.49705142], [0.60235918],
                  [0.73694557]],
                 [[0.57755166], [0.47387227], [0.40451217], [0.4876942],
                  [0.55843753]],
                 [[0.49037799], [0.4466258], [0.35829377], [0.40070742],
                  [0.37205362]],
                 [[0.47563809], [0.4075647], [0.34894851], [0.35470542],
                  [0.3322109]],
                 [[0.52914065], [0.45464769], [0.38156652], [0.32455513],
                  [0.33199897]]]]

    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, shape, name="a")
      output = nn.avg_pool(
          pa,
          ksize=[1, 5, 5, 1],
          strides=[1, 2, 2, 1],
          data_format='NHWC',
          padding='SAME',
          name="avg")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      fd = {pa: data}
      result = sess.run(output, fd)
      self.assertAllClose(result, expected)

      result = sess.run(report)
      self.assertEqual(len(result), 4)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      ok = ['__seed*', 'avg/custom-call.*/avgPool5x5']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testFullyConnectedWithBias(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[2, 2])
      weights = array_ops.placeholder(np.float32, shape=[2, 2])
      bias = array_ops.placeholder(np.float32, shape=[2])
      x_new = nn.xw_plus_b(x, weights, bias)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:

      sess.run(report)

      out = sess.run(x_new, {
          x: np.full([2, 2], 3),
          weights: np.full([2, 2], 4),
          bias: np.ones([2]),
      })
      self.assertAllClose(np.full([2, 2], 25), out)

      result = sess.run(report)
      self.assertEqual(len(result),
                       4)  # 1xcompile, 1xload, 1xdownload, 1xexecute

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = [
          '__seed*', 'host-exchange-local-copy',
          'xw_plus_b/MatMul/dot.*/Conv_1/Convolve',
          'xw_plus_b/fusion/addToChannel'
      ]
      self.assertTrue(tu.check_compute_sets_in_whitelist_entries(cs_list, ok))

  def testConvWithBnAndRelu(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())
        y = layers_norm.batch_normalization(y, fused=True)
        y = nn_ops.relu(y)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:

      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)
      self.assertEqual(len(result),
                       6)  # 2xcompile, 1xupload 1xload, 1xdownload, 1xexecute

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = [
          '__seed*', 'host-exchange-local-copy', 'Copy_',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1', 'vs/conv2d/BiasAdd',
          'vs/batch_normalization/FusedBatchNorm/batch-norm-inference.*/',
          'vs/Relu/fusion/Nonlinearity'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testBiasApplyFixedLR(self):
    input = np.ones((1, 4, 4, 2))

    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer(),
            bias_initializer=init_ops.ones_initializer(),
            name="a")
        y = nn.relu(y)

      loss = math_ops.reduce_sum(y)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(loss)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(report)
      fe = {
          x: input,
      }
      l, _ = sess.run((loss, train), fe)
      tvars = variables.global_variables()
      tvars_vals = sess.run(tvars)

      found = False
      for var, val in zip(tvars, tvars_vals):
        if var.name == "vs/a/bias:0":
          # Value computed using the CPU backend
          self.assertAllClose(val, [-0.6, -0.6])
          found = True
      self.assertTrue(found)

  def testBiasApplyVariableLR(self):
    input = np.ones((1, 4, 4, 2))

    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      lr = array_ops.placeholder(np.float32, shape=[])
      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer(),
            bias_initializer=init_ops.ones_initializer(),
            name="a")
        y = nn.relu(y)

      loss = math_ops.reduce_sum(y)
      optimizer = gradient_descent.GradientDescentOptimizer(lr)
      train = optimizer.minimize(loss)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(report)
      fe = {
          x: input,
          lr: 0.1,
      }
      l, _ = sess.run((loss, train), fe)
      tvars = variables.global_variables()
      tvars_vals = sess.run(tvars)

      found = False
      for var, val in zip(tvars, tvars_vals):
        if var.name == "vs/a/bias:0":
          # Value computed using the CPU backend
          self.assertAllClose(val, [-0.6, -0.6])
          found = True
      self.assertTrue(found)


if __name__ == "__main__":
  googletest.main()
