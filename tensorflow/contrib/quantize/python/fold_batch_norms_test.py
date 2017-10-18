# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for folding batch norm layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.quantize.python import fold_batch_norms
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest

batch_norm = layers.batch_norm
conv2d = layers.conv2d
fully_connected = layers.fully_connected
separable_conv2d = layers.separable_conv2d

_DEFAULT_BATCH_NORM_PARAMS = {
    'center': True,
    'scale': True,
    'decay': 1.0 - 0.003,
    'fused': False,
}


# TODO(suharshs): Use parameterized test once OSS TF supports it.
class FoldBatchNormsTest(test_util.TensorFlowTestCase):

  def _RunTestOverParameters(self, test_fn):
    parameters_list = [
        # (relu, relu_op_name, with_bypass)
        (nn_ops.relu6, 'Relu6', False),
        (nn_ops.relu, 'Relu', False),
        (nn_ops.relu6, 'Relu6', True),
        (nn_ops.relu, 'Relu', True),
    ]
    for parameters in parameters_list:
      test_fn(parameters[0], parameters[1], parameters[2])

  def testFailsWithFusedBatchNorm(self):
    self._RunTestOverParameters(self._TestFailsWithFusedBatchNorm)

  def _TestFailsWithFusedBatchNorm(self, relu, relu_op_name, with_bypass):
    """Tests that batch norm fails when fused batch norm ops are present."""
    g = ops.Graph()
    with g.as_default():
      batch_size, height, width = 5, 128, 128
      inputs = array_ops.zeros((batch_size, height, width, 3))
      out_depth = 3 if with_bypass else 32
      stride = 1 if with_bypass else 2
      activation_fn = None if with_bypass else relu
      batch_norm_params = _DEFAULT_BATCH_NORM_PARAMS.copy()
      batch_norm_params['fused'] = True
      scope = 'test/test2' if with_bypass else 'test'
      node = conv2d(inputs, out_depth, [5, 5], stride=stride, padding='SAME',
                    weights_initializer=self._WeightInit(0.09),
                    activation_fn=activation_fn,
                    normalizer_fn=batch_norm,
                    normalizer_params=batch_norm_params,
                    scope=scope)
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      with self.assertRaises(ValueError):
        fold_batch_norms.FoldBatchNorms(g)

  def _TestFoldConv2d(self, relu, relu_op_name, with_bypass):
    """Tests folding cases: inputs -> Conv2d with batch norm -> Relu*.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, height, width = 5, 128, 128
      inputs = array_ops.zeros((batch_size, height, width, 3))
      out_depth = 3 if with_bypass else 32
      stride = 1 if with_bypass else 2
      activation_fn = None if with_bypass else relu
      scope = 'test/test2' if with_bypass else 'test'
      node = conv2d(inputs, out_depth, [5, 5], stride=stride, padding='SAME',
                    weights_initializer=self._WeightInit(0.09),
                    activation_fn=activation_fn,
                    normalizer_fn=batch_norm,
                    normalizer_params=_DEFAULT_BATCH_NORM_PARAMS,
                    scope=scope)
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(g)

    folded_mul = g.get_operation_by_name(scope + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul,
                            [scope + '/weights/read',
                             scope + '/BatchNorm/batchnorm/mul'])
    self._AssertOutputGoesToOps(folded_mul, g, [scope + '/convolution_Fold'])

    folded_conv = g.get_operation_by_name(scope + '/convolution_Fold')
    self.assertEqual(folded_conv.type, 'Conv2D')
    self._AssertInputOpsAre(folded_conv,
                            [scope + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g, [scope + '/add_fold'])

    folded_add = g.get_operation_by_name(scope + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add,
                            [scope + '/convolution_Fold',
                             scope + '/BatchNorm/batchnorm/sub'])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)

  def testFoldConv2d(self):
    self._RunTestOverParameters(self._TestFoldConv2d)

  def _TestFoldConv2dUnknownShape(self, relu, relu_op_name, with_bypass):
    """Tests folding cases: inputs -> Conv2d with batch norm -> Relu*.

    Tests that folding works even with an input shape where some dimensions are
    not known (i.e. None).

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
    """
    g = ops.Graph()
    with g.as_default():
      inputs = array_ops.placeholder(dtypes.float32, shape=(5, None, None, 3))
      out_depth = 3 if with_bypass else 32
      stride = 1 if with_bypass else 2
      activation_fn = None if with_bypass else relu
      scope = 'test/test2' if with_bypass else 'test'
      node = conv2d(
          inputs,
          out_depth, [5, 5],
          stride=stride,
          padding='SAME',
          weights_initializer=self._WeightInit(0.09),
          activation_fn=activation_fn,
          normalizer_fn=batch_norm,
          normalizer_params=_DEFAULT_BATCH_NORM_PARAMS,
          scope=scope)
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(g)

    folded_mul = g.get_operation_by_name(scope + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul, [
        scope + '/weights/read', scope + '/BatchNorm/batchnorm/mul'
    ])
    self._AssertOutputGoesToOps(folded_mul, g, [scope + '/convolution_Fold'])

    folded_conv = g.get_operation_by_name(scope + '/convolution_Fold')
    self.assertEqual(folded_conv.type, 'Conv2D')
    self._AssertInputOpsAre(folded_conv, [scope + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g, [scope + '/add_fold'])

    folded_add = g.get_operation_by_name(scope + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add, [
        scope + '/convolution_Fold', scope + '/BatchNorm/batchnorm/sub'
    ])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)

  def testFoldConv2dUnknownShape(self):
    self._RunTestOverParameters(self._TestFoldConv2dUnknownShape)

  def _TestFoldConv2dWithoutScale(self, relu, relu_op_name, with_bypass):
    """Tests folding cases: inputs -> Conv2d with batch norm -> Relu*.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, height, width = 5, 128, 128
      inputs = array_ops.zeros((batch_size, height, width, 3))
      out_depth = 3 if with_bypass else 32
      stride = 1 if with_bypass else 2
      activation_fn = None if with_bypass else relu
      bn_params = copy.copy(_DEFAULT_BATCH_NORM_PARAMS)
      bn_params['scale'] = False
      scope = 'test/test2' if with_bypass else 'test'
      node = conv2d(inputs, out_depth, [5, 5], stride=stride, padding='SAME',
                    weights_initializer=self._WeightInit(0.09),
                    activation_fn=activation_fn,
                    normalizer_fn=batch_norm,
                    normalizer_params=bn_params,
                    scope=scope)
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(g)

    folded_mul = g.get_operation_by_name(scope + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul,
                            [scope + '/weights/read',
                             scope + '/BatchNorm/batchnorm/Rsqrt'])
    self._AssertOutputGoesToOps(folded_mul, g, [scope + '/convolution_Fold'])

    folded_conv = g.get_operation_by_name(scope + '/convolution_Fold')
    self.assertEqual(folded_conv.type, 'Conv2D')
    self._AssertInputOpsAre(folded_conv,
                            [scope + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g, [scope + '/add_fold'])

    folded_add = g.get_operation_by_name(scope + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add,
                            [scope + '/convolution_Fold',
                             scope + '/BatchNorm/batchnorm/sub'])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)

  def testFoldConv2dWithoutScale(self):
    self._RunTestOverParameters(self._TestFoldConv2dWithoutScale)

  def _TestFoldFullyConnectedLayer(self, relu, relu_op_name, with_bypass):
    """Tests folding cases: inputs -> FC with batch norm -> Relu*.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, depth = 5, 256
      inputs = array_ops.zeros((batch_size, depth))
      out_depth = 256 if with_bypass else 128
      activation_fn = None if with_bypass else relu
      scope = 'test/test2' if with_bypass else 'test'
      node = fully_connected(inputs, out_depth,
                             weights_initializer=self._WeightInit(0.03),
                             activation_fn=activation_fn,
                             normalizer_fn=batch_norm,
                             normalizer_params=_DEFAULT_BATCH_NORM_PARAMS,
                             scope=scope)
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(g)

    folded_mul = g.get_operation_by_name(scope + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul,
                            [scope + '/weights/read',
                             scope + '/BatchNorm/batchnorm/mul'])
    self._AssertOutputGoesToOps(folded_mul, g, [scope + '/MatMul_Fold'])

    folded_conv = g.get_operation_by_name(scope + '/MatMul_Fold')
    self.assertEqual(folded_conv.type, 'MatMul')
    self._AssertInputOpsAre(folded_conv,
                            [scope + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g, [scope + '/add_fold'])

    folded_add = g.get_operation_by_name(scope + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add,
                            [scope + '/MatMul_Fold',
                             scope + '/BatchNorm/batchnorm/sub'])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)

  def testFoldFullyConnectedLayer(self):
    self._RunTestOverParameters(self._TestFoldFullyConnectedLayer)

  def _TestFoldFullyConnectedLayerWithoutScale(self, relu, relu_op_name,
                                               with_bypass):
    """Tests folding cases: inputs -> FC with batch norm -> Relu*.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, depth = 5, 256
      inputs = array_ops.zeros((batch_size, depth))
      out_depth = 256 if with_bypass else 128
      activation_fn = None if with_bypass else relu
      bn_params = copy.copy(_DEFAULT_BATCH_NORM_PARAMS)
      bn_params['scale'] = False
      scope = 'test/test2' if with_bypass else 'test'
      node = fully_connected(inputs, out_depth,
                             weights_initializer=self._WeightInit(0.03),
                             activation_fn=activation_fn,
                             normalizer_fn=batch_norm,
                             normalizer_params=bn_params,
                             scope=scope)
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(g)

    folded_mul = g.get_operation_by_name(scope + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul,
                            [scope + '/weights/read',
                             scope + '/BatchNorm/batchnorm/Rsqrt'])
    self._AssertOutputGoesToOps(folded_mul, g, [scope + '/MatMul_Fold'])

    folded_conv = g.get_operation_by_name(scope + '/MatMul_Fold')
    self.assertEqual(folded_conv.type, 'MatMul')
    self._AssertInputOpsAre(folded_conv,
                            [scope + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g, [scope + '/add_fold'])

    folded_add = g.get_operation_by_name(scope + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add,
                            [scope + '/MatMul_Fold',
                             scope + '/BatchNorm/batchnorm/sub'])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)

  def testFoldFullyConnectedLayerWithoutScale(self):
    self._RunTestOverParameters(self._TestFoldFullyConnectedLayerWithoutScale)

  def _TestFoldDepthwiseConv2d(self, relu, relu_op_name, with_bypass):
    """Tests folding: inputs -> DepthwiseConv2d with batch norm -> Relu*.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, height, width = 5, 128, 128
      inputs = array_ops.zeros((batch_size, height, width, 3))
      stride = 1 if with_bypass else 2
      activation_fn = None if with_bypass else relu
      scope = 'test/test2' if with_bypass else 'test'
      node = separable_conv2d(inputs, None, [5, 5], stride=stride,
                              depth_multiplier=1.0, padding='SAME',
                              weights_initializer=self._WeightInit(0.09),
                              activation_fn=activation_fn,
                              normalizer_fn=batch_norm,
                              normalizer_params=_DEFAULT_BATCH_NORM_PARAMS,
                              scope=scope)
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(g)

    folded_mul = g.get_operation_by_name(scope + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul,
                            [scope + '/depthwise_weights/read',
                             scope + '/scale_reshape'])
    self._AssertOutputGoesToOps(folded_mul, g, [scope + '/depthwise_Fold'])

    scale_reshape = g.get_operation_by_name(scope + '/scale_reshape')
    self.assertEqual(scale_reshape.type, 'Reshape')
    self._AssertInputOpsAre(scale_reshape,
                            [scope + '/BatchNorm/batchnorm/mul',
                             scope + '/scale_reshape/shape'])
    self._AssertOutputGoesToOps(scale_reshape, g, [scope + '/mul_fold'])

    folded_conv = g.get_operation_by_name(scope + '/depthwise_Fold')
    self.assertEqual(folded_conv.type, 'DepthwiseConv2dNative')
    self._AssertInputOpsAre(folded_conv,
                            [scope + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g, [scope + '/add_fold'])

    folded_add = g.get_operation_by_name(scope + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add,
                            [scope + '/depthwise_Fold',
                             scope + '/BatchNorm/batchnorm/sub'])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)

  def testFoldDepthwiseConv2d(self):
    self._RunTestOverParameters(self._TestFoldDepthwiseConv2d)

  def _TestFoldDepthwiseConv2dWithoutScale(self, relu, relu_op_name,
                                           with_bypass):
    """Tests folding: inputs -> DepthwiseConv2d with batch norm -> Relu*.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, height, width = 5, 128, 128
      inputs = array_ops.zeros((batch_size, height, width, 3))
      stride = 1 if with_bypass else 2
      activation_fn = None if with_bypass else relu
      bn_params = copy.copy(_DEFAULT_BATCH_NORM_PARAMS)
      bn_params['scale'] = False
      scope = 'test/test2' if with_bypass else 'test'
      node = separable_conv2d(inputs, None, [5, 5], stride=stride,
                              depth_multiplier=1.0, padding='SAME',
                              weights_initializer=self._WeightInit(0.09),
                              activation_fn=activation_fn,
                              normalizer_fn=batch_norm,
                              normalizer_params=bn_params,
                              scope=scope)
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(g)

    folded_mul = g.get_operation_by_name(scope + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul,
                            [scope + '/depthwise_weights/read',
                             scope + '/scale_reshape'])
    self._AssertOutputGoesToOps(folded_mul, g, [scope + '/depthwise_Fold'])

    scale_reshape = g.get_operation_by_name(scope + '/scale_reshape')
    self.assertEqual(scale_reshape.type, 'Reshape')
    self._AssertInputOpsAre(scale_reshape,
                            [scope + '/BatchNorm/batchnorm/Rsqrt',
                             scope + '/scale_reshape/shape'])
    self._AssertOutputGoesToOps(scale_reshape, g, [scope + '/mul_fold'])

    folded_conv = g.get_operation_by_name(scope + '/depthwise_Fold')
    self.assertEqual(folded_conv.type, 'DepthwiseConv2dNative')
    self._AssertInputOpsAre(folded_conv,
                            [scope + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g, [scope + '/add_fold'])

    folded_add = g.get_operation_by_name(scope + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add,
                            [scope + '/depthwise_Fold',
                             scope + '/BatchNorm/batchnorm/sub'])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)

  def testFoldDepthwiseConv2dWithoutScale(self):
    self._RunTestOverParameters(self._TestFoldDepthwiseConv2dWithoutScale)

  def _WeightInit(self, stddev):
    """Returns a truncated normal variable initializer.

    Function is defined purely to shorten the name so that it stops wrapping.

    Args:
      stddev: Standard deviation of normal variable.

    Returns:
      An initializer that initializes with a truncated normal variable.
    """
    return init_ops.truncated_normal_initializer(stddev=stddev)

  def _AssertInputOpsAre(self, op, in_op_names):
    """Asserts that all inputs to op come from in_op_names (disregarding order).

    Args:
      op: Operation to check inputs for.
      in_op_names: List of strings, operations where all op's inputs should
        come from.
    """
    expected_inputs = [in_op_name + ':0' for in_op_name in in_op_names]
    self.assertItemsEqual([t.name for t in op.inputs], expected_inputs)

  def _AssertOutputGoesToOps(self, op, graph, out_op_names):
    """Asserts that outputs from op go to out_op_names (and perhaps others).

    Args:
      op: Operation to check outputs for.
      graph: Graph where output operations are located.
      out_op_names: List of strings, operations where op's outputs should go.
    """
    for out_op_name in out_op_names:
      out_op = graph.get_operation_by_name(out_op_name)
      self.assertIn(op.outputs[0].name, [str(t.name) for t in out_op.inputs])

if __name__ == '__main__':
  googletest.main()
