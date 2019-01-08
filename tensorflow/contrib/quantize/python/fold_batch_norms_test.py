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

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.quantize.python import fold_batch_norms
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import saver as saver_lib

batch_norm = layers.batch_norm
conv2d = layers.conv2d
fully_connected = layers.fully_connected
separable_conv2d = layers.separable_conv2d


# TODO(suharshs): Use parameterized test once OSS TF supports it.
class FoldBatchNormsTest(test_util.TensorFlowTestCase):

  def _RunTestOverParameters(self, test_fn):
    parameters_list = [
        # (relu, relu_op_name, with_bypass, has_scaling, fused_batch_norm,
        # freeze_batch_norm_delay, insert identity node)
        (nn_ops.relu6, 'Relu6', False, False, False, 100, False),
        (nn_ops.relu, 'Relu', False, False, False, None, False),
        (nn_ops.relu6, 'Relu6', True, False, False, 100, False),
        (nn_ops.relu, 'Relu', True, False, False, None, False),
        (nn_ops.relu6, 'Relu6', False, True, False, 100, False),
        (nn_ops.relu, 'Relu', False, True, False, None, False),
        (nn_ops.relu6, 'Relu6', True, True, False, 100, False),
        (nn_ops.relu, 'Relu', True, True, False, None, False),
        # Fused batch norm always has scaling enabled.
        (nn_ops.relu6, 'Relu6', False, True, True, None, False),
        (nn_ops.relu, 'Relu', False, True, True, 100, False),
        (nn_ops.relu6, 'Relu6', True, True, True, None, False),
        (nn_ops.relu, 'Relu', True, True, True, 100, False),
        (nn_ops.relu6, 'Relu6', False, True, True, None, True),
        (nn_ops.relu, 'Relu', False, True, True, 100, True),
        (nn_ops.relu6, 'Relu6', True, True, True, None, True),
        (nn_ops.relu, 'Relu', True, True, True, 100, True),
    ]
    for params in parameters_list:
      test_fn(params[0], params[1], params[2], params[3], params[4], params[5],
              params[6])

  def _TestFoldConv2d(self, relu, relu_op_name, with_bypass, has_scaling,
                      fused_batch_norm, freeze_batch_norm_delay,
                      insert_identity_node):
    """Tests folding cases: inputs -> Conv2d with batch norm -> Relu*.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
      has_scaling: Bool, when true the batch norm has scaling.
      fused_batch_norm: Bool, when true the batch norm is fused.
      freeze_batch_norm_delay: None or the number of steps after which training
      switches to using frozen mean and variance
      insert_identity_node: Bool, insert identity node between conv and batch
      norm
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, height, width = 5, 128, 128
      inputs = array_ops.zeros((batch_size, height, width, 3))
      out_depth = 3 if with_bypass else 32
      stride = 1 if with_bypass else 2
      activation_fn = None if with_bypass else relu
      name = 'test/test2' if with_bypass else 'test'
      if insert_identity_node:
        with g.name_scope(name):
          node = conv2d(
              inputs,
              out_depth, [5, 5],
              stride=stride,
              padding='SAME',
              weights_initializer=self._WeightInit(0.09),
              activation_fn=None,
              normalizer_fn=None,
              biases_initializer=None)
          conv_out = array_ops.identity(node, name='conv_out')

          node = batch_norm(
              conv_out,
              center=True,
              scale=has_scaling,
              decay=1.0 - 0.003,
              fused=fused_batch_norm)
          if activation_fn is not None:
            node = activation_fn(node)
          conv_name = name + '/Conv'
      else:
        node = conv2d(
            inputs,
            out_depth, [5, 5],
            stride=stride,
            padding='SAME',
            weights_initializer=self._WeightInit(0.09),
            activation_fn=activation_fn,
            normalizer_fn=batch_norm,
            normalizer_params=self._BatchNormParams(
                scale=has_scaling, fused=fused_batch_norm),
            scope=name)
        conv_name = name
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(
          g, is_training=True, freeze_batch_norm_delay=freeze_batch_norm_delay)

    folded_mul = g.get_operation_by_name(conv_name + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul, [
        conv_name + '/correction_mult',
        self._BatchNormMultiplierName(conv_name, has_scaling, fused_batch_norm)
    ])
    self._AssertOutputGoesToOps(folded_mul, g, [conv_name + '/Conv2D_Fold'])

    folded_conv = g.get_operation_by_name(conv_name + '/Conv2D_Fold')
    self.assertEqual(folded_conv.type, 'Conv2D')
    self._AssertInputOpsAre(folded_conv,
                            [conv_name + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g, [conv_name + '/post_conv_mul'])

    folded_add = g.get_operation_by_name(conv_name + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add, [
        conv_name + '/correction_add',
        self._BathNormBiasName(conv_name, fused_batch_norm)
    ])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)
    if freeze_batch_norm_delay is not None:
      self._AssertMovingAveragesAreFrozen(g, name)

    for op in g.get_operations():
      self.assertFalse('//' in op.name, 'Double slash in op %s' % op.name)

  def testFoldConv2d(self):
    self._RunTestOverParameters(self._TestFoldConv2d)

  def testMultipleLayerConv2d(self,
                              relu=nn_ops.relu,
                              relu_op_name='Relu',
                              has_scaling=True,
                              fused_batch_norm=False,
                              freeze_batch_norm_delay=None,
                              insert_identity_node=False):
    """Tests folding cases for a network with multiple layers.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      has_scaling: Bool, when true the batch norm has scaling.
      fused_batch_norm: Bool, when true the batch norm is fused.
      freeze_batch_norm_delay: None or the number of steps after which training
      switches to using frozen mean and variance
      insert_identity_node: Bool, insert identity node between conv and batch
      norm
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, height, width = 5, 128, 128
      inputs = array_ops.zeros((batch_size, height, width, 3))
      out_depth = 3
      stride = 1
      activation_fn = relu
      scope = 'topnet/testnet'
      with variable_scope.variable_scope(scope, [inputs]):
        layer1 = conv2d(
            inputs,
            out_depth, [5, 5],
            stride=stride,
            padding='SAME',
            weights_initializer=self._WeightInit(0.09),
            activation_fn=None,
            normalizer_fn=None,
            scope='testnet/layer1')
        # Add bn and relu with different scope
        layer1 = batch_norm(
            layer1, scale=has_scaling, fused=fused_batch_norm, scope='layer1')
        layer1 = activation_fn(layer1)
        layer2 = conv2d(
            layer1,
            2 * out_depth, [5, 5],
            stride=stride,
            padding='SAME',
            weights_initializer=self._WeightInit(0.09),
            activation_fn=activation_fn,
            normalizer_fn=batch_norm,
            normalizer_params=self._BatchNormParams(
                scale=has_scaling, fused=fused_batch_norm),
            scope='testnet/layer2')
        # Add bn and relu with different scope
        layer2 = batch_norm(
            layer2, scale=has_scaling, fused=fused_batch_norm, scope='layer2')
        _ = activation_fn(layer2)

      scope = 'topnet/testnet/testnet/layer2'

      fold_batch_norms.FoldBatchNorms(
          g, is_training=True, freeze_batch_norm_delay=freeze_batch_norm_delay)
    folded_mul = g.get_operation_by_name(scope + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul, [
        scope + '/correction_mult',
        self._BatchNormMultiplierName(scope, has_scaling, fused_batch_norm)
    ])
    self._AssertOutputGoesToOps(folded_mul, g, [scope + '/Conv2D_Fold'])

    folded_conv = g.get_operation_by_name(scope + '/Conv2D_Fold')
    self.assertEqual(folded_conv.type, 'Conv2D')
    # Remove :0 at end of name for tensor prior to comparison
    self._AssertInputOpsAre(folded_conv,
                            [scope + '/mul_fold', layer1.name[:-2]])
    self._AssertOutputGoesToOps(folded_conv, g, [scope + '/post_conv_mul'])

    folded_add = g.get_operation_by_name(scope + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add, [
        scope + '/correction_add',
        self._BathNormBiasName(scope, fused_batch_norm)
    ])
    output_op_names = [scope + '/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)
    if freeze_batch_norm_delay is not None:
      self._AssertMovingAveragesAreFrozen(g, scope)

    for op in g.get_operations():
      self.assertFalse('//' in op.name, 'Double slash in op %s' % op.name)

  def _TestFoldConv2dUnknownShape(self,
                                  relu,
                                  relu_op_name,
                                  with_bypass,
                                  has_scaling,
                                  fused_batch_norm,
                                  freeze_batch_norm_delay,
                                  insert_identity_node=False):
    """Tests folding cases: inputs -> Conv2d with batch norm -> Relu*.

    Tests that folding works even with an input shape where some dimensions are
    not known (i.e. None).

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
      has_scaling: Bool, when true the batch norm has scaling.
      fused_batch_norm: Bool, when true the batch norm is fused.
      freeze_batch_norm_delay: None or the number of steps after which training
      switches to using frozen mean and variance
      insert_identity_node: Bool, insert identity node between conv and batch
      norm
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
          normalizer_params=self._BatchNormParams(
              scale=has_scaling, fused=fused_batch_norm),
          scope=scope)
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(
          g, is_training=True, freeze_batch_norm_delay=freeze_batch_norm_delay)

    folded_mul = g.get_operation_by_name(scope + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul, [
        scope + '/correction_mult',
        self._BatchNormMultiplierName(scope, has_scaling, fused_batch_norm)
    ])
    self._AssertOutputGoesToOps(folded_mul, g, [scope + '/Conv2D_Fold'])

    folded_conv = g.get_operation_by_name(scope + '/Conv2D_Fold')
    self.assertEqual(folded_conv.type, 'Conv2D')
    self._AssertInputOpsAre(folded_conv, [scope + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g, [scope + '/post_conv_mul'])

    folded_add = g.get_operation_by_name(scope + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add, [
        scope + '/correction_add',
        self._BathNormBiasName(scope, fused_batch_norm)
    ])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)
    if freeze_batch_norm_delay is not None:
      self._AssertMovingAveragesAreFrozen(g, scope)

    for op in g.get_operations():
      self.assertFalse('//' in op.name, 'Double slash in op %s' % op.name)

  def testFoldConv2dUnknownShape(self):
    self._RunTestOverParameters(self._TestFoldConv2dUnknownShape)

  def _TestFoldFullyConnectedLayer(
      self, relu, relu_op_name, with_bypass, has_scaling, fused_batch_norm,
      freeze_batch_norm_delay, insert_identity_node):
    """Tests folding cases: inputs -> FC with batch norm -> Relu*.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
      has_scaling: Bool, when true the batch norm has scaling.
      fused_batch_norm: Bool, when true the batch norm is fused.
      freeze_batch_norm_delay: None or the number of steps after which training
      switches to using frozen mean and variance
      insert_identity_node: Bool, insert identity node between conv and batch
      norm
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, depth = 5, 256
      inputs = array_ops.zeros((batch_size, depth))
      out_depth = 256 if with_bypass else 128
      activation_fn = None if with_bypass else relu
      name = 'test/test2' if with_bypass else 'test'
      insert_identity_node = fused_batch_norm
      if insert_identity_node:
        with g.name_scope(name):
          node = fully_connected(
              inputs,
              out_depth,
              weights_initializer=self._WeightInit(0.03),
              activation_fn=None,
              normalizer_fn=None,
              biases_initializer=None)
          node = array_ops.identity(node, name='fc_out')

          node = batch_norm(
              node,
              center=True,
              scale=has_scaling,
              decay=1.0 - 0.003,
              fused=fused_batch_norm)
          if activation_fn is not None:
            node = activation_fn(node)
          fc_name = name + '/fully_connected'
      else:

        node = fully_connected(
            inputs,
            out_depth,
            weights_initializer=self._WeightInit(0.03),
            activation_fn=activation_fn,
            normalizer_fn=batch_norm,
            normalizer_params=self._BatchNormParams(
                scale=has_scaling, fused=fused_batch_norm),
            scope=name)
        fc_name = name
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(
          g, is_training=True, freeze_batch_norm_delay=freeze_batch_norm_delay)

    folded_mul = g.get_operation_by_name(fc_name + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    self._AssertInputOpsAre(folded_mul, [
        fc_name + '/correction_mult',
        self._BatchNormMultiplierName(fc_name, has_scaling, fused_batch_norm)
    ])
    self._AssertOutputGoesToOps(folded_mul, g, [fc_name + '/MatMul_Fold'])

    folded_conv = g.get_operation_by_name(fc_name + '/MatMul_Fold')
    self.assertEqual(folded_conv.type, 'MatMul')
    self._AssertInputOpsAre(folded_conv,
                            [fc_name + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g, [fc_name + '/post_conv_mul'])

    folded_add = g.get_operation_by_name(fc_name + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add, [
        fc_name + '/correction_add',
        self._BathNormBiasName(fc_name, fused_batch_norm)
    ])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)
    if freeze_batch_norm_delay is not None:
      self._AssertMovingAveragesAreFrozen(g, name)

    for op in g.get_operations():
      self.assertFalse('//' in op.name, 'Double slash in op %s' % op.name)

  def testFoldFullyConnectedLayer(self):
    self._RunTestOverParameters(self._TestFoldFullyConnectedLayer)

  def _TestFoldDepthwiseConv2d(self, relu, relu_op_name, with_bypass,
                               has_scaling, fused_batch_norm,
                               freeze_batch_norm_delay, insert_identity_node):
    """Tests folding: inputs -> DepthwiseConv2d with batch norm -> Relu*.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
      has_scaling: Bool, when true the batch norm has scaling.
      fused_batch_norm: Bool, when true the batch norm is fused.
      freeze_batch_norm_delay: None or the number of steps after which training
      insert_identity_node: Bool, insert identity node between conv and batch
        norm switches to using frozen mean and variance
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, height, width = 5, 128, 128
      inputs = array_ops.zeros((batch_size, height, width, 3))
      stride = 1 if with_bypass else 2
      activation_fn = None if with_bypass else relu
      name = 'test/test2' if with_bypass else 'test'
      if insert_identity_node:
        with g.name_scope(name):
          node = separable_conv2d(
              inputs,
              None, [5, 5],
              stride=stride,
              depth_multiplier=1.0,
              padding='SAME',
              weights_initializer=self._WeightInit(0.09),
              activation_fn=None,
              normalizer_fn=None,
              biases_initializer=None)
          node = array_ops.identity(node, name='sep_conv_out')

          node = batch_norm(
              node,
              center=True,
              scale=has_scaling,
              decay=1.0 - 0.003,
              fused=fused_batch_norm)
          if activation_fn is not None:
            node = activation_fn(node)
          sep_conv_name = name + '/SeparableConv2d'
      else:
        node = separable_conv2d(
            inputs,
            None, [5, 5],
            stride=stride,
            depth_multiplier=1.0,
            padding='SAME',
            weights_initializer=self._WeightInit(0.09),
            activation_fn=activation_fn,
            normalizer_fn=batch_norm,
            normalizer_params=self._BatchNormParams(
                scale=has_scaling, fused=fused_batch_norm),
            scope=name)
        sep_conv_name = name
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(
          g, is_training=True, freeze_batch_norm_delay=freeze_batch_norm_delay)

    folded_mul = g.get_operation_by_name(sep_conv_name + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    if fused_batch_norm:
      scale_reshape_op_name = sep_conv_name + '/BatchNorm_Fold/scale_reshape'
    else:
      scale_reshape_op_name = sep_conv_name + '/scale_reshape'
    self._AssertInputOpsAre(
        folded_mul, [sep_conv_name + '/correction_mult', scale_reshape_op_name])
    self._AssertOutputGoesToOps(folded_mul, g,
                                [sep_conv_name + '/depthwise_Fold'])

    scale_reshape = g.get_operation_by_name(scale_reshape_op_name)
    self.assertEqual(scale_reshape.type, 'Reshape')
    self._AssertInputOpsAre(scale_reshape, [
        self._BatchNormMultiplierName(sep_conv_name, has_scaling,
                                      fused_batch_norm),
        scale_reshape_op_name + '/shape'
    ])
    self._AssertOutputGoesToOps(scale_reshape, g, [sep_conv_name + '/mul_fold'])

    folded_conv = g.get_operation_by_name(sep_conv_name + '/depthwise_Fold')
    self.assertEqual(folded_conv.type, 'DepthwiseConv2dNative')
    self._AssertInputOpsAre(folded_conv,
                            [sep_conv_name + '/mul_fold', inputs.op.name])
    self._AssertOutputGoesToOps(folded_conv, g,
                                [sep_conv_name + '/post_conv_mul'])

    folded_add = g.get_operation_by_name(sep_conv_name + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add, [
        sep_conv_name + '/correction_add',
        self._BathNormBiasName(sep_conv_name, fused_batch_norm)
    ])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)
    if freeze_batch_norm_delay is not None:
      self._AssertMovingAveragesAreFrozen(g, name)

    for op in g.get_operations():
      self.assertFalse('//' in op.name, 'Double slash in op %s' % op.name)

  def testFoldDepthwiseConv2d(self):
    self._RunTestOverParameters(self._TestFoldDepthwiseConv2d)

  def _TestFoldAtrousConv2d(self, relu, relu_op_name, with_bypass, has_scaling,
                            fused_batch_norm, freeze_batch_norm_delay,
                            insert_identity_node):
    """Tests folding: inputs -> AtrousConv2d with batch norm -> Relu*.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
      has_scaling: Bool, when true the batch norm has scaling.
      fused_batch_norm: Bool, when true the batch norm is fused.
      freeze_batch_norm_delay: None or the number of steps after which training
        switches to using frozen mean and variance
      insert_identity_node: Bool, insert identity node between conv and batch
        norm
    """
    g = ops.Graph()
    with g.as_default():
      batch_size, height, width = 5, 128, 128
      inputs = array_ops.zeros((batch_size, height, width, 3))
      dilation_rate = 2
      activation_fn = None if with_bypass else relu
      name = 'test/test2' if with_bypass else 'test'
      if insert_identity_node:
        with g.name_scope(name):
          node = separable_conv2d(
              inputs,
              None, [3, 3],
              rate=dilation_rate,
              depth_multiplier=1.0,
              padding='SAME',
              weights_initializer=self._WeightInit(0.09),
              activation_fn=None,
              normalizer_fn=None,
              biases_initializer=None)
          node = array_ops.identity(node, name='sep_conv_out')

          node = batch_norm(
              node,
              center=True,
              scale=has_scaling,
              decay=1.0 - 0.003,
              fused=fused_batch_norm)
          if activation_fn is not None:
            node = activation_fn(node)
          sep_conv_name = name + '/SeparableConv2d'
      else:
        node = separable_conv2d(
            inputs,
            None, [3, 3],
            rate=dilation_rate,
            depth_multiplier=1.0,
            padding='SAME',
            weights_initializer=self._WeightInit(0.09),
            activation_fn=activation_fn,
            normalizer_fn=batch_norm,
            normalizer_params=self._BatchNormParams(
                scale=has_scaling, fused=fused_batch_norm),
            scope=name)
        sep_conv_name = name
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
        relu(node, name='test/' + relu_op_name)

      fold_batch_norms.FoldBatchNorms(
          g, is_training=True, freeze_batch_norm_delay=freeze_batch_norm_delay)

    folded_mul = g.get_operation_by_name(sep_conv_name + '/mul_fold')
    self.assertEqual(folded_mul.type, 'Mul')
    if fused_batch_norm:
      scale_reshape_op_name = sep_conv_name + '/BatchNorm_Fold/scale_reshape'
    else:
      scale_reshape_op_name = sep_conv_name + '/scale_reshape'
    self._AssertInputOpsAre(
        folded_mul, [sep_conv_name + '/correction_mult', scale_reshape_op_name])
    self._AssertOutputGoesToOps(folded_mul, g,
                                [sep_conv_name + '/depthwise_Fold'])

    scale_reshape = g.get_operation_by_name(scale_reshape_op_name)
    self.assertEqual(scale_reshape.type, 'Reshape')
    self._AssertInputOpsAre(scale_reshape, [
        self._BatchNormMultiplierName(sep_conv_name, has_scaling,
                                      fused_batch_norm),
        scale_reshape_op_name + '/shape'
    ])
    self._AssertOutputGoesToOps(scale_reshape, g, [sep_conv_name + '/mul_fold'])

    folded_conv = g.get_operation_by_name(sep_conv_name + '/depthwise_Fold')
    self.assertEqual(folded_conv.type, 'DepthwiseConv2dNative')
    self._AssertInputOpsAre(folded_conv, [
        sep_conv_name + '/mul_fold', sep_conv_name + '/depthwise/SpaceToBatchND'
    ])
    if fused_batch_norm:
      self._AssertOutputGoesToOps(folded_conv, g,
                                  [sep_conv_name + '/BatchToSpaceND_Fold'])
    else:
      self._AssertOutputGoesToOps(
          folded_conv, g, [sep_conv_name + '/depthwise/BatchToSpaceND_Fold'])

    folded_add = g.get_operation_by_name(sep_conv_name + '/add_fold')
    self.assertEqual(folded_add.type, 'Add')
    self._AssertInputOpsAre(folded_add, [
        sep_conv_name + '/correction_add',
        self._BathNormBiasName(sep_conv_name, fused_batch_norm)
    ])
    output_op_names = ['test/Add' if with_bypass else 'test/' + relu_op_name]
    self._AssertOutputGoesToOps(folded_add, g, output_op_names)
    if freeze_batch_norm_delay is not None:
      self._AssertMovingAveragesAreFrozen(g, name)

    for op in g.get_operations():
      self.assertFalse('//' in op.name, 'Double slash in op %s' % op.name)

  def testFoldAtrousConv2d(self):
    self._RunTestOverParameters(self._TestFoldAtrousConv2d)

  def _TestCompareFoldAndUnfolded(self,
                                  relu,
                                  relu_op_name,
                                  with_bypass,
                                  has_scaling,
                                  fused_batch_norm,
                                  freeze_batch_norm_delay,
                                  insert_identity_node=False):
    """Tests that running folded and unfolded BN returns the same results.

    Args:
      relu: Callable that returns an Operation, a factory method for the Relu*.
      relu_op_name: String, name of the Relu* operation.
      with_bypass: Bool, when true there is an extra connection added from
        inputs to just before Relu*.
      has_scaling: Bool, when true the batch norm has scaling.
      fused_batch_norm: Bool, when true the batch norm is fused.
      freeze_batch_norm_delay: None or the number of steps after which training
      switches to using frozen mean and variance
      insert_identity_node: Bool, insert identity node between conv and batch
      norm
    """
    random_seed.set_random_seed(1234)
    unfolded_g = ops.Graph()
    with unfolded_g.as_default():
      batch_size, height, width = 5, 128, 128
      inputs = random_ops.random_uniform(
          (batch_size, height, width, 3), dtype=dtypes.float32, seed=1234)
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
          normalizer_params=self._BatchNormParams(
              scale=has_scaling, fused=fused_batch_norm),
          scope=scope)
      if with_bypass:
        node = math_ops.add(inputs, node, name='test/Add')
      relu_node = relu(node, name='test/' + relu_op_name)
    folded_g = self._CopyGraph(unfolded_g)
    with folded_g.as_default():
      fold_batch_norms.FoldBatchNorms(
          folded_g,
          is_training=True,
          freeze_batch_norm_delay=freeze_batch_norm_delay)
    with session.Session(graph=unfolded_g) as sess:
      sess.run(variables.global_variables_initializer())
      grad_node = gradients.gradients(relu_node, inputs)
      results = sess.run([relu_node, grad_node])
      unfolded_forward, unfolded_backward = results[0], results[1]

    with session.Session(graph=folded_g) as sess:
      sess.run(variables.global_variables_initializer())
      relu_node = folded_g.get_tensor_by_name(relu_node.name)
      inputs = folded_g.get_tensor_by_name(inputs.name)
      grad_node = gradients.gradients(relu_node, inputs)
      results = sess.run([relu_node, grad_node])
      folded_forward, folded_backward = results[0], results[1]

    # Check that the folded and unfolded results match.
    self.assertAllClose(unfolded_forward, folded_forward, atol=1e-3)
    self.assertAllClose(unfolded_backward, folded_backward, atol=1e-3)

  def testCompareFoldAndUnfolded(self):
    self._RunTestOverParameters(self._TestCompareFoldAndUnfolded)

  def _BatchNormParams(self, scale=True, fused=False):
    return {
        'center': True,
        'scale': scale,
        'decay': 1.0 - 0.003,
        'fused': fused
    }

  def _BatchNormMultiplierName(self, scope, has_scaling, fused):
    if has_scaling:
      if fused:
        return scope + '/BatchNorm_Fold/mul'
      return scope + '/BatchNorm/batchnorm_1/mul'
    return scope + '/BatchNorm/batchnorm_1/Rsqrt'

  def _BathNormBiasName(self, scope, fused):
    if fused:
      return scope + '/BatchNorm_Fold/bias'
    return scope + '/BatchNorm/batchnorm_1/sub'

  def _WeightInit(self, stddev):
    """Returns a truncated normal variable initializer.

    Function is defined purely to shorten the name so that it stops wrapping.

    Args:
      stddev: Standard deviation of normal variable.

    Returns:
      An initializer that initializes with a truncated normal variable.
    """
    return init_ops.truncated_normal_initializer(stddev=stddev, seed=1234)

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

  def _AssertMovingAveragesAreFrozen(self, graph, scope):
    """Asserts to check if moving mean and variance are frozen.

    Args:
      graph: Graph where the operations are located.
      scope: Scope of batch norm op
    """
    moving_average_mult = graph.get_operation_by_name(
        scope + '/BatchNorm/AssignMovingAvg/mul')
    self.assertTrue(
        moving_average_mult.inputs[1].name.find('freeze_moving_mean/Merge') > 0)
    moving_var_mult = graph.get_operation_by_name(
        scope + '/BatchNorm/AssignMovingAvg_1/mul')
    self.assertTrue(
        moving_var_mult.inputs[1].name.find('freeze_moving_var/Merge') > 0)

  def _CopyGraph(self, graph):
    """Return a copy of graph."""
    meta_graph = saver_lib.export_meta_graph(
        graph=graph, collection_list=graph.get_all_collection_keys())
    graph_copy = ops.Graph()
    with graph_copy.as_default():
      _ = saver_lib.import_meta_graph(meta_graph)
    return graph_copy


if __name__ == '__main__':
  googletest.main()
