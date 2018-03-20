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
"""Unit tests for quantizing a Tensorflow graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.quantize.python import quantize
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest

conv2d = layers.conv2d
separable_conv2d = layers.separable_conv2d


class QuantizeTest(test_util.TensorFlowTestCase):

  def _RunTestOverParameters(self, test_fn):
    params = [True, False]
    for is_training in params:
      test_fn(is_training)

  def testInsertQuantOpFailsWhenOpsNotConnected(self):
    pass

  def _TestInsertQuantOpFailsWhenOpsNotConnected(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      batch_size, height, width, depth = 5, 128, 128, 3
      inputs = array_ops.zeros((batch_size, height, width, depth))
      conv = conv2d(inputs, 32, [5, 5], stride=2, padding='SAME',
                    weights_initializer=self._WeightInit(0.09),
                    activation_fn=None, scope='test')
      relu = nn_ops.relu6(inputs)

    # Inserting a quantization op between two unconnected ops should fail with
    # ValueError.
    with self.assertRaises(ValueError) as err:
      quantize._InsertQuantOp('test', is_training, conv.op, [relu.op],
                              'FailingQuantOp')
    self.assertEqual(
        str(err.exception), 'Some inputs not quantized for ops: [Relu6]')

  def testInsertQuantOpForAddAfterConv2d(self):
    self._RunTestOverParameters(self._TestInsertQuantOpForAddAfterConv2d)

  def _TestInsertQuantOpForAddAfterConv2d(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      batch_size, height, width, depth = 5, 128, 128, 3
      input1 = array_ops.zeros((batch_size, height, width, depth))
      input2 = array_ops.zeros((batch_size, height / 2, width / 2, 32))
      conv = conv2d(input1, 32, [5, 5], stride=2, padding='SAME',
                    weights_initializer=self._WeightInit(0.09),
                    activation_fn=None, scope='test/test')
      node = math_ops.add(conv, input2, name='test/add')
      node = array_ops.identity(node, name='test/identity')
      update_barrier = control_flow_ops.no_op(name='update_barrier')
      with ops.control_dependencies([update_barrier]):
        array_ops.identity(node, name='control_dependency')

    quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)

    quantization_node_name = 'FakeQuantWithMinMaxVars'
    add_quant = graph.get_operation_by_name('test/add_quant/' +
                                            quantization_node_name)
    self.assertEqual(add_quant.type, quantization_node_name)

  def testInsertQuantOpForAddAfterSeparableConv2d(self):
    self._RunTestOverParameters(
        self._TestInsertQuantOpForAddAfterSeparableConv2d)

  def _TestInsertQuantOpForAddAfterSeparableConv2d(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      batch_size, height, width, depth = 5, 128, 128, 3
      input1 = array_ops.zeros((batch_size, height, width, depth))
      input2 = array_ops.zeros((batch_size, height / 2, width / 2, depth))
      conv = separable_conv2d(input1, None, [5, 5], stride=2,
                              depth_multiplier=1.0, padding='SAME',
                              weights_initializer=self._WeightInit(0.09),
                              activation_fn=None, scope='test/test')
      node = math_ops.add(conv, input2, name='test/add')
      node = array_ops.identity(node, name='test/identity')
      update_barrier = control_flow_ops.no_op(name='update_barrier')
      with ops.control_dependencies([update_barrier]):
        array_ops.identity(node, name='control_dependency')

    quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)

    quantization_node_name = 'FakeQuantWithMinMaxVars'
    add_quant = graph.get_operation_by_name('test/add_quant/' +
                                            quantization_node_name)
    self.assertEqual(add_quant.type, quantization_node_name)

  def testFinalLayerQuantized(self):
    self._RunTestOverParameters(self._TestFinalLayerQuantized)

  def _TestFinalLayerQuantized(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      batch_size, height, width, depth = 5, 128, 128, 3
      input1 = array_ops.zeros((batch_size, height, width, depth))
      _ = conv2d(
          input1,
          32, [5, 5],
          stride=2,
          padding='SAME',
          weights_initializer=self._WeightInit(0.09),
          activation_fn=None,
          scope='test')
      # Ensure that the a FakeQuant operation is in the outputs of the BiasAdd.
      bias_add_op = graph.get_operation_by_name('test/BiasAdd')
      quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)
      self.assertTrue('FakeQuantWithMinMaxVars' in
                      [op.type for op in bias_add_op.outputs[0].consumers()])

  def testPostActivationBypassQuantized(self):
    self._RunTestOverParameters(self._TestPostActivationBypassQuantized)

  def _TestPostActivationBypassQuantized(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      batch_size, height, width, depth = 5, 128, 128, 3
      input1 = array_ops.zeros((batch_size, height, width, depth))
      input2 = array_ops.zeros((batch_size, height / 2, width / 2, 32))
      conv = conv2d(
          input1,
          32, [5, 5],
          stride=2,
          padding='SAME',
          weights_initializer=self._WeightInit(0.09),
          activation_fn=array_ops.identity,
          scope='test/test')
      bypass_tensor = math_ops.add(conv, input2, name='test/add')
      _ = array_ops.identity(bypass_tensor, name='test/output')

      quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)

      # Ensure that the bypass node is preceded and followed by
      # FakeQuantWithMinMaxVars operations.
      self.assertTrue('FakeQuantWithMinMaxVars' in
                      [c.type for c in bypass_tensor.consumers()])
      self.assertTrue('FakeQuantWithMinMaxVars' in
                      [i.op.type for i in bypass_tensor.op.inputs])

  def testWithNameScope(self):
    self._RunTestOverParameters(self._TestWithNameScope)

  def _TestWithNameScope(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      with graph.name_scope('name_scope'):
        batch_size, height, width, depth = 5, 128, 128, 3
        input1 = array_ops.zeros((batch_size, height, width, depth))
        _ = conv2d(
            input1,
            32, [5, 5],
            stride=2,
            padding='SAME',
            weights_initializer=self._WeightInit(0.09),
            activation_fn=None,
            scope='test')

        quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)

    for op in graph.get_operations():
      self.assertTrue(not op.name.startswith('name_scope/name_scope/'),
                      'Broken op: %s' % op.name)

  def _WeightInit(self, stddev):
    """Returns truncated normal variable initializer.

    Function is defined purely to shorten the name so that it stops wrapping.

    Args:
      stddev: Standard deviation of normal variable.

    Returns:
      An initialized that initializes with a truncated normal variable.
    """
    return init_ops.truncated_normal_initializer(stddev=stddev)

if __name__ == '__main__':
  googletest.main()
