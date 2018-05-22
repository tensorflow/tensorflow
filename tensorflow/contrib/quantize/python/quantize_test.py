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
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
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
      node = nn_ops.relu6(node, name='test/relu6')
      update_barrier = control_flow_ops.no_op(name='update_barrier')
      with ops.control_dependencies([update_barrier]):
        array_ops.identity(node, name='control_dependency')

    quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)

    quantization_node_name = 'FakeQuantWithMinMaxVars'
    conv_quant = graph.get_operation_by_name('test/test/conv_quant/' +
                                             quantization_node_name)
    self.assertEqual(conv_quant.type, quantization_node_name)

    # Scan through all FakeQuant operations, ensuring that the activation
    # isn't in the consumers of the operation. Since activations are folded
    # the preceding operation during inference, the FakeQuant operation after
    # the activation is all that is needed.
    for op in graph.get_operations():
      if op.type == quantization_node_name:
        quant_op = graph.get_operation_by_name(op.name)
        consumers = []
        for output in quant_op.outputs:
          consumers.extend(output.consumers())

        self.assertNotIn('test/relu6', [c.name for c in consumers])

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
      node = nn_ops.relu6(node, name='test/relu6')
      update_barrier = control_flow_ops.no_op(name='update_barrier')
      with ops.control_dependencies([update_barrier]):
        array_ops.identity(node, name='control_dependency')

    quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)

    quantization_node_name = 'FakeQuantWithMinMaxVars'
    conv_quant = graph.get_operation_by_name('test/test/conv_quant/' +
                                             quantization_node_name)
    self.assertEqual(conv_quant.type, quantization_node_name)

    for op in graph.get_operations():
      if op.type == quantization_node_name:
        quant_op = graph.get_operation_by_name(op.name)
        # Scan through all FakeQuant operations, ensuring that the activation
        # identity op isn't in the consumers of the operation.
        consumers = []
        for output in quant_op.outputs:
          consumers.extend(output.consumers())

        self.assertNotIn('test/relu6', [c.name for c in consumers])

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
          activation_fn=nn_ops.relu6,
          scope='test/test')
      bypass_tensor = math_ops.add(conv, input2, name='test/add')
      # The output of the post_activation bypass will be another layer.
      _ = conv2d(
          bypass_tensor,
          32, [5, 5],
          stride=2,
          padding='SAME',
          weights_initializer=self._WeightInit(0.09),
          activation_fn=nn_ops.relu6,
          scope='test/unused')

      quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)

      # Ensure that the bypass node is preceded by and followed by a
      # FakeQuantWithMinMaxVar operation, since the output of the Add isn't an
      # activation.
      self.assertTrue('FakeQuantWithMinMaxVars' in
                      [c.type for c in bypass_tensor.consumers()])
      self.assertTrue('FakeQuantWithMinMaxVars' in
                      [i.op.type for i in bypass_tensor.op.inputs])

  def testOverlappingPostActivationBypassQuantized(self):
    self._RunTestOverParameters(
        self._TestOverlappingPostActivationBypassQuantized)

  def _TestOverlappingPostActivationBypassQuantized(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      batch_size, height, width, depth = 5, 128, 128, 3
      conv_input = array_ops.zeros((batch_size, height, width, depth))
      conv1 = conv2d(
          conv_input,
          32, [5, 5],
          stride=2,
          padding='SAME',
          weights_initializer=self._WeightInit(0.09),
          activation_fn=nn_ops.relu6,
          scope='test/test1')

      # The bypass of this conv is the post activation bypass of the previous
      # conv.
      conv2 = conv2d(
          conv_input,
          32, [5, 5],
          stride=2,
          padding='SAME',
          weights_initializer=self._WeightInit(0.09),
          activation_fn=None,
          scope='test/test2')

      bypass_tensor = math_ops.add(conv1, conv2, name='test/add')
      _ = nn_ops.relu6(bypass_tensor, name='test/output')

      quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)

      # Ensure that the bypass node is preceded by a FakeQuantWithMinMaxVar
      # operation, and NOT followed by one.
      self.assertTrue('FakeQuantWithMinMaxVars' not in
                      [c.type for c in bypass_tensor.consumers()])
      self.assertTrue('FakeQuantWithMinMaxVars' in
                      [i.op.type for i in bypass_tensor.op.inputs])

      # Ensure that all the convs and activations are quantized.
      op_names = [op.name for op in graph.get_operations()]
      self.assertTrue(
          'test/test1/weights_quant/FakeQuantWithMinMaxVars' in op_names)
      self.assertTrue(
          'test/test2/weights_quant/FakeQuantWithMinMaxVars' in op_names)
      self.assertTrue(
          'test/test1/act_quant/FakeQuantWithMinMaxVars' in op_names)
      self.assertTrue('test/act_quant/FakeQuantWithMinMaxVars' in op_names)
      self.assertEqual(
          'Relu6',
          graph.get_operation_by_name(
              'test/test1/act_quant/FakeQuantWithMinMaxVars').inputs[0].op.type)
      self.assertEqual(
          'Relu6',
          graph.get_operation_by_name(
              'test/act_quant/FakeQuantWithMinMaxVars').inputs[0].op.type)

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

  def testWithNullNameScope(self):
    self._RunTestOverParameters(self._TestWithNullNameScope)

  def _TestWithNullNameScope(self, is_training):
    graph = ops.Graph()
    with graph.as_default():
      with graph.name_scope(None):
        batch_size, height, width, depth = 5, 128, 128, 32
        input1 = array_ops.zeros((batch_size, height, width, depth))
        _ = conv2d(
            input1,
            32, [5, 5],
            padding='SAME',
            weights_initializer=self._WeightInit(0.09),
            activation_fn=None,
            scope='test')

        quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)
        # Passes if Quantize() does not crash.

  def testWithNonMatchingNameScope(self):
    self._RunTestOverParameters(self._testWithNonMatchingNameScope)

  def _testWithNonMatchingNameScope(self, is_training):
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

    op_names_before_quantize = set([op.name for op in graph.get_operations()])
    quantize.Quantize(
        graph, is_training, weight_bits=8, activation_bits=8,
        scope='NonExisting/')
    op_names_after_quantize = set([op.name for op in graph.get_operations()])

    # No ops should be inserted or removed.
    self.assertEqual(op_names_before_quantize, op_names_after_quantize)

  def testSinglePartitionedVariable(self):
    self._RunTestOverParameters(self._testSinglePartitionedVariable)

  def _testSinglePartitionedVariable(self, is_training):
    # When weights are partitioned into a single partition, the weights variable
    # is followed by a identity -> identity (An additional identity node).
    partitioner = partitioned_variables.fixed_size_partitioner(1)
    graph = ops.Graph()
    with graph.as_default():
      with variable_scope.variable_scope('part', partitioner=partitioner):
        batch_size, height, width, depth = 5, 128, 128, 3
        input1 = array_ops.zeros((batch_size, height, width, depth))
        input2 = array_ops.zeros((batch_size, height / 2, width / 2, 32))
        conv = conv2d(
            input1,
            32, [5, 5],
            stride=2,
            padding='SAME',
            weights_initializer=self._WeightInit(0.09),
            activation_fn=None,
            scope='test/test')
        node = math_ops.add(conv, input2, name='test/add')
        node = nn_ops.relu6(node, name='test/relu6')

      quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)
      # Check that the weight's quant node was added.
      op_names = [op.name for op in graph.get_operations()]
      self.assertTrue(
          'part/test/test/weights_quant/FakeQuantWithMinMaxVars' in op_names)

  def testMultiplePartitionedVariables(self):
    self._RunTestOverParameters(self._testMultiplePartitionedVariables)

  def _testMultiplePartitionedVariables(self, is_training):
    # When weights are partitioned into multiple partitions the weights variable
    # is followed by a identity -> concat -> identity to group the partitions.
    partitioner = partitioned_variables.fixed_size_partitioner(2)
    graph = ops.Graph()
    with graph.as_default():
      with variable_scope.variable_scope('part', partitioner=partitioner):
        batch_size, height, width, depth = 5, 128, 128, 3
        input1 = array_ops.zeros((batch_size, height, width, depth))
        input2 = array_ops.zeros((batch_size, height / 2, width / 2, 32))
        conv = conv2d(
            input1,
            32, [5, 5],
            stride=2,
            padding='SAME',
            weights_initializer=self._WeightInit(0.09),
            activation_fn=None,
            scope='test/test')
        node = math_ops.add(conv, input2, name='test/add')
        node = nn_ops.relu6(node, name='test/relu6')

      quantize.Quantize(graph, is_training, weight_bits=8, activation_bits=8)
      # Check that the weight's quant node was added.
      op_names = [op.name for op in graph.get_operations()]
      self.assertTrue(
          'part/test/test/weights_quant/FakeQuantWithMinMaxVars' in op_names)

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
