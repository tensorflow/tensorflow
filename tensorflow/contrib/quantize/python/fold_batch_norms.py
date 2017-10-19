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
"""Logic to fold batch norm into preceding convolution or FC layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tensorflow.contrib import graph_editor
from tensorflow.contrib.quantize.python import common
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


def FoldBatchNorms(graph):
  """Finds batch norm layers in the graph, folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.

  Raises:
    ValueError: When batch norm folding fails.
  """
  # Fail immediately when the graph contains unsupported fused batch norm ops.
  if any(op for op in graph.get_operations() if op.type == 'FusedBatchNorm'):
    raise ValueError('Fused batch norm is not supported')

  input_to_ops_map = input_to_ops.InputToOps(graph)

  for bn in common.BatchNormGroups(graph):
    has_scaling = _HasScaling(graph, input_to_ops_map, bn)

    # The mangling code intimately depends on BatchNorm node's internals.
    original_op, folded_op = _CreateFoldedOp(graph, bn, has_scaling=has_scaling)

    activation = common.GetEndpointActivationOp(graph, bn)
    if activation:
      nodes_modified_count = graph_editor.reroute_ts([folded_op.outputs[0]],
                                                     [original_op.outputs[0]],
                                                     can_modify=[activation])
      if nodes_modified_count != 1:
        raise ValueError('Unexpected inputs to op: %s' % activation.name)
      continue

    # Treat consumer ops in bypass modules differently since they have Add
    # operations instead of Relu* above.
    add_bypass_ctx = re.search(r'^(.*)/([^/]+)', bn).group(1)
    add_bypass = graph.get_operation_by_name(add_bypass_ctx + '/Add')
    nodes_modified_count = graph_editor.reroute_ts([folded_op.outputs[0]],
                                                   [original_op.outputs[0]],
                                                   can_modify=[add_bypass])
    if nodes_modified_count != 1:
      raise ValueError('Unexpected inputs to op: %s' % add_bypass.name)


def _HasScaling(graph, input_to_ops_map, bn):
  r"""Checks if batch norm  has scaling enabled.

  Difference between batch norm with scaling and without is that with scaling:

  Rsqrt -> mul -> mul_1
              \-> mul_2

  where
    mul multiplies gamma by inverse square root of EMA of batch variance,
    mul_1 multiplies output of mul with output from the base operation
      (convolution, FC or depthwise convolution),
    mul_2 multiplies output of mul with EMA of batch mean,
  and without scaling:

  Rsqrt -> mul
       \-> mul_1

  where
    mul multiplies the inverse square root of EMA of batch variance with output
      from the base operation,
    mul_1 multiplies inverse square root of EMA of batch variance with EMA
      of batch mean.

  Args:
    graph: Graph to inspect.
    input_to_ops_map: InputToOps object containing mapping from tensor's name
      to ops that take it as input.
    bn: Batch norm layer prefix string.

  Returns:
    A boolean indicating whether this batch norm layer has scaling enabled.
  """
  rsqrt_op = graph.get_operation_by_name(bn + '/BatchNorm/batchnorm/Rsqrt')
  rsqrt_consumers = input_to_ops_map.ConsumerOperations(rsqrt_op)

  return sum(1 for op in rsqrt_consumers if op.type == 'Mul') == 1


def _CreateFoldedOp(graph, context, has_scaling):
  """Folds in batch norm layer into preceding convolution or FC layer.

  Creates 3 new nodes, connects their inputs and adds them to the graph:
  mul is cloned into mul_fold, Conv2D or MatMul, or DepthwiseConv2d is cloned
  into respective *_Fold, add is cloned into add_fold.

  Args:
    graph: Graph to modify.
    context: String, batch norm context, i.e. node into which BatchNorm is
        nested.
    has_scaling: Whether the batch norm has scaling enabled.

  Raises:
    ValueError: When operation type is not supported, or input and output tensor
        shapes mismatch for created operations: mul_fold, add_fold.

  Returns:
    A pair of Operations, the first is the original consumer node of the batch
        norm (../BatchNorm/batchnorm/add_1), the second is the consumer node of
        the folded graph (add_fold).
  """
  mul_scale_name = 'mul_1' if has_scaling else 'mul'
  mul_scale = graph.get_operation_by_name(context +
                                          '/BatchNorm/batchnorm/' +
                                          mul_scale_name)
  op_below = mul_scale.inputs[0].op
  weights = op_below.inputs[1]

  # Special handling for weights of depthwise convolution.
  if op_below.type == 'DepthwiseConv2dNative':
    new_shape = [weights.get_shape().as_list()[2],
                 weights.get_shape().as_list()[3]]
    scale_name = 'mul' if has_scaling else 'Rsqrt'
    scale = graph.get_operation_by_name(context + '/BatchNorm/batchnorm/' +
                                        scale_name)
    scale = array_ops.reshape(scale.outputs[0], new_shape,
                              context + '/scale_reshape')
    mul_fold = _CloneOp(mul_scale, context + '/mul_fold',
                        [(0, weights), (1, scale)])
  elif op_below.type in ['Conv2D', 'MatMul']:
    mul_fold = _CloneOp(mul_scale, context + '/mul_fold', [(0, weights)])
  else:
    raise ValueError('Cannot handle operation of type: %s' % op_below.op)
  _AssertShapesMatch('mul_fold', mul_fold.inputs[0], mul_fold.outputs[0])

  conv_or_fc_folded = _CloneOp(op_below, op_below.name + '_Fold',
                               [(1, mul_fold.outputs[0])])

  add_shift = graph.get_operation_by_name(context +
                                          '/BatchNorm/batchnorm/add_1')
  add_fold = _CloneOp(add_shift, context + '/add_fold',
                      [(0, conv_or_fc_folded.outputs[0])])
  _AssertShapesMatch('add_fold', add_fold.inputs[0], add_fold.outputs[0])
  return add_shift, add_fold


def _CloneOp(op, new_name, new_inputs):
  """Clones a given op, replaces its name and some of its inputs.

  Args:
    op: Operation to modify.
    new_name: String, a new name to set on cloned op.
    new_inputs: A list of tuples (idx, tensor), each input with corresponding
        index will be replaced by the given Tensor in the cloned op.

  Returns:
    Operation, the cloned op.

  Raises:
    TypeError: When Operation type is not supported.
    ValueError: When input shapes are incompatible.
  """
  inputs = list(op.inputs)
  for new_input in new_inputs:
    inputs[new_input[0]] = new_input[1]
  return _OP_CLONER.Clone(op, inputs, new_name)


class _OpCloner(object):
  """Helper class that clones tf.Operations based on their type."""

  def __init__(self):
    self.op_type_to_action = {
        'Mul': self._CloneMul,
        'Add': self._CloneAdd,
        'Conv2D': self._CloneConv2d,
        'DepthwiseConv2dNative': self._CloneDepthwiseConv2d,
        'MatMul': self._CloneMatMul,
    }

  def _CloneMul(self, op, inputs, new_name):
    del op  # Unused.
    return math_ops.multiply(inputs[0], inputs[1], name=new_name).op

  def _CloneAdd(self, op, inputs, new_name):
    del op  # Unused.
    return math_ops.add(inputs[0], inputs[1], name=new_name).op

  def _CloneConv2d(self, op, inputs, new_name):
    input_tensor = inputs[0]
    weights = inputs[1]
    self._AssertConvShapes(op.name, input_tensor, weights)
    return nn_ops.conv2d(
        input_tensor,
        weights,
        strides=op.get_attr('strides'),
        padding=op.get_attr('padding'),
        use_cudnn_on_gpu=op.get_attr('use_cudnn_on_gpu'),
        data_format=op.get_attr('data_format'),
        name=new_name).op

  def _CloneDepthwiseConv2d(self, op, inputs, new_name):
    input_tensor = inputs[0]
    weights = inputs[1]
    self._AssertConvShapes(op.name, input_tensor, weights)
    return nn.depthwise_conv2d(
        input_tensor,
        weights,
        strides=op.get_attr('strides'),
        padding=op.get_attr('padding'),
        name=new_name).op

  def _CloneMatMul(self, op, inputs, new_name):
    weights = inputs[0]
    input_tensor = inputs[1]
    self._AssertFCShapes(op.name, weights, input_tensor)
    return math_ops.matmul(
        weights,
        input_tensor,
        transpose_a=op.get_attr('transpose_a'),
        transpose_b=op.get_attr('transpose_b'),
        name=new_name).op

  def Clone(self, op, inputs, new_name):
    try:
      return self.op_type_to_action[op.type](op, inputs, new_name)
    except KeyError:
      raise TypeError('Unsupported operation type: %s' % op.type)

  def _AssertConvShapes(self, op_name, input_tensor, weights):
    """Makes sure that convolution inputs have compatible shapes.

    Args:
      op_name: Operation name, only used in error message.
      input_tensor: Input that is convolved.
      weights: Weights of the convolution filter.

    Raises:
      ValueError: When input shapes are incompatible.
    """
    input_shape = input_tensor.get_shape()
    weights_shape = weights.get_shape()
    if (len(input_shape) != 4 or len(weights_shape) != 4 or
        input_shape[3] != weights_shape[2]):
      raise ValueError('Incompatible shapes for op %s inputs: %s and %s' %
                       (op_name, input_shape, weights_shape))

  def _AssertFCShapes(self, op_name, weights, input_tensor):
    """Makes sure that FC layer inputs have compatible shapes.

    Args:
      op_name: Operation name, only used in error message.
      weights: Weights used in FC layer.
      input_tensor: Input into FC layer.

    Raises:
      ValueError: When input shapes are incompatible.
    """
    weights_shape = weights.get_shape()
    input_shape = input_tensor.get_shape()
    if (len(weights_shape) != 2 or len(input_shape) != 2 or
        weights_shape[1] != input_shape[0]):
      raise ValueError('Incompatible shapes for op %s inputs: %s and %s' %
                       (op_name, weights_shape, input_shape))

_OP_CLONER = _OpCloner()


def _AssertShapesMatch(op_name, in_tensor, out_tensor):
  """Makes sure that shapes of input and output tensors are compatible.

  Args:
    op_name: String, operation name, only used in error message.
    in_tensor: Tensor, input tensor.
    out_tensor: Tensor, output tensor.

  Raises:
    ValueError: When input and output tensors have different shapes.
  """
  in_shape = in_tensor.get_shape()
  out_shape = out_tensor.get_shape()

  if not in_shape.is_compatible_with(out_shape):
    raise ValueError('%s should not change tensor shape: input %s, '
                     'output %s' % (op_name, in_shape, out_shape))
