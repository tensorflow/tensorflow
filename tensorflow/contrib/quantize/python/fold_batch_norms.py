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
from tensorflow.contrib.quantize.python import graph_matcher
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


def FoldBatchNorms(graph):
  """Finds batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.

  Raises:
    ValueError: When batch norm folding fails.
  """
  _FoldFusedBatchNorms(graph)
  _FoldUnfusedBatchNorms(graph)


def _FoldFusedBatchNorms(graph):
  """Finds fused batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.

  Raises:
    ValueError: When batch norm folding fails.
  """
  for match in _FindFusedBatchNorms(graph):
    scope, sep, _ = match.layer_op.name.rpartition('/')
    # Make sure new ops are added to `graph` and put on the same device as
    # `bn_op`. The '/' (i.e. `sep`) ensures that we reuse the existing scope
    # named `scope`. Otherwise, TF creates a unique scope whose name starts with
    # `scope`.
    with graph.as_default(), graph.name_scope(scope + sep), ops.device(
        match.bn_op.device):
      with graph.name_scope(scope + sep + 'BatchNorm_Fold' + sep):
        # new weights = old weights * gamma / sqrt(variance + epsilon)
        # new biases = -mean * gamma / sqrt(variance + epsilon) + beta
        multiplier_tensor = match.gamma_tensor * math_ops.rsqrt(
            match.variance_tensor + match.bn_op.get_attr('epsilon'))
        bias_tensor = math_ops.subtract(
            match.beta_tensor,
            match.mean_tensor * multiplier_tensor,
            name='bias')

        # The shape of depthwise weights is different, so we need to reshape the
        # multiplier_tensor to ensure that the scaled_weight_tensor has the
        # expected shape.
        if match.layer_op.type == 'DepthwiseConv2dNative':
          new_shape = [
              match.weight_tensor.get_shape().as_list()[2],
              match.weight_tensor.get_shape().as_list()[3]
          ]
          multiplier_tensor = array_ops.reshape(
              multiplier_tensor, new_shape, name='scale_reshape')

      # TODO(suharshs): This naming of the following ops needs to carefully
      # follow the naming expected by quantize.py. Generalize the quantize code
      # to not require these delicate naming conventions.
      scaled_weight_tensor = math_ops.multiply(
          match.weight_tensor, multiplier_tensor, name='mul_fold')

      new_layer_tensor = _CloneWithNewOperands(
          match.layer_op, match.input_tensor, scaled_weight_tensor)

      bias_add_tensor = math_ops.add(
          new_layer_tensor, bias_tensor, name='add_fold')

      nodes_modified_count = graph_editor.reroute_ts(bias_add_tensor,
                                                     match.output_tensor)
      if nodes_modified_count != 1:
        raise ValueError(
            'Unexpected inputs to op: %s' % match.output_tensor.name)


def _CloneWithNewOperands(layer_op, input_tensor, weight_tensor):
  """Clones layer_op with input_tensor and weight_tensor as new inputs."""
  new_layer_name = layer_op.name.split('/')[-1] + '_Fold'
  if layer_op.type == 'Conv2D':
    return nn_ops.conv2d(
        input_tensor,
        weight_tensor,
        strides=layer_op.get_attr('strides'),
        padding=layer_op.get_attr('padding'),
        use_cudnn_on_gpu=layer_op.get_attr('use_cudnn_on_gpu'),
        data_format=layer_op.get_attr('data_format'),
        name=new_layer_name)
  elif layer_op.type == 'MatMul':
    return math_ops.matmul(
        input_tensor,
        weight_tensor,
        transpose_a=layer_op.get_attr('transpose_a'),
        transpose_b=layer_op.get_attr('transpose_b'),
        name=new_layer_name)
  elif layer_op.type == 'DepthwiseConv2dNative':
    return nn.depthwise_conv2d(
        input_tensor,
        weight_tensor,
        strides=layer_op.get_attr('strides'),
        padding=layer_op.get_attr('padding'),
        name=new_layer_name)
  else:
    raise ValueError('Cannot handle operation of type: %s' % layer_op.type)


def _FindFusedBatchNorms(graph):
  """Finds all ops and tensors related to found FusedBatchNorms.

  Args:
    graph: Graph to inspect.

  Yields:
    _FusedBatchNormMatches.
  """
  input_pattern = graph_matcher.OpTypePattern('*')
  weight_pattern = graph_matcher.OpTypePattern('*')
  gamma_pattern = graph_matcher.OpTypePattern('*')
  beta_pattern = graph_matcher.OpTypePattern('*')
  mean_pattern = graph_matcher.OpTypePattern('*')
  variance_pattern = graph_matcher.OpTypePattern('*')

  conv_pattern = graph_matcher.OpTypePattern(
      'Conv2D|DepthwiseConv2dNative', inputs=[input_pattern, weight_pattern])
  # MatMul has a Reshape between it and FusedBatchNorm.
  matmul_pattern = graph_matcher.OpTypePattern(
      'MatMul', inputs=[input_pattern, weight_pattern])
  matmul_reshape_pattern = graph_matcher.OpTypePattern(
      'Reshape', inputs=[matmul_pattern,
                         graph_matcher.OpTypePattern('*')])

  conv_batch_norm_pattern = graph_matcher.OpTypePattern(
      'FusedBatchNorm',
      inputs=[
          conv_pattern, gamma_pattern, beta_pattern, mean_pattern,
          variance_pattern
      ])
  matmul_batch_norm_pattern = graph_matcher.OpTypePattern(
      'FusedBatchNorm',
      inputs=[
          matmul_reshape_pattern, gamma_pattern, beta_pattern, mean_pattern,
          variance_pattern
      ])
  matmul_bn_output_reshape_pattern = graph_matcher.OpTypePattern(
      'Reshape',
      inputs=[matmul_batch_norm_pattern,
              graph_matcher.OpTypePattern('*')])

  conv_matcher = graph_matcher.GraphMatcher(conv_batch_norm_pattern)
  matmul_matcher = graph_matcher.GraphMatcher(matmul_bn_output_reshape_pattern)

  def _GetCommonTensors(match_result):
    """Gets tensors needed for FusedBatchNormMatch from match_result."""
    input_tensor = match_result.get_tensor(input_pattern)
    weight_tensor = match_result.get_tensor(weight_pattern)
    gamma_tensor = match_result.get_tensor(gamma_pattern)
    beta_tensor = match_result.get_tensor(beta_pattern)
    # FusedBatchNorm in training is different from that in inference. It takes
    # empty 'mean' and empty 'variance', and produces the mean and the variance
    # of the batch. Therefore, when is_training is true, mean_tensor and
    # variance_tensor point to 1st and 2nd (0-based) output of bn_op,
    # respectively; when is_training is false, they point to bn_op's inputs.
    is_training = bn_op.get_attr('is_training')
    if is_training:
      mean_tensor = bn_op.outputs[1]
      variance_tensor = bn_op.outputs[2]
    else:
      mean_tensor = match_result.get_tensor(mean_pattern)
      variance_tensor = match_result.get_tensor(variance_pattern)
    return (input_tensor, weight_tensor, gamma_tensor, beta_tensor, mean_tensor,
            variance_tensor)

  for match_result in conv_matcher.match_graph(graph):
    layer_op = match_result.get_op(conv_pattern)
    bn_op = match_result.get_op(conv_batch_norm_pattern)
    # In the case of convolution the output_tensor is the output of bn_op.
    output_tensor = bn_op.outputs[0]

    (input_tensor, weight_tensor, gamma_tensor, beta_tensor, mean_tensor,
     variance_tensor) = _GetCommonTensors(match_result)
    yield _FusedBatchNormMatch(
        layer_op=layer_op,
        bn_op=bn_op,
        output_tensor=output_tensor,
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        gamma_tensor=gamma_tensor,
        beta_tensor=beta_tensor,
        mean_tensor=mean_tensor,
        variance_tensor=variance_tensor)

  for match_result in matmul_matcher.match_graph(graph):
    layer_op = match_result.get_op(matmul_pattern)
    bn_op = match_result.get_op(matmul_batch_norm_pattern)
    # In the MatMul case, the output of batch norm is reshaped back into a
    # 2D tensor, so the output_tensor is the output of the Reshape op.
    output_reshape_op = match_result.get_op(matmul_bn_output_reshape_pattern)
    output_tensor = output_reshape_op.outputs[0]

    (input_tensor, weight_tensor, gamma_tensor, beta_tensor, mean_tensor,
     variance_tensor) = _GetCommonTensors(match_result)
    yield _FusedBatchNormMatch(
        layer_op=layer_op,
        bn_op=bn_op,
        output_tensor=output_tensor,
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        gamma_tensor=gamma_tensor,
        beta_tensor=beta_tensor,
        mean_tensor=mean_tensor,
        variance_tensor=variance_tensor)


class _FusedBatchNormMatch(object):
  """Contains all information related to a found FusedBatchNorm."""

  def __init__(self, layer_op, bn_op, output_tensor, input_tensor,
               weight_tensor, gamma_tensor, beta_tensor, mean_tensor,
               variance_tensor):
    self._layer_op = layer_op
    self._bn_op = bn_op
    self._output_tensor = output_tensor
    self._input_tensor = input_tensor
    self._weight_tensor = weight_tensor
    self._gamma_tensor = gamma_tensor
    self._beta_tensor = beta_tensor
    self._mean_tensor = mean_tensor
    self._variance_tensor = variance_tensor

  @property
  def layer_op(self):
    return self._layer_op

  @property
  def bn_op(self):
    return self._bn_op

  @property
  def output_tensor(self):
    return self._output_tensor

  @property
  def input_tensor(self):
    return self._input_tensor

  @property
  def weight_tensor(self):
    return self._weight_tensor

  @property
  def gamma_tensor(self):
    return self._gamma_tensor

  @property
  def beta_tensor(self):
    return self._beta_tensor

  @property
  def mean_tensor(self):
    return self._mean_tensor

  @property
  def variance_tensor(self):
    return self._variance_tensor


def _FoldUnfusedBatchNorms(graph):
  """Finds unfused batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.

  Raises:
    ValueError: When batch norm folding fails.
  """
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
