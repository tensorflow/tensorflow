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
"""Logic to update a TensorFlow model graph with quantization operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tensorflow.contrib import graph_editor
from tensorflow.contrib.quantize.python import common
from tensorflow.contrib.quantize.python import graph_matcher
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

# Quantizable operation types that are supported by the quantization rewrite.
_QUANTIZABLE_TYPES = {'Conv2D', 'MatMul', 'DepthwiseConv2dNative'}

# Activations that are supported by the quantization rewrite.
_ACTIVATION_TYPES = {'Relu', 'Relu6'}


def Quantize(graph,
             is_training,
             weight_bits=8,
             activation_bits=8,
             ema_decay=0.999,
             quant_delay=None,
             vars_collection=ops.GraphKeys.GLOBAL_VARIABLES,
             scope=None):
  """Updates graph with quantization operations.

  Currently we quantize the following tensors:
  * Conv/MatMul: Quantize the weights if it matches.
  * Activation: Quantize the output if it matches.
  * Bypass/Post-activation Bypass: Quantize both input and output
    if it matches.

  Args:
    graph: Graph to modify.
    is_training: Whether quantizing training graph or eval graph.
    weight_bits: Number of bits to use for quantizing weights.
    activation_bits: Number of bits to use for quantizing activations.
    ema_decay: (Optional) Float, EMA decay parameter.  EMA is used to update
      quantization intervals for quantizing activations (see here about EMA:
      https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average).
    quant_delay: (Optional, default None) Int, count of global steps for which
      to delay quantization.  This helps weights stabilize at the start of
      training.
    vars_collection: (Optional) Collection where to store the variables for
      quantization interval ends.
    scope: The scope to be transformed. If it's not None, only the ops which
      are in this scope will be transformed.
  Raises:
    ValueError: When quantization fails.
  """
  if scope and not scope.endswith('/'):
    scope += '/'

  input_to_ops_map = input_to_ops.InputToOps(graph)
  for layer_match in _FindLayersToQuantize(graph):
    # Quantize the weights.
    context = _GetContextFromOp(layer_match.layer_op)

    # If `scope` is given, only quantize it if the consumer of weights
    # (the layer op) is in the right scope.
    _InsertQuantOp(
        context,
        'weights_quant',
        layer_match.weight_tensor.op, [layer_match.layer_op],
        is_training,
        moving_avg=False,
        ema_decay=ema_decay,
        quant_delay=quant_delay,
        narrow_range=True,
        vars_collection=vars_collection,
        bits=weight_bits,
        consumer_scope=scope)

    # Quantize the activations.
    consumer_ops = input_to_ops_map.ConsumerOperations(
        layer_match.activation_op)
    add_context = context
    if layer_match.bypass_op:
      add_context = re.search(r'^(.*)/([^/]+)', context).group(1)

    # If `scope` is given, only quantize it if the producer of weights
    # (usually it's the layer op) is in the right scope.
    _InsertQuantOp(
        add_context,
        'act_quant',
        layer_match.activation_op,
        consumer_ops,
        is_training,
        moving_avg=True,
        ema_decay=ema_decay,
        quant_delay=quant_delay,
        vars_collection=vars_collection,
        bits=activation_bits,
        init_min=0.0,
        producer_scope=scope)

    # Quantize the inputs and output to the bypass (if it exists). The input to
    # the bypass is the bias add, and the output is the activation.
    if layer_match.bypass_op is not None:
      # If `scope` is given, only quantize it if the both the producer and the
      # consumer are in the right scope.
      _InsertQuantOp(
          context,
          'conv_quant',
          layer_match.bias_add_op, [layer_match.bypass_op],
          is_training,
          moving_avg=True,
          ema_decay=ema_decay,
          quant_delay=quant_delay,
          vars_collection=vars_collection,
          bits=activation_bits,
          producer_scope=scope,
          consumer_scope=scope)
      # Make sure the op following this isn't an activation. In which case, we
      # shouldn't quantize it, since the activation will be Fused into the
      # Add at inference time.
      consumers = input_to_ops_map.ConsumerOperations(layer_match.bypass_op)
      if any([consumer.type in _ACTIVATION_TYPES for consumer in consumers]):
        logging.info('Skipping %s, because its followed by an activation.',
                     layer_match.bypass_op.name)
      else:
        _InsertQuantOp(
            add_context,
            'add_quant',
            layer_match.bypass_op,
            input_to_ops_map.ConsumerOperations(layer_match.bypass_op),
            is_training,
            moving_avg=True,
            ema_decay=ema_decay,
            quant_delay=quant_delay,
            vars_collection=vars_collection,
            bits=activation_bits,
            producer_scope=scope,
            consumer_scope=scope)

    # Quantize bypass ops that occur after the activation.
    if layer_match.post_activation_bypass_op is not None:
      post_activation_bypass_context = re.search(
          r'^(.*)/([^/]+)', layer_match.post_activation_bypass_op.name).group(1)
      # If `scope` is given, only quantize it if the producer is in the right
      # scope.
      # Make sure the op following this isn't an activation. In which case, we
      # shouldn't quantize it, since the activation will be Fused into the
      # Add at inference time.
      consumers = input_to_ops_map.ConsumerOperations(
          layer_match.post_activation_bypass_op)
      if any([consumer.type in _ACTIVATION_TYPES for consumer in consumers]):
        logging.info('Skipping %s, because its followed by an activation.',
                     layer_match.post_activation_bypass_op.name)
      else:
        _InsertQuantOp(
            post_activation_bypass_context,
            'post_activation_bypass_quant',
            layer_match.post_activation_bypass_op,
            consumers,
            is_training,
            moving_avg=True,
            ema_decay=ema_decay,
            quant_delay=quant_delay,
            vars_collection=vars_collection,
            bits=activation_bits,
            producer_scope=scope)


def _FindLayersToQuantize(graph):
  """Matches layers in graph to quantize.

  The following patterns get matched. Nodes surrounded by [] will be
  optionally matched:

          weight|folded_weight
                /
         conv|fc
            |
      [batch_to_space_nd]
            |
    [post_conv_correction]
            |
     [biasadd|folded_bias]
            |
         [bypass]
            |
        activation
            |
   [post_activation_bypass]

  Match replacements:
    If weight|folded_weight is found, FakeQuant is added afterwards.
    If bypass is found, FakeQuant is added before and after.
    If activation is found, FakeQuant is added afterwards.
    If post_activation_bypass is found, FakeQuant is added afterwards.

  Args:
    graph: Graph to perform match on.

  Returns:
    list of _LayerMatches.
  """
  input_pattern = graph_matcher.OpTypePattern('*')
  weight_var_pattern = graph_matcher.OpTypePattern('Variable|VariableV2')
  weight_partition_identity_pattern = graph_matcher.OpTypePattern(
      'Identity', inputs=[weight_var_pattern])
  weight_partition_concat_pattern = graph_matcher.OpTypePattern(
      'ConcatV2', inputs=[weight_partition_identity_pattern, '*', '*'])
  weight_identity_pattern = graph_matcher.OpTypePattern(
      'Identity',
      inputs=[
          graph_matcher.OneofPattern([
              weight_partition_identity_pattern,
              weight_partition_concat_pattern,
              weight_var_pattern,
          ])
      ])
  weight_resource_var_pattern = graph_matcher.OpTypePattern('ReadVariableOp')
  folded_weight_pattern = graph_matcher.OpTypePattern('Mul')

  # The weights inputs to the layer operation can either be from the Variable or
  # the folded weight (Mul).
  layer_pattern = graph_matcher.OpTypePattern(
      '|'.join(_QUANTIZABLE_TYPES),
      inputs=[
          input_pattern,
          graph_matcher.OneofPattern([
              weight_identity_pattern, weight_resource_var_pattern,
              folded_weight_pattern
          ])
      ],
      ordered_inputs=False)

  # For atrous convolutions a BatchToSpaceND will occur after the depthwise
  # convolution.
  batch_to_space_pattern = graph_matcher.OpTypePattern(
      'BatchToSpaceND',
      inputs=[
          layer_pattern,
          graph_matcher.OpTypePattern('*'),
          graph_matcher.OpTypePattern('*')
      ])

  layer_output_pattern = graph_matcher.OneofPattern(
      [batch_to_space_pattern, layer_pattern])

  # For separable convolutions, we are looking for a conv, followed by a conv
  # with no activations between the two.
  sep_conv_pattern = graph_matcher.OpTypePattern(
      '|'.join(_QUANTIZABLE_TYPES),
      inputs=[
          graph_matcher.OneofPattern([layer_output_pattern]),
          graph_matcher.OpTypePattern('*')
      ],
      ordered_inputs=False)
  folded_bias_mul_pattern = graph_matcher.OpTypePattern(
      'Mul',
      inputs=[graph_matcher.OpTypePattern('*'), layer_output_pattern],
      ordered_inputs=False)
  post_layer_op_correction_pattern = graph_matcher.OpTypePattern(
      'Add',
      inputs=[folded_bias_mul_pattern,
              graph_matcher.OpTypePattern('*')],
      ordered_inputs=False)
  folded_bias_add_pattern = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          post_layer_op_correction_pattern,
          graph_matcher.OpTypePattern('*')
      ],
      ordered_inputs=False)

  # batch_norms with forced updates have an Identity operation at the end.
  # TODO(suharshs): Find a way to easily skip extra Identity operations. The
  # current issue is that doing so can often match patterns across many layers
  # incorrectly.
  batch_norm_identity = graph_matcher.OpTypePattern(
      'Identity', inputs=[folded_bias_add_pattern])

  bias_add_pattern = graph_matcher.OpTypePattern(
      'Add|BiasAdd', inputs=[layer_output_pattern, '*'], ordered_inputs=False)

  # The bias can come from the bias add or the folded bias add.
  bypass_pattern = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          graph_matcher.OneofPattern(
              [bias_add_pattern, folded_bias_add_pattern, batch_norm_identity]),
          '*'
      ],
      ordered_inputs=False)

  # The input to the activation can come from bias add, fold bias add, the
  # bypasses.
  # TODO(suharshs): We should ideally skip Identity operations instead of
  # treating them as activations.
  activation_pattern = graph_matcher.OpTypePattern(
      '|'.join(_ACTIVATION_TYPES) + '|Identity',
      inputs=[
          graph_matcher.OneofPattern([
              bias_add_pattern,
              folded_bias_add_pattern,
              batch_norm_identity,
              bypass_pattern,
              layer_pattern,
          ])
      ])

  post_activation_bypass_pattern = graph_matcher.OpTypePattern(
      'Add', inputs=['*', activation_pattern], ordered_inputs=False)

  # The order of the following matching blocks is very important. Since matches
  # aren't guaranteed to be disjoint, we structure matches from largest to
  # smallest to guarantee that the largest match always wins. Additionally, we
  # ensure that we don't match layers multiple times.

  layer_matches = []
  # We use matched_layer_set to ensure that layers aren't matched multiple
  # times.
  matched_layer_set = set()

  # First, we match layers that have a post activation bypass. We do this first
  # to ensure we don't match only the first part of this layer, missing the
  # post activation bypass node.
  post_activation_bypass_layer_matcher = graph_matcher.GraphMatcher(
      post_activation_bypass_pattern)
  for match_result in post_activation_bypass_layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    activation_op = match_result.get_op(activation_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    if bias_add_op is None:
      bias_add_op = match_result.get_op(folded_bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern)
    post_activation_bypass_op = match_result.get_op(
        post_activation_bypass_pattern)
    if layer_op not in matched_layer_set:
      matched_layer_set.add(layer_op)
      layer_matches.append(
          _LayerMatch(layer_op, weight_tensor, activation_op, bypass_op,
                      post_activation_bypass_op, bias_add_op))

  # Now, we match the basic layer ending at an activation. We may get duplicate
  # matches from above, but we don't add them to layer_matches.
  layer_matcher = graph_matcher.GraphMatcher(activation_pattern)
  for match_result in layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    activation_op = match_result.get_op(activation_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    if bias_add_op is None:
      bias_add_op = match_result.get_op(folded_bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern)
    if layer_op not in matched_layer_set:
      matched_layer_set.add(layer_op)
      layer_matches.append(
          _LayerMatch(layer_op, weight_tensor, activation_op, bypass_op, None,
                      bias_add_op))

  # Match the final layer, where there may not be an activation and instead
  # the output of the final BiasAdd must be quantized. So we treat the BiasAdd
  # as the 'activation_op' in the _LayerMatch, to ensure that it's output is
  # quantized.
  final_layer_matcher = graph_matcher.GraphMatcher(
      graph_matcher.OneofPattern([bias_add_pattern, folded_bias_add_pattern]))
  for match_result in final_layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    activation_op = match_result.get_op(bias_add_pattern)
    if activation_op is None:
      activation_op = match_result.get_op(folded_bias_add_pattern)
    if layer_op not in matched_layer_set:
      matched_layer_set.add(layer_op)
      layer_matches.append(
          _LayerMatch(layer_op, weight_tensor, activation_op, None, None, None))

  # Look for separable convolutions here
  sep_conv_matcher = graph_matcher.GraphMatcher(sep_conv_pattern)
  for match_result in sep_conv_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    activation_op = match_result.get_op(layer_pattern)
    if layer_op not in matched_layer_set:
      matched_layer_set.add(layer_op)
      layer_matches.append(
          _LayerMatch(layer_op, weight_tensor, activation_op, None, None, None))

  return layer_matches


class _LayerMatch(object):
  """Contains all information related to a matched Layer."""

  def __init__(self, layer_op, weight_tensor, activation_op, bypass_op,
               post_activation_bypass_op, bias_add_op):
    self._layer_op = layer_op
    self._weight_tensor = weight_tensor
    self._activation_op = activation_op
    self._bypass_op = bypass_op
    self._post_activation_bypass_op = post_activation_bypass_op
    self._bias_add_op = bias_add_op

  @property
  def layer_op(self):
    return self._layer_op

  @property
  def weight_tensor(self):
    return self._weight_tensor

  @property
  def activation_op(self):
    return self._activation_op

  @property
  def bypass_op(self):
    return self._bypass_op

  @property
  def post_activation_bypass_op(self):
    return self._post_activation_bypass_op

  @property
  def bias_add_op(self):
    return self._bias_add_op


def _InsertQuantOp(context,
                   name,
                   producer,
                   consumers,
                   is_training,
                   moving_avg=True,
                   init_min=-6.0,
                   init_max=6.0,
                   bits=8,
                   ema_decay=0.999,
                   quant_delay=None,
                   vars_collection=ops.GraphKeys.GLOBAL_VARIABLES,
                   narrow_range=False,
                   producer_scope=None,
                   consumer_scope=None):
  """Inserts a quant op between a producer op and (multiple) consumer ops.

  Args:
    context: Context where producer and consumer operations are nested.
    name: Name for the new quantization op within the context.
    producer: Producer operation of the pairs where quantization will be
      inserted.
    consumers: Consumer operations of the pairs.
    is_training: Whether quantizing training graph or eval graph.
    moving_avg: Specifies whether to use exponential moving average or just
      the last value seen.
    init_min: Starting minimum value for the new quantization op.
    init_max: Starting maximum value for the new quantization op.
    bits: Number of bits to use for quantization, must be between 2 and 8.
    ema_decay: (Optional) Float, EMA decay parameter.  EMA is used to update
      quantization intervals for quantizing activations (see here about EMA:
      https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average).
    quant_delay: (Optional, default None) Int, count of global steps for which
      to delay quantization.  This helps weights stabilize at the start of
      training.
    vars_collection: (Optional) Collection where to store the variables for
      quantization interval ends.
    narrow_range: Whether to use the narrow quantization range
      [1; 2^bits - 1] or wide range [0; 2^bits - 1].
    producer_scope: The restriction of producer scope. If not None, the new op
      will be inserted only when the producer is in this scope.
    consumer_scope: The restriction of producer scope. If not None, the new op
      will be inserted only when all the consumers are in this scope.
  Raises:
    ValueError: When producer operation is not directly connected to the
      consumer operation.
  """
  if producer_scope and not producer.name.startswith(producer_scope):
    logging.info(
        '_InsertQuantOp ignores context="%s" name="%s" '
        'because producer "%s" is not in scope "%s"',
        context, name, producer.name, producer_scope)
    return

  if consumer_scope:
    consumers_in_scope = []
    for consumer in consumers:
      if consumer.name.startswith(consumer_scope):
        consumers_in_scope.append(consumer)
      else:
        logging.info(
            '_InsertQuantOp context="%s" name="%s" ignores '
            'consumer "%s" because it is not in scope "%s"',
            context, name, consumer.name, consumer_scope)
        return
    consumers = consumers_in_scope

  name_prefix = _AddContextToName(context, name)
  # This is needed on TPU where name_scope == 'TPUReplicate/loop', and
  # name_prefix starts with 'TPUReplicate/loop/'; without dropping it
  # variables are created as TPUReplicate/loop/TPUReplicate/loop/..., which
  # breaks things later.
  name_scope = ops.get_name_scope()
  if name_scope:
    name_prefix = common.DropStringPrefix(name_prefix, name_scope + '/')

  inputs = producer.outputs[0]
  # Prevent ops from being quantized multiple times. Bypass ops can sometimes
  # overlap between multiple matches, so we need to ensure that we don't
  # add duplicate FakeQuant operations.
  fake_quant_ops = set([
      'FakeQuantWithMinMaxVars',
      'FakeQuantWithMinMaxArgs'
  ])
  if fake_quant_ops.intersection(set([c.type for c in inputs.consumers()])):
    return

  if moving_avg:
    quant = (
        quant_ops.MovingAvgQuantize(
            inputs,
            init_min=init_min,
            init_max=init_max,
            ema_decay=ema_decay,
            is_training=is_training,
            num_bits=bits,
            narrow_range=narrow_range,
            vars_collection=vars_collection,
            name_prefix=name_prefix))
  else:
    quant = (
        quant_ops.LastValueQuantize(
            inputs,
            init_min=init_min,
            init_max=init_max,
            is_training=is_training,
            num_bits=bits,
            narrow_range=narrow_range,
            vars_collection=vars_collection,
            name_prefix=name_prefix))

  if quant_delay and quant_delay > 0:
    activate_quant = math_ops.greater_equal(
        common.CreateOrGetQuantizationStep(),
        quant_delay,
        name=name_prefix + '/activate_quant')
    quant = control_flow_ops.cond(
        activate_quant,
        lambda: quant,
        lambda: inputs,
        name=name_prefix + '/delayed_quant')

  if consumers:
    tensors_modified_count = graph_editor.reroute_ts(
        [quant], [inputs], can_modify=consumers)
    # Some operations can have multiple output tensors going to the same
    # consumer. Since consumers is a set, we need to ensure that
    # tensors_modified_count is greater than or equal to the length of the set
    # of consumers.
    if tensors_modified_count < len(consumers):
      raise ValueError('No inputs quantized for ops: [%s]' % ', '.join(
          [consumer.name for consumer in consumers]))


def _GetContextFromOp(op):
  """Gets the root context name from the op name."""
  context_re = re.search(r'^(.*)/([^/]+)', op.name)
  if context_re:
    return context_re.group(1)
  return ''


def _AddContextToName(context, name):
  """Adds the context to the name if it exists."""
  if not context:
    return name
  return context + '/' + name
