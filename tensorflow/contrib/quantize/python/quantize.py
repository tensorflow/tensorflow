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
"""Logic to update a Tensorflow model graph with quantization operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tensorflow.contrib import graph_editor
from tensorflow.contrib.quantize.python import common
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_util

# Operation types used to select operations of interest.
_QUANTIZABLE_TYPES = {'Conv2D', 'MatMul', 'DepthwiseConv2dNative'}

# Custom key for storing and retrieving update ops used by quantizing nodes.
_UPDATE_QUANT_OPS = 'update_quant_ops'


def Quantize(graph,
             weight_bits=8,
             weight_narrow_range=False,
             activation_bits=8,
             ema_decay=0.999,
             quant_delay=None,
             vars_collection=ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
             is_training=True,
             quantize_folded_weights_use_ema=False):
  """Updates graph with quantization operations.

  Args:
    graph: Graph to modify.
    weight_bits: Number of bits to use for quantizing weights.
    weight_narrow_range: Whether to use a more efficient narrow range for
      weights quantization.  With weight_narrow_range true, the range is
      [1; 2^weight_bits - 1], with it false [0; 2^weight_bits - 1].
    activation_bits: Number of bits to use for quantizing activations.
    ema_decay: (Optional) Float, EMA decay parameter.  EMA is used to update
      quantization intervals for quantizing activations (see here about EMA:
      https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average).
    quant_delay: (Optional, default None) Int, count of global steps for which
      to delay quantization.  This helps weights stabilize at the start of
      training.
    vars_collection: (Optional) Collection where to store the variables for
      quantization interval ends.
    is_training: (Optional) Whether quantizing training graph or eval graph.
    quantize_folded_weights_use_ema: (Optional, default False) Whether to
      quantize weights after batchnorm-folding with exponential average
      quantization.
  Raises:
    ValueError: When quantization fails.
  """
  context = _QuantizeContext(graph, weight_bits, weight_narrow_range,
                             activation_bits, ema_decay, quant_delay,
                             vars_collection, is_training,
                             quantize_folded_weights_use_ema)

  graph_ops = graph.get_operations()

  # Filter out backprop and summary related operations, leave only interesting
  # op types.
  def _IsInterestingOpWithWeights(op):
    return (op.type in _QUANTIZABLE_TYPES and
            not op.name.startswith(common.SKIPPED_PREFIXES))

  for op in (op for op in graph_ops if _IsInterestingOpWithWeights(op)):
    if op.name.endswith('/depthwise'):
      # Separable convolution may consist of 2 convolution nodes. If so, skip
      # .../depthwise and only quantize the top one.
      separable_conv = context.GetOperationByNameDontThrow(
          op.name[:-len('/depthwise')])
      if separable_conv and separable_conv.type == 'Conv2D':
        continue
    # Quantize add ops that come after Conv2D or DepthwiseConv2dNative.
    if op.type in ['Conv2D', 'DepthwiseConv2dNative']:
      add_context_re = re.search(r'^(.*)/[^/]+/', op.name)
      if add_context_re is not None:
        context.add_contexts.add(add_context_re.group(1))
    if not op.name.endswith('_Fold'):
      folded_op = context.GetOperationByNameDontThrow(op.name + '_Fold')
      # Do nothing if found, it will be quantized when it is iterated over.
      if not folded_op:
        context.QuantizeOpWithWeights(op, folded=False)
    else:
      context.QuantizeOpWithWeights(op, folded=True)

  context.QuantizeAddContexts()

  # Once all quantization ops have been inserted in the graph, collect update
  # ops for their variables and modify the TF Slim update barrier (see
  # https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/learning.py)
  # to depend on them.
  try:
    update_barrier = graph.get_operation_by_name('update_barrier')
  except KeyError:
    # In evaluation graph, this barrier may not exist.
    return None
  update_quant_ops = graph.get_collection_ref(_UPDATE_QUANT_OPS)
  graph_editor.add_control_inputs(update_barrier, update_quant_ops)


class _QuantizeContext(object):
  """Context holds references needed for quantization."""

  def __init__(self,
               graph,
               weight_bits,
               weight_narrow_range,
               activation_bits,
               ema_decay=0.999,
               quant_delay=None,
               vars_collection=ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
               is_training=True,
               quantize_folded_weights_use_ema=False):
    """Initializes context to hold references needed for quantization.

    Args:
      graph: Graph to modify.
      weight_bits: Number of bits to use for quantizing weights.
      weight_narrow_range: Whether to use a more efficient narrow range for
        weights quantization.  With weight_narrow_range true, the range is
        [1; 2^weight_bits - 1], with it false [0; 2^weight_bits - 1].
      activation_bits: Number of bits to use for quantizing activations.
      ema_decay: (Optional) Float, EMA decay parameter.
      quant_delay: (Optional, default None) Int, count of global steps for which
        to delay quantization.  This helps weights stabilize at the start of
        training.
      vars_collection: (Optional) Collection where to store the variables for
        quantization interval ends.
      is_training: (Optional) Whether quantizing training or eval graph.
      quantize_folded_weights_use_ema: (Optional, default False) Whether to
        quantize weights after batchnorm-folding with exponential average
        quantization.
    """
    self.graph = graph
    self.weight_bits = weight_bits
    self.weight_narrow_range = weight_narrow_range
    self.activation_bits = activation_bits
    self.ema_decay = ema_decay
    self.quant_delay = quant_delay
    self.vars_collection = vars_collection
    self.is_training = is_training
    self.quantize_folded_weights_use_ema = quantize_folded_weights_use_ema
    self.input_to_ops_map = input_to_ops.InputToOps(graph)
    self.add_contexts = set()

  def QuantizeAddContexts(self):
    """Quantizes all add ops in self.add_contexts."""
    # Loop through sorted self.add_contexts so that op creation is
    # deterministic. This is needed when using multiple worker replicas so that
    # the ops can be initialized consistently.
    for add_context in sorted(self.add_contexts):
      add_op = self.GetOperationByNamesDontThrow([
          add_context + '/Add', add_context + '/add'])
      if add_op is not None:
        self._InsertQuantOp(
            add_context,
            add_op,
            self.input_to_ops_map.ConsumerOperations(add_op),
            name='add_quant',
            moving_avg=True,
            bits=self.activation_bits,
            narrow_range=False)

  def QuantizeOpWithWeights(self, op, folded):
    """Quantizes around the specific operation with or without batch norm.

    Args:
      op: Operation to quantize.
      folded: Operation has been folded and needs special handling if True.
    Raises:
      ValueError: When quantization fails.
    """
    # Op name component before the last slash will be used as context.
    context = re.search(r'^(.*)/([^/]+)', op.name).group(1)

    # Quantize weights.
    if folded:
      producer_op = self.graph.get_operation_by_name(context + '/mul_fold')
    else:
      try:
        input_idx = next(i for i, v in enumerate(op.inputs)
                         if '/weights/' in v.name or
                         '/depthwise_weights' in v.name)
      except StopIteration:
        raise ValueError('No inputs to quantize for op: %s' % op)
      producer_op = op.inputs[input_idx].op

    # If batch norm is used, the folded weights depend on the batch std, hence
    # it is sensible to use EMA during training to smooth out the noise. This is
    # controlled by the flag quantize_folded_weights_use_ema. Its default is
    # False for backward compatibility.
    # If there is no batch norm, weights do not depend on the batch and using
    # the latest value of min and max is more efficient.
    weight_use_ema = folded and self.quantize_folded_weights_use_ema
    self._InsertQuantOp(
        context,
        producer_op, [op],
        name='weights_quant',
        moving_avg=weight_use_ema,
        delay_requested=weight_use_ema,
        bits=self.weight_bits,
        narrow_range=self.weight_narrow_range)

    # Important: do not quantize biases here.  During inference they are
    # quantized to 32 bits, which is much finer than 8 bit quantization and
    # depends on weight and input activation ranges.

    # Find activation and (optionally) Add operations to quantize.
    activation_op, add_op, add_context = self._GetReluAndAddOperations(context,
                                                                       op)
    if add_op:
      original_context = context
      context = add_context

    # Quantize activation outputs.
    consumer_ops = self.input_to_ops_map.ConsumerOperations(activation_op)
    self._InsertQuantOp(
        context,
        activation_op,
        consumer_ops,
        name='act_quant',
        moving_avg=True,
        init_min=0.0,
        bits=self.activation_bits,
        narrow_range=False)

    # When a bypass connection was found, also quantize Add op input.
    if add_op:
      def _QuantizeAddInput(add_input):
        if folded:
          return add_input.op.name.endswith('/add_fold')
        else:
          return add_input.op.name.startswith(original_context + '/')

      for add_input in add_op.inputs:
        if _QuantizeAddInput(add_input):
          self._InsertQuantOp(
              original_context,
              add_input.op, [add_op],
              name='conv_quant',
              moving_avg=True,
              bits=self.activation_bits,
              narrow_range=False)

  def _GetReluAndAddOperations(self, context, op):
    """Looks up a Relu* and Add operations in given context.

    Args:
      context: Context where to look for operations.
      op: Operation to quantize.

    Returns:
      A triplet (Operation, Operation, string), the first element is an end
      point operation, the second is Add operation (optional), the third element
      is string context where the Add operation was found (optional).

    Raises:
      ValueError: When operations cannot be found.
    """
    activation_op = common.GetEndpointActivationOp(self.graph, context)
    if activation_op:
      return activation_op, None, None

    if '/' in context:
      # If no activation op is there, look for them one level up.
      add_context = re.search(r'^(.*)/([^/]+)', context).group(1)
      activation_op = common.GetEndpointActivationOp(self.graph, add_context)
    if not activation_op:
      # Still no Relu, can happen on the top layer, just find the next node up,
      # make sure it is BiasAdd.
      consumers = [c for outp in op.outputs for c in outp.consumers()]
      if len(consumers) != 1 or consumers[0].type != 'BiasAdd':
        raise ValueError('Failed to quantize op: %s, %s' % (op.name, op.type))
      return consumers[0], None, None
    if add_context:
      add_op = self.GetOperationByNamesDontThrow([
          add_context + '/Add', add_context + '/add'])
      return activation_op, add_op, add_context
    else:
      raise ValueError('Failed to quantize op: %s, %s' % (op.name, op.type))

  def GetOperationByNameDontThrow(self, name):
    """Returns an Operation with the given name.

    Args:
      name: Name of Operation to return.

    Returns:
      The Operation with the given name. None if the name does not correspond to
      any operation in the graph.
    """
    try:
      return self.graph.get_operation_by_name(name)
    except KeyError:
      return None

  def GetOperationByNamesDontThrow(self, names):
    """Returns an Operation with one of the given names.

    Args:
      names: Names of Operation to return.

    Returns:
      The Operation with one of the given names. None if none of the names
      corresponds to any operation in the graph.
    """
    for name in names:
      op = self.GetOperationByNameDontThrow(name)
      if op is not None:
        return op
    return None

  def _InsertQuantOp(
      self,
      context,
      producer,
      consumers,
      name,
      moving_avg=True,
      init_min=-6.0,
      init_max=6.0,
      delay_requested=True,
      bits=8,
      narrow_range=False,):
    """Inserts a quant op between a producer op and (multiple) consumer ops.

    Args:
      context: Context where producer and consumer operations are nested.
      producer: Producer operation of the pairs where quantization will be
        inserted.
      consumers: Consumer operations of the pairs.
      name: Name for the new quantization op within the context.
      moving_avg: Specifies whether to use exponential moving average or just
        the last value seen.
      init_min: Starting minimum value for the new quantization op.
      init_max: Starting maximum value for the new quantization op.
      delay_requested: If true, implement quantization delay where needed.
        False value explicitly disables delay quantization everywhere.
      bits: Number of bits to use for quantization, must be between 2 and 8.
      narrow_range: Whether to use the narrow quantization range
        [1; 2^bits - 1] or wide range [0; 2^bits - 1].
    Raises:
      ValueError: When producer operation is not directly connected to the
        consumer operation.
    """
    scope = context + '/' + name
    inputs = producer.outputs[0]
    if moving_avg:
      quant = (quant_ops.MovingAvgQuantize(
          inputs,
          init_min=init_min,
          init_max=init_max,
          ema_decay=self.ema_decay,
          is_training=self.is_training,
          num_bits=bits,
          narrow_range=narrow_range,
          updates_collection=_UPDATE_QUANT_OPS,
          vars_collection=self.vars_collection,
          scope=scope))
    else:
      quant = (quant_ops.LastValueQuantize(
          inputs,
          init_min=init_min,
          init_max=init_max,
          is_training=self.is_training,
          num_bits=bits,
          narrow_range=narrow_range,
          updates_collection=_UPDATE_QUANT_OPS,
          vars_collection=self.vars_collection,
          scope=scope))

    if delay_requested and self.quant_delay and self.quant_delay > 0:
      activate_quant = math_ops.greater_equal(
          training_util.get_or_create_global_step(),
          self.quant_delay,
          name=scope + '/activate_quant')
      quant = control_flow_ops.cond(
          activate_quant,
          lambda: quant,
          lambda: inputs,
          name=scope + '/delayed_quant')

    nodes_modified_count = graph_editor.reroute_ts(
        [quant], [inputs], can_modify=consumers)
    if nodes_modified_count != len(consumers):
      raise ValueError('Some inputs not quantized for ops: [%s]' %
                       ', '.join([consumer.name for consumer in consumers]))
