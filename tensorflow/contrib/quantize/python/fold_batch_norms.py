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
from tensorflow.contrib.quantize.python import common
from tensorflow.contrib.quantize.python import graph_matcher
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import compat


def FoldBatchNorms(graph, is_training, freeze_batch_norm_delay=None):
  """Finds batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.
    is_training: Bool, true if training.
    freeze_batch_norm_delay: How many steps to wait before freezing moving mean
      and variance and using them for batch normalization. This value is used
      only when is_training is True.
  Raises:
    ValueError: When batch norm folding fails.
  """
  _FoldFusedBatchNorms(
      graph, is_training, freeze_batch_norm_delay=freeze_batch_norm_delay)
  _FoldUnfusedBatchNorms(
      graph,
      is_training=is_training,
      freeze_batch_norm_delay=freeze_batch_norm_delay)


def _FoldFusedBatchNorms(graph, is_training, freeze_batch_norm_delay):
  """Finds fused batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.
    is_training: Bool, true if training.
    freeze_batch_norm_delay: How many steps to wait before freezing moving mean
      and variance and using them for batch normalization.

  Raises:
    ValueError: When batch norm folding fails.
  """
  for match in _FindFusedBatchNorms(graph):
    scope, sep, _ = match.layer_op.name.rpartition('/')
    # Make sure new ops are added to `graph` and put on the same device as
    # `bn_op`. The '/' (i.e. `sep`) ensures that we reuse the existing scope
    # named `scope`. Otherwise, TF creates a unique scope whose name starts with
    # `scope`.
    with graph.as_default(), graph.name_scope(scope + sep):
      with graph.name_scope(scope + sep + 'BatchNorm_Fold' + sep):
        # new weights = old weights * gamma / sqrt(variance + epsilon)
        # new biases = -mean * gamma / sqrt(variance + epsilon) + beta
        multiplier_tensor = match.gamma_tensor * math_ops.rsqrt(
            match.variance_tensor + match.bn_op.get_attr('epsilon'))
        bias_tensor = math_ops.subtract(
            match.beta_tensor,
            match.mean_tensor * multiplier_tensor,
            name='bias')

        correction_scale, correction_recip, correction_offset = None, None, None
        if is_training:
          correction_scale, correction_recip, correction_offset = (
              _ComputeBatchNormCorrections(
                  context='',
                  match=match,
                  freeze_batch_norm_delay=freeze_batch_norm_delay,
                  fused_batch_norm=True))
        # The shape of depthwise weights is different, so we need to reshape the
        # multiplier_tensor to ensure that the scaled_weight_tensor has the
        # expected shape.
        weights = match.weight_tensor
        if match.layer_op.type == 'DepthwiseConv2dNative':
          new_shape = [
              match.weight_tensor.get_shape().as_list()[2],
              match.weight_tensor.get_shape().as_list()[3]
          ]
          multiplier_tensor = array_ops.reshape(
              multiplier_tensor, new_shape, name='scale_reshape')

          if correction_scale is not None:
            correction_scale = array_ops.reshape(
                correction_scale, new_shape, name='correction_reshape')

      if correction_scale is not None:
        weights = math_ops.multiply(
            correction_scale, weights, name='correction_mult')

      scaled_weight_tensor = math_ops.multiply(
          weights, multiplier_tensor, name='mul_fold')

      new_layer_tensor = _CloneWithNewOperands(
          match.layer_op, match.input_tensor, scaled_weight_tensor,
          match.batch_to_space_op)

      if correction_recip is not None:
        new_layer_tensor = math_ops.multiply(
            correction_recip, new_layer_tensor, name='post_conv_mul')
        new_layer_tensor = math_ops.add(new_layer_tensor, (correction_offset),
                                        'correction_add')

      bias_add_tensor = math_ops.add(
          new_layer_tensor, bias_tensor, name='add_fold')

      nodes_modified_count = common.RerouteTensor(bias_add_tensor,
                                                  match.output_tensor)
      if nodes_modified_count == 0:
        raise ValueError('Folding batch norms failed, %s had no outputs.' %
                         match.output_tensor.name)


def _FindFusedBatchNorms(graph):
  """Finds all ops and tensors related to found FusedBatchNorms.

  Args:
    graph: Graph to inspect.

  Yields:
    _FusedBatchNormMatches.
  """
  input_pattern = graph_matcher.OpTypePattern('*')
  # In practice, the weight pattern can match a Variable or a SpaceToBatchND
  # operation that follows a variable for atrous convolutions.
  weight_pattern = graph_matcher.OpTypePattern('*')
  gamma_pattern = graph_matcher.OpTypePattern('*')
  beta_pattern = graph_matcher.OpTypePattern('*')
  mean_pattern = graph_matcher.OpTypePattern('*')
  variance_pattern = graph_matcher.OpTypePattern('*')

  moving_average_pattern = graph_matcher.OpTypePattern('*')
  bn_decay_pattern = graph_matcher.OpTypePattern('*')
  layer_pattern = graph_matcher.OpTypePattern(
      'Conv2D|DepthwiseConv2dNative|MatMul',
      inputs=[input_pattern, weight_pattern])
  batch_to_space_pattern = graph_matcher.OpTypePattern(
      'BatchToSpaceND',
      inputs=[
          layer_pattern,
          graph_matcher.OpTypePattern('*'),
          graph_matcher.OpTypePattern('*')
      ])
  layer_output_pattern = graph_matcher.OneofPattern(
      [layer_pattern, batch_to_space_pattern])
  # MatMul has a Reshape between it and FusedBatchNorm.
  matmul_reshape_pattern = graph_matcher.OpTypePattern(
      'Reshape',
      inputs=[layer_output_pattern,
              graph_matcher.OpTypePattern('*')])

  batch_norm_pattern = graph_matcher.OpTypePattern(
      'FusedBatchNorm',
      inputs=[
          graph_matcher.OneofPattern(
              [matmul_reshape_pattern, layer_output_pattern]), gamma_pattern,
          beta_pattern, mean_pattern, variance_pattern
      ])
  matmul_bn_output_reshape_pattern = graph_matcher.OpTypePattern(
      'Reshape', inputs=[batch_norm_pattern,
                         graph_matcher.OpTypePattern('*')])

  bn_matcher = graph_matcher.GraphMatcher(
      graph_matcher.OneofPattern(
          [matmul_bn_output_reshape_pattern, batch_norm_pattern]))

  moving_average_sub_pattern = graph_matcher.OpTypePattern(
      'Sub', inputs=[moving_average_pattern, batch_norm_pattern])
  moving_average_mul_pattern = graph_matcher.OpTypePattern(
      'Mul', inputs=[moving_average_sub_pattern, bn_decay_pattern])

  moving_avg_mul_matcher = graph_matcher.GraphMatcher(
      moving_average_mul_pattern)

  for match_result in bn_matcher.match_graph(graph):
    moving_mean_tensor = None
    moving_variance_tensor = None
    bn_decay_mean_tensor = None
    bn_decay_var_tensor = None
    batch_to_space_op = None
    layer_op = match_result.get_op(layer_pattern)
    layer_tensor = match_result.get_tensor(layer_pattern)
    bn_op = match_result.get_op(batch_norm_pattern)
    batch_epsilon = bn_op.get_attr('epsilon')

    # In the MatMul case, the output of batch norm is reshaped back into a
    # 2D tensor, so the output_tensor is the output of the Reshape op.
    output_tensor = bn_op.outputs[0]
    if layer_op.type == 'MatMul':
      output_reshape_op = match_result.get_op(matmul_bn_output_reshape_pattern)
      # If the matcher didn't match matmul_bn_output_reshape, there will be
      # another match for this 'MatMul' later, so we can skip this one.
      if output_reshape_op is None:
        continue
      output_tensor = output_reshape_op.outputs[0]

    # Ensure that the output tensor has consumers, otherwise this is a dangling
    # node and not a match.
    if not output_tensor.consumers():
      continue

    batch_to_space_op = match_result.get_op(batch_to_space_pattern)
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
      # FusedBatchNormGrad doesn't compute gradients of the batch_mean and
      # batch_variance outputs, so we need to substitute our own custom
      # gradient.
      # TODO(suharshs, raghuramank): Find a way to avoid needing this hack.
      # pylint: disable=protected-access
      bn_op._set_attr(
          '_gradient_op_type',
          attr_value_pb2.AttrValue(s=compat.as_bytes('FoldFusedBatchNormGrad')))
      # pylint: enable=protected-access
      mean_tensor = bn_op.outputs[1]
      # The batch variance used during forward and backward prop is biased,
      # i.e it is calculated as: V=sum(x(k)-mu)^2/N. For the moving average
      # calculation, the variance is corrected by the term N/N-1 (Bessel's
      # correction). The variance tensor read from FuseBatchNorm has Bessel's
      # correction applied, so we undo it here.
      scope, sep, _ = bn_op.name.rpartition('/')
      g = ops.get_default_graph()
      with g.as_default(), g.name_scope(scope + sep):
        n = math_ops.cast(
            array_ops.size(layer_tensor) / array_ops.size(mean_tensor),
            dtypes.float32)
        variance_tensor = math_ops.multiply(
            bn_op.outputs[2], (n - 1) / n, name='Undo_Bessel_Correction')
      # TODO(suharshs): Find a way to get rid of this inner match.
      for mul_match_result in moving_avg_mul_matcher.match_graph(graph):
        sub_op = mul_match_result.get_op(moving_average_sub_pattern)
        if sub_op.inputs[1].name == bn_op.outputs[1].name:
          # During training: Batch Mean is bn_op.outputs[1]
          moving_mean_tensor = sub_op.inputs[0]
          bn_decay_mean_tensor = mul_match_result.get_tensor(bn_decay_pattern)
        if sub_op.inputs[1].name == bn_op.outputs[2].name:
          # During training: Batch Var is bn_op.outputs[2]
          moving_variance_tensor = sub_op.inputs[0]
          bn_decay_var_tensor = mul_match_result.get_tensor(bn_decay_pattern)
    else:
      mean_tensor = match_result.get_tensor(mean_pattern)
      variance_tensor = match_result.get_tensor(variance_pattern)

    yield _BatchNormMatch(
        layer_op=layer_op,
        bn_op=bn_op,
        output_tensor=output_tensor,
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        gamma_tensor=gamma_tensor,
        beta_tensor=beta_tensor,
        mean_tensor=mean_tensor,
        variance_tensor=variance_tensor,
        moving_mean_tensor=moving_mean_tensor,
        moving_variance_tensor=moving_variance_tensor,
        bn_decay_mean_tensor=bn_decay_mean_tensor,
        bn_decay_var_tensor=bn_decay_var_tensor,
        batch_epsilon=batch_epsilon,
        batch_to_space_op=batch_to_space_op)


def _ComputeBatchNormCorrections(context, match, freeze_batch_norm_delay,
                                 fused_batch_norm):
  """Computes batch norm correction params.

     Before batch normalization is frozen:
     We use batch statistics for batch norm.
       correction_scale = sigma_b/sigma_mv
       correction_recip = 1/correction_scale
       correction_offset = 0

     After batch normalization is frozen:
      correction_scale = sigma_b/sigma_mv
      correction_recip = 1
      correction_offset =  gamma*(mu_b/sigma_b-mu_mv/sigma_mv).

     Batch norm is frozen if global_step > bn_freeze_delay.
     The corrections ensure that:
     a) The weights are quantized after scaling by gamma/sigma_mv. This enables
     smoother training as the scaling on the weights changes slowly, rather than
     jump across mini-batches
     b) Changing the values of the corrections allows for one to switch between
     using batch statistics to using moving mean and average, without requiring
     changes to batch_norm


  Args:
    context: The scope under which we look for batch norm params
    match: Object containing required batch norm tensors for correction
      computation.
    freeze_batch_norm_delay: Delay in steps at which computation switches
      from regular batch norm to frozen mean and variance.
    fused_batch_norm: Bool, true if fused batch norm is used.

  Returns:
    A tuple of correction_scale, correction_recip, correction_offset
  """

  g = ops.get_default_graph()
  prefix = '' if not context else context + '/'
  with g.name_scope(prefix + 'batch_norm_correction'):
    recip_sigma_mv = math_ops.rsqrt(
        match.moving_variance_tensor + match.batch_epsilon)
    recip_sigma = math_ops.rsqrt(match.variance_tensor + match.batch_epsilon)
    correction_scale = math_ops.divide(
        recip_sigma_mv, recip_sigma, name='scale_compute')
    correction_scale = array_ops.identity(
        correction_scale, name='correction_scale')
    correction_recip = math_ops.reciprocal(
        correction_scale, name='reciprocal_compute')
    correction_offset = math_ops.multiply(
        match.gamma_tensor,
        match.mean_tensor * recip_sigma -
        match.moving_mean_tensor * recip_sigma_mv,
        name='offset_compute')

    if freeze_batch_norm_delay is not None:
      use_mv_avg = math_ops.greater_equal(
          common.CreateOrGetQuantizationStep(),
          freeze_batch_norm_delay,
          name='use_moving_average')
    else:
      use_mv_avg = False

    bn_decay_zero = 0.0
    bn_decay_mean_consumers = list(match.bn_decay_mean_tensor.consumers())
    bn_decay_var_consumers = list(match.bn_decay_mean_tensor.consumers())

    bn_decay_mean_out = utils.smart_cond(
        use_mv_avg,
        lambda: bn_decay_zero,
        lambda: match.bn_decay_mean_tensor,
        name='freeze_moving_mean')

    common.RerouteTensor(
        bn_decay_mean_out,
        match.bn_decay_mean_tensor,
        can_modify=bn_decay_mean_consumers)

    bn_decay_var_consumers = list(match.bn_decay_var_tensor.consumers())
    bn_decay_var_out = utils.smart_cond(
        use_mv_avg,
        lambda: bn_decay_zero,
        lambda: match.bn_decay_var_tensor,
        name='freeze_moving_var')
    common.RerouteTensor(
        bn_decay_var_out,
        match.bn_decay_var_tensor,
        can_modify=bn_decay_var_consumers)

    correction_recip = utils.smart_cond(
        use_mv_avg,
        lambda: array_ops.ones(correction_scale.shape),
        lambda: correction_recip,
        name='correction_recip')

    correction_offset = utils.smart_cond(
        use_mv_avg,
        lambda: correction_offset,
        lambda: array_ops.zeros(correction_offset.shape),
        name='correction_offset')
  return correction_scale, correction_recip, correction_offset


def _CloneWithNewOperands(layer_op, input_tensor, weight_tensor,
                          batch_to_space_op):
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
    conv = nn.depthwise_conv2d(
        input_tensor,
        weight_tensor,
        rate=layer_op.get_attr('dilations'),
        strides=layer_op.get_attr('strides'),
        padding=layer_op.get_attr('padding'),
        name=new_layer_name)
    # Copy the batch to space operation if we have a atrous convolution.
    if batch_to_space_op:
      batch_to_space_op = layer_op.outputs[0].consumers()[0]
      # TODO(suharshs): It's hard to make this name match with the unfused name.
      # Restructure this code to not rely on scope at all.
      new_batch_to_space_name = batch_to_space_op.name.split('/')[-1] + '_Fold'
      conv = array_ops.batch_to_space_nd(
          conv,
          batch_to_space_op.inputs[1],
          batch_to_space_op.inputs[2],
          name=new_batch_to_space_name)
    return conv
  else:
    raise ValueError('Cannot handle operation of type: %s' % layer_op.type)


@ops.RegisterGradient('FoldFusedBatchNormGrad')
def _FoldFusedBatchNormGrad(op, unused_grad_y, grad_mean, grad_var, unused_1,
                            unused_2):
  x = op.inputs[0]
  n = math_ops.cast(
      array_ops.size(x) / array_ops.size(grad_mean), dtypes.float32)
  dmean_dx = grad_mean / n
  dvar_dx = 2 * grad_var * (x - op.outputs[1]) / (n - 1)
  return (dmean_dx + dvar_dx), None, None, None, None


def _FoldUnfusedBatchNorms(graph, is_training, freeze_batch_norm_delay):
  """Finds unfused batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.
    is_training: Bool, True if training.
    freeze_batch_norm_delay: How many steps to wait before freezing moving mean
      and variance and using them for batch normalization.

  Raises:
    ValueError: When batch norm folding fails.
  """
  input_to_ops_map = input_to_ops.InputToOps(graph)

  for bn in common.BatchNormGroups(graph):
    has_scaling = _HasScaling(graph, input_to_ops_map, bn)

    if not _IsValidUnfusedBatchNorm(graph, bn):
      continue

    # The mangling code intimately depends on BatchNorm node's internals.
    original_op, folded_op = _CreateFoldedOp(
        graph,
        bn,
        has_scaling=has_scaling,
        freeze_batch_norm_delay=freeze_batch_norm_delay,
        is_training=is_training)

    activation = common.GetEndpointActivationOp(graph, bn)
    if activation:
      nodes_modified_count = common.RerouteTensor(
          folded_op.outputs[0], original_op.outputs[0], can_modify=[activation])
      if nodes_modified_count != 1:
        raise ValueError('Unexpected inputs to op: %s' % activation.name)
      continue

    # Treat consumer ops in bypass modules differently since they have Add
    # operations instead of Relu* above.
    add_bypass_ctx = re.search(r'^(.*)/([^/]+)', bn).group(1)
    add_bypass = graph.get_operation_by_name(add_bypass_ctx + '/Add')
    nodes_modified_count = common.RerouteTensor(
        folded_op.outputs[0], original_op.outputs[0], can_modify=[add_bypass])
    if nodes_modified_count != 1:
      raise ValueError('Unexpected inputs to op: %s' % add_bypass.name)


def _IsValidUnfusedBatchNorm(graph, context):
  """Checks that the output of the unfused batch norm has consumers."""
  add_shift = graph.get_operation_by_name(
      context + '/BatchNorm/batchnorm_1/add_1')
  # Ensure that the output tensor of batch norm has consumers, otherwise this
  # is a dangling node and not a match.
  return bool(add_shift.outputs[0].consumers())


def _FindMatchingTensor(graph, match_pattern, scope):
  """Finds best match of ops matching match_pattern with scope.

     Example: _FindMatchingTensor(graph,'/BatchNorm/moments/Squeeze',
     'MobilenetV1/MobilenetV1/Conv2d_0/') returns:
      Tensor('MobilenetV1/Conv2d_0/BatchNorm/moments/Squeeze')

  Args:
    graph: Graph to inspect.
    match_pattern: Part of the name of the op that we need to match, should
    be present in the op's name
    scope: The scope of the op. All the elements of the scope need not be
    present in the op's name.

  Returns:
    Tensor from graph that provides the best match to the match_pattern and
    scope
  """

  oplist = graph.get_operations()
  split_context = set(scope.split('/'))
  match_dict = {}
  for op in oplist:
    if op.name.endswith(match_pattern):
      split_name = op.name.split('/')
      num_matches = len(set(split_name) & split_context)
      if num_matches > 0:
        match_dict[op.name] = num_matches
  # match_dict contains matching op names from graph with values being
  # number of matches to scope. We pick the key with the most matches
  if match_dict:
    max_key = max(match_dict, key=match_dict.get)
    return graph.get_tensor_by_name(max_key + ':0')
  else:
    return None


def _GetBatchNormParams(graph, context, has_scaling):
  """Extracts relevant tensors for folding batch norms.

  Args:
    graph: Graph to inspect.
    context: The scope under which we look for batch norm params
    has_scaling: Bool that specifies if scaling is done as part of batch norm.

  Returns:
    _BatchNormMatch containing all required batch norm parameters.
  """
  gamma_tensor = None
  batch_mean_tensor = None
  batch_variance_tensor = None
  moving_mean_tensor = None
  moving_variance_tensor = None
  batch_epsilon = None
  bn_decay_mean_tensor = None
  bn_decay_var_tensor = None

  # TODO(raghuramank) This code relies on string matching and needs to be
  # updated if unfused batch norm continues to be widely used
  # Matching variable names is brittle and relies on scoping
  # conventions. Fused batch norm folding is more robust. Support for unfused
  # batch norms will be deprecated as we move forward. Fused batch norms allow
  # for faster training and should be used whenever possible.
  # context contains part of the names of the tensors we are interested in:
  # For MobilenetV1, the context has repetitions:
  # MobilenetV1/MobilenetV1/Conv2d_3_depthwise
  # when the moving_mean tensor has the name:
  # MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/read
  # To pick the correct variable name, it is necessary to ignore the repeating
  # header.

  # For MobilenetV2, this problem does not exist:
  # The context is: MobilenetV2/expanded_conv_3/depthwise
  # and the names of the tensors start with a single MobilenetV2
  # The moving mean for example, has the name:
  # MobilenetV2/expanded_conv_3/depthwise/BatchNorm/moving_mean/read
  # We identify the best match for an op by checking for
  # 1. The suffix of the op is exactly matched
  # 2. Maximum number of matches with the context.The matching
  # score is given by the number of parts of context (split by /) that
  # are present in the parts of the tensor name (again split by /).
  # For example: scope= MobilenetV2/MobilenetV2/expanded_conv_3 and
  # op.name =  MobilenetV2/expanded_conv_3/depthwise/BatchNorm/moving_mean/read
  # will have 2 matches,scope with a different conv layer will have one match.

  op_suffix_mean = '/BatchNorm/moments/Squeeze'
  op_suffix_variance = '/BatchNorm/moments/Squeeze_1'
  op_suffix_epsilon = '/BatchNorm/batchnorm_1/add/y'
  op_suffix_bn_decay_mean = '/BatchNorm/AssignMovingAvg/decay'
  op_suffix_bn_decay_var = '/BatchNorm/AssignMovingAvg_1/decay'

  if variable_scope.get_variable_scope().use_resource:
    op_suffix_gamma = '/BatchNorm/gamma/Read/ReadVariableOp'
    op_suffix_moving_variance = (
        '/BatchNorm/moving_variance/Read/ReadVariableOp')
    op_suffix_moving_mean = ('/BatchNorm/moving_mean/Read/ReadVariableOp')
  else:
    op_suffix_gamma = '/BatchNorm/gamma'
    op_suffix_moving_variance = '/BatchNorm/moving_variance/read'
    op_suffix_moving_mean = '/BatchNorm/moving_mean/read'
  # Parse through list of ops to find relevant ops

  batch_mean_tensor = _FindMatchingTensor(graph, op_suffix_mean, context)
  batch_variance_tensor = _FindMatchingTensor(graph, op_suffix_variance,
                                              context)
  moving_mean_tensor = _FindMatchingTensor(graph, op_suffix_moving_mean,
                                           context)
  moving_variance_tensor = _FindMatchingTensor(graph, op_suffix_moving_variance,
                                               context)
  batch_epsilon = _FindMatchingTensor(graph, op_suffix_epsilon, context)
  bn_decay_mean_tensor = _FindMatchingTensor(graph, op_suffix_bn_decay_mean,
                                             context)
  bn_decay_var_tensor = _FindMatchingTensor(graph, op_suffix_bn_decay_var,
                                            context)
  if batch_mean_tensor is None and moving_mean_tensor is None:
    ValueError('Error folding unfused batch norms')
  if has_scaling:
    gamma_tensor = _FindMatchingTensor(graph, op_suffix_gamma, context)

  if not has_scaling:
    gamma_tensor = array_ops.ones(moving_mean_tensor.shape)

  return _BatchNormMatch(
      layer_op=None,
      bn_op=None,
      output_tensor=None,
      input_tensor=None,
      weight_tensor=None,
      gamma_tensor=gamma_tensor,
      beta_tensor=None,
      mean_tensor=batch_mean_tensor,
      variance_tensor=batch_variance_tensor,
      moving_mean_tensor=moving_mean_tensor,
      moving_variance_tensor=moving_variance_tensor,
      bn_decay_mean_tensor=bn_decay_mean_tensor,
      bn_decay_var_tensor=bn_decay_var_tensor,
      batch_epsilon=batch_epsilon,
      batch_to_space_op=None)


def _CreateFoldedOp(graph, context, has_scaling, freeze_batch_norm_delay,
                    is_training):
  """Folds in batch norm layer into preceding convolution or FC layer.

  Creates 3 new nodes, connects their inputs and adds them to the graph:
  mul is cloned into mul_fold, Conv2D or MatMul, or DepthwiseConv2d is cloned
  into respective *_Fold, add is cloned into add_fold.

  Args:
    graph: Graph to modify.
    context: String, batch norm context, i.e. node into which BatchNorm is
      nested.
    has_scaling: Whether the batch norm has scaling enabled.
    freeze_batch_norm_delay: How many steps to wait before freezing moving mean
      and variance and using them for batch normalization.
    is_training: Bool, true if training.

  Raises:
    ValueError: When operation type is not supported, or input and output tensor
      shapes mismatch for created operations: mul_fold, add_fold.

  Returns:
    A pair of Operations, the first is the original consumer node of the batch
      norm (../BatchNorm/batchnorm_1/add_1), the second is the consumer node of
      the folded graph (add_fold).
  """
  mul_scale_name = 'mul_1' if has_scaling else 'mul'
  mul_scale = graph.get_operation_by_name(context +
                                          '/BatchNorm/batchnorm_1/' +
                                          mul_scale_name)
  op_below = mul_scale.inputs[0].op
  # Skip over the BatchToSpace operation in the case of atrous convolutions.
  batch_to_space_op = None
  if op_below.type == 'BatchToSpaceND':
    batch_to_space_op = op_below
    op_below = op_below.inputs[0].op
  weights = op_below.inputs[1]
  match = _GetBatchNormParams(
      graph=graph, context=context, has_scaling=has_scaling)
  correction_scale, correction_recip, correction_offset = None, None, None
  if is_training:
    correction_scale, correction_recip, correction_offset = (
        _ComputeBatchNormCorrections(
            context=context,
            match=match,
            freeze_batch_norm_delay=freeze_batch_norm_delay,
            fused_batch_norm=False))
  # Special handling for weights of depthwise convolution.
  if op_below.type == 'DepthwiseConv2dNative':
    new_shape = [
        weights.get_shape().as_list()[2],
        weights.get_shape().as_list()[3]
    ]
    scale_name = 'mul' if has_scaling else 'Rsqrt'
    scale = graph.get_operation_by_name(
        context + '/BatchNorm/batchnorm_1/' + scale_name)
    scale = array_ops.reshape(scale.outputs[0], new_shape,
                              context + '/scale_reshape')

    if correction_scale is not None:
      correction_scale = array_ops.reshape(correction_scale, new_shape,
                                           context + '/correction_reshape')
      with ops.device(mul_scale.device):
        weights = math_ops.multiply(correction_scale, weights,
                                    context + '/correction_mult')

    mul_fold = _CloneOp(mul_scale, context + '/mul_fold', [(0, weights),
                                                           (1, scale)])
  elif op_below.type in ['Conv2D', 'MatMul']:

    if correction_scale is not None:
      with ops.device(mul_scale.device):
        weights = math_ops.multiply(correction_scale, weights,
                                    context + '/correction_mult')
    mul_fold = _CloneOp(mul_scale, context + '/mul_fold', [(0, weights)])
  else:
    raise ValueError('Cannot handle operation of type: %s' % op_below.type)
  _AssertShapesMatch('mul_fold', mul_fold.inputs[0], mul_fold.outputs[0])

  conv_or_fc_folded = _CloneOp(op_below, op_below.name + '_Fold',
                               [(1, mul_fold.outputs[0])])

  add_shift = graph.get_operation_by_name(
      context + '/BatchNorm/batchnorm_1/add_1')

  corrected_output = conv_or_fc_folded.outputs[0]
  # Copy the batch to space operation if we have a atrous convolution.
  if batch_to_space_op:
    corrected_output = array_ops.batch_to_space_nd(
        corrected_output,
        batch_to_space_op.inputs[1],
        batch_to_space_op.inputs[2],
        name=batch_to_space_op.name + '_Fold')
  if correction_offset is not None:
    with ops.device(conv_or_fc_folded.device):
      corrected_output = math_ops.multiply(correction_recip, corrected_output,
                                           context + '/post_conv_mul')
      corrected_output = math_ops.add(corrected_output, (correction_offset),
                                      context + '/correction_add')
  add_fold = _CloneOp(add_shift, context + '/add_fold', [(0, corrected_output)])
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
  rsqrt_op = graph.get_operation_by_name(bn + '/BatchNorm/batchnorm_1/Rsqrt')
  rsqrt_consumers = input_to_ops_map.ConsumerOperations(rsqrt_op)

  return sum(1 for op in rsqrt_consumers if op.type == 'Mul') == 1


class _BatchNormMatch(object):
  """Contains all information related to a found Fused/UnfusedBatchNorm."""

  def __init__(self, layer_op, bn_op, output_tensor, input_tensor,
               weight_tensor, gamma_tensor, beta_tensor, mean_tensor,
               variance_tensor, moving_mean_tensor, moving_variance_tensor,
               bn_decay_mean_tensor, bn_decay_var_tensor, batch_epsilon,
               batch_to_space_op):
    self._layer_op = layer_op
    self._bn_op = bn_op
    self._output_tensor = output_tensor
    self._input_tensor = input_tensor
    self._weight_tensor = weight_tensor
    self._gamma_tensor = gamma_tensor
    self._beta_tensor = beta_tensor
    self._mean_tensor = mean_tensor
    self._variance_tensor = variance_tensor
    self._moving_mean_tensor = moving_mean_tensor
    self._moving_variance_tensor = moving_variance_tensor
    self._bn_decay_mean_tensor = bn_decay_mean_tensor
    self._bn_decay_var_tensor = bn_decay_var_tensor
    self._batch_epsilon = batch_epsilon
    self._batch_to_space_op = batch_to_space_op

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

  @property
  def moving_mean_tensor(self):
    return self._moving_mean_tensor

  @property
  def moving_variance_tensor(self):
    return self._moving_variance_tensor

  @property
  def batch_epsilon(self):
    return self._batch_epsilon

  @property
  def bn_decay_mean_tensor(self):
    return self._bn_decay_mean_tensor

  @property
  def bn_decay_var_tensor(self):
    return self._bn_decay_var_tensor

  @property
  def batch_to_space_op(self):
    return self._batch_to_space_op
