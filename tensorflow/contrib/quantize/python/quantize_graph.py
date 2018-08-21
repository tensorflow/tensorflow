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
"""API to simulate quantization on a python graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.quantize.python import fold_batch_norms
from tensorflow.contrib.quantize.python import quantize
from tensorflow.python.framework import ops


def _create_graph(input_graph=None,
                  is_training=True,
                  weight_bits=8,
                  activation_bits=8,
                  quant_delay=None,
                  freeze_bn_delay=None,
                  scope=None):
  """Rewrites an input_graph in place for simulated quantization.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.
    is_training: Whether quantizing training or eval graph.
    weight_bits: Number of bits to use for quantizing weights.
    activation_bits: Number of bits to use for quantizing activations.
    quant_delay: Number of steps after which weights and activations are
      quantized during training.
    freeze_bn_delay: Number of steps after which moving mean and variance are
      frozen and used instead of batch statistics during training.
      freeze_bn_delay should be greater than quant_delay and should correspond
      to the number of steps when training has almost converged
    scope: The scope to be transformed. If it's not None, only the ops which
      are in this scope will be transformed.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
      tf.Operation.
  """

  if input_graph is None:
    input_graph = ops.get_default_graph()

  # Add check to see if graph has training ops, if so provide error message and
  # exit
  _check_for_training_ops(input_graph)
  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph,
        freeze_batch_norm_delay=freeze_bn_delay,
        is_training=is_training)
    quantize.Quantize(
        input_graph,
        is_training,
        quant_delay=quant_delay,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        scope=scope)


def create_training_graph(input_graph=None, quant_delay=0):
  """Rewrites a training input_graph in place for simulated quantization.

  Variables added by the rewrite get added to the global variables collection.

  This function must be invoked prior to insertion of gradient ops in a graph
  as quantization should be modeled in both forward and backward passes.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  The default value of quant_delay is suitable for finetuning an already trained
  floating point model (recommended).
  If one wants to train a quantized model from scratch, quant_delay should be
  set to the number of steps it take the floating point model to converge.
  Quantization will be activated at this point and effectively finetune the
  model. If quant_delay is not provided when training from scratch, training can
  often fail.

  Args:
    input_graph: The tf.Graph to be transformed.
    quant_delay: Number of steps after which weights and activations are
      quantized during training.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
      tf.Operation.
  """
  # TODO(raghuramank) Need to have freeze_bn_delay be a function of batch size
  # Currently the values below are hardcoded for mobilenetV1 on imagenet
  # Please use the experimental API if you need to tune these values.
  freeze_bn_delay = None
  _create_graph(
      input_graph=input_graph,
      is_training=True,
      quant_delay=quant_delay,
      freeze_bn_delay=freeze_bn_delay)


def create_eval_graph(input_graph=None):
  """Rewrites an eval input_graph in place for simulated quantization.

  Variables added by the rewrite get added to the global variables collection.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
      tf.Operation.
  """
  _create_graph(input_graph=input_graph, is_training=False)


def experimental_create_training_graph(input_graph=None,
                                       weight_bits=8,
                                       activation_bits=8,
                                       quant_delay=0,
                                       freeze_bn_delay=None,
                                       scope=None):
  """Rewrites a training input_graph in place for simulated quantization.

  This function must be invoked prior to insertion of gradient ops in a graph
  as quantization should be modeled in both forward and backward passes.

  Variables added by the rewrite get added to the global variables collection.

  This function has additional experimental options not (yet) available to
  create_training_graph. The resulting behavior may be undefined.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  The default value of quant_delay is suitable for finetuning an already trained
  floating point model (recommended).
  If one wants to train a quantized model from scratch, quant_delay should be
  set to the number of steps it take the floating point model to converge.
  Quantization will be activated at this point and effectively finetune the
  model. If quant_delay is not provided when training from scratch, training can
  often fail.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.
    weight_bits: Number of bits to use for quantizing weights.
    activation_bits: Number of bits to use for quantizing activations.
    quant_delay: Number of steps after which weights and activations are
      quantized during training.
    freeze_bn_delay: Number of steps after which moving mean and variance are
      frozen and used instead of batch statistics during training.
      freeze_bn_delay should be greater than quant_delay and should correspond
      to when training has almost converged
    scope: The scope to be transformed. If it's not None, only the ops which
      are in this scope will be transformed.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  """

  _create_graph(
      input_graph=input_graph,
      is_training=True,
      weight_bits=weight_bits,
      activation_bits=activation_bits,
      quant_delay=quant_delay,
      freeze_bn_delay=freeze_bn_delay,
      scope=scope)


def experimental_create_eval_graph(input_graph=None,
                                   weight_bits=8,
                                   activation_bits=8,
                                   quant_delay=None,
                                   scope=None):
  """Rewrites an eval input_graph in place for simulated quantization.

  Variables added by the rewrite get added to the global variables collection.

  This function has additional experimental options not (yet) available to
  create_eval_graph. The resulting behavior may be undefined.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.
    weight_bits: Number of bits to use for quantizing weights.
    activation_bits: Number of bits to use for quantizing activations.
    quant_delay: Number of steps after which weights and activations are
      quantized during eval.
    scope: The scope to be transformed. If it's not None, only the ops which
      are in this scope will be transformed.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
      tf.Operation.
  """
  _create_graph(
      input_graph=input_graph,
      is_training=False,
      weight_bits=weight_bits,
      activation_bits=activation_bits,
      quant_delay=quant_delay,
      scope=scope)


def _check_for_training_ops(g):
  """Check if training ops are present in the graph.

  Args:
   g: The tf.Graph on which the check for training ops needs to be
   performed.

  Raises:
    ValueError: If a training op is seen in the graph;
  """

  # The list here is obtained
  # from https://www.tensorflow.org/api_docs/cc/group/training-ops
  training_ops = frozenset([
      'ApplyAdagrad', 'ApplyAdagradDA', 'ApplyAdam', 'ApplyAddSign',
      'ApplyCenteredRMSProp', 'ApplyFtrl', 'ApplyFtrlV2',
      'ApplyGradientDescent', 'ApplyMomentum', 'ApplyPowerSign',
      'ApplyProximalAdagrad', 'ApplyProximalGradientDescent', 'ApplyRMSProp',
      'ResourceApplyAdadelta', 'ResourceApplyAdagrad', 'ResourceApplyAdagradDA',
      'ResourceApplyAdam', 'ResourceApplyAddSign',
      'ResourceApplyCenteredRMSProp', 'ResourceApplyFtrl',
      'ResourceApplyFtrlV2', 'ResourceApplyGradientDescent',
      'ResourceApplyMomentum', 'ResourceApplyPowerSign',
      'ResourceApplyProximalAdagrad', 'ResourceApplyProximalGradientDescent',
      'ResourceApplyRMSProp', 'ResourceSparseApplyAdadelta',
      'ResourceSparseApplyAdagrad', 'ResourceSparseApplyAdagradDA',
      'ResourceSparseApplyCenteredRMSProp', 'ResourceSparseApplyFtrl',
      'ResourceSparseApplyFtrlV2', 'ResourceSparseApplyMomentum',
      'ResourceSparseApplyProximalAdagrad',
      'ResourceSparseApplyProximalGradientDescent',
      'ResourceSparseApplyRMSProp', 'SparseApplyAdadelta', 'SparseApplyAdagrad',
      'SparseApplyAdagradDA', 'SparseApplyCenteredRMSProp', 'SparseApplyFtrl',
      'SparseApplyFtrlV2', 'SparseApplyMomentum', 'SparseApplyProximalAdagrad',
      'SparseApplyProximalGradientDescent', 'SparseApplyRMSProp'
  ])

  op_types = set([op.type for op in g.get_operations()])
  train_op_list = op_types.intersection(training_ops)
  if train_op_list:
    raise ValueError('Training op found in graph, exiting %s' % train_op_list)
