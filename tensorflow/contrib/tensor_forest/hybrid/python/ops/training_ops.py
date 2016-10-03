# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for hybrid model training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging

TRAINING_OPS_FILE = '_training_ops.so'

_training_ops = None
_ops_lock = threading.Lock()

# TODO(b/31222613): Some of these ops are probably differentiable, and
# there may be latent bugs here.
ops.NotDifferentiable('HardRoutingFunction')
ops.NotDifferentiable('RoutingGradient')
ops.NotDifferentiable('KFeatureDataGradient')
ops.NotDifferentiable('KFeatureRoutingGradient')
ops.NotDifferentiable('KFeatureWeightGradient')
ops.NotDifferentiable('UnpackPath')


ops.RegisterShape('RoutingFunction')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('KFeatureRoutingFunction')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('HardRoutingFunction')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('StochasticHardRoutingFunction')(
    common_shapes.call_cpp_shape_fn)
ops.RegisterShape('StochasticHardRoutingGradient')(
    common_shapes.call_cpp_shape_fn)
ops.RegisterShape('UnpackPath')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('RoutingGradient')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('KFeatureDataGradient')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('KFeatureRoutingGradient')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('KFeatureWeightGradient')(common_shapes.call_cpp_shape_fn)


@ops.RegisterGradient('RoutingFunction')
def _RoutingFunctionGradient(op, grad):
  """The gradient of RoutingFunction.

  Args:
    op: The RoutingFunction op.
    grad: Gradient with respect to the output of the RoutingFunction op.

  Returns:
    Gradients with respect to the input of the RoutingFunction op.
  """
  routing_gradient = _training_ops.routing_gradient

  input_data_tensor = op.inputs[0]
  tree_weights_tensor = op.inputs[1]
  tree_thresholds_tensor = op.inputs[2]

  routing_function_tensor = op.outputs[0]

  # The derivatives below are each defined over one or two of three dimensions:
  # (batch_size, num_nodes, num_features).  We explicitly expand each derivative
  # to three dimensions to ensure that they're broadcasted correctly.

  # dl / du is the derivative of the loss with respect to the output of the
  # routing function, which is provided by tensorflow.
  #
  # dl / du has dimension (batch_size, num_nodes), which we expand to
  # (batch_size, num_nodes, 1).
  dl_du = array_ops.expand_dims(grad, 2)

  # du / df is the derivative of the output of the routing function with respect
  # to the decision function at each node.  It is computed by
  # routing_gradient_op.cc.
  #
  # du / df has dimension (batch_size, num_nodes), which we expand to
  # (batch_size, num_nodes, 1).
  du_df = array_ops.expand_dims(
      routing_gradient(
          input_data_tensor,
          tree_weights_tensor,
          tree_thresholds_tensor,
          routing_function_tensor,
          max_nodes=op.get_attr('max_nodes')),
      2)

  # df / dx is the derivative of the decision function with respect to the input
  # data.  f_i(x) = (-t_i * x + b_i), so df_i / dx = -t_i.
  #
  # df / dx has dimension (num_nodes, num_features), which we expand to
  # (1, num_nodes, num_features).
  df_dx = -array_ops.expand_dims(tree_weights_tensor, 0)

  # df / dt is the derivative of the decision function with respect to its
  # parameters. f_i(x) = (-t_i * x + b_i), so df_i / d t_i = -x.
  #
  # df / dt has dimension (batch_size, num_features), which we expand to
  # (batch_size, 1, num_features).
  df_dt = -array_ops.expand_dims(input_data_tensor, 1)
  # df / dt is the derivative of the decision function with respect to its
  # bias parameter. f_i(x) = (-t_i * x + b_i), so df_i / d t_i = 1.
  #
  # df / db has dimension (num_nodes), which we expand to
  # (1, num_nodes, 1).
  df_db = array_ops.expand_dims(
      array_ops.expand_dims(array_ops.ones_like(tree_thresholds_tensor), 0), 2)

  # Compute the derivatives of the loss with respect to the inputs using the
  # chain rule (backpropagation).
  dl_dx = math_ops.reduce_mean(dl_du * du_df * df_dx, 1)
  dl_dt = math_ops.reduce_mean(dl_du * du_df * df_dt, 0)
  dl_db = math_ops.reduce_mean(array_ops.squeeze(dl_du * du_df * df_db, [2]), 0)

  input_gradients = [dl_dx, dl_dt, dl_db]

  return input_gradients


@ops.RegisterGradient('StochasticHardRoutingFunction')
def _StochasticHardRoutingFunctionGradient(op, routing_grad, unused_path_grad):
  """The gradient of RoutingFunction.

  Args:
    op: The RoutingFunction op.
    routing_grad: Gradient with respect to the output of the RoutingFunction op.

  Returns:
    Gradients with respect to the input of the RoutingFunction op.
  """
  gradient_op = _training_ops.stochastic_hard_routing_gradient
  unpack_path_op = _training_ops.unpack_path

  input_data_tensor = op.inputs[0]
  tree_weights_tensor = op.inputs[1]
  tree_thresholds_tensor = op.inputs[2]

  path_probability_tensor = op.outputs[0]
  path_tensor = op.outputs[1]

  # The derivatives below are each defined over one or two of three dimensions:
  # (batch_size, num_nodes, num_features).  We explicitly expand each derivative
  # to three dimensions to ensure that they're broadcasted correctly.
  du_df_raw, df_dx_raw, df_dt_raw, df_db_raw = gradient_op(
      input_data_tensor,
      tree_weights_tensor,
      tree_thresholds_tensor,
      path_probability_tensor,
      path_tensor,
      tree_depth=op.get_attr('tree_depth'))

  # dl / du is the derivative of the loss with respect to the output of the
  # routing function, which is provided by tensorflow.
  #
  # dl / du has dimension (batch_size, num_nodes), which we expand to
  # (batch_size, num_nodes, 1).
  dl_du = array_ops.expand_dims(unpack_path_op(path_tensor, routing_grad), 2)

  # du / df is the derivative of the output of the routing function with respect
  # to the decision function at each node.  It is computed by
  # single_feature_routing_gradient_op.cc.
  #
  # du / df has dimension (batch_size, num_nodes), which we expand to
  # (batch_size, num_nodes, 1).
  du_df = array_ops.expand_dims(du_df_raw, 2)

  # df / dx is the derivative of the decision function with respect to the input
  # data.  f(x) = (-t * x + b), so df / dx = -t for the selected features and
  # zero elsewhere.
  #
  # df / dx has dimension (num_nodes, num_features), which we expand to
  # (1, num_nodes, num_features).
  df_dx = array_ops.expand_dims(df_dx_raw, 0)

  # df / dt is the derivative of the decision function with respect to its
  # parameters. f(x) = (-t * x + b), so df / dt = -x[feature].
  #
  # df / dt has dimension (batch_size, num_nodes, num_features).
  df_dt = -df_dt_raw

  # df / dt is the derivative of the decision function with respect to its
  # bias parameter. f(x) = (-t * x + b), so df / dt = 1.
  #
  # df / db has dimension (num_nodes), which we expand to
  # (1, num_nodes, 1).
  df_db = array_ops.expand_dims(array_ops.expand_dims(df_db_raw, 0), 2)

  # Compute the derivatives of the loss with respect to the inputs using the
  # chain rule (backpropagation).
  dl_dx = math_ops.reduce_mean(dl_du * du_df * df_dx, 1)
  dl_dt = math_ops.reduce_mean(dl_du * du_df * df_dt, 0)
  dl_db = math_ops.reduce_mean(array_ops.squeeze(dl_du * du_df * df_db, [2]), 0)

  input_gradients = [dl_dx, dl_dt, dl_db]

  return input_gradients


@ops.RegisterGradient('KFeatureRoutingFunction')
def _KFeatureRoutingFunctionGradient(op, grad):
  """The gradient of RoutingFunction.

  Args:
    op: The RoutingFunction op.
    grad: Gradient with respect to the output of the RoutingFunction op.

  Returns:
    Gradients with respect to the input of the RoutingFunction op.
  """
  gradient_op = _training_ops.k_feature_gradient

  input_data_tensor = op.inputs[0]
  tree_weights_tensor = op.inputs[1]
  tree_thresholds_tensor = op.inputs[2]

  routing_function_tensor = op.outputs[0]

  # The derivatives below are each defined over one or two of three dimensions:
  # (batch_size, num_nodes, num_features).  We explicitly expand each derivative
  # to three dimensions to ensure that they're broadcasted correctly.
  du_df_raw, df_dx_raw, df_dt_raw = gradient_op(
      input_data_tensor,
      tree_weights_tensor,
      tree_thresholds_tensor,
      routing_function_tensor,
      layer_num=op.get_attr('layer_num'),
      random_seed=op.get_attr('random_seed'))

  # dl / du is the derivative of the loss with respect to the output of the
  # routing function, which is provided by tensorflow.
  #
  # dl / du has dimension (batch_size, num_nodes), which we expand to
  # (batch_size, num_nodes, 1).
  dl_du = array_ops.expand_dims(grad, 2)

  # du / df is the derivative of the output of the routing function with respect
  # to the decision function at each node.  It is computed by
  # single_feature_routing_gradient_op.cc.
  #
  # du / df has dimension (batch_size, num_nodes), which we expand to
  # (batch_size, num_nodes, 1).
  du_df = array_ops.expand_dims(du_df_raw, 2)

  # df / dx is the derivative of the decision function with respect to the input
  # data.  f(x) = (-t * x + b), so df / dx = -t for the selected features and
  # zero elsewhere.
  #
  # df / dx has dimension (num_nodes, num_features), which we expand to
  # (1, num_nodes, num_features).
  df_dx = array_ops.expand_dims(df_dx_raw, 0)

  # df / dt is the derivative of the decision function with respect to its
  # parameters. f(x) = (-t * x + b), so df / dt = -x[feature].
  #
  # df / dt has dimension (batch_size, num_nodes, num_features).
  df_dt = -df_dt_raw

  # df / dt is the derivative of the decision function with respect to its
  # bias parameter. f(x) = (-t * x + b), so df / dt = 1.
  #
  # df / db has dimension (num_nodes), which we expand to
  # (1, num_nodes, 1).
  df_db = array_ops.expand_dims(
      array_ops.expand_dims(array_ops.ones_like(tree_thresholds_tensor), 0), 2)

  # Compute the derivatives of the loss with respect to the inputs using the
  # chain rule (backpropagation).
  dl_dx = math_ops.reduce_mean(dl_du * du_df * df_dx, 1)
  dl_dt = math_ops.reduce_mean(dl_du * du_df * df_dt, 0)
  dl_db = math_ops.reduce_mean(array_ops.squeeze(dl_du * du_df * df_db, [2]), 0)

  input_gradients = [dl_dx, dl_dt, dl_db]

  return input_gradients


# Workaround for the fact that importing tensorflow imports contrib
# (even if a user isn't using this or any other contrib op), but
# there's not yet any guarantee that the shared object exists.
# In which case, "import tensorflow" will always crash, even for users that
# never use contrib.
def Load():
  """Load training ops library and return the loaded module."""
  with _ops_lock:
    global _training_ops
    if not _training_ops:
      ops_path = resource_loader.get_path_to_datafile(TRAINING_OPS_FILE)
      logging.info('data path: %s', ops_path)
      _training_ops = load_library.load_op_library(ops_path)

      assert _training_ops, 'Could not load _training_ops.so'
  return _training_ops
