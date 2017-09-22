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
"""Common functionality for canned estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned.optimizers import get_optimizer_instance
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training_util
from tensorflow.python.summary import summary
from tensorflow.python.ops import nn
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops


def add_layer_summary(value, tag):
  summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
  summary.histogram('%s/activation' % tag, value)

def _check_no_sync_replicas_optimizer(optimizer):
  if isinstance(optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
    raise ValueError(
      'SyncReplicasOptimizer does not support multi optimizers case. '
      'Therefore, it is not supported in DNNLinearCombined model. '
      'If you want to use this optimizer, please use either DNN or Linear '
      'model.')

def common_model_fn(
    features, labels, mode, head,
    name, logit_fn, optimizer, learning_rate=None, input_layer_partitioner=None,
    partitioner=None, config=None):
  """Deep Neural Net and Linear combined model_fn.

  Args:
    features: dict of `Tensor`.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
      `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    head: A `Head` instance.
    linear_feature_columns: An iterable containing all the feature columns used
      by the Linear model.
    linear_optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training the Linear model. Defaults to the Ftrl
      optimizer.
    dnn_feature_columns: An iterable containing all the feature columns used by
      the DNN model.
    dnn_optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training the DNN model. Defaults to the Adagrad
      optimizer.
    dnn_hidden_units: List of hidden units per DNN layer.
    dnn_activation_fn: Activation function applied to each DNN layer. If `None`,
      will use `tf.nn.relu`.
    dnn_dropout: When not `None`, the probability we will drop out a given DNN
      coordinate.
    input_layer_partitioner: Partitioner for input layer.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    `ModelFnOps`

  Raises:
    ValueError: If both `linear_feature_columns` and `dnn_features_columns`
      are empty at the same time, or `input_layer_partitioner` is missing,
      or features has the wrong type.
  """
  if not isinstance(features, dict):
    raise ValueError('features should be a dictionary of `Tensor`s. '
                     'Given type: {}'.format(type(features)))

  # if name is a string, then assume logit_fn, optimizer and learning_rate
  # parameters not collections as well and convert everything to lists
  if isinstance(name, six.string_types):
    name = [name]
    logit_fn = [logit_fn]
    optimizer = [optimizer]
    if learning_rate is not None:
      learning_rate = [learning_rate]

  # check if parameters have the same length (unless learning_rate is None)
  if not(len(name) == len(logit_fn) and
         len(logit_fn) == len(optimizer)):
    raise ValueError('Parameters name, logit_fn and optimizer must have the same length. '
                     'Given lengths: '
                     'len(name) = {}\n'
                     'len(logit_fn) = {}\n'
                     'len(optimizer) = {}\n'.
                     format(len(name), len(logit_fn), len(optimizer)))

  # if learning_rate is None, make a list with None values because of zip function bellow
  if learning_rate is None:
    learning_rate = [None] * len(name)

  if not(learning_rate is None or
         len(learning_rate) == len(optimizer)):
    raise ValueError('When learning_rate parameter is given, its length must be equal to'
                     'length of the optimizer parameter. '
                     'Given lengths are\n'
                     'len(learning_rate) = {}\n'
                     'len(optimizer) = {}\n'.format(len(learning_rate), len(optimizer)))

  num_ps_replicas = config.num_ps_replicas if config else 0

  partitioner = partitioner or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))

  input_layer_partitioner = input_layer_partitioner or (
    partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas,
      min_slice_size=64 << 20))

  logits = None
  train_op_params = []
  for parent_scope, logit_fn, optimizer, learning_rate in zip(name, logit_fn, optimizer, learning_rate):
    if logit_fn:
      # create optimizer and check for compatibility
      optimizer = get_optimizer_instance(optimizer, learning_rate=learning_rate)
      _check_no_sync_replicas_optimizer(optimizer)

      with variable_scope.variable_scope(
          parent_scope,
          values=tuple(six.itervalues(features)),
          partitioner=partitioner):

        cur_logits = logit_fn(
            features=features,
            mode=mode,
            input_layer_partitioner=input_layer_partitioner)
        logits = logits + cur_logits if logits is not None else cur_logits
        train_op_params.append((optimizer, parent_scope))

  if logits is None:
    raise ValueError('At least one item in logit_fns must be non-None.')

  def _train_op_fn(loss):
    """Returns the op to optimize the loss."""
    if len(train_op_params) is 1:
      optimizer, _ = train_op_params[0]
      return optimizer.minimize(
          loss,
          global_step=training_util.get_global_step())

    train_ops = []
    global_step = training_util.get_global_step()
    for optimizer, parent_scope in train_op_params:
      train_ops.append(
          optimizer.minimize(
              loss,
              var_list=ops.get_collection(
                  ops.GraphKeys.TRAINABLE_VARIABLES,
                  scope=parent_scope)))

    train_op = control_flow_ops.group(*train_ops)
    with ops.control_dependencies([train_op]):
      with ops.colocate_with(global_step):
        return state_ops.assign_add(global_step, 1)

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_train_op_fn,
      logits=logits)


def classifier_head(n_classes, weight_column, label_vocabulary):
  if n_classes == 2:
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
      weight_column=weight_column,
      label_vocabulary=label_vocabulary)
  else:
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
      n_classes, weight_column=weight_column,
      label_vocabulary=label_vocabulary)

  return head

def regression_head(label_dimension, weight_column):
  return head_lib._regression_head_with_mean_squared_error_loss(  # pylint: disable=protected-access
    label_dimension=label_dimension, weight_column=weight_column)
