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
"""TensorFlow estimators for Linear and DNN joined training models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import six

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util
from tensorflow.python.util.tf_export import estimator_export

# The default learning rates are a historical artifact of the initial
# implementation.
_DNN_LEARNING_RATE = 0.001
_LINEAR_LEARNING_RATE = 0.005


def _check_no_sync_replicas_optimizer(optimizer):
  if isinstance(optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
    raise ValueError(
        'SyncReplicasOptimizer does not support multi optimizers case. '
        'Therefore, it is not supported in DNNLinearCombined model. '
        'If you want to use this optimizer, please use either DNN or Linear '
        'model.')


def _linear_learning_rate(num_linear_feature_columns):
  """Returns the default learning rate of the linear model.

  The calculation is a historical artifact of this initial implementation, but
  has proven a reasonable choice.

  Args:
    num_linear_feature_columns: The number of feature columns of the linear
      model.

  Returns:
    A float.
  """
  default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
  return min(_LINEAR_LEARNING_RATE, default_learning_rate)


def _add_layer_summary(value, tag):
  summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
  summary.histogram('%s/activation' % tag, value)


def _dnn_linear_combined_model_fn(features,
                                  labels,
                                  mode,
                                  head,
                                  linear_feature_columns=None,
                                  linear_optimizer='Ftrl',
                                  dnn_feature_columns=None,
                                  dnn_optimizer='Adagrad',
                                  dnn_hidden_units=None,
                                  dnn_activation_fn=nn.relu,
                                  dnn_dropout=None,
                                  input_layer_partitioner=None,
                                  config=None,
                                  batch_norm=False,
                                  linear_sparse_combiner='sum'):
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
    batch_norm: Whether to use batch normalization after each hidden layer.
    linear_sparse_combiner: A string specifying how to reduce the linear model
      if a categorical column is multivalent.  One of "mean", "sqrtn", and
      "sum".
  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: If both `linear_feature_columns` and `dnn_features_columns`
      are empty at the same time, or `input_layer_partitioner` is missing,
      or features has the wrong type.
  """
  if not isinstance(features, dict):
    raise ValueError('features should be a dictionary of `Tensor`s. '
                     'Given type: {}'.format(type(features)))
  if not linear_feature_columns and not dnn_feature_columns:
    raise ValueError(
        'Either linear_feature_columns or dnn_feature_columns must be defined.')

  num_ps_replicas = config.num_ps_replicas if config else 0
  input_layer_partitioner = input_layer_partitioner or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))

  # Build DNN Logits.
  dnn_parent_scope = 'dnn'

  if not dnn_feature_columns:
    dnn_logits = None
  else:
    dnn_optimizer = optimizers.get_optimizer_instance(
        dnn_optimizer, learning_rate=_DNN_LEARNING_RATE)
    _check_no_sync_replicas_optimizer(dnn_optimizer)
    if not dnn_hidden_units:
      raise ValueError(
          'dnn_hidden_units must be defined when dnn_feature_columns is '
          'specified.')
    dnn_partitioner = (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas))
    with variable_scope.variable_scope(
        dnn_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=dnn_partitioner):

      dnn_logit_fn = dnn._dnn_logit_fn_builder(  # pylint: disable=protected-access
          units=head.logits_dimension,
          hidden_units=dnn_hidden_units,
          feature_columns=dnn_feature_columns,
          activation_fn=dnn_activation_fn,
          dropout=dnn_dropout,
          input_layer_partitioner=input_layer_partitioner,
          batch_norm=batch_norm)
      dnn_logits = dnn_logit_fn(features=features, mode=mode)

  linear_parent_scope = 'linear'

  if not linear_feature_columns:
    linear_logits = None
  else:
    linear_optimizer = optimizers.get_optimizer_instance(
        linear_optimizer,
        learning_rate=_linear_learning_rate(len(linear_feature_columns)))
    _check_no_sync_replicas_optimizer(linear_optimizer)
    with variable_scope.variable_scope(
        linear_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=input_layer_partitioner) as scope:
      logit_fn = linear._linear_logit_fn_builder(  # pylint: disable=protected-access
          units=head.logits_dimension,
          feature_columns=linear_feature_columns,
          sparse_combiner=linear_sparse_combiner)
      linear_logits = logit_fn(features=features)
      _add_layer_summary(linear_logits, scope.name)

  # Combine logits and build full model.
  if dnn_logits is not None and linear_logits is not None:
    logits = dnn_logits + linear_logits
  elif dnn_logits is not None:
    logits = dnn_logits
  else:
    logits = linear_logits

  def _train_op_fn(loss):
    """Returns the op to optimize the loss."""
    train_ops = []
    global_step = training_util.get_global_step()
    if dnn_logits is not None:
      train_ops.append(
          dnn_optimizer.minimize(
              loss,
              var_list=ops.get_collection(
                  ops.GraphKeys.TRAINABLE_VARIABLES,
                  scope=dnn_parent_scope)))
    if linear_logits is not None:
      train_ops.append(
          linear_optimizer.minimize(
              loss,
              var_list=ops.get_collection(
                  ops.GraphKeys.TRAINABLE_VARIABLES,
                  scope=linear_parent_scope)))

    train_op = control_flow_ops.group(*train_ops)
    with ops.control_dependencies([train_op]):
      return distribute_lib.increment_var(global_step)

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_train_op_fn,
      logits=logits)


@estimator_export('estimator.DNNLinearCombinedClassifier')
class DNNLinearCombinedClassifier(estimator.Estimator):
  """An estimator for TensorFlow Linear and DNN joined classification models.

  Note: This estimator is also known as wide-n-deep.

  Example:

  ```python
  numeric_feature = numeric_column(...)
  categorical_column_a = categorical_column_with_hash_bucket(...)
  categorical_column_b = categorical_column_with_hash_bucket(...)

  categorical_feature_a_x_categorical_feature_b = crossed_column(...)
  categorical_feature_a_emb = embedding_column(
      categorical_column=categorical_feature_a, ...)
  categorical_feature_b_emb = embedding_column(
      categorical_id_column=categorical_feature_b, ...)

  estimator = DNNLinearCombinedClassifier(
      # wide settings
      linear_feature_columns=[categorical_feature_a_x_categorical_feature_b],
      linear_optimizer=tf.train.FtrlOptimizer(...),
      # deep settings
      dnn_feature_columns=[
          categorical_feature_a_emb, categorical_feature_b_emb,
          numeric_feature],
      dnn_hidden_units=[1000, 500, 100],
      dnn_optimizer=tf.train.ProximalAdagradOptimizer(...),
      # warm-start settings
      warm_start_from="/path/to/checkpoint/dir")

  # To apply L1 and L2 regularization, you can set dnn_optimizer to:
  tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001,
      l2_regularization_strength=0.001)
  # To apply learning rate decay, you can set dnn_optimizer to a callable:
  lambda: tf.AdamOptimizer(
      learning_rate=tf.exponential_decay(
          learning_rate=0.1,
          global_step=tf.get_global_step(),
          decay_steps=10000,
          decay_rate=0.96)
  # It is the same for linear_optimizer.

  # Input builders
  def input_fn_train: # returns x, y
    pass
  estimator.train(input_fn=input_fn_train, steps=100)

  def input_fn_eval: # returns x, y
    pass
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  def input_fn_predict: # returns x, None
    pass
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
    - if `column` is a `_CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `_WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `_DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss is calculated by using softmax cross entropy.

  @compatibility(eager)
  Estimators can be used while eager execution is enabled. Note that `input_fn`
  and all hooks are executed inside a graph context, so they have to be written
  to be compatible with graph mode. Note that `input_fn` code using `tf.data`
  generally works in both graph and eager modes.
  @end_compatibility
  """

  def __init__(self,
               model_dir=None,
               linear_feature_columns=None,
               linear_optimizer='Ftrl',
               dnn_feature_columns=None,
               dnn_optimizer='Adagrad',
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               input_layer_partitioner=None,
               config=None,
               warm_start_from=None,
               loss_reduction=losses.Reduction.SUM,
               batch_norm=False,
               linear_sparse_combiner='sum'):
    """Initializes a DNNLinearCombinedClassifier instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set must be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. Can also be a string (one of 'Adagrad',
        'Adam', 'Ftrl', 'RMSProp', 'SGD'), or callable. Defaults to FTRL
        optimizer.
      dnn_feature_columns: An iterable containing all the feature columns used
        by deep part of the model. All items in the set must be instances of
        classes derived from `FeatureColumn`.
      dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the deep part of the model. Can also be a string (one of 'Adagrad',
        'Adam', 'Ftrl', 'RMSProp', 'SGD'), or callable. Defaults to Adagrad
        optimizer.
      dnn_hidden_units: List of hidden units per layer. All layers are fully
        connected.
      dnn_activation_fn: Activation function applied to each layer. If None,
        will use `tf.nn.relu`.
      dnn_dropout: When not None, the probability we will drop out
        a given coordinate.
      n_classes: Number of label classes. Defaults to 2, namely binary
        classification. Must be > 1.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      label_vocabulary: A list of strings represents possible label values. If
        given, labels must be string type and have any value in
        `label_vocabulary`. If it is not given, that means labels are
        already encoded as integer or float within [0, 1] for `n_classes=2` and
        encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
        Also there will be errors if vocabulary is not provided and labels are
        string.
      input_layer_partitioner: Partitioner for input layer. Defaults to
        `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
      config: RunConfig object to configure the runtime settings.
      warm_start_from: A string filepath to a checkpoint to warm-start from, or
        a `WarmStartSettings` object to fully configure warm-starting.  If the
        string filepath is provided instead of a `WarmStartSettings`, then all
        weights are warm-started, and it is assumed that vocabularies and Tensor
        names are unchanged.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM`.
      batch_norm: Whether to use batch normalization after each hidden layer.
      linear_sparse_combiner: A string specifying how to reduce the linear model
        if a categorical column is multivalent.  One of "mean", "sqrtn", and
        "sum" -- these are effectively different ways to do example-level
        normalization, which can be useful for bag-of-words features.  For more
        details, see `tf.feature_column.linear_model`.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
    """
    linear_feature_columns = linear_feature_columns or []
    dnn_feature_columns = dnn_feature_columns or []
    self._feature_columns = (
        list(linear_feature_columns) + list(dnn_feature_columns))
    if not self._feature_columns:
      raise ValueError('Either linear_feature_columns or dnn_feature_columns '
                       'must be defined.')
    if n_classes == 2:
      head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction)
    else:
      head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
          n_classes,
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction)

    def _model_fn(features, labels, mode, config):
      """Call the _dnn_linear_combined_model_fn."""
      return _dnn_linear_combined_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          linear_feature_columns=linear_feature_columns,
          linear_optimizer=linear_optimizer,
          dnn_feature_columns=dnn_feature_columns,
          dnn_optimizer=dnn_optimizer,
          dnn_hidden_units=dnn_hidden_units,
          dnn_activation_fn=dnn_activation_fn,
          dnn_dropout=dnn_dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config,
          batch_norm=batch_norm,
          linear_sparse_combiner=linear_sparse_combiner)

    super(DNNLinearCombinedClassifier, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config,
        warm_start_from=warm_start_from)


@estimator_export('estimator.DNNLinearCombinedRegressor')
class DNNLinearCombinedRegressor(estimator.Estimator):
  """An estimator for TensorFlow Linear and DNN joined models for regression.

  Note: This estimator is also known as wide-n-deep.

  Example:

  ```python
  numeric_feature = numeric_column(...)
  categorical_column_a = categorical_column_with_hash_bucket(...)
  categorical_column_b = categorical_column_with_hash_bucket(...)

  categorical_feature_a_x_categorical_feature_b = crossed_column(...)
  categorical_feature_a_emb = embedding_column(
      categorical_column=categorical_feature_a, ...)
  categorical_feature_b_emb = embedding_column(
      categorical_column=categorical_feature_b, ...)

  estimator = DNNLinearCombinedRegressor(
      # wide settings
      linear_feature_columns=[categorical_feature_a_x_categorical_feature_b],
      linear_optimizer=tf.train.FtrlOptimizer(...),
      # deep settings
      dnn_feature_columns=[
          categorical_feature_a_emb, categorical_feature_b_emb,
          numeric_feature],
      dnn_hidden_units=[1000, 500, 100],
      dnn_optimizer=tf.train.ProximalAdagradOptimizer(...),
      # warm-start settings
      warm_start_from="/path/to/checkpoint/dir")

  # To apply L1 and L2 regularization, you can set dnn_optimizer to:
  tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001,
      l2_regularization_strength=0.001)
  # To apply learning rate decay, you can set dnn_optimizer to a callable:
  lambda: tf.AdamOptimizer(
      learning_rate=tf.exponential_decay(
          learning_rate=0.1,
          global_step=tf.get_global_step(),
          decay_steps=10000,
          decay_rate=0.96)
  # It is the same for linear_optimizer.

  # Input builders
  def input_fn_train: # returns x, y
    pass
  estimator.train(input_fn=input_fn_train, steps=100)

  def input_fn_eval: # returns x, y
    pass
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  def input_fn_predict: # returns x, None
    pass
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
    - if `column` is a `_CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `_WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `_DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss is calculated by using mean squared error.

  @compatibility(eager)
  Estimators can be used while eager execution is enabled. Note that `input_fn`
  and all hooks are executed inside a graph context, so they have to be written
  to be compatible with graph mode. Note that `input_fn` code using `tf.data`
  generally works in both graph and eager modes.
  @end_compatibility
  """

  def __init__(self,
               model_dir=None,
               linear_feature_columns=None,
               linear_optimizer='Ftrl',
               dnn_feature_columns=None,
               dnn_optimizer='Adagrad',
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               label_dimension=1,
               weight_column=None,
               input_layer_partitioner=None,
               config=None,
               warm_start_from=None,
               loss_reduction=losses.Reduction.SUM,
               batch_norm=False,
               linear_sparse_combiner='sum'):
    """Initializes a DNNLinearCombinedRegressor instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set must be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. Can also be a string (one of 'Adagrad',
        'Adam', 'Ftrl', 'RMSProp', 'SGD'), or callable. Defaults to FTRL
        optimizer.
      dnn_feature_columns: An iterable containing all the feature columns used
        by deep part of the model. All items in the set must be instances of
        classes derived from `FeatureColumn`.
      dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the deep part of the model. Can also be a string (one of 'Adagrad',
        'Adam', 'Ftrl', 'RMSProp', 'SGD'), or callable. Defaults to Adagrad
        optimizer.
      dnn_hidden_units: List of hidden units per layer. All layers are fully
        connected.
      dnn_activation_fn: Activation function applied to each layer. If None,
        will use `tf.nn.relu`.
      dnn_dropout: When not None, the probability we will drop out
        a given coordinate.
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      input_layer_partitioner: Partitioner for input layer. Defaults to
        `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
      config: RunConfig object to configure the runtime settings.
      warm_start_from: A string filepath to a checkpoint to warm-start from, or
        a `WarmStartSettings` object to fully configure warm-starting.  If the
        string filepath is provided instead of a `WarmStartSettings`, then all
        weights are warm-started, and it is assumed that vocabularies and Tensor
        names are unchanged.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM`.
      batch_norm: Whether to use batch normalization after each hidden layer.
      linear_sparse_combiner: A string specifying how to reduce the linear model
        if a categorical column is multivalent.  One of "mean", "sqrtn", and
        "sum" -- these are effectively different ways to do example-level
        normalization, which can be useful for bag-of-words features.  For more
        details, see `tf.feature_column.linear_model`.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
    """
    linear_feature_columns = linear_feature_columns or []
    dnn_feature_columns = dnn_feature_columns or []
    self._feature_columns = (
        list(linear_feature_columns) + list(dnn_feature_columns))
    if not self._feature_columns:
      raise ValueError('Either linear_feature_columns or dnn_feature_columns '
                       'must be defined.')

    def _model_fn(features, labels, mode, config):
      """Call the _dnn_linear_combined_model_fn."""
      return _dnn_linear_combined_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head_lib._regression_head(  # pylint: disable=protected-access
              label_dimension=label_dimension, weight_column=weight_column,
              loss_reduction=loss_reduction),
          linear_feature_columns=linear_feature_columns,
          linear_optimizer=linear_optimizer,
          dnn_feature_columns=dnn_feature_columns,
          dnn_optimizer=dnn_optimizer,
          dnn_hidden_units=dnn_hidden_units,
          dnn_activation_fn=dnn_activation_fn,
          dnn_dropout=dnn_dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config,
          batch_norm=batch_norm,
          linear_sparse_combiner=linear_sparse_combiner)

    super(DNNLinearCombinedRegressor, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config,
        warm_start_from=warm_start_from)
