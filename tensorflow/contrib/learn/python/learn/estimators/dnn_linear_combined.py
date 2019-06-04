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

"""TensorFlow estimators for Linear and DNN joined training models (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import six

from tensorflow.contrib import layers
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.layers.python.layers import feature_column as feature_column_lib
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.python.feature_column import feature_column_lib as fc_core
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util


# The default learning rates are a historical artifact of the initial
# implementation, but seem a reasonable choice.
_DNN_LEARNING_RATE = 0.05
_LINEAR_LEARNING_RATE = 0.2


_FIX_GLOBAL_STEP_INCREMENT_DATE = "2017-04-15"
_FIX_GLOBAL_STEP_INCREMENT_INSTRUCTIONS = (
    "Please set fix_global_step_increment_bug=True and update training steps "
    "in your pipeline. See pydoc for details.")


def _as_iterable(preds, output):
  for pred in preds:
    yield pred[output]


def _get_feature_dict(features):
  if isinstance(features, dict):
    return features
  return {"": features}


def _get_optimizer(optimizer):
  if callable(optimizer):
    return optimizer()
  else:
    return optimizer


def _check_no_sync_replicas_optimizer(optimizer):
  if isinstance(optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
    raise ValueError(
        "SyncReplicasOptimizer is not supported in DNNLinearCombined model. "
        "If you want to use this optimizer, please use either DNN or Linear "
        "model.")


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


def _add_hidden_layer_summary(value, tag):
  summary.scalar("%s/fraction_of_zero_values" % tag, nn.zero_fraction(value))
  summary.histogram("%s/activation" % tag, value)


def _add_layer_summary(value, tag):
  summary.scalar("%s/fraction_of_zero_values" % tag, nn.zero_fraction(value))
  summary.histogram("%s/activation" % tag, value)


def _get_embedding_variable(column, collection_key, input_layer_scope):
  return ops.get_collection(collection_key,
                            input_layer_scope + "/" + column.name)


def _extract_embedding_lr_multipliers(embedding_lr_multipliers, collection_key,
                                      input_layer_scope):
  """Converts embedding lr multipliers to variable based gradient multiplier."""
  if not embedding_lr_multipliers:
    return None
  gradient_multipliers = {}
  for column, lr_mult in embedding_lr_multipliers.items():
    if not isinstance(column, feature_column_lib._EmbeddingColumn):  # pylint: disable=protected-access
      raise ValueError(
          "learning rate multipler can only be defined for embedding columns. "
          "It is defined for {}".format(column))
    embedding = _get_embedding_variable(
        column, collection_key, input_layer_scope)
    if not embedding:
      raise ValueError("Couldn't find a variable for column {}".format(column))
    for v in embedding:
      gradient_multipliers[v] = lr_mult
  return gradient_multipliers


def _dnn_linear_combined_model_fn(features, labels, mode, params, config=None):
  """Deep Neural Net and Linear combined model_fn.

  Args:
    features: `Tensor` or dict of `Tensor` (depends on data passed to `fit`).
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
      `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * head: A `Head` instance.
      * linear_feature_columns: An iterable containing all the feature columns
          used by the Linear model.
      * linear_optimizer: string, `Optimizer` object, or callable that defines
          the optimizer to use for training the Linear model. Defaults to the
          Ftrl optimizer.
      * joint_linear_weights: If True a single (possibly partitioned) variable
          will be used to store the linear model weights. It's faster, but
          requires all columns are sparse and have the 'sum' combiner.
      * dnn_feature_columns: An iterable containing all the feature columns used
          by the DNN model.
      * dnn_optimizer: string, `Optimizer` object, or callable that defines the
          optimizer to use for training the DNN model. Defaults to the Adagrad
          optimizer.
      * dnn_hidden_units: List of hidden units per DNN layer.
      * dnn_activation_fn: Activation function applied to each DNN layer. If
          `None`, will use `tf.nn.relu`.
      * dnn_dropout: When not `None`, the probability we will drop out a given
          DNN coordinate.
      * gradient_clip_norm: A float > 0. If provided, gradients are
          clipped to their global norm with this clipping ratio.
      * embedding_lr_multipliers: Optional. A dictionary from
          `EmbeddingColumn` to a `float` multiplier. Multiplier will be used to
          multiply with learning rate for the embedding variables.
      * input_layer_partitioner: Optional. Partitioner for input layer.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    `ModelFnOps`

  Raises:
    ValueError: If both `linear_feature_columns` and `dnn_features_columns`
      are empty at the same time, or `input_layer_partitioner` is missing.
  """
  head = params["head"]
  linear_feature_columns = params.get("linear_feature_columns")
  linear_optimizer = params.get("linear_optimizer") or "Ftrl"
  joint_linear_weights = params.get("joint_linear_weights")
  dnn_feature_columns = params.get("dnn_feature_columns")
  dnn_optimizer = params.get("dnn_optimizer") or "Adagrad"
  dnn_hidden_units = params.get("dnn_hidden_units")
  dnn_activation_fn = params.get("dnn_activation_fn") or nn.relu
  dnn_dropout = params.get("dnn_dropout")
  gradient_clip_norm = params.get("gradient_clip_norm")
  num_ps_replicas = config.num_ps_replicas if config else 0
  input_layer_partitioner = params.get("input_layer_partitioner") or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))
  embedding_lr_multipliers = params.get("embedding_lr_multipliers", {})
  fix_global_step_increment_bug = params.get(
      "fix_global_step_increment_bug", True)

  if not linear_feature_columns and not dnn_feature_columns:
    raise ValueError(
        "Either linear_feature_columns or dnn_feature_columns must be defined.")

  features = _get_feature_dict(features)

  linear_optimizer = _get_optimizer(linear_optimizer)
  _check_no_sync_replicas_optimizer(linear_optimizer)
  dnn_optimizer = _get_optimizer(dnn_optimizer)
  _check_no_sync_replicas_optimizer(dnn_optimizer)

  # Build DNN Logits.
  dnn_parent_scope = "dnn"

  if not dnn_feature_columns:
    dnn_logits = None
  else:
    if not dnn_hidden_units:
      raise ValueError(
          "dnn_hidden_units must be defined when dnn_feature_columns is "
          "specified.")
    dnn_partitioner = (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas))
    with variable_scope.variable_scope(
        dnn_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=dnn_partitioner):
      with variable_scope.variable_scope(
          "input_from_feature_columns",
          values=tuple(six.itervalues(features)),
          partitioner=input_layer_partitioner) as dnn_input_scope:
        if all(
            isinstance(fc, feature_column_lib._FeatureColumn)  # pylint: disable=protected-access
            for fc in dnn_feature_columns
        ):
          net = layers.input_from_feature_columns(
              columns_to_tensors=features,
              feature_columns=dnn_feature_columns,
              weight_collections=[dnn_parent_scope],
              scope=dnn_input_scope)
        else:
          net = fc_core.input_layer(
              features=features,
              feature_columns=dnn_feature_columns,
              weight_collections=[dnn_parent_scope])

      for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
        with variable_scope.variable_scope(
            "hiddenlayer_%d" % layer_id,
            values=(net,)) as dnn_hidden_layer_scope:
          net = layers.fully_connected(
              net,
              num_hidden_units,
              activation_fn=dnn_activation_fn,
              variables_collections=[dnn_parent_scope],
              scope=dnn_hidden_layer_scope)
          if dnn_dropout is not None and mode == model_fn.ModeKeys.TRAIN:
            net = layers.dropout(
                net,
                keep_prob=(1.0 - dnn_dropout))
        # TODO(b/31209633): Consider adding summary before dropout.
        _add_layer_summary(net, dnn_hidden_layer_scope.name)

      with variable_scope.variable_scope(
          "logits",
          values=(net,)) as dnn_logits_scope:
        dnn_logits = layers.fully_connected(
            net,
            head.logits_dimension,
            activation_fn=None,
            variables_collections=[dnn_parent_scope],
            scope=dnn_logits_scope)
      _add_layer_summary(dnn_logits, dnn_logits_scope.name)

  # Build Linear logits.
  linear_parent_scope = "linear"

  if not linear_feature_columns:
    linear_logits = None
  else:
    linear_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=64 << 20)
    with variable_scope.variable_scope(
        linear_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=linear_partitioner) as scope:
      if all(isinstance(fc, feature_column_lib._FeatureColumn)  # pylint: disable=protected-access
             for fc in linear_feature_columns):
        if joint_linear_weights:
          linear_logits, _, _ = layers.joint_weighted_sum_from_feature_columns(
              columns_to_tensors=features,
              feature_columns=linear_feature_columns,
              num_outputs=head.logits_dimension,
              weight_collections=[linear_parent_scope],
              scope=scope)
        else:
          linear_logits, _, _ = layers.weighted_sum_from_feature_columns(
              columns_to_tensors=features,
              feature_columns=linear_feature_columns,
              num_outputs=head.logits_dimension,
              weight_collections=[linear_parent_scope],
              scope=scope)
      else:
        linear_logits = fc_core.linear_model(
            features=features,
            feature_columns=linear_feature_columns,
            units=head.logits_dimension,
            weight_collections=[linear_parent_scope])

      _add_layer_summary(linear_logits, scope.name)

  # Combine logits and build full model.
  if dnn_logits is not None and linear_logits is not None:
    logits = dnn_logits + linear_logits
  elif dnn_logits is not None:
    logits = dnn_logits
  else:
    logits = linear_logits

  def _make_training_op(training_loss):
    """Training op for the DNN linear combined model."""
    train_ops = []
    global_step = training_util.get_global_step()
    if dnn_logits is not None:
      train_ops.append(
          optimizers.optimize_loss(
              loss=training_loss,
              global_step=global_step,
              learning_rate=_DNN_LEARNING_RATE,
              optimizer=dnn_optimizer,
              gradient_multipliers=_extract_embedding_lr_multipliers(  # pylint: disable=protected-access
                  embedding_lr_multipliers, dnn_parent_scope,
                  dnn_input_scope.name),
              clip_gradients=gradient_clip_norm,
              variables=ops.get_collection(dnn_parent_scope),
              name=dnn_parent_scope,
              # Empty summaries, because head already logs "loss" summary.
              summaries=[],
              increment_global_step=not fix_global_step_increment_bug))
    if linear_logits is not None:
      train_ops.append(
          optimizers.optimize_loss(
              loss=training_loss,
              global_step=global_step,
              learning_rate=_linear_learning_rate(len(linear_feature_columns)),
              optimizer=linear_optimizer,
              clip_gradients=gradient_clip_norm,
              variables=ops.get_collection(linear_parent_scope),
              name=linear_parent_scope,
              # Empty summaries, because head already logs "loss" summary.
              summaries=[],
              increment_global_step=not fix_global_step_increment_bug))

    train_op = control_flow_ops.group(*train_ops)
    if fix_global_step_increment_bug:
      with ops.control_dependencies([train_op]):
        with ops.colocate_with(global_step):
          return state_ops.assign_add(global_step, 1).op
    return train_op

  return head.create_model_fn_ops(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_make_training_op,
      logits=logits)


class DNNLinearCombinedEstimator(estimator.Estimator):
  """An estimator for TensorFlow Linear and DNN joined training models.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Note: New users must set `fix_global_step_increment_bug=True` when creating an
  estimator.

  Input of `fit`, `train`, and `evaluate` should have following features,
    otherwise there will be a `KeyError`:
      if `weight_column_name` is not `None`, a feature with
        `key=weight_column_name` whose value is a `Tensor`.
      for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `WeightedSparseColumn`, two features: the first with
        `key` the id column name, the second with `key` the weight column
        name. Both features' `value` must be a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
  """

  @deprecated_arg_values(
      _FIX_GLOBAL_STEP_INCREMENT_DATE,
      _FIX_GLOBAL_STEP_INCREMENT_INSTRUCTIONS,
      fix_global_step_increment_bug=False)
  def __init__(self,  # _joint_linear_weights pylint: disable=invalid-name
               head,
               model_dir=None,
               linear_feature_columns=None,
               linear_optimizer=None,
               _joint_linear_weights=False,
               dnn_feature_columns=None,
               dnn_optimizer=None,
               dnn_hidden_units=None,
               dnn_activation_fn=None,
               dnn_dropout=None,
               gradient_clip_norm=None,
               config=None,
               feature_engineering_fn=None,
               embedding_lr_multipliers=None,
               fix_global_step_increment_bug=False,
               input_layer_partitioner=None):
    """Initializes a DNNLinearCombinedEstimator instance.

    Note: New users must set `fix_global_step_increment_bug=True` when creating
    an estimator.

    Args:
      head: A _Head object.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set should be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. If `None`, will use a FTRL optimizer.
      _joint_linear_weights: If True will use a single (possibly partitioned)
        variable to store all weights for the linear model. More efficient if
        there are many columns, however requires all columns are sparse and
        have the 'sum' combiner.
      dnn_feature_columns: An iterable containing all the feature columns used
        by deep part of the model. All items in the set should be instances of
        classes derived from `FeatureColumn`.
      dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the deep part of the model. If `None`, will use an Adagrad optimizer.
      dnn_hidden_units: List of hidden units per layer. All layers are fully
        connected.
      dnn_activation_fn: Activation function applied to each layer. If `None`,
        will use `tf.nn.relu`.
      dnn_dropout: When not None, the probability we will drop out
        a given coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      config: RunConfig object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      embedding_lr_multipliers: Optional. A dictionary from `EmbeddingColumn` to
        a `float` multiplier. Multiplier will be used to multiply with
        learning rate for the embedding variables.
      fix_global_step_increment_bug: If `False`, the estimator needs two fit
        steps to optimize both linear and dnn parts. If `True`, this bug is
        fixed. New users must set this to `True`, but the default value is
        `False` for backwards compatibility.
      input_layer_partitioner: Optional. Partitioner for input layer.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
    """
    linear_feature_columns = tuple(linear_feature_columns or [])
    dnn_feature_columns = tuple(dnn_feature_columns or [])
    if not linear_feature_columns + dnn_feature_columns:
      raise ValueError("Either linear_feature_columns or dnn_feature_columns "
                       "must be defined.")
    super(DNNLinearCombinedEstimator, self).__init__(
        model_fn=_dnn_linear_combined_model_fn,
        model_dir=model_dir,
        config=config,
        params={
            "head": head,
            "linear_feature_columns": linear_feature_columns,
            "linear_optimizer": linear_optimizer,
            "joint_linear_weights": _joint_linear_weights,
            "dnn_feature_columns": dnn_feature_columns,
            "dnn_optimizer": dnn_optimizer,
            "dnn_hidden_units": dnn_hidden_units,
            "dnn_activation_fn": dnn_activation_fn,
            "dnn_dropout": dnn_dropout,
            "gradient_clip_norm": gradient_clip_norm,
            "embedding_lr_multipliers": embedding_lr_multipliers,
            "fix_global_step_increment_bug": fix_global_step_increment_bug,
            "input_layer_partitioner": input_layer_partitioner
        },
        feature_engineering_fn=feature_engineering_fn)


class DNNLinearCombinedClassifier(estimator.Estimator):
  """A classifier for TensorFlow Linear and DNN joined training models.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Note: New users must set `fix_global_step_increment_bug=True` when creating an
  estimator.

  Example:

  ```python
  sparse_feature_a = sparse_column_with_hash_bucket(...)
  sparse_feature_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,
                                          ...)
  sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b,
                                          ...)

  estimator = DNNLinearCombinedClassifier(
      # common settings
      n_classes=n_classes,
      weight_column_name=weight_column_name,
      # wide settings
      linear_feature_columns=[sparse_feature_a_x_sparse_feature_b],
      linear_optimizer=tf.compat.v1.train.FtrlOptimizer(...),
      # deep settings
      dnn_feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
      dnn_hidden_units=[1000, 500, 100],
      dnn_optimizer=tf.compat.v1.train.AdagradOptimizer(...))

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    ...
  def input_fn_eval: # returns x, y (where y represents label's class index).
    ...
  def input_fn_predict: # returns x, None.
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  # predict_classes returns class indices.
  estimator.predict_classes(input_fn=input_fn_predict)
  ```

  If the user specifies `label_keys` in constructor, labels must be strings from
  the `label_keys` vocabulary. Example:

  ```python
  label_keys = ['label0', 'label1', 'label2']
  estimator = DNNLinearCombinedClassifier(
      n_classes=n_classes,
      linear_feature_columns=[sparse_feature_a_x_sparse_feature_b],
      dnn_feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
      dnn_hidden_units=[1000, 500, 100],
      label_keys=label_keys)

  def input_fn_train: # returns x, y (where y is one of label_keys).
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, y (where y is one of label_keys).
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  def input_fn_predict: # returns x, None
  # predict_classes returns one of label_keys.
  estimator.predict_classes(input_fn=input_fn_predict)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column_name` is not `None`, a feature with
        `key=weight_column_name` whose value is a `Tensor`.
  * for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `WeightedSparseColumn`, two features: the first with
        `key` the id column name, the second with `key` the weight column name.
        Both features' `value` must be a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
  """

  @deprecated_arg_values(
      _FIX_GLOBAL_STEP_INCREMENT_DATE,
      _FIX_GLOBAL_STEP_INCREMENT_INSTRUCTIONS,
      fix_global_step_increment_bug=False)
  def __init__(self,  # _joint_linear_weights pylint: disable=invalid-name
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               linear_feature_columns=None,
               linear_optimizer=None,
               _joint_linear_weights=False,
               dnn_feature_columns=None,
               dnn_optimizer=None,
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               gradient_clip_norm=None,
               enable_centered_bias=False,
               config=None,
               feature_engineering_fn=None,
               embedding_lr_multipliers=None,
               input_layer_min_slice_size=None,
               label_keys=None,
               fix_global_step_increment_bug=False):
    """Constructs a DNNLinearCombinedClassifier instance.

    Note: New users must set `fix_global_step_increment_bug=True` when creating
    an estimator.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
        Note that class labels are integers representing the class index (i.e.
        values from 0 to n_classes-1). For arbitrary label values (e.g. string
        labels), convert to class indices first.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training.
        It will be multiplied by the loss of the example.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set must be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. If `None`, will use a FTRL optimizer.
      _joint_linear_weights: If True a single (possibly partitioned) variable
        will be used to store the linear model weights. It's faster, but
        requires all columns are sparse and have the 'sum' combiner.
      dnn_feature_columns: An iterable containing all the feature columns used
        by deep part of the model. All items in the set must be instances of
        classes derived from `FeatureColumn`.
      dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the deep part of the model. If `None`, will use an Adagrad optimizer.
      dnn_hidden_units: List of hidden units per layer. All layers are fully
        connected.
      dnn_activation_fn: Activation function applied to each layer. If `None`,
        will use `tf.nn.relu`.
      dnn_dropout: When not None, the probability we will drop out
        a given coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      config: RunConfig object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      embedding_lr_multipliers: Optional. A dictionary from `EmbeddingColumn` to
        a `float` multiplier. Multiplier will be used to multiply with
        learning rate for the embedding variables.
      input_layer_min_slice_size: Optional. The min slice size of input layer
        partitions. If not provided, will use the default of 64M.
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary. Only supported for `n_classes` > 2.
      fix_global_step_increment_bug: If `False`, the estimator needs two fit
        steps to optimize both linear and dnn parts. If `True`, this bug is
        fixed. New users must set this to `True`, but it the default value is
        `False` for backwards compatibility.

    Raises:
      ValueError: If `n_classes` < 2.
      ValueError: If both `linear_feature_columns` and `dnn_features_columns`
        are empty at the same time.
    """
    head = head_lib.multi_class_head(
        n_classes=n_classes,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias,
        label_keys=label_keys)
    linear_feature_columns = tuple(linear_feature_columns or [])
    dnn_feature_columns = tuple(dnn_feature_columns or [])
    self._feature_columns = linear_feature_columns + dnn_feature_columns
    if not self._feature_columns:
      raise ValueError("Either linear_feature_columns or dnn_feature_columns "
                       "must be defined.")

    # TODO(b/35922130): Replace with `input_layer_partitioner` arg.
    input_layer_partitioner = None
    if input_layer_min_slice_size is not None:
      input_layer_partitioner = (
          partitioned_variables.min_max_variable_partitioner(
              max_partitions=config.num_ps_replicas if config else 0,
              min_slice_size=input_layer_min_slice_size))

    super(DNNLinearCombinedClassifier, self).__init__(
        model_fn=_dnn_linear_combined_model_fn,
        model_dir=model_dir,
        config=config,
        params={
            "head": head,
            "linear_feature_columns": linear_feature_columns,
            "linear_optimizer": linear_optimizer,
            "joint_linear_weights": _joint_linear_weights,
            "dnn_feature_columns": dnn_feature_columns,
            "dnn_optimizer": dnn_optimizer,
            "dnn_hidden_units": dnn_hidden_units,
            "dnn_activation_fn": dnn_activation_fn,
            "dnn_dropout": dnn_dropout,
            "gradient_clip_norm": gradient_clip_norm,
            "embedding_lr_multipliers": embedding_lr_multipliers,
            "input_layer_partitioner": input_layer_partitioner,
            "fix_global_step_increment_bug": fix_global_step_increment_bug,
        },
        feature_engineering_fn=feature_engineering_fn)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  @deprecated_arg_values(
      "2017-03-01",
      "Please switch to predict_classes, or set `outputs` argument.",
      outputs=None)
  def predict(self, x=None, input_fn=None, batch_size=None, outputs=None,
              as_iterable=True):
    """Returns predictions for given features.

    By default, returns predicted classes. But this default will be dropped
    soon. Users should either pass `outputs`, or call `predict_classes` method.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.
      outputs: list of `str`, name of the output to predict.
        If `None`, returns classes.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted classes with shape [batch_size] (or an iterable
      of predicted classes if as_iterable is True). Each predicted class is
      represented by its class index (i.e. integer from 0 to n_classes-1).
      If `outputs` is set, returns a dict of predictions.
    """
    if not outputs:
      return self.predict_classes(
          x=x,
          input_fn=input_fn,
          batch_size=batch_size,
          as_iterable=as_iterable)
    return super(DNNLinearCombinedClassifier, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=outputs,
        as_iterable=as_iterable)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_classes(self, x=None, input_fn=None, batch_size=None,
                      as_iterable=True):
    """Returns predicted classes for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted classes with shape [batch_size] (or an iterable
      of predicted classes if as_iterable is True). Each predicted class is
      represented by its class index (i.e. integer from 0 to n_classes-1).
    """
    key = prediction_key.PredictionKey.CLASSES
    preds = super(DNNLinearCombinedClassifier, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key],
        as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=key)
    return preds[key].reshape(-1)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_proba(
      self, x=None, input_fn=None, batch_size=None, as_iterable=True):
    """Returns prediction probabilities for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x and y must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted probabilities with shape [batch_size, n_classes]
      (or an iterable of predicted probabilities if as_iterable is True).
    """
    key = prediction_key.PredictionKey.PROBABILITIES
    preds = super(DNNLinearCombinedClassifier, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key],
        as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=key)
    return preds[key]

  @deprecated("2017-03-25", "Please use Estimator.export_savedmodel() instead.")
  def export(self,
             export_dir,
             input_fn=None,
             input_feature_key=None,
             use_deprecated_input_fn=True,
             signature_fn=None,
             default_batch_size=1,
             exports_to_keep=None):
    """See BasEstimator.export."""
    def default_input_fn(unused_estimator, examples):
      return layers.parse_feature_columns_from_examples(
          examples, self._feature_columns)
    return super(DNNLinearCombinedClassifier, self).export(
        export_dir=export_dir,
        input_fn=input_fn or default_input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=use_deprecated_input_fn,
        signature_fn=(signature_fn or
                      export.classification_signature_fn_with_prob),
        prediction_key=prediction_key.PredictionKey.PROBABILITIES,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)


class DNNLinearCombinedRegressor(estimator.Estimator):
  """A regressor for TensorFlow Linear and DNN joined training models.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Note: New users must set `fix_global_step_increment_bug=True` when creating an
  estimator.

  Example:

  ```python
  sparse_feature_a = sparse_column_with_hash_bucket(...)
  sparse_feature_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,
                                          ...)
  sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b,
                                          ...)

  estimator = DNNLinearCombinedRegressor(
      # common settings
      weight_column_name=weight_column_name,
      # wide settings
      linear_feature_columns=[sparse_feature_a_x_sparse_feature_b],
      linear_optimizer=tf.compat.v1.train.FtrlOptimizer(...),
      # deep settings
      dnn_feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
      dnn_hidden_units=[1000, 500, 100],
      dnn_optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(...))

  # To apply L1 and L2 regularization, you can set optimizers as follows:
  tf.compat.v1.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001,
      l2_regularization_strength=0.001)
  # It is same for FtrlOptimizer.

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...
  def input_fn_predict: # returns x, None
    ...
  estimator.train(input_fn_train)
  estimator.evaluate(input_fn_eval)
  estimator.predict(input_fn_predict)
  ```

  Input of `fit`, `train`, and `evaluate` should have following features,
    otherwise there will be a `KeyError`:
      if `weight_column_name` is not `None`, a feature with
        `key=weight_column_name` whose value is a `Tensor`.
      for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `WeightedSparseColumn`, two features: the first with
        `key` the id column name, the second with `key` the weight column name.
        Both features' `value` must be a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
  """

  @deprecated_arg_values(
      _FIX_GLOBAL_STEP_INCREMENT_DATE,
      _FIX_GLOBAL_STEP_INCREMENT_INSTRUCTIONS,
      fix_global_step_increment_bug=False)
  def __init__(self,  # _joint_linear_weights pylint: disable=invalid-name
               model_dir=None,
               weight_column_name=None,
               linear_feature_columns=None,
               linear_optimizer=None,
               _joint_linear_weights=False,
               dnn_feature_columns=None,
               dnn_optimizer=None,
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               gradient_clip_norm=None,
               enable_centered_bias=False,
               label_dimension=1,
               config=None,
               feature_engineering_fn=None,
               embedding_lr_multipliers=None,
               input_layer_min_slice_size=None,
               fix_global_step_increment_bug=False):
    """Initializes a DNNLinearCombinedRegressor instance.

    Note: New users must set `fix_global_step_increment_bug=True` when creating
    an estimator.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set must be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. If `None`, will use a FTRL optimizer.
      _joint_linear_weights: If True a single (possibly partitioned) variable
        will be used to store the linear model weights. It's faster, but
        requires that all columns are sparse and have the 'sum' combiner.
      dnn_feature_columns: An iterable containing all the feature columns used
        by deep part of the model. All items in the set must be instances of
        classes derived from `FeatureColumn`.
      dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the deep part of the model. If `None`, will use an Adagrad optimizer.
      dnn_hidden_units: List of hidden units per layer. All layers are fully
        connected.
      dnn_activation_fn: Activation function applied to each layer. If None,
        will use `tf.nn.relu`.
      dnn_dropout: When not None, the probability we will drop out
        a given coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      config: RunConfig object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      embedding_lr_multipliers: Optional. A dictionary from `EmbeddingColumn` to
        a `float` multiplier. Multiplier will be used to multiply with
        learning rate for the embedding variables.
      input_layer_min_slice_size: Optional. The min slice size of input layer
        partitions. If not provided, will use the default of 64M.
      fix_global_step_increment_bug: If `False`, the estimator needs two fit
        steps to optimize both linear and dnn parts. If `True`, this bug is
        fixed. New users must set this to `True`, but it the default value is
        `False` for backwards compatibility.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
    """
    linear_feature_columns = linear_feature_columns or []
    dnn_feature_columns = dnn_feature_columns or []
    self._feature_columns = linear_feature_columns + dnn_feature_columns
    if not self._feature_columns:
      raise ValueError("Either linear_feature_columns or dnn_feature_columns "
                       "must be defined.")

    # TODO(b/35922130): Replace with `input_layer_partitioner` arg.
    input_layer_partitioner = None
    if input_layer_min_slice_size is not None:
      input_layer_partitioner = (
          partitioned_variables.min_max_variable_partitioner(
              max_partitions=config.num_ps_replicas if config else 0,
              min_slice_size=input_layer_min_slice_size))

    head = head_lib.regression_head(
        weight_column_name=weight_column_name,
        label_dimension=label_dimension,
        enable_centered_bias=enable_centered_bias)
    super(DNNLinearCombinedRegressor, self).__init__(
        model_fn=_dnn_linear_combined_model_fn,
        model_dir=model_dir,
        config=config,
        params={
            "head": head,
            "linear_feature_columns": linear_feature_columns,
            "linear_optimizer": linear_optimizer,
            "joint_linear_weights": _joint_linear_weights,
            "dnn_feature_columns": dnn_feature_columns,
            "dnn_optimizer": dnn_optimizer,
            "dnn_hidden_units": dnn_hidden_units,
            "dnn_activation_fn": dnn_activation_fn,
            "dnn_dropout": dnn_dropout,
            "gradient_clip_norm": gradient_clip_norm,
            "embedding_lr_multipliers": embedding_lr_multipliers,
            "input_layer_partitioner": input_layer_partitioner,
            "fix_global_step_increment_bug": fix_global_step_increment_bug,
        },
        feature_engineering_fn=feature_engineering_fn)

  def evaluate(self,
               x=None,
               y=None,
               input_fn=None,
               feed_fn=None,
               batch_size=None,
               steps=None,
               metrics=None,
               name=None,
               checkpoint_path=None,
               hooks=None):
    """See evaluable.Evaluable."""
    # TODO(zakaria): remove once deprecation is finished (b/31229024)
    custom_metrics = {}
    if metrics:
      for key, metric in six.iteritems(metrics):
        if (not isinstance(metric, metric_spec.MetricSpec) and
            not isinstance(key, tuple)):
          custom_metrics[(key, prediction_key.PredictionKey.SCORES)] = metric
        else:
          custom_metrics[key] = metric

    return super(DNNLinearCombinedRegressor, self).evaluate(
        x=x,
        y=y,
        input_fn=input_fn,
        feed_fn=feed_fn,
        batch_size=batch_size,
        steps=steps,
        metrics=custom_metrics,
        name=name,
        checkpoint_path=checkpoint_path,
        hooks=hooks)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  @deprecated_arg_values(
      "2017-03-01",
      "Please switch to predict_scores, or set `outputs` argument.",
      outputs=None)
  def predict(self, x=None, input_fn=None, batch_size=None, outputs=None,
              as_iterable=True):
    """Returns predictions for given features.

    By default, returns predicted scores. But this default will be dropped
    soon. Users should either pass `outputs`, or call `predict_scores` method.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.
      outputs: list of `str`, name of the output to predict.
        If `None`, returns scores.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted scores (or an iterable of predicted scores if
      as_iterable is True). If `label_dimension == 1`, the shape of the output
      is `[batch_size]`, otherwise the shape is `[batch_size, label_dimension]`.
      If `outputs` is set, returns a dict of predictions.
    """
    if not outputs:
      return self.predict_scores(
          x=x,
          input_fn=input_fn,
          batch_size=batch_size,
          as_iterable=as_iterable)
    return super(DNNLinearCombinedRegressor, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=outputs,
        as_iterable=as_iterable)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_scores(self, x=None, input_fn=None, batch_size=None,
                     as_iterable=True):
    """Returns predicted scores for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted scores (or an iterable of predicted scores if
      as_iterable is True). If `label_dimension == 1`, the shape of the output
      is `[batch_size]`, otherwise the shape is `[batch_size, label_dimension]`.
    """
    key = prediction_key.PredictionKey.SCORES
    preds = super(DNNLinearCombinedRegressor, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key],
        as_iterable=as_iterable)
    if as_iterable:
      return (pred[key] for pred in preds)
    return preds[key]

  @deprecated("2017-03-25", "Please use Estimator.export_savedmodel() instead.")
  def export(self,
             export_dir,
             input_fn=None,
             input_feature_key=None,
             use_deprecated_input_fn=True,
             signature_fn=None,
             default_batch_size=1,
             exports_to_keep=None):
    """See BaseEstimator.export."""
    def default_input_fn(unused_estimator, examples):
      return layers.parse_feature_columns_from_examples(
          examples, self._feature_columns)
    return super(DNNLinearCombinedRegressor, self).export(
        export_dir=export_dir,
        input_fn=input_fn or default_input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=use_deprecated_input_fn,
        signature_fn=signature_fn or export.regression_signature_fn,
        prediction_key=prediction_key.PredictionKey.SCORES,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)

# Aliases
# TODO(zakaria): Remove these aliases, See b/34751732
_DNNLinearCombinedEstimator = DNNLinearCombinedEstimator
