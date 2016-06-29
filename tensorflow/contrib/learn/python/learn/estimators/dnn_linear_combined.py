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

"""TensorFlow estimators for Linear and DNN joined training models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math

import numpy as np
import six

from tensorflow.contrib import layers
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import logistic_regressor
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import training


class _ComposableModel(object):
  """ABC for building blocks that can be used to create estimators.

  Subclasses need to implement the following methods:
    - build_model
    - _get_optimizer
  See below for the required signatures.
  _ComposableModel and its subclasses are not part of the public tf.learn API.
  """

  def __init__(self,
               num_label_columns,
               optimizer,
               weight_collection_name,
               gradient_clip_norm):
    """Common initialization for all _ComposableModel objects.

    Args:
      num_label_columns: The number of label/target columns.
      optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the model. If `None`, will use a FTRL optimizer.
      weight_collection_name: A string defining the name to use for the
        collection of weights (e.g. 'dnn').
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
    """
    self._num_label_columns = num_label_columns
    self._optimizer = optimizer
    self._weight_collection_name = weight_collection_name
    self._gradient_clip_norm = gradient_clip_norm
    self._feature_columns=None

  def build_model(self, features, feature_columns, is_training):
    """Builds the model that can calculate the logits.

    Args:
      features: A mapping from feature columns to tensors.
      feature_columns: An iterable containing all the feature columns used
        by the model. All items in the set should be instances of
        classes derived from `FeatureColumn`.
      is_training: Set to True when training, False otherwise.

    Returns:
      The logits for this model.
    """
    raise NotImplementedError

  def get_train_step(self, loss):
    """Returns the ops to run to perform a training step on this estimator.

    Args:
      loss: The loss to use when calculating gradients.

    Returns:
      The ops to run to perform a training step.
    """
    my_vars = self._get_vars()
    if not (self._get_feature_columns() or my_vars):
      return []

    grads = gradients.gradients(loss, my_vars)
    if self._gradient_clip_norm:
      grads, _ = clip_ops.clip_by_global_norm(grads, self._gradient_clip_norm)
    self._optimizer = self._get_optimizer()
    return [self._optimizer.apply_gradients(zip(grads, my_vars))]

  def _get_feature_columns(self):
    if not self._feature_columns:
      return None
    feature_column_ops.check_feature_columns(self._feature_columns)
    return sorted(set(self._feature_columns), key=lambda x: x.key)

  def _get_feature_dict(self, features):
    if isinstance(features, dict):
      return features
    return {"": features}

  def _get_vars(self):
    if self._get_feature_columns():
      return ops.get_collection(self._weight_collection_name)
    return []

  def _get_optimizer(self):
    raise NotImplementedError


class _LinearComposableModel(_ComposableModel):
  """A _ComposableModel that implements linear regression.

  Instances of this class can be used to build estimators through the use
  of composition.
  """

  def __init__(self,
               num_label_columns,
               optimizer=None,
               gradient_clip_norm=None):
    """Initializes _LinearComposableModel objects.

    Args:
      num_label_columns: The number of label/target columns.
      optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the model. If `None`, will use a FTRL optimizer.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
    """
    super(_LinearComposableModel, self).__init__(
        num_label_columns=num_label_columns,
        optimizer=optimizer,
        weight_collection_name="linear",
        gradient_clip_norm=gradient_clip_norm)

  def build_model(self, features, feature_columns, is_training):
    """See base class."""
    features = self._get_feature_dict(features)
    self._feature_columns = feature_columns

    logits, _, _ = layers.weighted_sum_from_feature_columns(
        columns_to_tensors=features,
        feature_columns=self._get_feature_columns(),
        num_outputs=self._num_label_columns,
        weight_collections=[self._weight_collection_name],
        name="linear")
    return logits

  def _get_optimizer(self):
    if self._optimizer is None:
      self._optimizer = "Ftrl"
    if isinstance(self._optimizer, six.string_types):
      default_learning_rate = 1. / math.sqrt(len(self._get_feature_columns()))
      self._optimizer = layers.OPTIMIZER_CLS_NAMES[self._optimizer](
          learning_rate=default_learning_rate)
    return self._optimizer


class _DNNComposableModel(_ComposableModel):
  """A _ComposableModel that implements a DNN.

  Instances of this class can be used to build estimators through the use
  of composition.
  """

  def __init__(self,
               num_label_columns,
               hidden_units,
               optimizer=None,
               activation_fn=nn.relu,
               dropout=None,
               gradient_clip_norm=None,
               config=None):
    """Initializes _DNNComposableModel objects.

    Args:
      num_label_columns: The number of label/target columns.
      hidden_units: List of hidden units per layer. All layers are fully
        connected.
      optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the model. If `None`, will use a FTRL optimizer.
      activation_fn: Activation function applied to each layer. If `None`,
        will use `tf.nn.relu`.
      dropout: When not None, the probability we will drop out
        a given coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      config: RunConfig object to configure the runtime settings.
    """
    super(_DNNComposableModel, self).__init__(
        num_label_columns=num_label_columns,
        optimizer=optimizer,
        weight_collection_name="DNN",
        gradient_clip_norm=gradient_clip_norm)
    self._hidden_units = hidden_units
    self._activation_fn = activation_fn
    self._dropout = dropout
    self._config = config

  def _add_hidden_layer_summary(self, value, tag):
    # TODO(zakaria): Move this code to tf.learn and add test.
    logging_ops.scalar_summary("%s:fraction_of_zero_values" % tag,
                               nn.zero_fraction(value))
    logging_ops.histogram_summary("%s:activation" % tag, value)

  def build_model(self, features, feature_columns, is_training):
    """See base class."""
    features = self._get_feature_dict(features)
    self._feature_columns = feature_columns

    net = layers.input_from_feature_columns(
        features,
        self._get_feature_columns(),
        weight_collections=[self._weight_collection_name])
    for layer_id, num_hidden_units in enumerate(self._hidden_units):
      with variable_scope.variable_op_scope(
          [net], "hiddenlayer_%d" % layer_id,
          partitioner=partitioned_variables.min_max_variable_partitioner(
              max_partitions=self._config.num_ps_replicas)) as scope:
        net = layers.fully_connected(
            net,
            num_hidden_units,
            activation_fn=self._activation_fn,
            variables_collections=[self._weight_collection_name],
            scope=scope)
        if self._dropout is not None and is_training:
          net = layers.dropout(
              net,
              keep_prob=(1.0 - self._dropout))
      self._add_hidden_layer_summary(net, scope.name)
    with variable_scope.variable_op_scope(
        [net], "dnn_logits",
        partitioner=partitioned_variables.min_max_variable_partitioner(
            max_partitions=self._config.num_ps_replicas)) as scope:
      logits = layers.fully_connected(
          net,
          self._num_label_columns,
          activation_fn=None,
          variables_collections=[self._weight_collection_name],
          scope=scope)
    self._add_hidden_layer_summary(logits, "dnn_logits")
    return logits

  def _get_optimizer(self):
    if self._optimizer is None:
      self._optimizer = "Adagrad"
    if isinstance(self._optimizer, six.string_types):
      self._optimizer = layers.OPTIMIZER_CLS_NAMES[self._optimizer](
          learning_rate=0.05)
    return self._optimizer


# TODO(ispir): Increase test coverage
class _DNNLinearCombinedBaseEstimator(estimator.BaseEstimator):
  """An estimator for TensorFlow Linear and DNN joined training models.

    Input of `fit`, `train`, and `evaluate` should have following features,
      otherwise there will be a `KeyError`:
        if `weight_column_name` is not `None`, a feature with
          `key=weight_column_name` whose value is a `Tensor`.
        for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
        - if `column` is a `SparseColumn`, a feature with `key=column.name`
          whose `value` is a `SparseTensor`.
        - if `column` is a `RealValuedColumn, a feature with `key=column.name`
          whose `value` is a `Tensor`.
  """

  def __init__(self,
               target_column,
               model_dir=None,
               linear_feature_columns=None,
               linear_optimizer=None,
               dnn_feature_columns=None,
               dnn_optimizer=None,
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               gradient_clip_norm=None,
               enable_centered_bias=True,
               config=None):
    """Initializes a _DNNLinearCombinedBaseEstimator instance.

    Args:
      target_column: A _TargetColumn object.
      model_dir: Directory to save model parameters, graph and etc.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set should be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. If `None`, will use a FTRL optimizer.
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
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      config: RunConfig object to configure the runtime settings.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
    """
    super(_DNNLinearCombinedBaseEstimator, self).__init__(model_dir=model_dir,
                                                          config=config)

    self._linear_model = _LinearComposableModel(
        num_label_columns=target_column.num_label_columns,
        optimizer=linear_optimizer,
        gradient_clip_norm=gradient_clip_norm)

    self._dnn_model = _DNNComposableModel(
        num_label_columns=target_column.num_label_columns,
        hidden_units=dnn_hidden_units,
        optimizer=dnn_optimizer,
        activation_fn=dnn_activation_fn,
        dropout=dnn_dropout,
        gradient_clip_norm=gradient_clip_norm,
        config=self._config) if dnn_hidden_units else None

    self._linear_feature_columns = linear_feature_columns
    self._linear_optimizer = linear_optimizer
    self._dnn_feature_columns = dnn_feature_columns
    self._dnn_optimizer = dnn_optimizer
    self._dnn_hidden_units = dnn_hidden_units
    self._dnn_activation_fn = dnn_activation_fn
    if self._dnn_activation_fn is None:
      self._dnn_activation_fn = nn.relu
    self._dnn_dropout = dnn_dropout
    self._dnn_weight_collection = "DNNLinearCombined_dnn"
    self._linear_weight_collection = "DNNLinearCombined_linear"
    self._centered_bias_weight_collection = "centered_bias"
    self._gradient_clip_norm = gradient_clip_norm
    self._enable_centered_bias = enable_centered_bias
    self._target_column = target_column

  @property
  def linear_weights_(self):
    """Returns weights per feature of the linear part."""
    all_variables = self.get_variable_names()
    # TODO(ispir): Figure out a better way to retrieve variables for features.
    # for example using feature info / columns.
    values = {}
    for name in all_variables:
      if (name.startswith("linear/") and name.rfind("/") == 6 and
          name != "linear/bias_weight"):
        values[name] = self.get_variable_value(name)
    if len(values) == 1:
      return values[list(values.keys())[0]]
    return values

  @property
  def linear_bias_(self):
    """Returns bias of the linear part."""
    return (self.get_variable_value("linear/bias_weight") +
            self.get_variable_value("centered_bias_weight"))

  @property
  def dnn_weights_(self):
    """Returns weights of deep neural network part."""
    return [self.get_variable_value("hiddenlayer_%d/weights" % i)
            for i, _ in enumerate(self._dnn_hidden_units)] + [
                self.get_variable_value("dnn_logits/weights")]

  @property
  def dnn_bias_(self):
    """Returns bias of deep neural network part."""
    return [self.get_variable_value("hiddenlayer_%d/biases" % i)
            for i, _ in enumerate(self._dnn_hidden_units)] + [
                self.get_variable_value("dnn_logits/biases"),
                self.get_variable_value("centered_bias_weight")]

  def _get_train_ops(self, features, targets):
    """See base class."""
    global_step = contrib_variables.get_global_step()
    assert global_step
    logits = self._logits(features, is_training=True)
    if self._enable_centered_bias:
      centered_bias_step = [self._centered_bias_step(targets, features)]
    else:
      centered_bias_step = []
    with ops.control_dependencies(centered_bias_step):
      loss = self._loss(logits, targets, features)
    logging_ops.scalar_summary("loss", loss)

    linear_train_step = self._linear_model.get_train_step(loss)
    dnn_train_step = (self._dnn_model.get_train_step(loss)
                      if self._dnn_model else [])

    with ops.control_dependencies(linear_train_step + dnn_train_step):
      with ops.get_default_graph().colocate_with(global_step):
        return state_ops.assign_add(global_step, 1).op, loss

  def _run_metrics(self, predictions, targets, metrics, weights):
    result = {}
    targets = math_ops.cast(targets, predictions.dtype)
    for name, metric in six.iteritems(metrics or {}):
      if "weights" in inspect.getargspec(metric)[0]:
        result[name] = metric(predictions, targets, weights=weights)
      else:
        result[name] = metric(predictions, targets)

    return result

  def _get_eval_ops(self, features, targets, metrics=None):
    raise NotImplementedError

  def _get_predict_ops(self, features):
    """See base class."""
    logits = self._logits(features)
    return self._target_column.logits_to_predictions(logits, proba=True)

  def _get_feature_ops_from_example(self, examples_batch):
    column_types = layers.create_feature_spec_for_parsing((
        self._get_linear_feature_columns() or []) + (
            self._get_dnn_feature_columns() or []))
    features = parsing_ops.parse_example(examples_batch, column_types)
    return features

  def _get_linear_feature_columns(self):
    if not self._linear_feature_columns:
      return None
    feature_column_ops.check_feature_columns(self._linear_feature_columns)
    return sorted(set(self._linear_feature_columns), key=lambda x: x.key)

  def _get_dnn_feature_columns(self):
    if not self._dnn_feature_columns:
      return None
    feature_column_ops.check_feature_columns(self._dnn_feature_columns)
    return sorted(set(self._dnn_feature_columns), key=lambda x: x.key)

  def _dnn_logits(self, features, is_training):
    return self._dnn_model.build_model(
        features, self._dnn_feature_columns, is_training)

  def _linear_logits(self, features, is_training):
    return self._linear_model.build_model(
        features, self._linear_feature_columns, is_training)

  def _get_feature_dict(self, features):
    if isinstance(features, dict):
      return features
    return {"": features}

  def _centered_bias(self):
    centered_bias = variables.Variable(
        array_ops.zeros([self._target_column.num_label_columns]),
        collections=[self._centered_bias_weight_collection,
                     ops.GraphKeys.VARIABLES],
        name="centered_bias_weight")
    logging_ops.scalar_summary(
        ["centered_bias_%d" % cb for cb in range(
            self._target_column.num_label_columns)],
        array_ops.reshape(centered_bias, [-1]))
    return centered_bias

  def _centered_bias_step(self, targets, features):
    centered_bias = ops.get_collection(self._centered_bias_weight_collection)
    batch_size = array_ops.shape(targets)[0]
    logits = array_ops.reshape(
        array_ops.tile(centered_bias[0], [batch_size]),
        [batch_size, self._target_column.num_label_columns])
    loss = self._loss(logits, targets, features)
    # Learn central bias by an optimizer. 0.1 is a convervative lr for a single
    # variable.
    return training.AdagradOptimizer(0.1).minimize(loss, var_list=centered_bias)

  def _logits(self, features, is_training=False):
    linear_feature_columns = self._get_linear_feature_columns()
    dnn_feature_columns = self._get_dnn_feature_columns()
    if not (linear_feature_columns or dnn_feature_columns):
      raise ValueError("Either linear_feature_columns or dnn_feature_columns "
                       "should be defined.")

    features = self._get_feature_dict(features)
    if linear_feature_columns and dnn_feature_columns:
      logits = (self._linear_logits(features, is_training) +
                self._dnn_logits(features, is_training))
    elif dnn_feature_columns:
      logits = self._dnn_logits(features, is_training)
    else:
      logits = self._linear_logits(features, is_training)

    if self._enable_centered_bias:
      return nn.bias_add(logits, self._centered_bias())
    else:
      return logits

  def _loss(self, logits, target, features):
    return self._target_column.loss(logits, target,
                                    self._get_feature_dict(features))

  def _get_linear_vars(self):
    if self._get_linear_feature_columns():
      return ops.get_collection(self._linear_weight_collection)
    return []

  def _get_linear_training_ops(self, linear_grads, linear_vars):
    if self._get_linear_feature_columns():
      self._linear_optimizer = self._get_optimizer(
          self._linear_optimizer,
          default_optimizer="Ftrl",
          default_learning_rate=1. / math.sqrt(len(
              self._get_linear_feature_columns())))
      return [
          self._linear_optimizer.apply_gradients(zip(linear_grads, linear_vars))
      ]
    return []

  def _get_dnn_vars(self):
    if self._get_dnn_feature_columns():
      return ops.get_collection(self._dnn_weight_collection)
    return []

  def _get_dnn_training_ops(self, dnn_grads, dnn_vars):
    if self._get_dnn_feature_columns():
      self._dnn_optimizer = self._get_optimizer(self._dnn_optimizer,
                                                default_optimizer="Adagrad",
                                                default_learning_rate=0.05)
      return [self._dnn_optimizer.apply_gradients(zip(dnn_grads, dnn_vars))]
    return []

  def _get_optimizer(self, optimizer, default_optimizer, default_learning_rate):
    if optimizer is None:
      optimizer = default_optimizer
    if isinstance(optimizer, six.string_types):
      optimizer = layers.OPTIMIZER_CLS_NAMES[optimizer](
          learning_rate=default_learning_rate)
    return optimizer


class DNNLinearCombinedClassifier(_DNNLinearCombinedBaseEstimator):
  """A classifier for TensorFlow Linear and DNN joined training models.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_x_occupation = crossed_column(columns=[education, occupation],
                                          hash_bucket_size=10000)
  education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                   combiner="sum")
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")

  estimator = DNNLinearCombinedClassifier(
      # common settings
      n_classes=n_classes,
      weight_column_name=weight_column_name,
      # wide settings
      linear_feature_columns=[education_x_occupation],
      linear_optimizer=tf.train.FtrlOptimizer(...),
      # deep settings
      dnn_feature_columns=[education_emb, occupation_emb],
      dnn_hidden_units=[1000, 500, 100],
      dnn_optimizer=tf.train.AdagradOptimizer(...))

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:
      if `weight_column_name` is not `None`, a feature with
        `key=weight_column_name` whose value is a `Tensor`.
      for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
  """

  def __init__(self,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               linear_feature_columns=None,
               linear_optimizer=None,
               dnn_feature_columns=None,
               dnn_optimizer=None,
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               gradient_clip_norm=None,
               enable_centered_bias=True,
               config=None):
    """Constructs a DNNLinearCombinedClassifier instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc.
      n_classes: number of target classes. Default is binary classification.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training.
        It will be multiplied by the loss of the example.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set must be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. If `None`, will use a FTRL optimizer.
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

    Raises:
      ValueError: If `n_classes` < 2.
      ValueError: If both `linear_feature_columns` and `dnn_features_columns`
        are empty at the same time.
    """

    if n_classes < 2:
      raise ValueError("n_classes should be greater than 1. Given: {}".format(
          n_classes))
    target_column = layers.multi_class_target(
        n_classes=n_classes,
        weight_column_name=weight_column_name)
    super(DNNLinearCombinedClassifier, self).__init__(
        model_dir=model_dir,
        linear_feature_columns=linear_feature_columns,
        linear_optimizer=linear_optimizer,
        dnn_feature_columns=dnn_feature_columns,
        dnn_optimizer=dnn_optimizer,
        dnn_hidden_units=dnn_hidden_units,
        dnn_activation_fn=dnn_activation_fn,
        dnn_dropout=dnn_dropout,
        gradient_clip_norm=gradient_clip_norm,
        enable_centered_bias=enable_centered_bias,
        target_column=target_column,
        config=config)

  def predict(self, x=None, input_fn=None, batch_size=None):
    """Returns predictions for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.

    Returns:
      Numpy array of predicted classes or regression values.
    """
    predictions = super(DNNLinearCombinedClassifier, self).predict(
        x=x, input_fn=input_fn, batch_size=batch_size)
    predictions = np.argmax(predictions, axis=1)
    return predictions

  def predict_proba(self, x=None, input_fn=None, batch_size=None):
    """Returns prediction probabilities for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x and y must be None.
      batch_size: Override default batch size.

    Returns:
      Numpy array of predicted probabilities.
    """
    return super(DNNLinearCombinedClassifier, self).predict(
        x=x, input_fn=input_fn, batch_size=batch_size)

  def _get_eval_ops(self, features, targets, metrics=None):
    """See base class."""
    logits = self._logits(features)
    result = {"loss": metrics_lib.streaming_mean(self._loss(
        logits, targets, features))}

    # Adds default metrics.
    if metrics is None:
      # TODO(b/29366811): This currently results in both an "accuracy" and an
      # "accuracy/threshold_0.500000_mean" metric for binary classification.
      metrics = {("accuracy", "classes"): metrics_lib.streaming_accuracy}

    # Adds additional useful metrics for the special case of binary
    # classification.
    # TODO(zakaria): Move LogisticRegressor.get_default_metrics to metrics
    #   and handle eval metric from targetcolumn.
    if self._target_column.num_label_columns == 1:
      predictions = math_ops.sigmoid(logits)
      targets_float = math_ops.to_float(targets)
      default_metrics = (
          logistic_regressor.LogisticRegressor.get_default_metrics())
      for metric_name, metric_op in default_metrics.items():
        result[metric_name] = metric_op(predictions, targets_float)

    if metrics:
      class_metrics = {}
      proba_metrics = {}
      for name, metric_op in six.iteritems(metrics):
        if isinstance(name, tuple):
          if len(name) != 2:
            raise ValueError("Ignoring metric {}. It returned a tuple with "
                             "len {}, expected 2.".format(name, len(name)))
          else:
            if name[1] not in ["classes", "probabilities"]:
              raise ValueError("Ignoring metric {}. The 2nd element of its "
                               "name should be either 'classes' or "
                               "'probabilities'.".format(name))
            elif name[1] == "classes":
              class_metrics[name[0]] = metric_op
            else:
              proba_metrics[name[0]] = metric_op
        elif isinstance(name, str):
          class_metrics[name] = metric_op
        else:
          raise ValueError("Ignoring metric {}. Its name is not in the correct "
                           "form.".format(name))
      if class_metrics:
        predictions = self._target_column.logits_to_predictions(logits,
                                                                proba=False)
        result.update(self._run_metrics(predictions, targets, class_metrics,
                                        self._target_column.get_weight_tensor(
                                            features)))
      if proba_metrics:
        predictions = self._target_column.logits_to_predictions(logits,
                                                                proba=True)
        result.update(self._run_metrics(predictions, targets, proba_metrics,
                                        self._target_column.get_weight_tensor(
                                            features)))

    return result


class DNNLinearCombinedRegressor(_DNNLinearCombinedBaseEstimator):
  """A regressor for TensorFlow Linear and DNN joined training models.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_x_occupation = crossed_column(columns=[education, occupation],
                                          hash_bucket_size=10000)
  education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                   combiner="sum")
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")

  estimator = DNNLinearCombinedClassifier(
      # common settings
      n_classes=n_classes,
      weight_column_name=weight_column_name,
      # wide settings
      linear_feature_columns=[education_x_occupation],
      linear_optimizer=tf.train.FtrlOptimizer(...),
      # deep settings
      dnn_feature_columns=[education_emb, occupation_emb],
      dnn_hidden_units=[1000, 500, 100],
      dnn_optimizer=tf.train.ProximalAdagradOptimizer(...))

  # To apply L1 and L2 regularization, you can set optimizers as follows:
  tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001,
      l2_regularization_strength=0.001)
  # It is same for FtrlOptimizer.

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...
  estimator.train(input_fn_train)
  estimator.evaluate(input_fn_eval)
  estimator.predict(x)
  ```

  Input of `fit`, `train`, and `evaluate` should have following features,
    otherwise there will be a `KeyError`:
      if `weight_column_name` is not `None`, a feature with
        `key=weight_column_name` whose value is a `Tensor`.
      for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
  """

  def __init__(self,
               model_dir=None,
               weight_column_name=None,
               linear_feature_columns=None,
               linear_optimizer=None,
               dnn_feature_columns=None,
               dnn_optimizer=None,
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               gradient_clip_norm=None,
               enable_centered_bias=True,
               target_dimension=1,
               config=None):
    """Initializes a DNNLinearCombinedRegressor instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      linear_feature_columns: An iterable containing all the feature columns
        used by linear part of the model. All items in the set must be
        instances of classes derived from `FeatureColumn`.
      linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. If `None`, will use a FTRL optimizer.
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
      target_dimension: TODO(zakaria): dimension of the target for multilabels.
      config: RunConfig object to configure the runtime settings.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
    """
    super(DNNLinearCombinedRegressor, self).__init__(
        model_dir=model_dir,
        linear_feature_columns=linear_feature_columns,
        linear_optimizer=linear_optimizer,
        dnn_feature_columns=dnn_feature_columns,
        dnn_optimizer=dnn_optimizer,
        dnn_hidden_units=dnn_hidden_units,
        dnn_activation_fn=dnn_activation_fn,
        dnn_dropout=dnn_dropout,
        gradient_clip_norm=gradient_clip_norm,
        enable_centered_bias=enable_centered_bias,
        target_column=layers.regression_target(
            weight_column_name=weight_column_name,
            target_dimension=target_dimension),
        config=config)

  def predict(self, x=None, input_fn=None, batch_size=None):
    """Returns predictions for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.

    Returns:
      Numpy array of predicted classes or regression values.
    """
    return super(DNNLinearCombinedRegressor, self).predict(
        x=x, input_fn=input_fn, batch_size=batch_size)

  def _get_eval_ops(self, features, targets, metrics=None):
    """See base class."""
    logits = self._logits(features)
    result = {"loss": metrics_lib.streaming_mean(self._loss(
        logits, targets, features))}

    if metrics:
      predictions = self._target_column.logits_to_predictions(logits,
                                                              proba=False)
      result.update(self._run_metrics(predictions, targets, metrics,
                                      self._target_column.get_weight_tensor(
                                          features)))

    return result
