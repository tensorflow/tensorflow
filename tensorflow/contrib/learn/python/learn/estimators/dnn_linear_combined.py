# pylint: disable=g-bad-file-header
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
from tensorflow.contrib.framework.python.ops import variables as variables
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops


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

  Parameters:
    model_dir: Directory to save model parameters, graph and etc.
    n_classes: number of target classes. Default is binary classification.
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    linear_feature_columns: An iterable containing all the feature columns used
      by linear part of the model. All items in the set should be instances of
      classes derived from `FeatureColumn`.
    linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
      the linear part of the model. If `None`, will use a FTRL optimizer.
    dnn_feature_columns: An iterable containing all the feature columns used by
      deep part of the model. All items in the set should be instances of
      classes derived from `FeatureColumn`.
    dnn_hidden_units: List of hidden units per layer. All layers are fully
      connected.
    dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to the
      deep part of the model. If `None`, will use an Adagrad optimizer.
    dnn_activation_fn: Activation function applied to each layer. If `None`,
      will use `tf.nn.relu`.
    config: RunConfig object to configure the runtime settings.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
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
               config=None):
    super(_DNNLinearCombinedBaseEstimator, self).__init__(model_dir=model_dir,
                                                          config=config)
    self._n_classes = n_classes
    self._weight_column_name = weight_column_name
    self._linear_feature_columns = linear_feature_columns
    self._linear_optimizer = linear_optimizer
    self._dnn_feature_columns = dnn_feature_columns
    self._dnn_optimizer = dnn_optimizer
    self._dnn_hidden_units = dnn_hidden_units
    self._dnn_activation_fn = dnn_activation_fn
    if self._dnn_activation_fn is None:
      self._dnn_activation_fn = nn.relu
    self._dnn_weight_collection = "DNNLinearCombined_dnn"
    self._linear_weight_collection = "DNNLinearCombined_linear"

  def predict(self, x=None, input_fn=None, batch_size=None):
    """Returns predictions for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.

    Returns:
      Numpy array of predicted classes or regression values.
    """
    predictions = self._infer_model(x=x,
                                    input_fn=input_fn,
                                    batch_size=batch_size)
    if self._n_classes > 1:
      predictions = np.argmax(predictions, axis=1)
    return predictions

  def predict_proba(self, x=None, input_fn=None, batch_size=None):
    """Returns prediction probabilities for given features (classification).

    Args:
      x: features.
      input_fn: Input function. If set, x and y must be None.
      batch_size: Override default batch size.

    Returns:
      Numpy array of predicted probabilities.
    """
    return self._infer_model(x=x, input_fn=input_fn, batch_size=batch_size)

  def _get_train_ops(self, features, targets):
    """See base class."""
    global_step = variables.get_global_step()
    assert global_step
    loss = self._loss(
        self._logits(features), targets, self._get_weight_tensor(features))
    logging_ops.scalar_summary("loss", loss)

    linear_vars = self._get_linear_vars()
    dnn_vars = self._get_dnn_vars()
    grads = gradients.gradients(loss, dnn_vars + linear_vars)
    dnn_grads = grads[0:len(dnn_vars)]
    linear_grads = grads[len(dnn_vars):]

    train_ops = self._get_linear_training_ops(
        linear_grads, linear_vars) + self._get_dnn_training_ops(dnn_grads,
                                                                dnn_vars)

    train_step = control_flow_ops.group(*train_ops, name="combined_training_op")
    with ops.control_dependencies([train_step]):
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
    """See base class."""
    logits = self._logits(features)
    result = {"loss": metrics_lib.streaming_mean(self._loss(
        logits, targets,
        weight_tensor=self._get_weight_tensor(features)))}

    # Adding default metrics
    if metrics is None and self._n_classes > 1:
      metrics = {"accuracy": metrics_lib.streaming_accuracy}

    if self._n_classes == 2:
      predictions = math_ops.sigmoid(logits)
      result["eval_auc"] = metrics_lib.streaming_auc(predictions, targets)

    if metrics:
      predictions = self._logits_to_predictions(logits, proba=False)
      result.update(self._run_metrics(predictions, targets, metrics,
                                      self._get_weight_tensor(features)))

    return result

  def _get_predict_ops(self, features):
    """See base class."""
    logits = self._logits(features)
    return self._logits_to_predictions(logits, proba=True)

  def _logits_to_predictions(self, logits, proba=False):
    if self._n_classes < 2:
      return array_ops.reshape(logits, [-1])

    if self._n_classes == 2:
      logits = array_ops.concat(1, [array_ops.zeros_like(logits), logits])

    if proba:
      return nn.softmax(logits)
    else:
      return math_ops.argmax(logits, 1)

  def _get_feature_ops_from_example(self, examples_batch):
    column_types = layers.create_dict_for_parse_example(
        (self._get_linear_feature_columns() or []) +
        (self._get_dnn_feature_columns() or []))
    features = parsing_ops.parse_example(examples_batch, column_types)
    return features

  def _num_label_columns(self):
    return 1 if self._n_classes <= 2 else self._n_classes

  def _get_linear_feature_columns(self):
    return sorted(
        set(self._linear_feature_columns),
        key=lambda x: x.key) if self._linear_feature_columns else None

  def _get_dnn_feature_columns(self):
    return sorted(set(
        self._dnn_feature_columns)) if self._dnn_feature_columns else None

  def _dnn_logits(self, features):
    net = layers.input_from_feature_columns(
        features,
        self._get_dnn_feature_columns(),
        weight_collections=[self._dnn_weight_collection])
    for layer_id, num_hidden_units in enumerate(self._dnn_hidden_units):
      net = layers.legacy_fully_connected(
          net,
          num_hidden_units,
          activation_fn=self._dnn_activation_fn,
          weight_collections=[self._dnn_weight_collection],
          bias_collections=[self._dnn_weight_collection],
          name="hiddenlayer_%d" % layer_id)
      self._add_hidden_layer_summary(net, "hiddenlayer_%d" % layer_id)
    logit = layers.legacy_fully_connected(
        net,
        self._num_label_columns(),
        weight_collections=[self._dnn_weight_collection],
        bias_collections=[self._dnn_weight_collection],
        name="dnn_logit")
    self._add_hidden_layer_summary(logit, "dnn_logit")
    return logit

  def _add_hidden_layer_summary(self, value, tag):
    # TODO(zakaria): Move this code to tf.learn and add test.
    logging_ops.scalar_summary("%s:fraction_of_zero_values" % tag,
                               nn.zero_fraction(value))
    logging_ops.histogram_summary("%s:activation" % tag, value)

  def _linear_logits(self, features):
    logits, _, _ = layers.weighted_sum_from_feature_columns(
        columns_to_tensors=features,
        feature_columns=self._get_linear_feature_columns(),
        num_outputs=self._num_label_columns(),
        weight_collections=[self._linear_weight_collection],
        name="linear")
    return logits

  def _get_feature_dict(self, features):
    if isinstance(features, dict):
      return features
    return {"": features}

  def _logits(self, features):
    if not (self._get_linear_feature_columns() or
            self._get_dnn_feature_columns()):
      raise ValueError("Either linear_feature_columns or dnn_feature_columns "
                       "should be defined.")

    features = self._get_feature_dict(features)
    if self._get_linear_feature_columns() and self._get_dnn_feature_columns():
      return self._linear_logits(features) + self._dnn_logits(features)
    elif self._get_dnn_feature_columns():
      return self._dnn_logits(features)
    else:
      return self._linear_logits(features)

  def _get_weight_tensor(self, features):
    if not self._weight_column_name:
      return None
    else:
      return array_ops.reshape(
          math_ops.to_float(features[self._weight_column_name]),
          shape=(-1,))

  def _loss(self, logits, target, weight_tensor):
    if self._n_classes < 2:
      loss_vec = math_ops.square(logits - math_ops.to_float(target))
    elif self._n_classes == 2:
      loss_vec = nn.sigmoid_cross_entropy_with_logits(logits,
                                                      math_ops.to_float(target))
    else:
      loss_vec = nn.sparse_softmax_cross_entropy_with_logits(
          logits, array_ops.reshape(target, [-1]))

    if weight_tensor is None:
      return math_ops.reduce_mean(loss_vec, name="loss")
    else:
      loss_vec = array_ops.reshape(loss_vec, shape=(-1,))
      loss_vec = math_ops.mul(
          loss_vec, array_ops.reshape(weight_tensor, shape=(-1,)))
      return math_ops.div(
          math_ops.reduce_sum(loss_vec),
          math_ops.to_float(math_ops.reduce_sum(weight_tensor)),
          name="loss")

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
    ```
    installed_app_id = sparse_column_with_hash_bucket("installed_id", 1e6)
    impression_app_id = sparse_column_with_hash_bucket("impression_id", 1e6)

    installed_x_impression = crossed_column(
        [installed_app_id, impression_app_id])

    installed_emb = embedding_column(installed_app_id, dimension=16,
                                     combiner="sum")
    impression_emb = embedding_column(impression_app_id, dimension=16,
                                      combiner="sum")

    estimator = DNNLinearCombinedClassifier(
        # common settings
        n_classes, weight_column_name,
        # wide settings
        linear_feature_columns=[installed_x_impression],
        linear_optimizer=tf.train.FtrlOptimizer(...),
        # deep settings
        dnn_feature_columns=[installed_emb, impression_emb],
        dnn_hidden_units=[1000, 500, 100],
        dnn_optimizer=tf.train.AdagradOptimizer(...))

    # Input builders
    def input_fn_train: # returns X, Y
      ...
    def input_fn_eval: # returns X, Y
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

  Parameters:
    model_dir: Directory to save model parameters, graph and etc.
    n_classes: number of target classes. Default is binary classification.
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    linear_feature_columns: An iterable containing all the feature columns used
      by linear part of the model. All items in the set must be instances of
      classes derived from `FeatureColumn`.
    linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
      the linear part of the model. If `None`, will use a FTRL optimizer.
    dnn_feature_columns: An iterable containing all the feature columns used by
      deep part of the model. All items in the set must be instances of
      classes derived from `FeatureColumn`.
    dnn_hidden_units: List of hidden units per layer. All layers are fully
      connected.
    dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to the
      deep part of the model. If `None`, will use an Adagrad optimizer.
    dnn_activation_fn: Activation function applied to each layer. If `None`,
      will use `tf.nn.relu`.
    config: RunConfig object to configure the runtime settings.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
      ValueError: If both n_classes < 2.
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
               config=None):
    if n_classes < 2:
      raise ValueError("n_classes should be greater than 1. Given: {}".format(
          n_classes))

    super(DNNLinearCombinedClassifier, self).__init__(
        model_dir=model_dir,
        n_classes=n_classes,
        weight_column_name=weight_column_name,
        linear_feature_columns=linear_feature_columns,
        linear_optimizer=linear_optimizer,
        dnn_feature_columns=dnn_feature_columns,
        dnn_optimizer=dnn_optimizer,
        dnn_hidden_units=dnn_hidden_units,
        dnn_activation_fn=dnn_activation_fn,
        config=config)


class DNNLinearCombinedRegressor(_DNNLinearCombinedBaseEstimator):
  """A regressor for TensorFlow Linear and DNN joined training models.

  Example:
    ```
    installed_app_id = sparse_column_with_hash_bucket("installed_id", 1e6)
    impression_app_id = sparse_column_with_hash_bucket("impression_id", 1e6)

    installed_x_impression = crossed_column(
        [installed_app_id, impression_app_id])

    installed_emb = embedding_column(installed_app_id, dimension=16,
                                     combiner="sum")
    impression_emb = embedding_column(impression_app_id, dimension=16,
                                      combiner="sum")

    estimator = DNNLinearCombinedClassifier(
        # common settings
        n_classes, weight_column_name,
        # wide settings
        linear_feature_columns=[installed_x_impression],
        linear_optimizer=tf.train.FtrlOptimizer(...),
        # deep settings
        dnn_feature_columns=[installed_emb, impression_emb],
        dnn_hidden_units=[1000, 500, 100],
        dnn_optimizer=tf.train.AdagradOptimizer(...))

    # Input builders
    def input_fn_train: # returns X, Y
      ...
    def input_fn_eval: # returns X, Y
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

  Parameters:
    model_dir: Directory to save model parameters, graph and etc.
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    linear_feature_columns: An iterable containing all the feature columns used
      by linear part of the model. All items in the set must be instances of
      classes derived from `FeatureColumn`.
    linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
      the linear part of the model. If `None`, will use a FTRL optimizer.
    dnn_feature_columns: An iterable containing all the feature columns used by
      deep part of the model. All items in the set must be instances of
      classes derived from `FeatureColumn`.
    dnn_hidden_units: List of hidden units per layer. All layers are fully
      connected.
    dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to the
      deep part of the model. If `None`, will use an Adagrad optimizer.
    dnn_activation_fn: Activation function applied to each layer. If None, will
      use `tf.nn.relu`.
    config: RunConfig object to configure the runtime settings.

    Raises:
      ValueError: If both linear_feature_columns and dnn_features_columns are
        empty at the same time.
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
               config=None):
    super(DNNLinearCombinedRegressor, self).__init__(
        model_dir=model_dir,
        n_classes=0,
        weight_column_name=weight_column_name,
        linear_feature_columns=linear_feature_columns,
        linear_optimizer=linear_optimizer,
        dnn_feature_columns=dnn_feature_columns,
        dnn_optimizer=dnn_optimizer,
        dnn_hidden_units=dnn_hidden_units,
        dnn_activation_fn=dnn_activation_fn,
        config=config)
