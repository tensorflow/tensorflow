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
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import training


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
    dnn_dropout: When not None, the probability we will drop out
      a given coordinate.
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
               dnn_dropout=None,
               config=None):
    super(_DNNLinearCombinedBaseEstimator, self).__init__(model_dir=model_dir,
                                                          config=config)
    self._weight_column_name = weight_column_name
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
                self.get_variable_value("dnn_logit/weights")]

  @property
  def dnn_bias_(self):
    """Returns bias of deep neural network part."""
    return [self.get_variable_value("hiddenlayer_%d/bias" % i)
            for i, _ in enumerate(self._dnn_hidden_units)] + [
                self.get_variable_value("dnn_logit/bias"),
                self.get_variable_value("centered_bias_weight")]

  def _get_train_ops(self, features, targets):
    """See base class."""
    global_step = contrib_variables.get_global_step()
    assert global_step
    logits = self._logits(features, is_training=True)
    with ops.control_dependencies([self._centered_bias_step(
        targets, self._get_weight_tensor(features))]):
      loss = self._loss(logits, targets, self._get_weight_tensor(features))
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
    raise NotImplementedError

  def _get_predict_ops(self, features):
    """See base class."""
    logits = self._logits(features)
    return self._logits_to_predictions(logits, proba=True)

  def _logits_to_predictions(self, logits, proba=False):
    raise NotImplementedError

  def _get_feature_ops_from_example(self, examples_batch):
    column_types = layers.create_feature_spec_for_parsing((
        self._get_linear_feature_columns() or []) + (
            self._get_dnn_feature_columns() or []))
    features = parsing_ops.parse_example(examples_batch, column_types)
    return features

  def _num_label_columns(self):
    raise NotImplementedError

  def _get_linear_feature_columns(self):
    return sorted(
        set(self._linear_feature_columns),
        key=lambda x: x.key) if self._linear_feature_columns else None

  def _get_dnn_feature_columns(self):
    return sorted(set(
        self._dnn_feature_columns)) if self._dnn_feature_columns else None

  def _dnn_logits(self, features, is_training=False):
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
      if self._dnn_dropout is not None and is_training:
        net = layers.dropout(
            net,
            keep_prob=(1.0 - self._dnn_dropout))
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

  def _centered_bias(self):
    centered_bias = variables.Variable(
        array_ops.zeros([self._num_label_columns()]),
        collections=[self._centered_bias_weight_collection,
                     ops.GraphKeys.VARIABLES],
        name="centered_bias_weight")
    logging_ops.scalar_summary(
        ["centered_bias_%d" % cb for cb in range(self._num_label_columns())],
        array_ops.reshape(centered_bias, [-1]))
    return centered_bias

  def _centered_bias_step(self, targets, weight_tensor):
    centered_bias = ops.get_collection(self._centered_bias_weight_collection)
    batch_size = array_ops.shape(targets)[0]
    logits = array_ops.reshape(
        array_ops.tile(centered_bias[0], [batch_size]),
        [batch_size, self._num_label_columns()])
    loss = self._loss(logits, targets, weight_tensor)
    # Learn central bias by an optimizer. 0.1 is a convervative lr for a single
    # variable.
    return training.AdagradOptimizer(0.1).minimize(loss, var_list=centered_bias)

  def _logits(self, features, is_training=False):
    if not (self._get_linear_feature_columns() or
            self._get_dnn_feature_columns()):
      raise ValueError("Either linear_feature_columns or dnn_feature_columns "
                       "should be defined.")

    features = self._get_feature_dict(features)
    if self._get_linear_feature_columns() and self._get_dnn_feature_columns():
      logits = (self._linear_logits(features) +
                self._dnn_logits(features, is_training=is_training))
    elif self._get_dnn_feature_columns():
      logits = self._dnn_logits(features, is_training=is_training)
    else:
      logits = self._linear_logits(features)

    return nn.bias_add(logits, self._centered_bias())

  def _get_weight_tensor(self, features):
    if not self._weight_column_name:
      return None
    else:
      return array_ops.reshape(
          math_ops.to_float(features[self._weight_column_name]),
          shape=(-1,))

  def _loss_vec(self, logits, targets):
    raise NotImplementedError

  def _loss(self, logits, target, weight_tensor):
    loss_vec = self._loss_vec(logits, target)

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
               config=None):

    if n_classes < 2:
      raise ValueError("n_classes should be greater than 1. Given: {}".format(
          n_classes))

    self._n_classes = n_classes
    super(DNNLinearCombinedClassifier, self).__init__(
        model_dir=model_dir,
        weight_column_name=weight_column_name,
        linear_feature_columns=linear_feature_columns,
        linear_optimizer=linear_optimizer,
        dnn_feature_columns=dnn_feature_columns,
        dnn_optimizer=dnn_optimizer,
        dnn_hidden_units=dnn_hidden_units,
        dnn_activation_fn=dnn_activation_fn,
        dnn_dropout=dnn_dropout,
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

  def _loss_vec(self, logits, target):
    # Check that we got int32/int64 for classification.
    if (not target.dtype.is_compatible_with(dtypes.int64) and
        not target.dtype.is_compatible_with(dtypes.int32)):
      raise ValueError("Target's dtype should be int32, int64 or compatible. "
                       "Instead got %s." % target.dtype)

    if self._n_classes == 2:
      # sigmoid_cross_entropy_with_logits requires [batch_size, 1] target.
      if len(target.get_shape()) == 1:
        target = array_ops.expand_dims(target, dim=[1])
      loss_vec = nn.sigmoid_cross_entropy_with_logits(
          logits, math_ops.to_float(target))
    else:
      # sparse_softmax_cross_entropy_with_logits requires [batch_size] target.
      if len(target.get_shape()) == 2:
        target = array_ops.squeeze(target, squeeze_dims=[1])
      loss_vec = nn.sparse_softmax_cross_entropy_with_logits(
          logits, target)
    return loss_vec

  def _logits_to_predictions(self, logits, proba=False):
    if self._n_classes == 2:
      logits = array_ops.concat(1, [array_ops.zeros_like(logits), logits])

    if proba:
      return nn.softmax(logits)
    else:
      return math_ops.argmax(logits, 1)

  def _num_label_columns(self):
    return 1 if self._n_classes == 2 else self._n_classes

  def _get_eval_ops(self, features, targets, metrics=None):
    """See base class."""
    logits = self._logits(features)
    result = {"loss": metrics_lib.streaming_mean(self._loss(
        logits, targets,
        weight_tensor=self._get_weight_tensor(features)))}

    # Adding default metrics
    if metrics is None:
      metrics = {"accuracy": metrics_lib.streaming_accuracy}

    if self._n_classes == 2:
      predictions = math_ops.sigmoid(logits)
      result["eval_auc"] = metrics_lib.streaming_auc(predictions, targets)

    if metrics:
      predictions = self._logits_to_predictions(logits, proba=False)
      result.update(self._run_metrics(predictions, targets, metrics,
                                      self._get_weight_tensor(features)))

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
               config=None):
    super(DNNLinearCombinedRegressor, self).__init__(
        model_dir=model_dir,
        weight_column_name=weight_column_name,
        linear_feature_columns=linear_feature_columns,
        linear_optimizer=linear_optimizer,
        dnn_feature_columns=dnn_feature_columns,
        dnn_optimizer=dnn_optimizer,
        dnn_hidden_units=dnn_hidden_units,
        dnn_activation_fn=dnn_activation_fn,
        dnn_dropout=dnn_dropout,
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

  def _loss_vec(self, logits, target):
    # To prevent broadcasting inside "-".
    if len(target.get_shape()) == 1:
      target = array_ops.expand_dims(target, dim=[1])
    logits.get_shape().assert_is_compatible_with(target.get_shape())
    return math_ops.square(logits - math_ops.to_float(target))

  def _logits_to_predictions(self, logits, proba=False):
    # TODO(ispir): Add target column support.
    if self._targets_info is None or len(self._targets_info.shape) == 1:
      return array_ops.squeeze(logits, squeeze_dims=[1])
    return logits

  def _num_label_columns(self):
    # TODO(ispir): Add target column support.
    if self._targets_info is None or len(self._targets_info.shape) == 1:
      return 1
    return int(self._targets_info.shape[1])

  def _get_eval_ops(self, features, targets, metrics=None):
    """See base class."""
    logits = self._logits(features)
    result = {"loss": metrics_lib.streaming_mean(self._loss(
        logits, targets,
        weight_tensor=self._get_weight_tensor(features)))}

    # Adding default metrics
    if metrics:
      predictions = self._logits_to_predictions(logits, proba=False)
      result.update(self._run_metrics(predictions, targets, metrics,
                                      self._get_weight_tensor(features)))

    return result

