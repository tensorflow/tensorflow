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

"""Deep Neural Network estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from tensorflow.contrib import layers
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.framework import list_variables
from tensorflow.contrib.framework import load_variable
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn import session_run_hook
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.utils import checkpoints
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.contrib.losses.python.losses import loss_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import training as train


_CENTERED_BIAS = "centered_bias"
_CENTERED_BIAS_WEIGHT = "centered_bias_weight"
_CLASSES = "classes"
_LOGISTIC = "logistic"
_PROBABILITIES = "probabilities"

# The default learning rate of 0.05 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.05


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


def _add_hidden_layer_summary(value, tag):
  logging_ops.scalar_summary("%s:fraction_of_zero_values" % tag,
                             nn.zero_fraction(value))
  logging_ops.histogram_summary("%s:activation" % tag, value)


def _centered_bias(num_label_columns):
  centered_bias = variables.Variable(
      array_ops.zeros([num_label_columns]),
      collections=[_CENTERED_BIAS, ops.GraphKeys.VARIABLES],
      name=_CENTERED_BIAS_WEIGHT)
  logging_ops.scalar_summary(
      ["centered_bias %d" % cb for cb in range(num_label_columns)],
      array_ops.reshape(centered_bias, [-1]))
  return centered_bias


def _centered_bias_step(targets, loss_fn, num_label_columns):
  centered_bias = ops.get_collection(_CENTERED_BIAS)
  batch_size = array_ops.shape(targets)[0]
  logits = array_ops.reshape(
      array_ops.tile(centered_bias[0], [batch_size]),
      [batch_size, num_label_columns])
  loss = loss_fn(logits, targets)
  return train.AdagradOptimizer(0.1).minimize(loss, var_list=centered_bias)


def _get_weight_tensor(features, weight_column_name):
  """Returns the weight tensor of shape [batch_size] or 1."""
  if weight_column_name is None:
    return 1.0
  else:
    return array_ops.reshape(
        math_ops.to_float(features[weight_column_name]),
        shape=(-1,))


def _rescale_eval_loss(loss, weights):
  """Rescales evaluation loss according to the given weights.

  The rescaling is needed because in the training loss weights are not
  considered in the denominator, whereas  for the evaluation loss we should
  divide by the sum of weights.

  The rescaling factor is:
    R = sum_{i} 1 / sum_{i} w_{i}

  Args:
    loss: the scalar weighted loss.
    weights: weight coefficients. Either a scalar, or a `Tensor` of shape
      [batch_size].

  Returns:
    The given loss multiplied by the rescaling factor.
  """
  rescaling_factor = math_ops.reduce_mean(weights)
  return math_ops.div(loss, rescaling_factor)


def _predictions(logits, n_classes):
  """Returns predictions for the given logits and n_classes."""
  predictions = {}
  if n_classes == 2:
    predictions[_LOGISTIC] = math_ops.sigmoid(logits)
    logits = array_ops.concat(1, [array_ops.zeros_like(logits), logits])
  predictions[_PROBABILITIES] = nn.softmax(logits)
  predictions[_CLASSES] = array_ops.reshape(
      math_ops.argmax(logits, 1), shape=(-1, 1))
  return predictions


def _dnn_classifier_model_fn(features, targets, mode, params):
  """Deep Neural Net model_fn.

  Args:
    features: `Tensor` or dict of `Tensor` (depends on data passed to `fit`).
    targets: `Tensor` of shape [batch_size, 1] or [batch_size] target labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * hidden_units: List of hidden units per layer.
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * n_classes: number of target classes.
      * weight_column_name: A string defining the weight feature column, or
          None if there are no weights.
      * optimizer: string, `Optimizer` object, or callable that defines the
          optimizer to use for training.
      * activation_fn: Activation function applied to each layer. If `None`,
          will use `tf.nn.relu`.
      * dropout: When not `None`, the probability we will drop out a given
          coordinate.
      * gradient_clip_norm: A float > 0. If provided, gradients are
          clipped to their global norm with this clipping ratio.
      * enable_centered_bias: A bool. If True, estimator will learn a centered
          bias variable for each class. Rest of the model structure learns the
          residual after centered bias.
      * num_ps_replicas: The number of parameter server replicas.

  Returns:
    predictions: A dict of `Tensor` objects.
    loss: A scalar containing the loss of the step.
    train_op: The op for training.
  """
  hidden_units = params["hidden_units"]
  feature_columns = params["feature_columns"]
  n_classes = params["n_classes"]
  weight_column_name = params["weight_column_name"]
  optimizer = params["optimizer"]
  activation_fn = params["activation_fn"]
  dropout = params["dropout"]
  gradient_clip_norm = params["gradient_clip_norm"]
  enable_centered_bias = params["enable_centered_bias"]
  num_ps_replicas = params["num_ps_replicas"]

  features = _get_feature_dict(features)
  parent_scope = "dnn"
  num_label_columns = 1 if n_classes == 2 else n_classes
  if n_classes == 2:
    loss_fn = loss_ops.sigmoid_cross_entropy
  else:
    loss_fn = loss_ops.sparse_softmax_cross_entropy

  input_layer_partitioner = (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))
  with variable_scope.variable_scope(
      parent_scope + "/input_from_feature_columns",
      values=features.values(),
      partitioner=input_layer_partitioner) as scope:
    net = layers.input_from_feature_columns(
        columns_to_tensors=features,
        feature_columns=feature_columns,
        weight_collections=[parent_scope],
        scope=scope)

  hidden_layer_partitioner = (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas))
  for layer_id, num_hidden_units in enumerate(hidden_units):
    with variable_scope.variable_scope(
        parent_scope + "/hiddenlayer_%d" % layer_id,
        values=[net],
        partitioner=hidden_layer_partitioner) as scope:
      net = layers.fully_connected(
          net,
          num_hidden_units,
          activation_fn=activation_fn,
          variables_collections=[parent_scope],
          scope=scope)
      if dropout is not None and mode == estimator.ModeKeys.TRAIN:
        net = layers.dropout(
            net,
            keep_prob=(1.0 - dropout))
    _add_hidden_layer_summary(net, scope.name)

  with variable_scope.variable_scope(
      parent_scope + "/logits",
      values=[net],
      partitioner=hidden_layer_partitioner) as scope:
    logits = layers.fully_connected(
        net,
        num_label_columns,
        activation_fn=None,
        variables_collections=[parent_scope],
        scope=scope)
  _add_hidden_layer_summary(logits, scope.name)

  if enable_centered_bias:
    logits = nn.bias_add(logits, _centered_bias(num_label_columns))

  if mode == estimator.ModeKeys.TRAIN:
    loss = loss_fn(logits, targets,
                   weight=_get_weight_tensor(features, weight_column_name))

    train_ops = [optimizers.optimize_loss(
        loss=loss, global_step=contrib_variables.get_global_step(),
        learning_rate=_LEARNING_RATE, optimizer=_get_optimizer(optimizer),
        clip_gradients=gradient_clip_norm, name=parent_scope)]
    if enable_centered_bias:
      train_ops.append(_centered_bias_step(targets, loss_fn, num_label_columns))

    return None, loss, control_flow_ops.group(*train_ops)

  elif mode == estimator.ModeKeys.EVAL:
    predictions = _predictions(logits=logits, n_classes=n_classes)

    weight = _get_weight_tensor(features, weight_column_name)
    training_loss = loss_fn(logits, targets, weight=weight)
    loss = _rescale_eval_loss(training_loss, weight)

    return predictions, loss, []

  else:  # mode == estimator.ModeKeys.INFER:
    predictions = _predictions(logits=logits, n_classes=n_classes)

    return predictions, None, []


class DNNClassifier(evaluable.Evaluable, trainable.Trainable):
  """A classifier for TensorFlow DNN models.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                   combiner="sum")
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")

  estimator = DNNClassifier(
      feature_columns=[education_emb, occupation_emb],
      hidden_units=[1024, 512, 256])

  # Or estimator using the ProximalAdagradOptimizer optimizer with
  # regularization.
  estimator = DNNClassifier(
      feature_columns=[education_emb, occupation_emb],
      hidden_units=[1024, 512, 256],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Input builders
  def input_fn_train: # returns x, Y
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, Y
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column_name` is not `None`, a feature with
     `key=weight_column_name` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.
  """

  def __init__(self,
               hidden_units,
               feature_columns,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               optimizer=None,
               activation_fn=nn.relu,
               dropout=None,
               gradient_clip_norm=None,
               enable_centered_bias=None,
               config=None):
    """Initializes a DNNClassifier instance.

    Args:
      hidden_units: List of hidden units per layer. All layers are fully
        connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
        has 32.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can also
        be used to load checkpoints from the directory into a estimator to continue
        training a previously saved model.
      n_classes: number of target classes. Default is binary classification.
        It must be greater than 1.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      optimizer: An instance of `tf.Optimizer` used to train the model. If
        `None`, will use an Adagrad optimizer.
      activation_fn: Activation function applied to each layer. If `None`, will
        use `tf.nn.relu`.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are
        clipped to their global norm with this clipping ratio. See
        `tf.clip_by_global_norm` for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      A `DNNClassifier` estimator.

    Raises:
      ValueError: If `n_classes` < 2.
    """
    if enable_centered_bias is None:
      enable_centered_bias = True
      dnn_linear_combined._changing_default_center_bias()  # pylint: disable=protected-access
    self._hidden_units = hidden_units
    self._feature_columns = feature_columns
    self._model_dir = model_dir or tempfile.mkdtemp()
    if n_classes <= 1:
      raise ValueError(
          "Classification requires n_classes >= 2. Given: {}".format(n_classes))
    self._n_classes = n_classes
    self._weight_column_name = weight_column_name
    optimizer = optimizer or "Adagrad"
    num_ps_replicas = config.num_ps_replicas if config else 0

    self._estimator = estimator.Estimator(
        model_fn=_dnn_classifier_model_fn,
        model_dir=self._model_dir,
        config=config,
        params={
            "hidden_units": hidden_units,
            "feature_columns": feature_columns,
            "n_classes": n_classes,
            "weight_column_name": weight_column_name,
            "optimizer": optimizer,
            "activation_fn": activation_fn,
            "dropout": dropout,
            "gradient_clip_norm": gradient_clip_norm,
            "enable_centered_bias": enable_centered_bias,
            "num_ps_replicas": num_ps_replicas,
        })

  def fit(self, x=None, y=None, input_fn=None, steps=None, batch_size=None,
          monitors=None, max_steps=None):
    """See trainable.Trainable."""
    # TODO(roumposg): Remove when deprecated monitors are removed.
    if monitors is not None:
      deprecated_monitors = [
          m for m in monitors
          if not isinstance(m, session_run_hook.SessionRunHook)
      ]
      for monitor in deprecated_monitors:
        monitor.set_estimator(self)
        monitor._lock_estimator()  # pylint: disable=protected-access

    result = self._estimator.fit(x=x, y=y, input_fn=input_fn, steps=steps,
                                 batch_size=batch_size, monitors=monitors,
                                 max_steps=max_steps)

    if monitors is not None:
      for monitor in deprecated_monitors:
        monitor._unlock_estimator()  # pylint: disable=protected-access

    return result

  def evaluate(self, x=None, y=None, input_fn=None, feed_fn=None,
               batch_size=None, steps=None, metrics=None, name=None):
    """See evaluable.Evaluable."""
    if metrics is None:
      metrics = {}
    metrics.update({
        "accuracy": metric_spec.MetricSpec(
            metric_fn=metrics_lib.streaming_accuracy,
            prediction_key=_CLASSES,
            weight_key=self._weight_column_name)})
    if self._n_classes == 2:
      metrics.update({
          "auc": metric_spec.MetricSpec(
              metric_fn=metrics_lib.streaming_auc,
              prediction_key=_LOGISTIC,
              weight_key=self._weight_column_name)})
    return self._estimator.evaluate(
        x=x, y=y, input_fn=input_fn, feed_fn=feed_fn, batch_size=batch_size,
        steps=steps, metrics=metrics, name=name)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict(self, x=None, input_fn=None, batch_size=None, as_iterable=False):
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
      Numpy array of predicted classes (or an iterable of predicted classes if
      as_iterable is True).
    """
    preds = self._estimator.predict(x=x, input_fn=input_fn,
                                    batch_size=batch_size, outputs=[_CLASSES],
                                    as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=_CLASSES)
    return preds[_CLASSES].reshape(-1)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_proba(
      self, x=None, input_fn=None, batch_size=None, as_iterable=False):
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
      Numpy array of predicted probabilities (or an iterable of predicted
      probabilities if as_iterable is True).
    """
    preds = self._estimator.predict(x=x, input_fn=input_fn,
                                    batch_size=batch_size,
                                    outputs=[_PROBABILITIES],
                                    as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=_PROBABILITIES)
    return preds[_PROBABILITIES]

  def get_variable_names(self):
    """Returns list of all variable names in this model.

    Returns:
      List of names.
    """
    return [name for name, _ in list_variables(self._model_dir)]

  def get_variable_value(self, name):
    """Returns value of the variable given by name.

    Args:
      name: string, name of the tensor.

    Returns:
      `Tensor` object.
    """
    return load_variable(self._model_dir, name)

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
    return self._estimator.export(
        export_dir=export_dir,
        input_fn=input_fn or default_input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=use_deprecated_input_fn,
        signature_fn=(
            signature_fn or export.classification_signature_fn_with_prob),
        prediction_key=_PROBABILITIES,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)

  @property
  def model_dir(self):
    return self._model_dir

  @property
  @deprecated("2016-10-13", "This method inspects the private state of the "
              "object, and should not be used")
  def weights_(self):
    hiddenlayer_weights = [checkpoints.load_variable(
        self._model_dir, name=("dnn/hiddenlayer_%d/weights" % i))
                           for i, _ in enumerate(self._hidden_units)]
    logits_weights = [checkpoints.load_variable(
        self._model_dir, name="dnn/logits/weights")]
    return hiddenlayer_weights + logits_weights

  @property
  @deprecated("2016-10-13", "This method inspects the private state of the "
              "object, and should not be used")
  def bias_(self):
    hiddenlayer_bias = [checkpoints.load_variable(
        self._model_dir, name=("dnn/hiddenlayer_%d/biases" % i))
                        for i, _ in enumerate(self._hidden_units)]
    logits_bias = [checkpoints.load_variable(
        self._model_dir, name="dnn/logits/biases")]
    centered_bias = [checkpoints.load_variable(
        self._model_dir, name=_CENTERED_BIAS_WEIGHT)]
    return hiddenlayer_bias + logits_bias + centered_bias

  @property
  def config(self):
    return self._estimator.config


class DNNRegressor(dnn_linear_combined.DNNLinearCombinedRegressor):
  """A regressor for TensorFlow DNN models.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                   combiner="sum")
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")

  estimator = DNNRegressor(
      feature_columns=[education_emb, occupation_emb],
      hidden_units=[1024, 512, 256])

  # Or estimator using the ProximalAdagradOptimizer optimizer with
  # regularization.
  estimator = DNNRegressor(
      feature_columns=[education_emb, occupation_emb],
      hidden_units=[1024, 512, 256],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Input builders
  def input_fn_train: # returns x, Y
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, Y
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column_name` is not `None`, a feature with
    `key=weight_column_name` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.
  """

  def __init__(self,
               hidden_units,
               feature_columns,
               model_dir=None,
               weight_column_name=None,
               optimizer=None,
               activation_fn=nn.relu,
               dropout=None,
               gradient_clip_norm=None,
               enable_centered_bias=None,
               config=None):
    """Initializes a `DNNRegressor` instance.

    Args:
      hidden_units: List of hidden units per layer. All layers are fully
        connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
        has 32.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can also
        be used to load checkpoints from the directory into a estimator to continue
        training a previously saved model.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      optimizer: An instance of `tf.Optimizer` used to train the model. If
        `None`, will use an Adagrad optimizer.
      activation_fn: Activation function applied to each layer. If `None`, will
        use `tf.nn.relu`.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      gradient_clip_norm: A `float` > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        `tf.clip_by_global_norm` for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      A `DNNRegressor` estimator.
    """
    if enable_centered_bias is None:
      enable_centered_bias = True
      dnn_linear_combined._changing_default_center_bias()  # pylint: disable=protected-access
    super(DNNRegressor, self).__init__(
        model_dir=model_dir,
        weight_column_name=weight_column_name,
        dnn_feature_columns=feature_columns,
        dnn_optimizer=optimizer,
        dnn_hidden_units=hidden_units,
        dnn_activation_fn=activation_fn,
        dnn_dropout=dropout,
        gradient_clip_norm=gradient_clip_norm,
        enable_centered_bias=enable_centered_bias,
        config=config)
    self.feature_columns = feature_columns
    self.optimizer = optimizer
    self.activation_fn = activation_fn
    self.dropout = dropout
    self.hidden_units = hidden_units
    self._feature_columns_inferred = False

  @property
  def weights_(self):
    return self.dnn_weights_

  @property
  def bias_(self):
    return self.dnn_bias_
