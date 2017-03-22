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

import six

from tensorflow.contrib import layers
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary

# The default learning rate of 0.05 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.05


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
  summary.scalar("%s_fraction_of_zero_values" % tag, nn.zero_fraction(value))
  summary.histogram("%s_activation" % tag, value)


def _dnn_model_fn(features, labels, mode, params, config=None):
  """Deep Neural Net model_fn.

  Args:
    features: `Tensor` or dict of `Tensor` (depends on data passed to `fit`).
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * head: A `_Head` instance.
      * hidden_units: List of hidden units per layer.
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * optimizer: string, `Optimizer` object, or callable that defines the
          optimizer to use for training. If `None`, will use the Adagrad
          optimizer with a default learning rate of 0.05.
      * activation_fn: Activation function applied to each layer. If `None`,
          will use `tf.nn.relu`.
      * dropout: When not `None`, the probability we will drop out a given
          coordinate.
      * gradient_clip_norm: A float > 0. If provided, gradients are
          clipped to their global norm with this clipping ratio.
      * embedding_lr_multipliers: Optional. A dictionary from
          `EmbeddingColumn` to a `float` multiplier. Multiplier will be used to
          multiply with learning rate for the embedding variables.
      * input_layer_min_slice_size: Optional. The min slice size of input layer
          partitions. If not provided, will use the default of 64M.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    predictions: A dict of `Tensor` objects.
    loss: A scalar containing the loss of the step.
    train_op: The op for training.
  """
  head = params["head"]
  hidden_units = params["hidden_units"]
  feature_columns = params["feature_columns"]
  optimizer = params.get("optimizer") or "Adagrad"
  activation_fn = params.get("activation_fn")
  dropout = params.get("dropout")
  gradient_clip_norm = params.get("gradient_clip_norm")
  input_layer_min_slice_size = (
      params.get("input_layer_min_slice_size") or 64 << 20)
  num_ps_replicas = config.num_ps_replicas if config else 0
  embedding_lr_multipliers = params.get("embedding_lr_multipliers", {})

  features = _get_feature_dict(features)
  parent_scope = "dnn"

  partitioner = partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas)
  with variable_scope.variable_scope(
      parent_scope,
      values=tuple(six.itervalues(features)),
      partitioner=partitioner):
    input_layer_partitioner = (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=input_layer_min_slice_size))
    with variable_scope.variable_scope(
        "input_from_feature_columns",
        values=tuple(six.itervalues(features)),
        partitioner=input_layer_partitioner) as input_layer_scope:
      net = layers.input_from_feature_columns(
          columns_to_tensors=features,
          feature_columns=feature_columns,
          weight_collections=[parent_scope],
          scope=input_layer_scope)

    for layer_id, num_hidden_units in enumerate(hidden_units):
      with variable_scope.variable_scope(
          "hiddenlayer_%d" % layer_id,
          values=(net,)) as hidden_layer_scope:
        net = layers.fully_connected(
            net,
            num_hidden_units,
            activation_fn=activation_fn,
            variables_collections=[parent_scope],
            scope=hidden_layer_scope)
        if dropout is not None and mode == model_fn.ModeKeys.TRAIN:
          net = layers.dropout(net, keep_prob=(1.0 - dropout))
      _add_hidden_layer_summary(net, hidden_layer_scope.name)

    with variable_scope.variable_scope(
        "logits",
        values=(net,)) as logits_scope:
      logits = layers.fully_connected(
          net,
          head.logits_dimension,
          activation_fn=None,
          variables_collections=[parent_scope],
          scope=logits_scope)
    _add_hidden_layer_summary(logits, logits_scope.name)

    def _train_op_fn(loss):
      """Returns the op to optimize the loss."""
      return optimizers.optimize_loss(
          loss=loss,
          global_step=contrib_variables.get_global_step(),
          learning_rate=_LEARNING_RATE,
          optimizer=_get_optimizer(optimizer),
          gradient_multipliers=(
              dnn_linear_combined._extract_embedding_lr_multipliers(  # pylint: disable=protected-access
                  embedding_lr_multipliers, parent_scope,
                  input_layer_scope.name)),
          clip_gradients=gradient_clip_norm,
          name=parent_scope,
          # Empty summaries to prevent optimizers from logging training_loss.
          summaries=[])

    return head.create_model_fn_ops(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)


class DNNClassifier(estimator.Estimator):
  """A classifier for TensorFlow DNN models.

  Example:

  ```python
  sparse_feature_a = sparse_column_with_hash_bucket(...)
  sparse_feature_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,
                                          ...)
  sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b,
                                          ...)

  estimator = DNNClassifier(
      feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
      hidden_units=[1024, 512, 256])

  # Or estimator using the ProximalAdagradOptimizer optimizer with
  # regularization.
  estimator = DNNClassifier(
      feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
      hidden_units=[1024, 512, 256],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, y (where y represents label's class index).
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  def input_fn_predict: # returns x, None
  # predict_classes returns class indices.
  estimator.predict_classes(input_fn=input_fn_predict)
  ```

  If the user specifies `label_keys` in constructor, labels must be strings from
  the `label_keys` vocabulary. Example:

  ```python
  label_keys = ['label0', 'label1', 'label2']
  estimator = DNNClassifier(
      feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
      hidden_units=[1024, 512, 256],
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
               enable_centered_bias=False,
               config=None,
               feature_engineering_fn=None,
               embedding_lr_multipliers=None,
               input_layer_min_slice_size=None,
               label_keys=None):
    """Initializes a DNNClassifier instance.

    Args:
      hidden_units: List of hidden units per layer. All layers are fully
        connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
        has 32.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
        It must be greater than 1. Note: Class labels are integers representing
        the class index (i.e. values from 0 to n_classes-1). For arbitrary
        label values (e.g. string labels), convert to class indices first.
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
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      embedding_lr_multipliers: Optional. A dictionary from `EmbeddingColumn` to
        a `float` multiplier. Multiplier will be used to multiply with learning
        rate for the embedding variables.
      input_layer_min_slice_size: Optional. The min slice size of input layer
        partitions. If not provided, will use the default of 64M.
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary. Only supported for `n_classes` > 2.

    Returns:
      A `DNNClassifier` estimator.

    Raises:
      ValueError: If `n_classes` < 2.
    """
    self._feature_columns = tuple(feature_columns or [])
    super(DNNClassifier, self).__init__(
        model_fn=_dnn_model_fn,
        model_dir=model_dir,
        config=config,
        params={
            "head":
                head_lib.multi_class_head(
                    n_classes,
                    weight_column_name=weight_column_name,
                    enable_centered_bias=enable_centered_bias,
                    label_keys=label_keys),
            "hidden_units": hidden_units,
            "feature_columns": self._feature_columns,
            "optimizer": optimizer,
            "activation_fn": activation_fn,
            "dropout": dropout,
            "gradient_clip_norm": gradient_clip_norm,
            "embedding_lr_multipliers": embedding_lr_multipliers,
            "input_layer_min_slice_size": input_layer_min_slice_size,
        },
        feature_engineering_fn=feature_engineering_fn)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE,
      estimator.AS_ITERABLE_INSTRUCTIONS,
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
    return super(DNNClassifier, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=outputs,
        as_iterable=as_iterable)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE,
      estimator.AS_ITERABLE_INSTRUCTIONS,
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
    preds = super(DNNClassifier, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key],
        as_iterable=as_iterable)
    if as_iterable:
      return (pred[key] for pred in preds)
    return preds[key].reshape(-1)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE,
      estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_proba(self,
                    x=None,
                    input_fn=None,
                    batch_size=None,
                    as_iterable=True):
    """Returns predicted probabilities for given features.

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
    preds = super(DNNClassifier, self).predict(
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
      return layers.parse_feature_columns_from_examples(examples,
                                                        self._feature_columns)

    return super(DNNClassifier, self).export(
        export_dir=export_dir,
        input_fn=input_fn or default_input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=use_deprecated_input_fn,
        signature_fn=(signature_fn or
                      export.classification_signature_fn_with_prob),
        prediction_key=prediction_key.PredictionKey.PROBABILITIES,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)


class DNNRegressor(estimator.Estimator):
  """A regressor for TensorFlow DNN models.

  Example:

  ```python
  sparse_feature_a = sparse_column_with_hash_bucket(...)
  sparse_feature_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,
                                          ...)
  sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b,
                                          ...)

  estimator = DNNRegressor(
      feature_columns=[sparse_feature_a, sparse_feature_b],
      hidden_units=[1024, 512, 256])

  # Or estimator using the ProximalAdagradOptimizer optimizer with
  # regularization.
  estimator = DNNRegressor(
      feature_columns=[sparse_feature_a, sparse_feature_b],
      hidden_units=[1024, 512, 256],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Input builders
  def input_fn_train: # returns x, y
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, y
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  def input_fn_predict: # returns x, None
    pass
  estimator.predict_scores(input_fn=input_fn_predict)
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
               enable_centered_bias=False,
               config=None,
               feature_engineering_fn=None,
               label_dimension=1,
               embedding_lr_multipliers=None,
               input_layer_min_slice_size=None):
    """Initializes a `DNNRegressor` instance.

    Args:
      hidden_units: List of hidden units per layer. All layers are fully
        connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
        has 32.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
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
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      embedding_lr_multipliers: Optional. A dictionary from `EbeddingColumn` to
          a `float` multiplier. Multiplier will be used to multiply with
          learning rate for the embedding variables.
      input_layer_min_slice_size: Optional. The min slice size of input layer
          partitions. If not provided, will use the default of 64M.

    Returns:
      A `DNNRegressor` estimator.
    """
    self._feature_columns = tuple(feature_columns or [])
    super(DNNRegressor, self).__init__(
        model_fn=_dnn_model_fn,
        model_dir=model_dir,
        config=config,
        params={
            "head":
                head_lib.regression_head(
                    label_dimension=label_dimension,
                    weight_column_name=weight_column_name,
                    enable_centered_bias=enable_centered_bias),
            "hidden_units": hidden_units,
            "feature_columns": self._feature_columns,
            "optimizer": optimizer,
            "activation_fn": activation_fn,
            "dropout": dropout,
            "gradient_clip_norm": gradient_clip_norm,
            "embedding_lr_multipliers": embedding_lr_multipliers,
            "input_layer_min_slice_size": input_layer_min_slice_size,
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

    return super(DNNRegressor, self).evaluate(
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
      estimator.AS_ITERABLE_DATE,
      estimator.AS_ITERABLE_INSTRUCTIONS,
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
    return super(DNNRegressor, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=outputs,
        as_iterable=as_iterable)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE,
      estimator.AS_ITERABLE_INSTRUCTIONS,
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
    preds = super(DNNRegressor, self).predict(
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
      return layers.parse_feature_columns_from_examples(examples,
                                                        self._feature_columns)

    return super(DNNRegressor, self).export(
        export_dir=export_dir,
        input_fn=input_fn or default_input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=use_deprecated_input_fn,
        signature_fn=signature_fn or export.regression_signature_fn,
        prediction_key=prediction_key.PredictionKey.SCORES,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)


class DNNEstimator(estimator.Estimator):
  """A Estimator for TensorFlow DNN models with user specified _Head.

  Example:

  ```python
  sparse_feature_a = sparse_column_with_hash_bucket(...)
  sparse_feature_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,
                                          ...)
  sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b,
                                          ...)
  To create a DNNEstimator for binary classification, where
  estimator = DNNEstimator(
      feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
      head=tf.contrib.learn.multi_class_head(n_classes=2),
      hidden_units=[1024, 512, 256])

  If your label is keyed with "y" in your labels dict, and weights are keyed
  with "w" in features dict, and you want to enable centered bias,
  head = tf.contrib.learn.multi_class_head(
      n_classes=2,
      label_name="x",
      weight_column_name="w",
      enable_centered_bias=True)
  estimator = DNNEstimator(
      feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
      head=head,
      hidden_units=[1024, 512, 256])

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, y (where y represents label's class index).
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x) # returns predicted labels (i.e. label's class index).
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
               head,
               hidden_units,
               feature_columns,
               model_dir=None,
               optimizer=None,
               activation_fn=nn.relu,
               dropout=None,
               gradient_clip_norm=None,
               config=None,
               feature_engineering_fn=None,
               embedding_lr_multipliers=None,
               input_layer_min_slice_size=None):
    """Initializes a `DNNEstimator` instance.

    Args:
      head: `Head` instance.
      hidden_units: List of hidden units per layer. All layers are fully
        connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
        has 32.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      optimizer: An instance of `tf.Optimizer` used to train the model. If
        `None`, will use an Adagrad optimizer.
      activation_fn: Activation function applied to each layer. If `None`, will
        use `tf.nn.relu`.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are
        clipped to their global norm with this clipping ratio. See
        `tf.clip_by_global_norm` for more details.
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.
      embedding_lr_multipliers: Optional. A dictionary from `EmbeddingColumn` to
          a `float` multiplier. Multiplier will be used to multiply with
          learning rate for the embedding variables.
      input_layer_min_slice_size: Optional. The min slice size of input layer
          partitions. If not provided, will use the default of 64M.

    Returns:
      A `DNNEstimator` estimator.
    """
    super(DNNEstimator, self).__init__(
        model_fn=_dnn_model_fn,
        model_dir=model_dir,
        config=config,
        params={
            "head": head,
            "hidden_units": hidden_units,
            "feature_columns": feature_columns,
            "optimizer": optimizer,
            "activation_fn": activation_fn,
            "dropout": dropout,
            "gradient_clip_norm": gradient_clip_norm,
            "embedding_lr_multipliers": embedding_lr_multipliers,
            "input_layer_min_slice_size": input_layer_min_slice_size,
        },
        feature_engineering_fn=feature_engineering_fn)
