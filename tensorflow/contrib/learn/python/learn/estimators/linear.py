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

"""Linear Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math
import re
import tempfile

import six

from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import target_column
from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn import session_run_hook
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.utils import checkpoints
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import training as train

_CLASSES = "classes"
_LOGISTIC = "logistic"
_PROBABILITIES = "probabilities"


def _get_metric_args(metric):
  if hasattr(metric, "__code__"):
    return inspect.getargspec(metric).args
  elif hasattr(metric, "func") and hasattr(metric, "keywords"):
    return [arg for arg in inspect.getargspec(metric.func).args
            if arg not in metric.keywords.keys()]


def _wrap_metric(metric):
  """Wraps metrics for mismatched prediction/target types."""
  def wrapped(preds, targets):
    targets = math_ops.cast(targets, preds.dtype)
    return metric(preds, targets)

  def wrapped_weights(preds, targets, weights=None):
    targets = math_ops.cast(targets, preds.dtype)
    if weights is not None:
      weights = array_ops.reshape(math_ops.to_float(weights), shape=(-1,))
    return metric(preds, targets, weights)

  return wrapped_weights if "weights" in _get_metric_args(metric) else wrapped


def _get_optimizer(spec):
  if isinstance(spec, six.string_types):
    return layers.OPTIMIZER_CLS_NAMES[spec](
        learning_rate=0.2)
  elif callable(spec):
    return spec()
  return spec


# TODO(ispir): Remove this function by fixing '_infer_model' with single outputs
# and as_iteable case.
def _as_iterable(preds, output):
  for pred in preds:
    yield pred[output]


def _centered_bias_step(targets, loss_fn, num_label_columns):
  centered_bias = ops.get_collection("centered_bias")
  batch_size = array_ops.shape(targets)[0]
  logits = array_ops.reshape(
      array_ops.tile(centered_bias[0], [batch_size]),
      [batch_size, num_label_columns])
  loss = loss_fn(logits, targets)
  return train.AdagradOptimizer(0.1).minimize(loss, var_list=centered_bias)


def _centered_bias(num_outputs):
  centered_bias = variables.Variable(
      array_ops.zeros([num_outputs]),
      collections=["centered_bias", ops.GraphKeys.VARIABLES],
      name="centered_bias_weight")
  logging_ops.scalar_summary(
      ["centered_bias_%d" % cb for cb in range(num_outputs)],
      array_ops.reshape(centered_bias, [-1]))
  return centered_bias


def _add_bias_column(feature_columns, columns_to_tensors, bias_variable,
                     targets, columns_to_variables):
  # TODO(b/31008490): Move definition to a common constants place.
  bias_column_name = "tf_virtual_bias_column"
  if any(col.name is bias_column_name for col in feature_columns):
    raise ValueError("%s is a reserved column name." % bias_column_name)
  bias_column = layers.real_valued_column(bias_column_name)
  columns_to_tensors[bias_column] = array_ops.ones_like(targets,
                                                        dtype=dtypes.float32)
  columns_to_variables[bias_column] = [bias_variable]


def _log_loss_with_two_classes(logits, target):
  check_shape_op = control_flow_ops.Assert(
      math_ops.less_equal(array_ops.rank(target), 2),
      ["target's shape should be either [batch_size, 1] or [batch_size]"])
  with ops.control_dependencies([check_shape_op]):
    target = array_ops.reshape(target, shape=[array_ops.shape(target)[0], 1])
  return nn.sigmoid_cross_entropy_with_logits(
      logits, math_ops.to_float(target))


def _softmax_cross_entropy_loss(logits, target):
  check_shape_op = control_flow_ops.Assert(
      math_ops.less_equal(array_ops.rank(target), 2),
      ["target's shape should be either [batch_size, 1] or [batch_size]"])
  with ops.control_dependencies([check_shape_op]):
    target = array_ops.reshape(target, shape=[array_ops.shape(target)[0]])
  return nn.sparse_softmax_cross_entropy_with_logits(logits, target)


def _hinge_loss(logits, target):
  check_shape_op = control_flow_ops.Assert(
      math_ops.less_equal(array_ops.rank(target), 2),
      ["target's shape should be either [batch_size, 1] or [batch_size]"])
  with ops.control_dependencies([check_shape_op]):
    target = array_ops.reshape(target, shape=[array_ops.shape(target)[0], 1])
  return losses.hinge_loss(logits, target)


def _weighted_loss(loss, weight_tensor):
  unweighted_loss = array_ops.reshape(loss, shape=(-1,))
  weighted_loss = math_ops.mul(
      unweighted_loss, array_ops.reshape(weight_tensor, shape=(-1,)))
  return math_ops.div(
      math_ops.reduce_sum(weighted_loss),
      math_ops.to_float(math_ops.reduce_sum(weight_tensor)),
      name="loss")


def _linear_classifier_model_fn(features, targets, mode, params):
  """Estimator's linear model_fn."""
  n_classes = params["n_classes"]
  weight_column_name = params["weight_column_name"]
  feature_columns = params["feature_columns"]
  optimizer = params["optimizer"]
  gradient_clip_norm = params.get("gradient_clip_norm", None)
  enable_centered_bias = params.get("enable_centered_bias", True)
  num_ps_replicas = params.get("num_ps_replicas", 0)
  joint_weights = params.get("joint_weights", False)

  if not isinstance(features, dict):
    features = {"": features}

  num_label_columns = 1 if n_classes == 2 else n_classes
  loss_fn = _softmax_cross_entropy_loss
  if n_classes == 2:
    loss_fn = _log_loss_with_two_classes

  feat_values = (features.values() if isinstance(features, dict)
                 else [features])
  partitioner = partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas,
      min_slice_size=64 << 20)
  with variable_scope.variable_op_scope(
      feat_values, "linear", partitioner=partitioner) as scope:
    if joint_weights:
      logits, _, _ = (
          layers.joint_weighted_sum_from_feature_columns(
              columns_to_tensors=features,
              feature_columns=feature_columns,
              num_outputs=num_label_columns,
              weight_collections=["linear"],
              scope=scope))
    else:
      logits, _, _ = (
          layers.weighted_sum_from_feature_columns(
              columns_to_tensors=features,
              feature_columns=feature_columns,
              num_outputs=num_label_columns,
              weight_collections=["linear"],
              scope=scope))

  if enable_centered_bias:
    logits = nn.bias_add(logits, _centered_bias(num_label_columns))

  loss = None
  if mode != estimator.ModeKeys.INFER:
    loss = loss_fn(logits, targets)
    if weight_column_name:
      weight_tensor = array_ops.reshape(
          math_ops.to_float(features[weight_column_name]), shape=(-1,))
      loss = _weighted_loss(loss, weight_tensor)
    else:
      loss = math_ops.reduce_mean(loss, name="loss")
    logging_ops.scalar_summary("loss", loss)

  train_ops = []
  if mode == estimator.ModeKeys.TRAIN:
    global_step = contrib_variables.get_global_step()

    my_vars = ops.get_collection("linear")
    grads = gradients.gradients(loss, my_vars)
    if gradient_clip_norm:
      grads, _ = clip_ops.clip_by_global_norm(grads, gradient_clip_norm)
    train_ops.append(optimizer.apply_gradients(
        zip(grads, my_vars), global_step=global_step))
    if enable_centered_bias:
      train_ops.append(
          _centered_bias_step(targets, loss_fn, num_label_columns))

  predictions = {}
  if n_classes == 2:
    predictions[_LOGISTIC] = math_ops.sigmoid(logits)
    logits = array_ops.concat(1, [array_ops.zeros_like(logits), logits])
  predictions[_PROBABILITIES] = nn.softmax(logits)
  predictions[_CLASSES] = math_ops.argmax(logits, 1)

  return predictions, loss, control_flow_ops.group(*train_ops)


def sdca_classifier_model_fn(features, targets, mode, params):
  """Estimator's linear model_fn."""
  feature_columns = params["feature_columns"]
  optimizer = params["optimizer"]
  weight_column_name = params["weight_column_name"]
  loss_type = params["loss_type"]

  if not isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
    raise ValueError("Optimizer must be of type SDCAOptimizer")

  loss_fn = {
      "logistic_loss": _log_loss_with_two_classes,
      "hinge_loss": _hinge_loss,
  }[loss_type]

  logits, columns_to_variables, bias = (
      layers.weighted_sum_from_feature_columns(
          columns_to_tensors=features,
          feature_columns=feature_columns,
          num_outputs=1))

  _add_bias_column(feature_columns, features, bias, targets,
                   columns_to_variables)

  loss = None
  if mode != estimator.ModeKeys.INFER:
    loss = math_ops.reduce_mean(loss_fn(logits, targets), name="loss")
    logging_ops.scalar_summary("loss", loss)

  train_op = None
  if mode == estimator.ModeKeys.TRAIN:
    global_step = contrib_variables.get_global_step()
    train_op = optimizer.get_train_step(
        columns_to_variables, weight_column_name, loss_type, features,
        targets, global_step)

  predictions = {}
  predictions[_LOGISTIC] = math_ops.sigmoid(logits)
  logits = array_ops.concat(1, [array_ops.zeros_like(logits), logits])
  predictions[_PROBABILITIES] = nn.softmax(logits)
  predictions[_CLASSES] = math_ops.argmax(logits, 1)

  return predictions, loss, train_op


# Ensures consistency with LinearComposableModel.
def _get_default_optimizer(feature_columns):
  learning_rate = min(0.2, 1.0 / math.sqrt(len(feature_columns)))
  return train.FtrlOptimizer(learning_rate=learning_rate)


class LinearClassifier(evaluable.Evaluable, trainable.Trainable):
  """Linear classifier model.

  Train a linear model to classify instances into one of multiple possible
  classes. When number of possible classes is 2, this is binary classification.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_x_occupation = crossed_column(columns=[education, occupation],
                                          hash_bucket_size=10000)

  # Estimator using the default optimizer.
  estimator = LinearClassifier(
      feature_columns=[occupation, education_x_occupation])

  # Or estimator using the FTRL optimizer with regularization.
  estimator = LinearClassifier(
      feature_columns=[occupation, education_x_occupation],
      optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Or estimator using the SDCAOptimizer.
  estimator = LinearClassifier(
     feature_columns=[occupation, education_x_occupation],
     optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
       example_id_column='example_id',
       num_loss_partitions=...,
       symmetric_l2_regularization=2.0
     ))

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

  def __init__(self,  # _joint_weight pylint: disable=invalid-name
               feature_columns,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               optimizer=None,
               gradient_clip_norm=None,
               enable_centered_bias=None,
               _joint_weight=False,
               config=None):
    """Construct a `LinearClassifier` estimator object.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      n_classes: number of target classes. Default is binary classification.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      optimizer: The optimizer used to train the model. If specified, it should
        be either an instance of `tf.Optimizer` or the SDCAOptimizer. If `None`,
        the Ftrl optimizer will be used.
      gradient_clip_norm: A `float` > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        `tf.clip_by_global_norm` for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      _joint_weight: If True, the weights for all columns will be stored in a
        single (possibly partitioned) variable. It's more efficient, but it's
        incompatible with SDCAOptimizer, and requires all feature columns are
        sparse and use the 'sum' combiner.
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      A `LinearClassifier` estimator.

    Raises:
      ValueError: if n_classes < 2.
    """
    # TODO(zoy): Give an unsupported error if enable_centered_bias is
    #    requested for SDCA once its default changes to False.
    if enable_centered_bias is None:
      enable_centered_bias = True
      dnn_linear_combined._changing_default_center_bias()  # pylint: disable=protected-access
    self._model_dir = model_dir or tempfile.mkdtemp()
    if n_classes < 2:
      raise ValueError("Classification requires n_classes >= 2")
    self._n_classes = n_classes
    self._feature_columns = feature_columns
    assert self._feature_columns
    self._weight_column_name = weight_column_name
    self._optimizer = _get_default_optimizer(feature_columns)
    if optimizer:
      self._optimizer = _get_optimizer(optimizer)
    num_ps_replicas = config.num_ps_replicas if config else 0

    if isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
      assert not _joint_weight, ("_joint_weight is incompatible with the"
                                 " SDCAOptimizer")
      model_fn = sdca_classifier_model_fn
      params = {
          "feature_columns": feature_columns,
          "optimizer": self._optimizer,
          "weight_column_name": weight_column_name,
          "loss_type": "logistic_loss",
      }
    else:
      model_fn = _linear_classifier_model_fn
      params = {
          "n_classes": n_classes,
          "weight_column_name": weight_column_name,
          "feature_columns": feature_columns,
          "optimizer": self._optimizer,
          "gradient_clip_norm": gradient_clip_norm,
          "enable_centered_bias": enable_centered_bias,
          "num_ps_replicas": num_ps_replicas,
          "joint_weights": _joint_weight,
      }

    self._estimator = estimator.Estimator(
        model_fn=model_fn,
        model_dir=self._model_dir,
        config=config,
        params=params)

  def get_estimator(self):
    return self._estimator

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

  # TODO(ispir): Simplify evaluate by aligning this logic with custom Estimator.
  def evaluate(self, x=None, y=None, input_fn=None, feed_fn=None,
               batch_size=None, steps=None, metrics=None, name=None):
    """See evaluable.Evaluable."""
    if not metrics:
      metrics = {}
      metrics["accuracy"] = metric_spec.MetricSpec(
          metric_fn=_wrap_metric(metrics_lib.streaming_accuracy),
          prediction_key=_CLASSES,
          weight_key=self._weight_column_name)
    if self._n_classes == 2:
      additional_metrics = (
          target_column.get_default_binary_metrics_for_eval([0.5]))
      additional_metrics = {
          name: metric_spec.MetricSpec(metric_fn=metric,
                                       prediction_key=_LOGISTIC,
                                       weight_key=self._weight_column_name)
          for name, metric in additional_metrics.items()
      }
      metrics.update(additional_metrics)

    # TODO(b/31229024): Remove this loop
    for metric_name, metric in metrics.items():
      if isinstance(metric, metric_spec.MetricSpec):
        continue

      if isinstance(metric_name, tuple):
        if len(metric_name) != 2:
          raise ValueError("Ignoring metric %s. It returned a tuple with len  "
                           "%s, expected 2." % (metric_name, len(metric_name)))

        valid_keys = {_CLASSES, _LOGISTIC, _PROBABILITIES}
        if metric_name[1] not in valid_keys:
          raise ValueError("Ignoring metric %s. The 2nd element of its name "
                           "should be in %s" % (metric_name, valid_keys))
      elif isinstance(metric_name, str):
        metrics.pop(metric_name)
        metric_name = (metric_name, _CLASSES)
      else:
        raise ValueError("Ignoring metric %s. Its name is not in the correct "
                         "form." % metric_name)
      metrics[metric_name] = _wrap_metric(metric)

    return self._estimator.evaluate(x=x, y=y, input_fn=input_fn,
                                    feed_fn=feed_fn, batch_size=batch_size,
                                    steps=steps, metrics=metrics, name=name)

  def predict(self, x=None, input_fn=None, batch_size=None, as_iterable=False):
    """Runs inference to determine the predicted class."""
    preds = self._estimator.predict(x=x, input_fn=input_fn,
                                    batch_size=batch_size, outputs=[_CLASSES],
                                    as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=_CLASSES)
    return preds[_CLASSES]

  def predict_proba(self, x=None, input_fn=None, batch_size=None, outputs=None,
                    as_iterable=False):
    """Runs inference to determine the class probability predictions."""
    preds = self._estimator.predict(x=x, input_fn=input_fn,
                                    batch_size=batch_size,
                                    outputs=[_PROBABILITIES],
                                    as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=_PROBABILITIES)
    return preds[_PROBABILITIES]

  def get_variable_names(self):
    return [name for name, _ in checkpoints.list_variables(self._model_dir)]

  def get_variable_value(self, name):
    return checkpoints.load_variable(self.model_dir, name)

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
  def weights_(self):
    values = {}
    optimizer_regex = r".*/"+self._optimizer.get_name() + r"(_\d)?$"
    for name, _ in checkpoints.list_variables(self._model_dir):
      if (name.startswith("linear/") and
          name != "linear/bias_weight" and
          not re.match(optimizer_regex, name)):
        values[name] = checkpoints.load_variable(self._model_dir, name)
    if len(values) == 1:
      return values[list(values.keys())[0]]
    return values

  @property
  def bias_(self):
    return checkpoints.load_variable(self._model_dir,
                                     name="linear/bias_weight")

  @property
  def config(self):
    return self._estimator.config

  @property
  def model_dir(self):
    return self._model_dir


# TODO(zoy): Use model_fn similar to LinearClassifier.
class LinearRegressor(dnn_linear_combined.DNNLinearCombinedRegressor):
  """Linear regressor model.

  Train a linear regression model to predict target variable value given
  observation of feature values.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_x_occupation = crossed_column(columns=[education, occupation],
                                          hash_bucket_size=10000)

  estimator = LinearRegressor(
      feature_columns=[occupation, education_x_occupation])

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
    otherwise there will be a KeyError:

  * if `weight_column_name` is not `None`:
    key=weight_column_name, value=a `Tensor`
  * for column in `feature_columns`:
    - if isinstance(column, `SparseColumn`):
        key=column.name, value=a `SparseTensor`
    - if isinstance(column, `WeightedSparseColumn`):
        {key=id column name, value=a `SparseTensor`,
         key=weight column name, value=a `SparseTensor`}
    - if isinstance(column, `RealValuedColumn`):
        key=column.name, value=a `Tensor`
  """

  def __init__(self,  # _joint_weights: pylint: disable=invalid-name
               feature_columns,
               model_dir=None,
               weight_column_name=None,
               optimizer=None,
               gradient_clip_norm=None,
               enable_centered_bias=None,
               target_dimension=1,
               _joint_weights=False,
               config=None):
    """Construct a `LinearRegressor` estimator object.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph, etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      optimizer: An instance of `tf.Optimizer` used to train the model. If
        `None`, will use an Ftrl optimizer.
      gradient_clip_norm: A `float` > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        `tf.clip_by_global_norm` for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      target_dimension: dimension of the target for multilabels.
      _joint_weights: If True use a single (possibly partitioned) variable to
        store the weights. It's faster, but requires all feature columns are
        sparse and have the 'sum' combiner. Incompatible with SDCAOptimizer.
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      A `LinearRegressor` estimator.
    """
    if enable_centered_bias is None:
      enable_centered_bias = True
      dnn_linear_combined._changing_default_center_bias()  # pylint: disable=protected-access
    self._joint_weights = _joint_weights
    super(LinearRegressor, self).__init__(
        model_dir=model_dir,
        weight_column_name=weight_column_name,
        linear_feature_columns=feature_columns,
        linear_optimizer=optimizer,
        _joint_linear_weights=_joint_weights,
        gradient_clip_norm=gradient_clip_norm,
        enable_centered_bias=enable_centered_bias,
        target_dimension=target_dimension,
        config=config)

  def _get_train_ops(self, features, targets):
    """See base class."""
    if not isinstance(self._linear_optimizer, sdca_optimizer.SDCAOptimizer):
      return super(LinearRegressor, self)._get_train_ops(features, targets)
    assert not self._joint_weights, ("_joint_weights is incompatible with"
                                     " SDCAOptimizer.")
    global_step = contrib_variables.get_or_create_global_step()

    logits, columns_to_variables, bias = (
        layers.weighted_sum_from_feature_columns(
            columns_to_tensors=features,
            feature_columns=self._linear_feature_columns,
            num_outputs=self._target_column.num_label_columns,
            weight_collections=[self._linear_model.get_scope_name()],
            scope=self._linear_model.get_scope_name()))
    with ops.control_dependencies([self._centered_bias()]):
      loss = self._target_column.loss(logits, targets, features)
      logging_ops.scalar_summary("loss", loss)

      _add_bias_column(self._linear_feature_columns, features, bias, targets,
                       columns_to_variables)

    train_op = self._linear_optimizer.get_train_step(
        columns_to_variables, self._target_column.weight_column_name,
        self._loss_type(), features, targets, global_step)
    return train_op, loss

  def _loss_type(self):
    return "squared_loss"

  @property
  def weights_(self):
    return self.linear_weights_

  @property
  def bias_(self):
    return self.linear_bias_
