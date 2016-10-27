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

import math
import re
import tempfile

import six

from tensorflow.contrib import layers
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.framework import list_variables
from tensorflow.contrib.framework import load_variable
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training as train


# The default learning rate of 0.2 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.2


def _get_optimizer(spec):
  if isinstance(spec, six.string_types):
    return layers.OPTIMIZER_CLS_NAMES[spec](
        learning_rate=_LEARNING_RATE)
  elif callable(spec):
    return spec()
  return spec


# TODO(ispir): Remove this function by fixing '_infer_model' with single outputs
# and as_iteable case.
def _as_iterable(preds, output):
  for pred in preds:
    yield pred[output]


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


def _linear_classifier_model_fn(features, targets, mode, params):
  """Linear classifier model_fn.

  Args:
    features: `Tensor` or dict of `Tensor` (depends on data passed to `fit`).
    targets: `Tensor` of shape [batch_size, 1] or [batch_size] target labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * n_classes: number of target classes.
      * weight_column_name: A string defining the weight feature column, or
          None if there are no weights.
      * optimizer: string, `Optimizer` object, or callable that defines the
          optimizer to use for training.
      * gradient_clip_norm: A float > 0. If provided, gradients are
          clipped to their global norm with this clipping ratio.
      * enable_centered_bias: A bool. If True, estimator will learn a centered
          bias variable for each class. Rest of the model structure learns the
          residual after centered bias.
      * num_ps_replicas: The number of parameter server replicas.
      * joint_weights: If True, the weights for all columns will be stored in a
        single (possibly partitioned) variable. It's more efficient, but it's
        incompatible with SDCAOptimizer, and requires all feature columns are
        sparse and use the 'sum' combiner.

  Returns:
    predictions: A dict of `Tensor` objects.
    loss: A scalar containing the loss of the step.
    train_op: The op for training.

  Raises:
    ValueError: If mode is not any of the `ModeKeys`.
  """
  feature_columns = params["feature_columns"]
  optimizer = params["optimizer"]
  gradient_clip_norm = params.get("gradient_clip_norm", None)
  num_ps_replicas = params.get("num_ps_replicas", 0)
  joint_weights = params.get("joint_weights", False)

  head = params.get("head", None)
  if not head:
    # TODO(zakaria): Remove these params and make head mandatory
    head = head_lib._multi_class_head(  # pylint: disable=protected-access
        params.get("n_classes"),
        weight_column_name=params["weight_column_name"],
        enable_centered_bias=params.get("enable_centered_bias", False))

  if not isinstance(features, dict):
    features = {"": features}

  parent_scope = "linear"
  partitioner = partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas,
      min_slice_size=64 << 20)

  with variable_scope.variable_op_scope(
      features.values(), parent_scope, partitioner=partitioner) as scope:
    if joint_weights:
      logits, _, _ = (
          layers.joint_weighted_sum_from_feature_columns(
              columns_to_tensors=features,
              feature_columns=feature_columns,
              num_outputs=head.logits_dimension,
              weight_collections=[parent_scope],
              scope=scope))
    else:
      logits, _, _ = (
          layers.weighted_sum_from_feature_columns(
              columns_to_tensors=features,
              feature_columns=feature_columns,
              num_outputs=head.logits_dimension,
              weight_collections=[parent_scope],
              scope=scope))

  def _train_op_fn(loss):
    global_step = contrib_variables.get_global_step()
    my_vars = ops.get_collection("linear")
    grads = gradients.gradients(loss, my_vars)
    if gradient_clip_norm:
      grads, _ = clip_ops.clip_by_global_norm(grads, gradient_clip_norm)
    return (optimizer.apply_gradients(
        zip(grads, my_vars), global_step=global_step))

  return head.head_ops(features, targets, mode, _train_op_fn, logits)


def sdca_classifier_model_fn(features, targets, mode, params):
  """Linear classifier model_fn that uses the SDCA optimizer.

  Args:
    features: A dict of `Tensor` keyed by column name.
    targets: `Tensor` of shape [batch_size, 1] or [batch_size] target labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * optimizer: An `SDCAOptimizer` instance.
      * weight_column_name: A string defining the weight feature column, or
          None if there are no weights.
      * loss_type: A string. Must be either "logistic_loss" or "hinge_loss".
      * update_weights_hook: A `SessionRunHook` object or None. Used to update
          model weights.

  Returns:
    predictions: A dict of `Tensor` objects.
    loss: A scalar containing the loss of the step.
    train_op: The op for training.

  Raises:
    ValueError: If `optimizer` is not an `SDCAOptimizer` instance.
    ValueError: If mode is not any of the `ModeKeys`.
  """
  feature_columns = params["feature_columns"]
  optimizer = params["optimizer"]
  weight_column_name = params["weight_column_name"]
  loss_type = params.get("loss_type", None)
  update_weights_hook = params.get("update_weights_hook")

  if not isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
    raise ValueError("Optimizer must be of type SDCAOptimizer")

  logits, columns_to_variables, bias = (
      layers.weighted_sum_from_feature_columns(
          columns_to_tensors=features,
          feature_columns=feature_columns,
          num_outputs=1))

  _add_bias_column(feature_columns, features, bias, targets,
                   columns_to_variables)

  if loss_type is "hinge_loss":
    head = head_lib._binary_svm_head(  # pylint: disable=protected-access
        weight_column_name=weight_column_name,
        enable_centered_bias=False)
  else:
    # pylint: disable=protected-access
    head = head_lib._multi_class_head(2,  # pylint: disable=protected-access
                                      weight_column_name=weight_column_name,
                                      enable_centered_bias=False)
  def _train_op_fn(unused_loss):
    global_step = contrib_variables.get_global_step()
    sdca_model, train_op = optimizer.get_train_step(columns_to_variables,
                                                    weight_column_name,
                                                    loss_type, features,
                                                    targets, global_step)
    if update_weights_hook is not None:
      update_weights_hook.set_parameters(sdca_model, train_op)
    return train_op

  return head.head_ops(features, targets, mode, _train_op_fn, logits)


# Ensures consistency with LinearComposableModel.
def _get_default_optimizer(feature_columns):
  learning_rate = min(_LEARNING_RATE, 1.0 / math.sqrt(len(feature_columns)))
  return train.FtrlOptimizer(learning_rate=learning_rate)


class _SdcaUpdateWeightsHook(session_run_hook.SessionRunHook):
  """SessionRunHook to update and shrink SDCA model weights."""

  def __init__(self):
    pass

  def set_parameters(self, sdca_model, train_op):
    self._sdca_model = sdca_model
    self._train_op = train_op

  def begin(self):
    """Construct the update_weights op.

    The op is implicitly added to the default graph.
    """
    self._update_op = self._sdca_model.update_weights(self._train_op)

  def before_run(self, run_context):
    """Return the update_weights op so that it is executed during this run."""
    return session_run_hook.SessionRunArgs(self._update_op)


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
               enable_centered_bias=False,
               _joint_weight=False,
               config=None,
               feature_engineering_fn=None):
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
      feature_engineering_fn: Feature engineering function. Takes features and
                        targets which are the output of `input_fn` and
                        returns features and targets which will be fed
                        into the model.

    Returns:
      A `LinearClassifier` estimator.

    Raises:
      ValueError: if n_classes < 2.
    """
    # TODO(zoy): Give an unsupported error if enable_centered_bias is
    #    requested for SDCA once its default changes to False.
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

    chief_hook = None
    if isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
      assert not _joint_weight, ("_joint_weight is incompatible with the"
                                 " SDCAOptimizer")
      assert n_classes == 2, "SDCA only applies to binary classification."

      model_fn = sdca_classifier_model_fn
      # We use a hook to perform the weight update and shrink step only on the
      # chief. Because the SdcaModel constructed by the estimator within the
      # call to fit() but we need to pass the hook to fit(), we pass the hook
      # as a parameter to the model_fn and have that propagate the model to the
      # hook.
      chief_hook = _SdcaUpdateWeightsHook()
      params = {
          "feature_columns": feature_columns,
          "optimizer": self._optimizer,
          "weight_column_name": weight_column_name,
          "loss_type": "logistic_loss",
          "update_weights_hook": chief_hook,
      }
    else:
      model_fn = _linear_classifier_model_fn
      head = head_lib._multi_class_head(  # pylint: disable=protected-access
          n_classes,
          weight_column_name=weight_column_name,
          enable_centered_bias=enable_centered_bias)
      params = {
          "head": head,
          "feature_columns": feature_columns,
          "optimizer": self._optimizer,
          "gradient_clip_norm": gradient_clip_norm,
          "num_ps_replicas": num_ps_replicas,
          "joint_weights": _joint_weight,
      }

    self._estimator = estimator.Estimator(
        model_fn=model_fn,
        model_dir=self._model_dir,
        config=config,
        params=params,
        feature_engineering_fn=feature_engineering_fn)

    self._additional_run_hook = None
    if self._estimator.config.is_chief:
      self._additional_run_hook = chief_hook

  def get_estimator(self):
    return self._estimator

  def fit(self, x=None, y=None, input_fn=None, steps=None, batch_size=None,
          monitors=None, max_steps=None):
    """See trainable.Trainable."""
    # TODO(roumposg): Remove when deprecated monitors are removed.
    if monitors is None:
      monitors = []
    deprecated_monitors = [
        m for m in monitors
        if not isinstance(m, session_run_hook.SessionRunHook)
    ]
    for monitor in deprecated_monitors:
      monitor.set_estimator(self)
      monitor._lock_estimator()  # pylint: disable=protected-access

    if self._additional_run_hook:
      monitors.append(self._additional_run_hook)
    result = self._estimator.fit(x=x, y=y, input_fn=input_fn, steps=steps,
                                 batch_size=batch_size, monitors=monitors,
                                 max_steps=max_steps)

    for monitor in deprecated_monitors:
      monitor._unlock_estimator()  # pylint: disable=protected-access

    return result

  # TODO(ispir): Simplify evaluate by aligning this logic with custom Estimator.
  def evaluate(self, x=None, y=None, input_fn=None, feed_fn=None,
               batch_size=None, steps=None, metrics=None, name=None):
    """See evaluable.Evaluable."""
    return self._estimator.evaluate(x=x, y=y, input_fn=input_fn,
                                    feed_fn=feed_fn, batch_size=batch_size,
                                    steps=steps, metrics=metrics, name=name)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict(self, x=None, input_fn=None, batch_size=None, as_iterable=True):
    """Runs inference to determine the predicted class."""
    preds = self._estimator.predict(x=x, input_fn=input_fn,
                                    batch_size=batch_size,
                                    outputs=[head_lib.PredictionKey.CLASSES],
                                    as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=head_lib.PredictionKey.CLASSES)
    return preds[head_lib.PredictionKey.CLASSES]

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_proba(self, x=None, input_fn=None, batch_size=None, outputs=None,
                    as_iterable=True):
    """Runs inference to determine the class probability predictions."""
    preds = self._estimator.predict(x=x, input_fn=input_fn,
                                    batch_size=batch_size,
                                    outputs=[
                                        head_lib.PredictionKey.PROBABILITIES],
                                    as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=head_lib.PredictionKey.PROBABILITIES)
    return preds[head_lib.PredictionKey.PROBABILITIES]

  def get_variable_names(self):
    return [name for name, _ in list_variables(self._model_dir)]

  def get_variable_value(self, name):
    return load_variable(self.model_dir, name)

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
        prediction_key=head_lib.PredictionKey.PROBABILITIES,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def weights_(self):
    values = {}
    optimizer_regex = r".*/"+self._optimizer.get_name() + r"(_\d)?$"
    for name, _ in list_variables(self._model_dir):
      if (name.startswith("linear/") and
          name != "linear/bias_weight" and
          not re.match(optimizer_regex, name)):
        values[name] = load_variable(self._model_dir, name)
    if len(values) == 1:
      return values[list(values.keys())[0]]
    return values

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def bias_(self):
    return load_variable(self._model_dir, name="linear/bias_weight")

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
               enable_centered_bias=False,
               target_dimension=1,
               _joint_weights=False,
               config=None,
               feature_engineering_fn=None):
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
      feature_engineering_fn: Feature engineering function. Takes features and
                        targets which are the output of `input_fn` and
                        returns features and targets which will be fed
                        into the model.

    Returns:
      A `LinearRegressor` estimator.
    """
    if isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
      enable_centered_bias = False
      logging.warning("centered_bias is not supported with SDCA, "
                      "please disable it explicitly.")
    self._weight_column_name = weight_column_name
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
        config=config,
        feature_engineering_fn=feature_engineering_fn)

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
            num_outputs=self._head.logits_dimension,
            weight_collections=[self._linear_model.get_scope_name()],
            scope=self._linear_model.get_scope_name()))
    _add_bias_column(self._linear_feature_columns, features, bias, targets,
                     columns_to_variables)

    def _train_op_fn(unused_loss):
      sdca_model, train_op = self._linear_optimizer.get_train_step(
          columns_to_variables, self._weight_column_name,
          self._loss_type(), features, targets, global_step)
      return sdca_model.update_weights(train_op)

    model_fn_ops = self._head.head_ops(features, targets,
                                       estimator.ModeKeys.TRAIN, _train_op_fn,
                                       logits=logits)
    return model_fn_ops.training_op, model_fn_ops.loss

  def _loss_type(self):
    return "squared_loss"

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def weights_(self):
    return self.linear_weights_

  @property
  @deprecated("2016-10-30",
              "This method will be removed after the deprecation date. "
              "To inspect variables, use get_variable_names() and "
              "get_variable_value().")
  def bias_(self):
    return self.linear_bias_
