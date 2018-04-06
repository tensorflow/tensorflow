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

"""Linear Estimators (deprecated).

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
from tensorflow.python.training import training_util
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer
from tensorflow.python.feature_column import feature_column as fc_core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
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
                     columns_to_variables):
  """Adds a fake bias feature column filled with all 1s."""
  # TODO(b/31008490): Move definition to a common constants place.
  bias_column_name = "tf_virtual_bias_column"
  if any(col.name is bias_column_name for col in feature_columns):
    raise ValueError("%s is a reserved column name." % bias_column_name)
  if not feature_columns:
    raise ValueError("feature_columns can't be empty.")

  # Loop through input tensors until we can figure out batch_size.
  batch_size = None
  for column in columns_to_tensors.values():
    if isinstance(column, tuple):
      column = column[0]
    if isinstance(column, sparse_tensor.SparseTensor):
      shape = tensor_util.constant_value(column.dense_shape)
      if shape is not None:
        batch_size = shape[0]
        break
    else:
      batch_size = array_ops.shape(column)[0]
      break
  if batch_size is None:
    raise ValueError("Could not infer batch size from input features.")

  bias_column = layers.real_valued_column(bias_column_name)
  columns_to_tensors[bias_column] = array_ops.ones([batch_size, 1],
                                                   dtype=dtypes.float32)
  columns_to_variables[bias_column] = [bias_variable]


def _linear_model_fn(features, labels, mode, params, config=None):
  """A model_fn for linear models that use a gradient-based optimizer.

  Args:
    features: `Tensor` or dict of `Tensor` (depends on data passed to `fit`).
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * head: A `Head` instance.
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * optimizer: string, `Optimizer` object, or callable that defines the
          optimizer to use for training. If `None`, will use a FTRL optimizer.
      * gradient_clip_norm: A float > 0. If provided, gradients are
          clipped to their global norm with this clipping ratio.
      * joint_weights: If True, the weights for all columns will be stored in a
        single (possibly partitioned) variable. It's more efficient, but it's
        incompatible with SDCAOptimizer, and requires all feature columns are
        sparse and use the 'sum' combiner.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    A `ModelFnOps` instance.

  Raises:
    ValueError: If mode is not any of the `ModeKeys`.
  """
  head = params["head"]
  feature_columns = params["feature_columns"]
  optimizer = params.get("optimizer") or _get_default_optimizer(feature_columns)
  gradient_clip_norm = params.get("gradient_clip_norm", None)
  num_ps_replicas = config.num_ps_replicas if config else 0
  joint_weights = params.get("joint_weights", False)

  if not isinstance(features, dict):
    features = {"": features}

  parent_scope = "linear"
  partitioner = partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas,
      min_slice_size=64 << 20)

  with variable_scope.variable_scope(
      parent_scope,
      values=tuple(six.itervalues(features)),
      partitioner=partitioner) as scope:
    if all([isinstance(fc, feature_column._FeatureColumn)  # pylint: disable=protected-access
            for fc in feature_columns]):
      if joint_weights:
        layer_fn = layers.joint_weighted_sum_from_feature_columns
      else:
        layer_fn = layers.weighted_sum_from_feature_columns
      logits, _, _ = layer_fn(
          columns_to_tensors=features,
          feature_columns=feature_columns,
          num_outputs=head.logits_dimension,
          weight_collections=[parent_scope],
          scope=scope)
    else:
      logits = fc_core.linear_model(
          features=features,
          feature_columns=feature_columns,
          units=head.logits_dimension,
          weight_collections=[parent_scope])

    def _train_op_fn(loss):
      global_step = training_util.get_global_step()
      my_vars = ops.get_collection(parent_scope)
      grads = gradients.gradients(loss, my_vars)
      if gradient_clip_norm:
        grads, _ = clip_ops.clip_by_global_norm(grads, gradient_clip_norm)
      return (_get_optimizer(optimizer).apply_gradients(
          zip(grads, my_vars), global_step=global_step))

    return head.create_model_fn_ops(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)


def sdca_model_fn(features, labels, mode, params):
  """A model_fn for linear models that use the SDCA optimizer.

  Args:
    features: A dict of `Tensor` keyed by column name.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * head: A `Head` instance. Type must be one of `_BinarySvmHead`,
          `_RegressionHead` or `_BinaryLogisticHead`.
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * optimizer: An `SDCAOptimizer` instance.
      * weight_column_name: A string defining the weight feature column, or
          None if there are no weights.
      * update_weights_hook: A `SessionRunHook` object or None. Used to update
          model weights.

  Returns:
    A `ModelFnOps` instance.

  Raises:
    ValueError: If `optimizer` is not an `SDCAOptimizer` instance.
    ValueError: If the type of head is neither `_BinarySvmHead`, nor
      `_RegressionHead` nor `_MultiClassHead`.
    ValueError: If mode is not any of the `ModeKeys`.
  """
  head = params["head"]
  feature_columns = params["feature_columns"]
  optimizer = params["optimizer"]
  weight_column_name = params["weight_column_name"]
  update_weights_hook = params.get("update_weights_hook", None)

  if not isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
    raise ValueError("Optimizer must be of type SDCAOptimizer")

  if isinstance(head, head_lib._BinarySvmHead):  # pylint: disable=protected-access
    loss_type = "hinge_loss"
  elif isinstance(head, head_lib._BinaryLogisticHead):  # pylint: disable=protected-access
    loss_type = "logistic_loss"
  elif isinstance(head, head_lib._RegressionHead):  # pylint: disable=protected-access
    assert head.logits_dimension == 1, ("SDCA only applies for "
                                        "logits_dimension=1.")
    loss_type = "squared_loss"
  else:
    raise ValueError("Unsupported head type: {}".format(head))

  parent_scope = "linear"

  with variable_scope.variable_scope(
      values=features.values(), name_or_scope=parent_scope) as scope:
    features = features.copy()
    features.update(layers.transform_features(features, feature_columns))
    logits, columns_to_variables, bias = (
        layers.weighted_sum_from_feature_columns(
            columns_to_tensors=features,
            feature_columns=feature_columns,
            num_outputs=1,
            scope=scope))

    _add_bias_column(feature_columns, features, bias, columns_to_variables)

  def _train_op_fn(unused_loss):
    global_step = training_util.get_global_step()
    sdca_model, train_op = optimizer.get_train_step(columns_to_variables,
                                                    weight_column_name,
                                                    loss_type, features,
                                                    labels, global_step)
    if update_weights_hook is not None:
      update_weights_hook.set_parameters(sdca_model, train_op)
    return train_op

  model_fn_ops = head.create_model_fn_ops(
      features=features,
      labels=labels,
      mode=mode,
      train_op_fn=_train_op_fn,
      logits=logits)
  if update_weights_hook is not None:
    return model_fn_ops._replace(
        training_chief_hooks=(model_fn_ops.training_chief_hooks +
                              [update_weights_hook]))
  return model_fn_ops


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


class LinearClassifier(estimator.Estimator):
  """Linear classifier model.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Train a linear model to classify instances into one of multiple possible
  classes. When number of possible classes is 2, this is binary classification.

  Example:

  ```python
  sparse_column_a = sparse_column_with_hash_bucket(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  # Estimator using the default optimizer.
  estimator = LinearClassifier(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

  # Or estimator using the FTRL optimizer with regularization.
  estimator = LinearClassifier(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
      optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Or estimator using the SDCAOptimizer.
  estimator = LinearClassifier(
     feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
     optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
       example_id_column='example_id',
       num_loss_partitions=...,
       symmetric_l2_regularization=2.0
     ))

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
  estimator = LinearClassifier(
      n_classes=n_classes,
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
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
               feature_engineering_fn=None,
               label_keys=None):
    """Construct a `LinearClassifier` estimator object.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
        Note that class labels are integers representing the class index (i.e.
        values from 0 to n_classes-1). For arbitrary label values (e.g. string
        labels), convert to class indices first.
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
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary. Only supported for `n_classes` > 2.

    Returns:
      A `LinearClassifier` estimator.

    Raises:
      ValueError: if n_classes < 2.
      ValueError: if enable_centered_bias=True and optimizer is SDCAOptimizer.
    """
    if (isinstance(optimizer, sdca_optimizer.SDCAOptimizer) and
        enable_centered_bias):
      raise ValueError("enable_centered_bias is not supported with SDCA")

    self._feature_columns = tuple(feature_columns or [])
    assert self._feature_columns

    chief_hook = None
    head = head_lib.multi_class_head(
        n_classes,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias,
        label_keys=label_keys)
    params = {
        "head": head,
        "feature_columns": feature_columns,
        "optimizer": optimizer,
    }

    if isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
      assert not _joint_weight, ("_joint_weight is incompatible with the"
                                 " SDCAOptimizer")
      assert n_classes == 2, "SDCA only applies to binary classification."

      model_fn = sdca_model_fn
      # The model_fn passes the model parameters to the chief_hook. We then use
      # the hook to update weights and shrink step only on the chief.
      chief_hook = _SdcaUpdateWeightsHook()
      params.update({
          "weight_column_name": weight_column_name,
          "update_weights_hook": chief_hook,
      })
    else:
      model_fn = _linear_model_fn
      params.update({
          "gradient_clip_norm": gradient_clip_norm,
          "joint_weights": _joint_weight,
      })

    super(LinearClassifier, self).__init__(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
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
    return super(LinearClassifier, self).predict(
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
    preds = super(LinearClassifier, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key],
        as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=key)
    return preds[key]

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_proba(self, x=None, input_fn=None, batch_size=None,
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
    preds = super(LinearClassifier, self).predict(
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
    """See BaseEstimator.export."""
    def default_input_fn(unused_estimator, examples):
      return layers.parse_feature_columns_from_examples(
          examples, self._feature_columns)

    return super(LinearClassifier, self).export(
        export_dir=export_dir,
        input_fn=input_fn or default_input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=use_deprecated_input_fn,
        signature_fn=(signature_fn or
                      export.classification_signature_fn_with_prob),
        prediction_key=prediction_key.PredictionKey.PROBABILITIES,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)


class LinearRegressor(estimator.Estimator):
  """Linear regressor model.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Train a linear regression model to predict label value given observation of
  feature values.

  Example:

  ```python
  sparse_column_a = sparse_column_with_hash_bucket(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  estimator = LinearRegressor(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

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
               label_dimension=1,
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
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      _joint_weights: If True use a single (possibly partitioned) variable to
        store the weights. It's faster, but requires all feature columns are
        sparse and have the 'sum' combiner. Incompatible with SDCAOptimizer.
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.

    Returns:
      A `LinearRegressor` estimator.
    """
    self._feature_columns = tuple(feature_columns or [])
    assert self._feature_columns

    chief_hook = None
    if (isinstance(optimizer, sdca_optimizer.SDCAOptimizer) and
        enable_centered_bias):
      enable_centered_bias = False
      logging.warning("centered_bias is not supported with SDCA, "
                      "please disable it explicitly.")
    head = head_lib.regression_head(
        weight_column_name=weight_column_name,
        label_dimension=label_dimension,
        enable_centered_bias=enable_centered_bias)
    params = {
        "head": head,
        "feature_columns": feature_columns,
        "optimizer": optimizer,
    }

    if isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
      assert label_dimension == 1, "SDCA only applies for label_dimension=1."
      assert not _joint_weights, ("_joint_weights is incompatible with"
                                  " SDCAOptimizer.")

      model_fn = sdca_model_fn
      # The model_fn passes the model parameters to the chief_hook. We then use
      # the hook to update weights and shrink step only on the chief.
      chief_hook = _SdcaUpdateWeightsHook()
      params.update({
          "weight_column_name": weight_column_name,
          "update_weights_hook": chief_hook,
      })
    else:
      model_fn = _linear_model_fn
      params.update({
          "gradient_clip_norm": gradient_clip_norm,
          "joint_weights": _joint_weights,
      })

    super(LinearRegressor, self).__init__(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
        feature_engineering_fn=feature_engineering_fn)

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
    return super(LinearRegressor, self).predict(
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
    preds = super(LinearRegressor, self).predict(
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
    """See BaseEstimator.export."""
    def default_input_fn(unused_estimator, examples):
      return layers.parse_feature_columns_from_examples(
          examples, self._feature_columns)

    return super(LinearRegressor, self).export(
        export_dir=export_dir,
        input_fn=input_fn or default_input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=use_deprecated_input_fn,
        signature_fn=(signature_fn or export.regression_signature_fn),
        prediction_key=prediction_key.PredictionKey.SCORES,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)


class LinearEstimator(estimator.Estimator):
  """Linear model with user specified head.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Train a generalized linear model to predict label value given observation of
  feature values.

  Example:
  To do poisson regression,

  ```python
  sparse_column_a = sparse_column_with_hash_bucket(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  estimator = LinearEstimator(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
      head=head_lib.poisson_regression_head())

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
               head,
               model_dir=None,
               weight_column_name=None,
               optimizer=None,
               gradient_clip_norm=None,
               _joint_weights=False,
               config=None,
               feature_engineering_fn=None):
    """Construct a `LinearEstimator` object.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      head: An instance of _Head class.
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
      _joint_weights: If True use a single (possibly partitioned) variable to
        store the weights. It's faster, but requires all feature columns are
        sparse and have the 'sum' combiner. Incompatible with SDCAOptimizer.
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.

    Returns:
      A `LinearEstimator` estimator.

    Raises:
      ValueError: if optimizer is not supported, e.g., SDCAOptimizer
    """
    assert feature_columns
    if isinstance(optimizer, sdca_optimizer.SDCAOptimizer):
      raise ValueError("LinearEstimator does not support SDCA optimizer.")

    params = {
        "head": head,
        "feature_columns": feature_columns,
        "optimizer": optimizer,
        "gradient_clip_norm": gradient_clip_norm,
        "joint_weights": _joint_weights,
    }
    super(LinearEstimator, self).__init__(
        model_fn=_linear_model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
        feature_engineering_fn=feature_engineering_fn)
