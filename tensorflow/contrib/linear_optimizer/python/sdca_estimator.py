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
"""Linear Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.python.training import training_util
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import session_run_hook


def _head_is_valid_for_sdca(head):
  """Returns true if the provided head is supported by SDCAOptimizer."""
  # pylint: disable=protected-access
  return isinstance(head, head_lib._BinaryLogisticHead) or isinstance(
      head, head_lib._BinarySvmHead) or isinstance(head,
                                                   head_lib._RegressionHead)
  # pylint: enable=protected-access


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
  columns_to_tensors[bias_column] = array_ops.ones(
      [batch_size, 1], dtype=dtypes.float32)
  columns_to_variables[bias_column] = [bias_variable]


def sdca_model_fn(features, labels, mode, params, config=None):
  """A model_fn for linear models that use the SDCA optimizer.

  Args:
    features: A dict of `Tensor` keyed by column name.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
      dtype `int32` or `int64` with values in the set {0, 1}.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * head: A `Head` instance. Type must be one of `_BinarySvmHead`,
          `_RegressionHead` or `_BinaryLogisticHead`.
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * l1_regularization: Global (across all examples) L1-regularization
          parameter.
      * l2_regularization: Global (across all examples) L2-regularization
          parameter.
      * num_loss_partitions: Number of partitions of the global loss function
          optimized by `SDCAOptimizer`.
      * weight_column_name: A string defining the weight feature column, or
          None if there are no weights.
      * update_weights_hook: A `SessionRunHook` object or None. Used to update
          model weights.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    A `ModelFnOps` instance.

  Raises:
    ValueError: If the type of head is not one of `_BinarySvmHead`,
      `_RegressionHead` or `_MultiClassHead`.
    ValueError: If mode is not any of the `ModeKeys`.
  """
  head = params["head"]
  feature_columns = params["feature_columns"]
  example_id_column = params["example_id_column"]
  l1_regularization = params["l1_regularization"]
  l2_regularization = params["l2_regularization"]
  num_loss_partitions = params["num_loss_partitions"]
  weight_column_name = params["weight_column_name"]
  update_weights_hook = params.get("update_weights_hook", None)
  partitioner = params["partitioner"]

  loss_type = None
  if isinstance(head, head_lib._BinarySvmHead):  # pylint: disable=protected-access
    loss_type = "hinge_loss"
  elif isinstance(head, head_lib._BinaryLogisticHead):  # pylint: disable=protected-access
    loss_type = "logistic_loss"
  elif isinstance(head, head_lib._RegressionHead):  # pylint: disable=protected-access
    loss_type = "squared_loss"
  else:
    raise ValueError("Unsupported head type: {}".format(type(head)))

  assert head.logits_dimension == 1, (
      "SDCA only applies to logits_dimension=1.")

  # Update num_loss_partitions based on number of workers.
  n_loss_partitions = num_loss_partitions or max(1, config.num_worker_replicas)
  optimizer = sdca_optimizer.SDCAOptimizer(
      example_id_column=example_id_column,
      num_loss_partitions=n_loss_partitions,
      symmetric_l1_regularization=l1_regularization,
      symmetric_l2_regularization=l2_regularization,
      partitioner=partitioner)

  parent_scope = "linear"

  with variable_scope.variable_scope(
      values=features.values(), name_or_scope=parent_scope,
      partitioner=partitioner) as scope:
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
    sdca_model, train_op = optimizer.get_train_step(
        columns_to_variables, weight_column_name, loss_type, features, labels,
        global_step)
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
    return model_fn_ops._replace(training_chief_hooks=(
        model_fn_ops.training_chief_hooks + [update_weights_hook]))
  return model_fn_ops


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


class _SDCAEstimator(estimator.Estimator):
  """Base estimator class for linear models using the SDCA optimizer.

  This class should not be used directly. Rather, users should call one of the
  derived estimators.
  """

  def __init__(self,
               example_id_column,
               feature_columns,
               weight_column_name=None,
               model_dir=None,
               head=None,
               l1_regularization=0.0,
               l2_regularization=1.0,
               num_loss_partitions=None,
               config=None,
               feature_engineering_fn=None,
               partitioner=None):
    """Construct a `_SDCAEstimator` estimator object.

    Args:
      example_id_column: A string defining the feature column name representing
        example ids. Used to initialize the underlying SDCA optimizer.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      model_dir: Directory to save model parameters, graph etc. This can also be
        used to load checkpoints from the directory into an estimator to
        continue training a previously saved model.
      head: type of head. Currently, _BinaryLogisticHead and _BinarySvmHead are
        supported for classification and _RegressionHead for regression. It
        should be a subclass of _SingleHead.
      l1_regularization: L1-regularization parameter. Refers to global L1
        regularization (across all examples).
      l2_regularization: L2-regularization parameter. Refers to global L2
        regularization (across all examples).
      num_loss_partitions: number of partitions of the (global) loss function
        optimized by the underlying optimizer (SDCAOptimizer).
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      partitioner: Variable partitioner for the primal weights (`div`
        partitioning strategy will be used).

    Returns:
      A `_SDCAEstimator` estimator.

    Raises:
      ValueError: if head is not supported by SDCA.
    """
    self._feature_columns = tuple(feature_columns or [])
    assert self._feature_columns

    if not _head_is_valid_for_sdca(head):
      raise ValueError(
          "head type: {} is not supported. Supported head types: "
          "_BinaryLogisticHead, _BinarySvmHead and _RegressionHead.".format(
              type(head)))
    assert head.logits_dimension == 1

    params = {
        "head": head,
        "feature_columns": feature_columns,
        "example_id_column": example_id_column,
        "num_loss_partitions": num_loss_partitions,
        "l1_regularization": l1_regularization,
        "l2_regularization": l2_regularization,
        "weight_column_name": weight_column_name,
        "update_weights_hook": _SdcaUpdateWeightsHook(),
        "partitioner": partitioner,
    }

    super(_SDCAEstimator, self).__init__(
        model_fn=sdca_model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
        feature_engineering_fn=feature_engineering_fn)


class SDCALogisticClassifier(_SDCAEstimator):
  """Logistic regression binary classifier using the SDCA optimizer.

  Example usage:

  ```python
  sparse_column_a = sparse_column_with_hash_bucket(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  classifier = SDCALogisticClassifier(
      example_id_column='example_id',
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b]),
      weight_column_name=...,
      l2_regularization=...,
      num_loss_partitions=...,
  )

  # Input builders
  # returns x, y (where y is the label Tensor (with 0/1 values)
  def input_fn_{train, eval}:

  # returns x (features dict)
  def input_fn_test:
    ...
  classifier.fit(input_fn=input_fn_train)
  classifier.evaluate(input_fn=input_fn_eval)
  # Returns predicted classes.
  classifier.predict_classes(input_fn=input_fn_test)
  # Returns predicted probabilities.
  classifier.predict_proba(input_fn=input_fn_test)
  ```

  The input_fn provided to `fit`, `evaluate` and predict_* methods should return
  the following features, otherwise there will be a `KeyError`:
    * A feature with `key=example_id_column` whose value is a `Tensor` of dtype
      string.
    * If `weight_column_name` is not `None`, a feature with
      `key=weight_column_name` whose value is a `Tensor`.
    * For each `column` in `feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name` whose
        `value` is a `SparseTensor`
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`
      - if `column` is a `WeightedSparseColumn`, two features: the first with
        `key` the id column name, the second with `key` the weight column name.
        Both features' `value` must be a `SparseTensor`
  """

  def __init__(self,
               example_id_column,
               feature_columns,
               weight_column_name=None,
               model_dir=None,
               l1_regularization=0.0,
               l2_regularization=1.0,
               num_loss_partitions=None,
               config=None,
               feature_engineering_fn=None,
               partitioner=None):
    """Construct a `SDCALogisticClassifier` object.

    Args:
      example_id_column: A string defining the feature column name representing
        example ids. Used to initialize the underlying SDCA optimizer.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the iterable should derive from `FeatureColumn`.
        Note that the order of the items is ignored at model construction time.
      weight_column_name: A string defining feature column name representing
        weights. It is used to downweight or boost examples during training. It
        will be multiplied by the loss of the example.
      model_dir: Directory to save model parameters, graph etc. This can also be
        used to load checkpoints from the directory into an estimator to
        continue training a previously saved model.
      l1_regularization: L1-regularization parameter. Refers to global L1
        regularization (across all examples).
      l2_regularization: L2-regularization parameter. Refers to global L2
        regularization (across all examples).
      num_loss_partitions: Number of partitions of the global loss function
        optimized by the underlying optimizer (SDCAOptimizer).
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      partitioner: Variable partitioner for the primal weights (`div`
        partitioning strategy will be used).

    Returns:
      A `SDCALogisiticClassifier` estimator.
    """
    super(SDCALogisticClassifier, self).__init__(
        example_id_column=example_id_column,
        feature_columns=feature_columns,
        weight_column_name=weight_column_name,
        model_dir=model_dir,
        head=head_lib.multi_class_head(
            n_classes=2, weight_column_name=weight_column_name),
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization,
        num_loss_partitions=num_loss_partitions,
        config=config,
        feature_engineering_fn=None,
        partitioner=partitioner)

  def predict_classes(self, input_fn=None):
    """Runs inference to determine the predicted class.

    Args:
      input_fn: The input function providing features.

    Returns:
      A generator of predicted classes for the features provided by input_fn.
    """
    key = prediction_key.PredictionKey.CLASSES
    predictions = super(SDCALogisticClassifier, self).predict(
        input_fn=input_fn, outputs=[key])
    return (pred[key] for pred in predictions)

  def predict_proba(self, input_fn=None):
    """Runs inference to determine the class probability predictions.

    Args:
      input_fn: The input function providing features.

    Returns:
      A generator of predicted class probabilities for the features provided by
        input_fn.
    """
    key = prediction_key.PredictionKey.PROBABILITIES
    predictions = super(SDCALogisticClassifier, self).predict(
        input_fn=input_fn, outputs=[key])
    return (pred[key] for pred in predictions)


class SDCALinearRegressor(_SDCAEstimator):
  """Linear regression model using SDCA to solve the underlying optimization.

  Example usage:

  ```python
  real_column_a = real_valued_column(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  regressor = SDCALinearRegressor(
      example_id_column='example_id',
      feature_columns=[real_column_a, sparse_column_b]),
      weight_column_name=...,
      l2_regularization=...,
      num_loss_partitions=...,
  )

  # Input builders
  # returns x, y (where y is the label Tensor (with 0/1 values)
  def input_fn_{train, eval}:

  # returns x (features dict)
  def input_fn_test:
    ...
  regressor.fit(input_fn=input_fn_train)
  regressor.evaluate(input_fn=input_fn_eval)
  regressor.predict_scores(input_fn=input_fn_test) # returns predicted scores.
  ```

  The input_fn provided to `fit`, `evaluate` and predict_* methods should return
  the following features, otherwise there will be a `KeyError`:
    * A feature with `key=example_id_column` whose value is a `Tensor` of dtype
      string.
    * If `weight_column_name` is not `None`, a feature with
      `key=weight_column_name` whose value is a `Tensor`.
    * For each `column` in `feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name` whose
        `value` is a `SparseTensor`
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`
      - if `column` is a `WeightedSparseColumn`, two features: the first with
        `key` the id column name, the second with `key` the weight column name.
        Both features' `value` must be a `SparseTensor`

  """

  def __init__(self,
               example_id_column,
               feature_columns,
               weight_column_name=None,
               model_dir=None,
               l1_regularization=0.0,
               l2_regularization=1.0,
               num_loss_partitions=None,
               config=None,
               feature_engineering_fn=None,
               partitioner=None):
    """Construct a `SDCALinearRegressor` estimator object.


    Args:
      example_id_column: A string defining the feature column name representing
        example ids. Used to initialize the underlying SDCA optimizer.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the iterable should derive from `FeatureColumn`.
        Note that the order of the items is ignored at model construction time.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      model_dir: Directory to save model parameters, graph etc. This can also be
        used to load checkpoints from the directory into an estimator to
        continue training a previously saved model.
      l1_regularization: L1-regularization parameter. Refers to global L1
        regularization (across all examples).
      l2_regularization: L2-regularization parameter. Refers to global L2
        regularization (across all examples).
      num_loss_partitions: number of partitions of the (global) loss function
        optimized by the underlying optimizer (SDCAOptimizer).
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      partitioner: Variable partitioner for the primal weights (`div`
        partitioning strategy will be used).

    Returns:
      A `SDCALinearRegressor` estimator.
    """
    super(SDCALinearRegressor, self).__init__(
        example_id_column=example_id_column,
        feature_columns=feature_columns,
        weight_column_name=weight_column_name,
        model_dir=model_dir,
        head=head_lib.regression_head(weight_column_name=weight_column_name),
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization,
        num_loss_partitions=num_loss_partitions,
        config=config,
        feature_engineering_fn=None,
        partitioner=partitioner)

  def predict_scores(self, input_fn):
    """Returns predicted scores for given features.

    Args:
      input_fn: The input function providing features.

    Returns:
      A generator of predicted scores for the features provided by input_fn.
    """
    key = prediction_key.PredictionKey.SCORES
    predictions = super(SDCALinearRegressor, self).predict(
        input_fn=input_fn, outputs=[key])
    return (pred[key] for pred in predictions)
