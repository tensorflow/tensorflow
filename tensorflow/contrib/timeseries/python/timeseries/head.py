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
"""Timeseries head."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.training import training_util
from tensorflow.contrib.layers.python.layers import optimizers

from tensorflow.contrib.timeseries.python.timeseries import feature_keys

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.export import export_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def time_series_regression_head(model,
                                state_manager,
                                optimizer,
                                input_statistics_generator=None):
  """Creates a `_Head` for time series regression.

  Args:
    model: A model for time series regression.
    state_manager: A state manager.
    optimizer: An optimizer.
    input_statistics_generator: A input statistics generator.

  Returns:
    An instance of `_Head` for time series regression.
  """
  return _TimeSeriesRegressionHead(model, state_manager, optimizer,
                                   input_statistics_generator)


class _TimeSeriesRegressionHead(head_lib._Head):  # pylint:disable=protected-access
  """See `time_series_regression_head`."""

  def __init__(self,
               model,
               state_manager,
               optimizer,
               input_statistics_generator=None,
               name=None):
    self.model = model
    self.state_manager = state_manager
    self.optimizer = optimizer
    self.input_statistics_generator = input_statistics_generator
    self._name = name

  @property
  def name(self):
    return self._name

  def create_loss(self, features, mode, logits=None, labels=None):
    """See `_Head`."""
    with variable_scope.variable_scope("model", reuse=variable_scope.AUTO_REUSE):
      model_outputs = self.state_manager.define_loss(
          self.model, features, mode)
    return model_outputs

  @property
  def logits_dimension(self):
    """See `_Head`."""
    return 1

  def _train_ops(self, features):
    """Add training ops to the graph."""
    mode = estimator_lib.ModeKeys.TRAIN
    model_outputs = self.create_loss(features, mode)
    train_op = optimizers.optimize_loss(
        model_outputs.loss,
        global_step=training_util.get_global_step(),
        optimizer=self.optimizer,
        # Learning rate is set in the Optimizer object
        learning_rate=None)
    return estimator_lib.EstimatorSpec(
        loss=model_outputs.loss,
        mode=mode,
        train_op=train_op)

  def _evaluate_ops(self, features):
    """Add ops for evaluation (aka filtering) to the graph."""
    mode = estimator_lib.ModeKeys.EVAL
    model_outputs = self.create_loss(features, mode)
    metrics = {}
    # Just output in-sample predictions for the last chunk seen
    for prediction_key, prediction_value in model_outputs.predictions.items():
      metrics[prediction_key] = _identity_metric_single(prediction_key,
                                                        prediction_value)
    metrics[feature_keys.FilteringResults.TIMES] = _identity_metric_single(
        feature_keys.FilteringResults.TIMES, model_outputs.prediction_times)
    metrics[feature_keys.FilteringResults.STATE_TUPLE] = (
        _identity_metric_nested(feature_keys.FilteringResults.STATE_TUPLE,
                                model_outputs.end_state))
    return estimator_lib.EstimatorSpec(
        loss=model_outputs.loss,
        mode=mode,
        eval_metric_ops=metrics,
        predictions={})

  def _predict_ops(self, features):
    """Add ops for prediction to the graph."""
    with variable_scope.variable_scope("model"):
      prediction = self.model.predict(features=features)
    prediction[feature_keys.PredictionResults.TIMES] = features[
        feature_keys.PredictionFeatures.TIMES]
    return estimator_lib.EstimatorSpec(
        predictions=prediction, mode=estimator_lib.ModeKeys.PREDICT)

  def _serving_ops(self, features):
    """Add ops for serving to the graph."""
    with variable_scope.variable_scope("model"):
      prediction_outputs = self.model.predict(features=features)
    filtering_outputs = self.create_loss(features, estimator_lib.ModeKeys.EVAL)
    return estimator_lib.EstimatorSpec(
        mode=estimator_lib.ModeKeys.PREDICT,
        export_outputs={
            feature_keys.SavedModelLabels.PREDICT:
                export_lib.PredictOutput(prediction_outputs),
            feature_keys.SavedModelLabels.FILTER:
                export_lib.PredictOutput(
                    state_to_dictionary(filtering_outputs.end_state))
        },
        # Likely unused, but it is necessary to return `predictions` to satisfy
        # the Estimator's error checking.
        predictions={})

  def _convert_feature_to_tensor(self, name, value):
    """Casts features to the correct dtype based on their name."""
    if name in [
        feature_keys.TrainEvalFeatures.TIMES,
        feature_keys.PredictionFeatures.TIMES
    ]:
      return math_ops.cast(value, dtypes.int64)
    if name == feature_keys.TrainEvalFeatures.VALUES:
      return math_ops.cast(value, self.model.dtype)
    if name == feature_keys.PredictionFeatures.STATE_TUPLE:
      return value  # Correct dtypes are model-dependent
    return ops.convert_to_tensor(value)

  def _gather_state(self, features):
    """Returns `features` with state packed, indicates if packing was done."""
    prefixed_state_re = re.compile(r"^" + feature_keys.State.STATE_PREFIX +
                                   r"_(\d+)$")
    numbered_state = []
    for key, tensor in features.items():
      search_result = prefixed_state_re.search(key)
      if search_result:
        numbered_state.append((int(search_result.group(1)), key, tensor))
    if not numbered_state:
      return features, False
    features = features.copy()
    for _, key, _ in numbered_state:
      del features[key]
    numbered_state.sort(key=lambda number, *_: number)
    features[feature_keys.State.STATE_TUPLE] = nest.pack_sequence_as(
        structure=self.model.get_start_state(),
        flat_sequence=[tensor for _, _, tensor in numbered_state])
    return features, True

  def create_estimator_spec(self, features, mode, labels=None):
    """Performs basic error checking and returns an EstimatorSpec."""
    with ops.name_scope(self._name, "head"):
      if labels:
        raise ValueError(
            "The model received a `labels` dictionary, which is "
            "not supported. Pass '{}' and '{}' as "
            "features.".format(feature_keys.TrainEvalFeatures.TIMES,
                               feature_keys.TrainEvalFeatures.VALUES))
      del labels
      features = {
          name: self._convert_feature_to_tensor(name=name, value=value)
          for name, value in features.items()
      }
      if self.input_statistics_generator is not None:
        input_statistics = self.input_statistics_generator.initialize_graph(
            features, update_statistics=(mode == estimator_lib.ModeKeys.TRAIN))
      else:
        input_statistics = None
      self.model.initialize_graph(input_statistics=input_statistics)

      # _gather_state requires the model to have its graph initialized (so it
      # has access to the structure of the model's state)
      features, passed_flat_state = self._gather_state(features)
      if (mode == estimator_lib.ModeKeys.TRAIN or
          mode == estimator_lib.ModeKeys.EVAL):
        _check_train_eval_features(features, self.model)
      elif mode == estimator_lib.ModeKeys.PREDICT:
        _check_predict_features(features)
      else:
        raise ValueError("Unknown mode '{}' passed to model_fn.".format(mode))

      self.state_manager.initialize_graph(
          model=self.model, input_statistics=input_statistics)

      if mode == estimator_lib.ModeKeys.TRAIN:
        return self._train_ops(features)
      elif mode == estimator_lib.ModeKeys.EVAL:
        return self._evaluate_ops(features)
      elif mode == estimator_lib.ModeKeys.PREDICT and not passed_flat_state:
        return self._predict_ops(features)
      elif mode == estimator_lib.ModeKeys.PREDICT and passed_flat_state:
        # The mode is PREDICT, but we're actually in export_savedmodel for
        # serving. We want to return two graphs: one for filtering (state + data
        # -> state) and one for predicting (state -> prediction).
        return self._serving_ops(features)


def _check_feature_shapes_compatible_with(features,
                                          compatible_with_name,
                                          compatible_with_value,
                                          ignore=None):
  """Checks all features are compatible with the given time-like feature."""
  if ignore is None:
    ignore = set()
  for name, value in features.items():
    if name in ignore:
      continue
    feature_shape = value.get_shape()
    if feature_shape.ndims is None:
      continue
    if feature_shape.ndims < 2:
      raise ValueError(
          ("Features must have shape (batch dimension, window size, ...) "
           "(got rank {} for feature '{}')").format(feature_shape.ndims, name))
    if not feature_shape[:2].is_compatible_with(
        compatible_with_value.get_shape()):
      raise ValueError(
          ("Features must have shape (batch dimension, window size, ...) "
           "where batch dimension and window size match the "
           "'{times_feature}' feature (got shape {feature_shape} for "
           "feature '{feature_name}' but shape {times_shape} for feature "
           "'{times_feature}')").format(
               times_feature=compatible_with_name,
               feature_shape=feature_shape,
               feature_name=name,
               times_shape=compatible_with_value.get_shape()))


def _check_predict_features(features):
  """Raises errors if features are not suitable for prediction."""
  if feature_keys.PredictionFeatures.TIMES not in features:
    raise ValueError("Expected a '{}' feature for prediction.".format(
        feature_keys.PredictionFeatures.TIMES))
  if feature_keys.PredictionFeatures.STATE_TUPLE not in features:
    raise ValueError("Expected a '{}' feature for prediction.".format(
        feature_keys.PredictionFeatures.STATE_TUPLE))
  times_feature = features[feature_keys.PredictionFeatures.TIMES]
  if not times_feature.get_shape().is_compatible_with([None, None]):
    raise ValueError(
        ("Expected shape (batch dimension, window size) for feature '{}' "
         "(got shape {})").format(feature_keys.PredictionFeatures.TIMES,
                                  times_feature.get_shape()))
  _check_feature_shapes_compatible_with(
      features=features,
      compatible_with_name=feature_keys.PredictionFeatures.TIMES,
      compatible_with_value=times_feature,
      ignore=set([
          feature_keys.PredictionFeatures.STATE_TUPLE  # Model-dependent shapes
      ]))


def _check_train_eval_features(features, model):
  """Raise errors if features are not suitable for training/evaluation."""
  if feature_keys.TrainEvalFeatures.TIMES not in features:
    raise ValueError("Expected a '{}' feature for training/evaluation.".format(
        feature_keys.TrainEvalFeatures.TIMES))
  if feature_keys.TrainEvalFeatures.VALUES not in features:
    raise ValueError("Expected a '{}' feature for training/evaluation.".format(
        feature_keys.TrainEvalFeatures.VALUES))
  times_feature = features[feature_keys.TrainEvalFeatures.TIMES]
  if not times_feature.get_shape().is_compatible_with([None, None]):
    raise ValueError(
        ("Expected shape (batch dimension, window size) for feature '{}' "
         "(got shape {})").format(feature_keys.TrainEvalFeatures.TIMES,
                                  times_feature.get_shape()))
  values_feature = features[feature_keys.TrainEvalFeatures.VALUES]
  if not values_feature.get_shape().is_compatible_with(
      [None, None, model.num_features]):
    raise ValueError(
        ("Expected shape (batch dimension, window size, {num_features}) "
         "for feature '{feature_name}', since the model was configured "
         "with num_features={num_features} (got shape {got_shape})").format(
             num_features=model.num_features,
             feature_name=feature_keys.TrainEvalFeatures.VALUES,
             got_shape=times_feature.get_shape()))
  _check_feature_shapes_compatible_with(
      features=features,
      compatible_with_name=feature_keys.TrainEvalFeatures.TIMES,
      compatible_with_value=times_feature,
      ignore=set([
          feature_keys.State.STATE_TUPLE  # Model-dependent shapes
      ]))


def _identity_metric_single(name, input_tensor):
  """A metric which takes on its last updated value.

  This keeps evaluation metrics in sync with one another, since update ops are
  run separately from their result Tensors. Simply returning (input_tensor,
  no_op) as a metric with a value but no update means that a metric will come
  from a different batch of data than metrics which cache values in a Variable
  (e.g. the default loss metric).

  Args:
    name: A name for the metric.
    input_tensor: Any Tensor.
  Returns:
    A tuple of (value, update_op).
  """
  metric_variable = variable_scope.variable(
      name="{}_identity_metric".format(name),
      initial_value=array_ops.zeros([], dtype=input_tensor.dtype),
      collections=[ops.GraphKeys.LOCAL_VARIABLES],
      validate_shape=False)
  update_op = state_ops.assign(
      metric_variable, input_tensor, validate_shape=False)
  # This shape will be correct once the first update runs (but may be
  # incomplete, so is not helpful for initializing the variable).
  metric_variable.set_shape(input_tensor.get_shape())
  return (metric_variable.value(), update_op)


def _identity_metric_nested(name, input_tensors):
  """Create identity metrics for a nested tuple of Tensors."""
  update_ops = []
  value_tensors = []
  for tensor_number, tensor in enumerate(nest.flatten(input_tensors)):
    value_tensor, update_op = _identity_metric_single(
        name="{}_{}".format(name, tensor_number), input_tensor=tensor)
    update_ops.append(update_op)
    value_tensors.append(value_tensor)
  return (nest.pack_sequence_as(input_tensors, value_tensors),
          control_flow_ops.group(*update_ops))


def state_to_dictionary(state_tuple):
  """Flatten model state into a dictionary with string keys."""
  flattened = {}
  for state_number, state_value in enumerate(nest.flatten(state_tuple)):
    prefixed_state_name = "{}_{:02d}".format(feature_keys.State.STATE_PREFIX,
                                             state_number)
    flattened[prefixed_state_name] = state_value
  return flattened
