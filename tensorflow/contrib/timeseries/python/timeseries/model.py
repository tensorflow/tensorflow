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
"""Base class for time series models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six

from tensorflow.contrib.timeseries.python.timeseries import math_utils
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import PredictionFeatures
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import TrainEvalFeatures

from tensorflow.python.feature_column import feature_column_lib as feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope

from tensorflow.python.util import nest


ModelOutputs = collections.namedtuple(  # pylint: disable=invalid-name
    typename="ModelOutputs",
    field_names=[
        "loss",  # The scalar value to be minimized during training.
        "end_state",  # A nested tuple specifying the model's state after
                      # running on the specified data
        "predictions",  # A dictionary of predictions, each with shape prefixed
                        # by the shape of `prediction_times`.
        "prediction_times"  # A [batch size x window size] integer Tensor
                            # indicating times for which values in `predictions`
                            # were computed.
    ])


@six.add_metaclass(abc.ABCMeta)
class TimeSeriesModel(object):
  """Base class for creating generative time series models."""

  def __init__(self,
               num_features,
               exogenous_feature_columns=None,
               dtype=dtypes.float32):
    """Constructor for generative models.

    Args:
      num_features: Number of features for the time series
      exogenous_feature_columns: A list of `tf.feature_column`s (for example
           `tf.feature_column.embedding_column`) corresponding to exogenous
           features which provide extra information to the model but are not
           part of the series to be predicted. Passed to
           `tf.compat.v1.feature_column.input_layer`.
      dtype: The floating point datatype to use.
    """
    if exogenous_feature_columns:
      self._exogenous_feature_columns = exogenous_feature_columns
    else:
      self._exogenous_feature_columns = []
    self.num_features = num_features
    self.dtype = dtype
    self._input_statistics = None
    self._graph_initialized = False
    self._stats_means = None
    self._stats_sigmas = None

  @property
  def exogenous_feature_columns(self):
    """`tf.feature_colum`s for features which are not predicted."""
    return self._exogenous_feature_columns

  # TODO(allenl): Move more of the generic machinery for generating and
  # predicting into TimeSeriesModel, and possibly share it between generate()
  # and predict()
  def generate(self, number_of_series, series_length,
               model_parameters=None, seed=None):
    """Sample synthetic data from model parameters, with optional substitutions.

    Returns `number_of_series` possible sequences of future values, sampled from
    the generative model with each conditioned on the previous. Samples are
    based on trained parameters, except for those parameters explicitly
    overridden in `model_parameters`.

    For distributions over future observations, see predict().

    Args:
      number_of_series: Number of time series to create.
      series_length: Length of each time series.
      model_parameters: A dictionary mapping model parameters to values, which
          replace trained parameters when generating data.
      seed: If specified, return deterministic time series according to this
          value.
    Returns:
      A dictionary with keys TrainEvalFeatures.TIMES (mapping to an array with
      shape [number_of_series, series_length]) and TrainEvalFeatures.VALUES
      (mapping to an array with shape [number_of_series, series_length,
      num_features]).
    """
    raise NotImplementedError("This model does not support generation.")

  def initialize_graph(self, input_statistics=None):
    """Define ops for the model, not depending on any previously defined ops.

    Args:
      input_statistics: A math_utils.InputStatistics object containing input
          statistics. If None, data-independent defaults are used, which may
          result in longer or unstable training.
    """
    self._graph_initialized = True
    self._input_statistics = input_statistics
    if self._input_statistics:
      self._stats_means, variances = (
          self._input_statistics.overall_feature_moments)
      self._stats_sigmas = math_ops.sqrt(variances)

  def _scale_data(self, data):
    """Scale data according to stats (input scale -> model scale)."""
    if self._input_statistics is not None:
      return (data - self._stats_means) / self._stats_sigmas
    else:
      return data

  def _scale_variance(self, variance):
    """Scale variances according to stats (input scale -> model scale)."""
    if self._input_statistics is not None:
      return variance / self._input_statistics.overall_feature_moments.variance
    else:
      return variance

  def _scale_back_data(self, data):
    """Scale back data according to stats (model scale -> input scale)."""
    if self._input_statistics is not None:
      return (data * self._stats_sigmas) + self._stats_means
    else:
      return data

  def _scale_back_variance(self, variance):
    """Scale back variances according to stats (model scale -> input scale)."""
    if self._input_statistics is not None:
      return variance * self._input_statistics.overall_feature_moments.variance
    else:
      return variance

  def _check_graph_initialized(self):
    if not self._graph_initialized:
      raise ValueError(
          "TimeSeriesModels require initialize_graph() to be called before "
          "use. This defines variables and ops in the default graph, and "
          "allows Tensor-valued input statistics to be specified.")

  def define_loss(self, features, mode):
    """Default loss definition with state replicated across a batch.

    Time series passed to this model have a batch dimension, and each series in
    a batch can be operated on in parallel. This loss definition assumes that
    each element of the batch represents an independent sample conditioned on
    the same initial state (i.e. it is simply replicated across the batch). A
    batch size of one provides sequential operations on a single time series.

    More complex processing may operate instead on get_start_state() and
    get_batch_loss() directly.

    Args:
      features: A dictionary (such as is produced by a chunker) with at minimum
        the following key/value pairs (others corresponding to the
        `exogenous_feature_columns` argument to `__init__` may be included
        representing exogenous regressors):
        TrainEvalFeatures.TIMES: A [batch size x window size] integer Tensor
            with times for each observation. If there is no artificial chunking,
            the window size is simply the length of the time series.
        TrainEvalFeatures.VALUES: A [batch size x window size x num features]
            Tensor with values for each observation.
      mode: The tf.estimator.ModeKeys mode to use (TRAIN, EVAL). For INFER,
        see predict().
    Returns:
      A ModelOutputs object.
    """
    self._check_graph_initialized()
    start_state = math_utils.replicate_state(
        start_state=self.get_start_state(),
        batch_size=array_ops.shape(features[TrainEvalFeatures.TIMES])[0])
    return self.get_batch_loss(features=features, mode=mode, state=start_state)

  # TODO(vitalyk,allenl): Better documentation surrounding options for chunking,
  # references to papers, etc.
  @abc.abstractmethod
  def get_start_state(self):
    """Returns a tuple of state for the start of the time series.

    For example, a mean and covariance. State should not have a batch
    dimension, and will often be TensorFlow Variables to be learned along with
    the rest of the model parameters.
    """
    pass

  @abc.abstractmethod
  def get_batch_loss(self, features, mode, state):
    """Return predictions, losses, and end state for a time series.

    Args:
      features: A dictionary with times, values, and (optionally) exogenous
          regressors. See `define_loss`.
      mode: The tf.estimator.ModeKeys mode to use (TRAIN, EVAL, INFER).
      state: Model-dependent state, each with size [batch size x ...]. The
          number and type will typically be fixed by the model (for example a
          mean and variance).
    Returns:
      A ModelOutputs object.
    """
    pass

  @abc.abstractmethod
  def predict(self, features):
    """Returns predictions of future observations given an initial state.

    Computes distributions for future observations. For sampled draws from the
    model where each is conditioned on the previous, see generate().

    Args:
      features: A dictionary with at minimum the following key/value pairs
        (others corresponding to the `exogenous_feature_columns` argument to
        `__init__` may be included representing exogenous regressors):
        PredictionFeatures.TIMES: A [batch size x window size] Tensor with
          times to make predictions for. Times must be increasing within each
          part of the batch, and must be greater than the last time `state` was
          updated.
        PredictionFeatures.STATE_TUPLE: Model-dependent state, each with size
          [batch size x ...]. The number and type will typically be fixed by the
          model (for example a mean and variance). Typically these will be the
          end state returned by get_batch_loss, predicting beyond that data.
    Returns:
      A dictionary with model-dependent predictions corresponding to the
      requested times. Keys indicate the type of prediction, and values have
      shape [batch size x window size x ...]. For example state space models
      return a "predicted_mean" and "predicted_covariance".
    """
    pass

  def _get_exogenous_embedding_shape(self):
    """Computes the shape of the vector returned by _process_exogenous_features.

    Returns:
      The shape as a list. Does not include a batch dimension.
    """
    if not self._exogenous_feature_columns:
      return (0,)
    with ops.Graph().as_default():
      parsed_features = (
          feature_column.make_parse_example_spec(
              self._exogenous_feature_columns))
      placeholder_features = parsing_ops.parse_example(
          serialized=array_ops.placeholder(shape=[None], dtype=dtypes.string),
          features=parsed_features)
      embedded = feature_column.input_layer(
          features=placeholder_features,
          feature_columns=self._exogenous_feature_columns)
      return embedded.get_shape().as_list()[1:]

  def _process_exogenous_features(self, times, features):
    """Create a single vector from exogenous features.

    Args:
      times: A [batch size, window size] vector of times for this batch,
          primarily used to check the shape information of exogenous features.
      features: A dictionary of exogenous features corresponding to the columns
          in self._exogenous_feature_columns. Each value should have a shape
          prefixed by [batch size, window size].
    Returns:
      A Tensor with shape [batch size, window size, exogenous dimension], where
      the size of the exogenous dimension depends on the exogenous feature
      columns passed to the model's constructor.
    Raises:
      ValueError: If an exogenous feature has an unknown rank.
    """
    if self._exogenous_feature_columns:
      exogenous_features_single_batch_dimension = {}
      for name, tensor in features.items():
        if tensor.get_shape().ndims is None:
          # input_from_feature_columns does not support completely unknown
          # feature shapes, so we save on a bit of logic and provide a better
          # error message by checking that here.
          raise ValueError(
              ("Features with unknown rank are not supported. Got shape {} for "
               "feature {}.").format(tensor.get_shape(), name))
        tensor_shape_dynamic = array_ops.shape(tensor)
        tensor = array_ops.reshape(
            tensor,
            array_ops.concat([[tensor_shape_dynamic[0]
                               * tensor_shape_dynamic[1]],
                              tensor_shape_dynamic[2:]], axis=0))
        # Avoid shape warnings when embedding "scalar" exogenous features (those
        # with only batch and window dimensions); input_from_feature_columns
        # expects input ranks to match the embedded rank.
        if tensor.get_shape().ndims == 1 and tensor.dtype != dtypes.string:
          exogenous_features_single_batch_dimension[name] = tensor[:, None]
        else:
          exogenous_features_single_batch_dimension[name] = tensor
      embedded_exogenous_features_single_batch_dimension = (
          feature_column.input_layer(
              features=exogenous_features_single_batch_dimension,
              feature_columns=self._exogenous_feature_columns,
              trainable=True))
      exogenous_regressors = array_ops.reshape(
          embedded_exogenous_features_single_batch_dimension,
          array_ops.concat(
              [
                  array_ops.shape(times), array_ops.shape(
                      embedded_exogenous_features_single_batch_dimension)[1:]
              ],
              axis=0))
      exogenous_regressors.set_shape(times.get_shape().concatenate(
          embedded_exogenous_features_single_batch_dimension.get_shape()[1:]))
      exogenous_regressors = math_ops.cast(
          exogenous_regressors, dtype=self.dtype)
    else:
      # Not having any exogenous features is a special case so that models can
      # avoid superfluous updates, which may not be free of side effects due to
      # bias terms in transformations.
      exogenous_regressors = None
    return exogenous_regressors


# TODO(allenl): Add a superclass of SequentialTimeSeriesModel which fuses
# filtering/prediction/exogenous into one step, and move looping constructs to
# that class.
class SequentialTimeSeriesModel(TimeSeriesModel):
  """Base class for recurrent generative models.

  Models implementing this interface have three main functions, corresponding to
  abstract methods:
    _filtering_step: Updates state based on observations and computes a loss.
    _prediction_step: Predicts a batch of observations and new model state.
    _imputation_step: Updates model state across a gap.
    _exogenous_input_step: Updates state to account for exogenous regressors.

  Models may also specify a _window_initializer to prepare for a window of data.

  See StateSpaceModel for a concrete example of a model implementing this
  interface.

  """

  def __init__(self,
               train_output_names,
               predict_output_names,
               num_features,
               normalize_features=False,
               dtype=dtypes.float32,
               exogenous_feature_columns=None,
               exogenous_update_condition=None,
               static_unrolling_window_size_threshold=None):
    """Initialize a SequentialTimeSeriesModel.

    Args:
      train_output_names: A list of products/predictions returned from
          _filtering_step.
      predict_output_names: A list of products/predictions returned from
          _prediction_step.
      num_features: Number of features for the time series
      normalize_features: Boolean. If True, `values` are passed normalized to
          the model (via self._scale_data). Scaling is done for the whole window
          as a batch, which is slightly more efficient than scaling inside the
          window loop. The model must then define _scale_back_predictions, which
          may use _scale_back_data or _scale_back_variance to return predictions
          to the input scale.
      dtype: The floating point datatype to use.
      exogenous_feature_columns: A list of `tf.feature_column`s objects. See
          `TimeSeriesModel`.
      exogenous_update_condition: A function taking two Tensor arguments `times`
          (shape [batch size]) and `features` (a dictionary mapping exogenous
          feature keys to Tensors with shapes [batch size, ...]) and returning a
          boolean Tensor with shape [batch size] indicating whether state should
          be updated using exogenous features for each part of the batch. Where
          it is False, no exogenous update is performed. If None (default),
          exogenous updates are always performed. Useful for avoiding "leaky"
          frequent exogenous updates when sparse updates are desired. Called
          only during graph construction.
      static_unrolling_window_size_threshold: Controls whether a `tf.while_loop`
          is used when looping over a window of data. If
          `static_unrolling_window_size_threshold` is None, a `tf.while_loop` is
          always used. Otherwise it must be an integer, and the graph is
          replicated for each step taken whenever the window size is less than
          or equal to this value (if the window size is available in the static
          shape information of the TrainEvalFeatures.TIMES feature). Static
          unrolling generally decreases the per-step time for small window/batch
          sizes, but increases graph construction time.
    """
    super(SequentialTimeSeriesModel, self).__init__(
        num_features=num_features, dtype=dtype,
        exogenous_feature_columns=exogenous_feature_columns)
    self._exogenous_update_condition = exogenous_update_condition
    self._train_output_names = train_output_names
    self._predict_output_names = predict_output_names
    self._normalize_features = normalize_features
    self._static_unrolling_window_size_threshold = (
        static_unrolling_window_size_threshold)

  def _scale_back_predictions(self, predictions):
    """Return a window of predictions to input scale.

    Args:
      predictions: A dictionary mapping from prediction names to Tensors.
    Returns:
      A dictionary with values corrected for input normalization (e.g. with
      self._scale_back_mean and possibly self._scale_back_variance). May be a
      mutated version of the argument.
    """
    raise NotImplementedError(
        "SequentialTimeSeriesModel normalized input data"
        " (normalize_features=True), but no method was provided to transform "
        "the predictions back to the input scale.")

  @abc.abstractmethod
  def _filtering_step(self, current_times, current_values, state, predictions):
    """Compute a single-step loss for a batch of data.

    Args:
      current_times: A [batch size] Tensor of times for each observation.
      current_values: A [batch size] Tensor of values for each observation.
      state: Model state, updated to current_times.
      predictions: The outputs of _prediction_step
    Returns:
      A tuple of (updated state, outputs):
        updated state: Model state taking current_values into account.
        outputs: A dictionary of Tensors with keys corresponding to
            self._train_output_names, plus a special "loss" key. The value
            corresponding to "loss" is minimized during training. Other outputs
            may include one-step-ahead predictions, for example a predicted
            location and scale.
    """
    pass

  @abc.abstractmethod
  def _prediction_step(self, current_times, state):
    """Compute a batch of single-step predictions.

    Args:
      current_times: A [batch size] Tensor of times for each observation.
      state: Model state, imputed to one step before current_times.
    Returns:
      A tuple of (updated state, outputs):
        updated state: Model state updated to current_times.
        outputs: A dictionary of Tensors with keys corresponding to
            self._predict_output_names.
    """
    pass

  @abc.abstractmethod
  def _imputation_step(self, current_times, state):
    """Update model state across missing values.

    Called to prepare model state for _filtering_step and _prediction_step.

    Args:
      current_times: A [batch size] Tensor; state will be imputed up to, but not
          including, these timesteps.
      state: The pre-imputation model state, Tensors with shape [batch size x
          ...].
    Returns:
      Updated/imputed model state, corresponding to `state`.
    """
    pass

  @abc.abstractmethod
  def _exogenous_input_step(
      self, current_times, current_exogenous_regressors, state):
    """Update state to account for exogenous regressors.

    Args:
      current_times: A [batch size] Tensor of times for the exogenous values
          being input.
      current_exogenous_regressors: A [batch size x exogenous input dimension]
          Tensor of exogenous values for each part of the batch.
      state: Model state, a possibly nested list of Tensors, each with shape
          [batch size x ...].
    Returns:
      Updated model state, structure and shapes matching the `state` argument.
    """
    pass

  # TODO(allenl): Move regularization to a separate object (optional and
  # configurable)
  def _loss_additions(self, times, values, mode):
    """Additions to per-observation normalized loss, e.g. regularization.

    Args:
      times: A [batch size x window size] Tensor with times for each
          observation.
      values: A [batch size x window size x num features] Tensor with values for
          each observation.
      mode: The tf.estimator.ModeKeys mode to use (TRAIN, EVAL, INFER).
    Returns:
      A scalar value to add to the per-observation normalized loss.
    """
    del times, values, mode
    return 0.

  def _window_initializer(self, times, state):
    """Prepare for training or prediction on a window of data.

    Args:
      times: A [batch size x window size] Tensor with times for each
          observation.
      state: Model-dependent state, each with size [batch size x ...]. The
          number and type will typically be fixed by the model (for example a
          mean and variance).
    Returns:
      Nothing
    """
    pass

  def get_batch_loss(self, features, mode, state):
    """Calls self._filtering_step. See TimeSeriesModel.get_batch_loss."""
    per_observation_loss, state, outputs = self.per_step_batch_loss(
        features, mode, state)
    # per_step_batch_loss returns [batch size, window size, ...] state, whereas
    # get_batch_loss is expected to return [batch size, ...] state for the last
    # element of a window
    state = nest.pack_sequence_as(
        state,
        [state_element[:, -1] for state_element in nest.flatten(state)])
    outputs["observed"] = features[TrainEvalFeatures.VALUES]
    return ModelOutputs(
        loss=per_observation_loss,
        end_state=state,
        predictions=outputs,
        prediction_times=features[TrainEvalFeatures.TIMES])

  def _apply_exogenous_update(
      self, current_times, step_number, state, raw_features,
      embedded_exogenous_regressors):
    """Performs a conditional state update based on exogenous features."""
    if embedded_exogenous_regressors is None:
      return state
    else:
      current_exogenous_regressors = embedded_exogenous_regressors[
          :, step_number, :]
      exogenous_updated_state = self._exogenous_input_step(
          current_times=current_times,
          current_exogenous_regressors=current_exogenous_regressors,
          state=state)
      if self._exogenous_update_condition is not None:
        current_raw_exogenous_features = {
            key: value[:, step_number] for key, value in raw_features.items()
            if key not in [PredictionFeatures.STATE_TUPLE,
                           TrainEvalFeatures.TIMES,
                           TrainEvalFeatures.VALUES]}
        conditionally_updated_state_flat = []
        for updated_state_element, original_state_element in zip(
            nest.flatten(exogenous_updated_state),
            nest.flatten(state)):
          conditionally_updated_state_flat.append(
              array_ops.where(
                  self._exogenous_update_condition(
                      times=current_times,
                      features=current_raw_exogenous_features),
                  updated_state_element,
                  original_state_element))
        return nest.pack_sequence_as(state, conditionally_updated_state_flat)
      else:
        return exogenous_updated_state

  def per_step_batch_loss(self, features, mode, state):
    """Computes predictions, losses, and intermediate model states.

    Args:
      features: A dictionary with times, values, and (optionally) exogenous
          regressors. See `define_loss`.
      mode: The tf.estimator.ModeKeys mode to use (TRAIN, EVAL, INFER).
      state: Model-dependent state, each with size [batch size x ...]. The
          number and type will typically be fixed by the model (for example a
          mean and variance).
    Returns:
      A tuple of (loss, filtered_states, predictions)
        loss: Average loss values across the batch.
        filtered_states: For each Tensor in `state` with shape [batch size x
            ...], `filtered_states` has a Tensor with shape [batch size x window
            size x ...] with filtered state for each part of the batch and
            window.
        predictions: A dictionary with model-dependent one-step-ahead (or
            at-least-one-step-ahead with missing values) predictions, with keys
            indicating the type of prediction and values having shape [batch
            size x window size x ...]. For example state space models provide
            "mean", "covariance", and "log_likelihood".

    """
    self._check_graph_initialized()
    times = math_ops.cast(features[TrainEvalFeatures.TIMES], dtype=dtypes.int64)
    values = math_ops.cast(features[TrainEvalFeatures.VALUES], dtype=self.dtype)
    if self._normalize_features:
      values = self._scale_data(values)
    exogenous_regressors = self._process_exogenous_features(
        times=times,
        features={key: value for key, value in features.items()
                  if key not in [TrainEvalFeatures.TIMES,
                                 TrainEvalFeatures.VALUES]})
    def _batch_loss_filtering_step(step_number, current_times, state):
      """Make a prediction and update it based on data."""
      current_values = values[:, step_number, :]
      state = self._apply_exogenous_update(
          step_number=step_number, current_times=current_times, state=state,
          raw_features=features,
          embedded_exogenous_regressors=exogenous_regressors)
      predicted_state, predictions = self._prediction_step(
          current_times=current_times,
          state=state)
      filtered_state, outputs = self._filtering_step(
          current_times=current_times,
          current_values=current_values,
          state=predicted_state,
          predictions=predictions)
      return filtered_state, outputs
    state, outputs = self._state_update_loop(
        times=times, state=state, state_update_fn=_batch_loss_filtering_step,
        outputs=["loss"] + self._train_output_names)
    outputs["loss"].set_shape(times.get_shape())
    loss_sum = math_ops.reduce_sum(outputs["loss"])
    per_observation_loss = (loss_sum / math_ops.cast(
        math_ops.reduce_prod(array_ops.shape(times)), dtype=self.dtype))
    per_observation_loss += self._loss_additions(times, values, mode)
    # Since we have window-level additions to the loss, its per-step value is
    # misleading, so we avoid returning it.
    del outputs["loss"]
    if self._normalize_features:
      outputs = self._scale_back_predictions(outputs)
    return per_observation_loss, state, outputs

  def predict(self, features):
    """Calls self._prediction_step in a loop. See TimeSeriesModel.predict."""
    predict_times = ops.convert_to_tensor(features[PredictionFeatures.TIMES],
                                          dtypes.int64)
    start_state = features[PredictionFeatures.STATE_TUPLE]
    exogenous_regressors = self._process_exogenous_features(
        times=predict_times,
        features={
            key: value
            for key, value in features.items()
            if key not in
            [PredictionFeatures.TIMES, PredictionFeatures.STATE_TUPLE]
        })
    def _call_prediction_step(step_number, current_times, state):
      state = self._apply_exogenous_update(
          step_number=step_number, current_times=current_times, state=state,
          raw_features=features,
          embedded_exogenous_regressors=exogenous_regressors)
      state, outputs = self._prediction_step(
          current_times=current_times, state=state)
      return state, outputs
    _, predictions = self._state_update_loop(
        times=predict_times, state=start_state,
        state_update_fn=_call_prediction_step,
        outputs=self._predict_output_names)
    if self._normalize_features:
      predictions = self._scale_back_predictions(predictions)
    return predictions

  class _FakeTensorArray(object):
    """An interface for Python lists that is similar to TensorArray.

    Used for easy switching between static and dynamic looping.
    """

    def __init__(self):
      self.values = []

    def write(self, unused_position, value):
      del unused_position
      self.values.append(value)
      return self

  def _state_update_loop(self, times, state, state_update_fn, outputs):
    """Iterates over `times`, calling `state_update_fn` to collect outputs.

    Args:
      times: A [batch size x window size] Tensor of integers to iterate over.
      state: A list of model-specific state Tensors, each with shape [batch size
          x ...].
      state_update_fn: A callback taking the following arguments
            step_number; A scalar integer Tensor indicating the current position
              in the window.
            current_times; A [batch size] vector of Integers indicating times
              for each part of the batch.
            state; Current model state.
          It returns a tuple of (updated state, output_values), output_values
          being a dictionary of Tensors with keys corresponding to `outputs`.
      outputs: A list of strings indicating values which will be saved while
          iterating. Must match the keys of the dictionary returned by
          state_update_fn.
    Returns:
      A tuple of (state, output_dict)
      state: The final model state.
      output_dict: A dictionary of outputs corresponding to those specified in
        `outputs` and computed in state_update_fn.
    """
    times = ops.convert_to_tensor(times, dtype=dtypes.int64)
    window_static_shape = tensor_shape.dimension_value(times.shape[1])
    if self._static_unrolling_window_size_threshold is None:
      static_unroll = False
    else:
      # The user has specified a threshold for static loop unrolling.
      if window_static_shape is None:
        # We don't have static shape information for the window size, so dynamic
        # looping is our only option.
        static_unroll = False
      elif window_static_shape <= self._static_unrolling_window_size_threshold:
        # The threshold is satisfied; unroll statically
        static_unroll = True
      else:
        # A threshold was set but not satisfied
        static_unroll = False

    self._window_initializer(times, state)

    def _run_condition(step_number, *unused):
      del unused  # not part of while loop run condition
      return math_ops.less(step_number, window_size)

    def _state_update_step(
        step_number, state, state_accumulators, output_accumulators,
        reuse=False):
      """Impute, then take one state_update_fn step, accumulating outputs."""
      with variable_scope.variable_scope("state_update_step", reuse=reuse):
        current_times = times[:, step_number]
        state = self._imputation_step(current_times=current_times, state=state)
        output_accumulators_dict = {
            accumulator_key: accumulator
            for accumulator_key, accumulator
            in zip(outputs, output_accumulators)}
        step_state, output_values = state_update_fn(
            step_number=step_number,
            current_times=current_times,
            state=state)
        assert set(output_values.keys()) == set(outputs)
        new_output_accumulators = []
        for output_key in outputs:
          accumulator = output_accumulators_dict[output_key]
          output_value = output_values[output_key]
          new_output_accumulators.append(
              accumulator.write(step_number, output_value))
        flat_step_state = nest.flatten(step_state)
        assert len(state_accumulators) == len(flat_step_state)
        new_state_accumulators = []
        new_state_flat = []
        for step_state_value, state_accumulator, original_state in zip(
            flat_step_state, state_accumulators, nest.flatten(state)):
          # Make sure the static shape information is complete so while_loop
          # does not complain about shape information changing.
          step_state_value.set_shape(original_state.get_shape())
          new_state_flat.append(step_state_value)
          new_state_accumulators.append(state_accumulator.write(
              step_number, step_state_value))
        step_state = nest.pack_sequence_as(state, new_state_flat)
        return (step_number + 1, step_state,
                new_state_accumulators, new_output_accumulators)

    window_size = array_ops.shape(times)[1]

    def _window_size_tensor_array(dtype):
      if static_unroll:
        return self._FakeTensorArray()
      else:
        return tensor_array_ops.TensorArray(
            dtype=dtype, size=window_size, dynamic_size=False)

    initial_loop_arguments = [
        array_ops.zeros([], dtypes.int32),
        state,
        [_window_size_tensor_array(element.dtype)
         for element in nest.flatten(state)],
        [_window_size_tensor_array(self.dtype) for _ in outputs]]
    if static_unroll:
      arguments = initial_loop_arguments
      for step_number in range(tensor_shape.dimension_value(times.shape[1])):
        arguments = _state_update_step(
            array_ops.constant(step_number, dtypes.int32), *arguments[1:],
            reuse=(step_number > 0))  # Variable sharing between steps
    else:
      arguments = control_flow_ops.while_loop(
          cond=_run_condition,
          body=_state_update_step,
          loop_vars=initial_loop_arguments)
    (_, _, state_loop_result, outputs_loop_result) = arguments

    def _stack_and_transpose(tensor_array):
      """Stack and re-order the dimensions of a TensorArray."""
      if static_unroll:
        return array_ops.stack(tensor_array.values, axis=1)
      else:
        # TensorArrays from while_loop stack with window size as the first
        # dimension, so this function swaps it and the batch dimension to
        # maintain the [batch x window size x ...] convention used elsewhere.
        stacked = tensor_array.stack()
        return array_ops.transpose(
            stacked,
            perm=array_ops.concat([[1, 0], math_ops.range(
                2, array_ops.rank(stacked))], 0))

    outputs_dict = {output_key: _stack_and_transpose(output)
                    for output_key, output
                    in zip(outputs, outputs_loop_result)}
    full_state = nest.pack_sequence_as(
        state,
        [_stack_and_transpose(state_element)
         for state_element in state_loop_result])
    return full_state, outputs_dict
