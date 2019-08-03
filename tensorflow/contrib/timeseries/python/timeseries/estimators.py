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
"""Estimators for time series models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.contrib.timeseries.python.timeseries import ar_model
from tensorflow.contrib.timeseries.python.timeseries import feature_keys
from tensorflow.contrib.timeseries.python.timeseries import head as ts_head_lib
from tensorflow.contrib.timeseries.python.timeseries import math_utils
from tensorflow.contrib.timeseries.python.timeseries import state_management
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import state_space_model
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import structural_ensemble
from tensorflow.contrib.timeseries.python.timeseries.state_space_models.filtering_postprocessor import StateInterpolatingAnomalyDetector

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.estimator.export import export_lib
from tensorflow.python.feature_column import feature_column_lib as feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.training import training as train
from tensorflow.python.util import nest


class TimeSeriesRegressor(estimator_lib.Estimator):
  """An Estimator to fit and evaluate a time series model."""

  def __init__(self,
               model,
               state_manager=None,
               optimizer=None,
               model_dir=None,
               config=None,
               head_type=ts_head_lib.TimeSeriesRegressionHead):
    """Initialize the Estimator.

    Args:
      model: The time series model to wrap (inheriting from TimeSeriesModel).
      state_manager: The state manager to use, or (by default)
        PassthroughStateManager if none is needed.
      optimizer: The optimization algorithm to use when training, inheriting
        from tf.train.Optimizer. Defaults to Adam with step size 0.02.
      model_dir: See `Estimator`.
      config: See `Estimator`.
      head_type: The kind of head to use for the model (inheriting from
        `TimeSeriesRegressionHead`).
    """
    input_statistics_generator = math_utils.InputStatisticsFromMiniBatch(
        dtype=model.dtype, num_features=model.num_features)
    if state_manager is None:
      if isinstance(model, ar_model.ARModel):
        state_manager = state_management.FilteringOnlyStateManager()
      else:
        state_manager = state_management.PassthroughStateManager()
    if optimizer is None:
      optimizer = train.AdamOptimizer(0.02)
    self._model = model
    ts_regression_head = head_type(
        model=model,
        state_manager=state_manager,
        optimizer=optimizer,
        input_statistics_generator=input_statistics_generator)
    model_fn = ts_regression_head.create_estimator_spec
    super(TimeSeriesRegressor, self).__init__(
        model_fn=model_fn, model_dir=model_dir, config=config)

  def _model_start_state_placeholders(self,
                                      batch_size_tensor,
                                      static_batch_size=None):
    """Creates placeholders with zeroed start state for the current model."""
    gathered_state = {}
    # Models may not know the shape of their state without creating some
    # variables/ops. Avoid polluting the default graph by making a new one. We
    # use only static metadata from the returned Tensors.
    with ops.Graph().as_default():
      self._model.initialize_graph()

      # Evaluate the initial state as same-dtype "zero" values. These zero
      # constants aren't used, but are necessary for feeding to
      # placeholder_with_default for the "cold start" case where state is not
      # fed to the model.
      def _zeros_like_constant(tensor):
        return tensor_util.constant_value(array_ops.zeros_like(tensor))

      start_state = nest.map_structure(_zeros_like_constant,
                                       self._model.get_start_state())
    for prefixed_state_name, state in ts_head_lib.state_to_dictionary(
        start_state).items():
      state_shape_with_batch = tensor_shape.TensorShape(
          (static_batch_size,)).concatenate(state.shape)
      default_state_broadcast = array_ops.tile(
          state[None, ...],
          multiples=array_ops.concat([
              batch_size_tensor[None],
              array_ops.ones(len(state.shape), dtype=dtypes.int32)
          ],
                                     axis=0))
      gathered_state[prefixed_state_name] = array_ops.placeholder_with_default(
          input=default_state_broadcast,
          name=prefixed_state_name,
          shape=state_shape_with_batch)
    return gathered_state

  def build_one_shot_parsing_serving_input_receiver_fn(self,
                                                       filtering_length,
                                                       prediction_length,
                                                       default_batch_size=None,
                                                       values_input_dtype=None,
                                                       truncate_values=False):
    """Build an input_receiver_fn for export_savedmodel accepting tf.Examples.

    Only compatible with `OneShotPredictionHead` (see `head`).

    Args:
      filtering_length: The number of time steps used as input to the model, for
        which values are provided. If more than `filtering_length` values are
        provided (via `truncate_values`), only the first `filtering_length`
        values are used.
      prediction_length: The number of time steps requested as predictions from
        the model. Times and all exogenous features must be provided for these
        steps.
      default_batch_size: If specified, must be a scalar integer. Sets the batch
        size in the static shape information of all feature Tensors, which means
        only this batch size will be accepted by the exported model. If None
        (default), static shape information for batch sizes is omitted.
      values_input_dtype: An optional dtype specification for values in the
        tf.Example protos (either float32 or int64, since these are the numeric
        types supported by tf.Example). After parsing, values are cast to the
        model's dtype (float32 or float64).
      truncate_values: If True, expects `filtering_length + prediction_length`
        values to be provided, but only uses the first `filtering_length`. If
        False (default), exactly `filtering_length` values must be provided.

    Returns:
      An input_receiver_fn which may be passed to the Estimator's
      export_savedmodel.

      Expects features contained in a vector of serialized tf.Examples with
      shape [batch size] (dtype `tf.string`), each tf.Example containing
      features with the following shapes:
        times: [filtering_length + prediction_length] integer
        values: [filtering_length, num features] floating point. If
          `truncate_values` is True, expects `filtering_length +
          prediction_length` values but only uses the first `filtering_length`.
        all exogenous features: [filtering_length + prediction_length, ...]
          (various dtypes)
    """
    if values_input_dtype is None:
      values_input_dtype = dtypes.float32
    if truncate_values:
      values_proto_length = filtering_length + prediction_length
    else:
      values_proto_length = filtering_length

    def _serving_input_receiver_fn():
      """A receiver function to be passed to export_savedmodel."""
      times_column = feature_column.numeric_column(
          key=feature_keys.TrainEvalFeatures.TIMES, dtype=dtypes.int64)
      values_column = feature_column.numeric_column(
          key=feature_keys.TrainEvalFeatures.VALUES,
          dtype=values_input_dtype,
          shape=(self._model.num_features,))
      parsed_features_no_sequence = (
          feature_column.make_parse_example_spec(
              list(self._model.exogenous_feature_columns) +
              [times_column, values_column]))
      parsed_features = {}
      for key, feature_spec in parsed_features_no_sequence.items():
        if isinstance(feature_spec, parsing_ops.FixedLenFeature):
          if key == feature_keys.TrainEvalFeatures.VALUES:
            parsed_features[key] = feature_spec._replace(
                shape=((values_proto_length,) + feature_spec.shape))
          else:
            parsed_features[key] = feature_spec._replace(
                shape=((filtering_length + prediction_length,) +
                       feature_spec.shape))
        elif feature_spec.dtype == dtypes.string:
          parsed_features[key] = parsing_ops.FixedLenFeature(
              shape=(filtering_length + prediction_length,),
              dtype=dtypes.string)
        else:  # VarLenFeature
          raise ValueError("VarLenFeatures not supported, got %s for key %s" %
                           (feature_spec, key))
      tfexamples = array_ops.placeholder(
          shape=[default_batch_size], dtype=dtypes.string, name="input")
      features = parsing_ops.parse_example(
          serialized=tfexamples, features=parsed_features)
      features[feature_keys.TrainEvalFeatures.TIMES] = array_ops.squeeze(
          features[feature_keys.TrainEvalFeatures.TIMES], axis=-1)
      features[feature_keys.TrainEvalFeatures.VALUES] = math_ops.cast(
          features[feature_keys.TrainEvalFeatures.VALUES],
          dtype=self._model.dtype)[:, :filtering_length]
      features.update(
          self._model_start_state_placeholders(
              batch_size_tensor=array_ops.shape(
                  features[feature_keys.TrainEvalFeatures.TIMES])[0],
              static_batch_size=default_batch_size))
      return export_lib.ServingInputReceiver(features, {"examples": tfexamples})

    return _serving_input_receiver_fn

  def build_raw_serving_input_receiver_fn(self,
                                          default_batch_size=None,
                                          default_series_length=None):
    """Build an input_receiver_fn for export_savedmodel which accepts arrays.

    Automatically creates placeholders for exogenous `FeatureColumn`s passed to
    the model.

    Args:
      default_batch_size: If specified, must be a scalar integer. Sets the batch
        size in the static shape information of all feature Tensors, which means
        only this batch size will be accepted by the exported model. If None
        (default), static shape information for batch sizes is omitted.
      default_series_length: If specified, must be a scalar integer. Sets the
        series length in the static shape information of all feature Tensors,
        which means only this series length will be accepted by the exported
        model. If None (default), static shape information for series length is
        omitted.

    Returns:
      An input_receiver_fn which may be passed to the Estimator's
      export_savedmodel.
    """

    def _serving_input_receiver_fn():
      """A receiver function to be passed to export_savedmodel."""
      placeholders = {}
      time_placeholder = array_ops.placeholder(
          name=feature_keys.TrainEvalFeatures.TIMES,
          dtype=dtypes.int64,
          shape=[default_batch_size, default_series_length])
      placeholders[feature_keys.TrainEvalFeatures.TIMES] = time_placeholder
      # Values are only necessary when filtering. For prediction the default
      # value will be ignored.
      placeholders[feature_keys.TrainEvalFeatures.VALUES] = (
          array_ops.placeholder_with_default(
              name=feature_keys.TrainEvalFeatures.VALUES,
              input=array_ops.zeros(
                  shape=[
                      default_batch_size if default_batch_size else 0,
                      default_series_length if default_series_length else 0,
                      self._model.num_features
                  ],
                  dtype=self._model.dtype),
              shape=(default_batch_size, default_series_length,
                     self._model.num_features)))
      if self._model.exogenous_feature_columns:
        with ops.Graph().as_default():
          # Default placeholders have only an unknown batch dimension. Make them
          # in a separate graph, then splice in the series length to the shapes
          # and re-create them in the outer graph.
          parsed_features = (
              feature_column.make_parse_example_spec(
                  self._model.exogenous_feature_columns))
          placeholder_features = parsing_ops.parse_example(
              serialized=array_ops.placeholder(
                  shape=[None], dtype=dtypes.string),
              features=parsed_features)
          exogenous_feature_shapes = {
              key: (value.get_shape(), value.dtype) for key, value
              in placeholder_features.items()}
        for feature_key, (batch_only_feature_shape,
                          value_dtype) in (exogenous_feature_shapes.items()):
          batch_only_feature_shape = (
              batch_only_feature_shape.with_rank_at_least(1).as_list())
          feature_shape = ([default_batch_size, default_series_length] +
                           batch_only_feature_shape[1:])
          placeholders[feature_key] = array_ops.placeholder(
              dtype=value_dtype, name=feature_key, shape=feature_shape)
      batch_size_tensor = array_ops.shape(time_placeholder)[0]
      placeholders.update(
          self._model_start_state_placeholders(
              batch_size_tensor, static_batch_size=default_batch_size))
      return export_lib.ServingInputReceiver(placeholders, placeholders)

    return _serving_input_receiver_fn


class ARRegressor(TimeSeriesRegressor):
  """An Estimator for an (optionally non-linear) autoregressive model.

  ARRegressor is a window-based model, inputting fixed windows of length
  `input_window_size` and outputting fixed windows of length
  `output_window_size`. These two parameters must add up to the window_size
  passed to the `Chunker` used to create an `input_fn` for training or
  evaluation. `RandomWindowInputFn` is suggested for both training and
  evaluation, although it may be seeded for deterministic evaluation.
  """

  def __init__(self,
               periodicities,
               input_window_size,
               output_window_size,
               num_features,
               exogenous_feature_columns=None,
               num_time_buckets=10,
               loss=ar_model.ARModel.NORMAL_LIKELIHOOD_LOSS,
               hidden_layer_sizes=None,
               anomaly_prior_probability=None,
               anomaly_distribution=None,
               optimizer=None,
               model_dir=None,
               config=None):
    """Initialize the Estimator.

    Args:
      periodicities: periodicities of the input data, in the same units as the
        time feature. Note this can be a single value or a list of values for
        multiple periodicities.
      input_window_size: Number of past time steps of data to look at when doing
        the regression.
      output_window_size: Number of future time steps to predict. Note that
        setting it to > 1 empirically seems to give a better fit.
      num_features: The dimensionality of the time series (one for univariate,
        more than one for multivariate).
      exogenous_feature_columns: A list of `tf.feature_column`s (for example
        `tf.feature_column.embedding_column`) corresponding to exogenous
        features which provide extra information to the model but are not part
        of the series to be predicted. Passed to
        `tf.compat.v1.feature_column.input_layer`.
      num_time_buckets: Number of buckets into which to divide (time %
        periodicity) for generating time based features.
      loss: Loss function to use for training. Currently supported values are
        SQUARED_LOSS and NORMAL_LIKELIHOOD_LOSS. Note that for
        NORMAL_LIKELIHOOD_LOSS, we train the covariance term as well. For
        SQUARED_LOSS, the evaluation loss is reported based on un-scaled
        observations and predictions, while the training loss is computed on
        normalized data.
      hidden_layer_sizes: list of sizes of hidden layers.
      anomaly_prior_probability: If specified, constructs a mixture model under
        which anomalies (modeled with `anomaly_distribution`) have this prior
        probability. See `AnomalyMixtureARModel`.
      anomaly_distribution: May not be specified unless
        anomaly_prior_probability is specified and is not None. Controls the
        distribution of anomalies under the mixture model. Currently either
        `ar_model.AnomalyMixtureARModel.GAUSSIAN_ANOMALY` or
        `ar_model.AnomalyMixtureARModel.CAUCHY_ANOMALY`. See
        `AnomalyMixtureARModel`. Defaults to `GAUSSIAN_ANOMALY`.
      optimizer: The optimization algorithm to use when training, inheriting
        from tf.train.Optimizer. Defaults to Adagrad with step size 0.1.
      model_dir: See `Estimator`.
      config: See `Estimator`.

    Raises:
      ValueError: For invalid combinations of arguments.
    """
    if optimizer is None:
      optimizer = train.AdagradOptimizer(0.1)
    if anomaly_prior_probability is None and anomaly_distribution is not None:
      raise ValueError("anomaly_prior_probability is required if "
                       "anomaly_distribution is specified.")
    if anomaly_prior_probability is None:
      if anomaly_distribution is None:
        anomaly_distribution = ar_model.AnomalyMixtureARModel.GAUSSIAN_ANOMALY
      model = ar_model.ARModel(
          periodicities=periodicities,
          num_features=num_features,
          prediction_model_factory=functools.partial(
              ar_model.FlatPredictionModel,
              hidden_layer_sizes=hidden_layer_sizes),
          exogenous_feature_columns=exogenous_feature_columns,
          num_time_buckets=num_time_buckets,
          input_window_size=input_window_size,
          output_window_size=output_window_size,
          loss=loss)
    else:
      if loss != ar_model.ARModel.NORMAL_LIKELIHOOD_LOSS:
        raise ValueError(
            "AnomalyMixtureARModel only supports "
            "ar_model.ARModel.NORMAL_LIKELIHOOD_LOSS for its loss argument.")
      model = ar_model.AnomalyMixtureARModel(
          periodicities=periodicities,
          input_window_size=input_window_size,
          output_window_size=output_window_size,
          num_features=num_features,
          prediction_model_factory=functools.partial(
              ar_model.FlatPredictionModel,
              hidden_layer_sizes=hidden_layer_sizes),
          exogenous_feature_columns=exogenous_feature_columns,
          num_time_buckets=num_time_buckets,
          anomaly_prior_probability=anomaly_prior_probability,
          anomaly_distribution=anomaly_distribution)
    state_manager = state_management.FilteringOnlyStateManager()
    super(ARRegressor, self).__init__(
        model=model,
        state_manager=state_manager,
        optimizer=optimizer,
        model_dir=model_dir,
        config=config)


# TODO(b/113684821): Add detailed documentation on what the input_fn should do.
# Add an example of making and returning a Dataset object. Determine if
# endogenous features can be passed in as FeatureColumns. Move ARModel's loss
# functions into a more general location.
class LSTMAutoRegressor(TimeSeriesRegressor):
  """An Estimator for an LSTM autoregressive model.

  LSTMAutoRegressor is a window-based model, inputting fixed windows of length
  `input_window_size` and outputting fixed windows of length
  `output_window_size`. These two parameters must add up to the window_size
  of data returned by the `input_fn`.

  Each periodicity in the `periodicities` arg is divided by the `num_timesteps`
  into timesteps that are represented as time features added to the model.

  A good heuristic for picking an appropriate periodicity for a given data set
  would be the length of cycles in the data. For example, energy usage in a
  home is typically cyclic each day. If the time feature in a home energy
  usage dataset is in the unit of hours, then 24 would be an appropriate
  periodicity. Similarly, a good heuristic for `num_timesteps` is how often the
  data is expected to change within the cycle. For the aforementioned home
  energy usage dataset and periodicity of 24, then 48 would be a reasonable
  value if usage is expected to change every half hour.

  Each feature's value for a given example with time t is the difference
  between t and the start of the timestep it falls under. If it doesn't fall
  under a feature's associated timestep, then that feature's value is zero.

  For example: if `periodicities` = (9, 12) and `num_timesteps` = 3, then 6
  features would be added to the model, 3 for periodicity 9 and 3 for
  periodicity 12.

  For an example data point where t = 17:
  - It's in the 3rd timestep for periodicity 9 (2nd period is 9-18 and 3rd
    timestep is 15-18)
  - It's in the 2nd timestep for periodicity 12 (2nd period is 12-24 and
    2nd timestep is between 16-20).

  Therefore the 6 added features for this row with t = 17 would be:

  # Feature name (periodicity#_timestep#), feature value
  P9_T1, 0 # not in first timestep
  P9_T2, 0 # not in second timestep
  P9_T3, 2 # 17 - 15 since 15 is the start of the 3rd timestep
  P12_T1, 0 # not in first timestep
  P12_T2, 1 # 17 - 16 since 16 is the start of the 2nd timestep
  P12_T3, 0 # not in third timestep

  Example Code:

  ```python
  extra_feature_columns = (
      feature_column.numeric_column("exogenous_variable"),
  )

  estimator = LSTMAutoRegressor(
      periodicities=10,
      input_window_size=10,
      output_window_size=5,
      model_dir="/path/to/model/dir",
      num_features=1,
      extra_feature_columns=extra_feature_columns,
      num_timesteps=50,
      num_units=10,
      optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(...))

  # Input builders
  def input_fn_train():
    return {
      "times": tf.range(15)[None, :],
      "values": tf.random.normal(shape=[1, 15, 1])
    }
  estimator.train(input_fn=input_fn_train, steps=100)

  def input_fn_eval():
    pass
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)

  def input_fn_predict():
    pass
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```
  """

  def __init__(self,
               periodicities,
               input_window_size,
               output_window_size,
               model_dir=None,
               num_features=1,
               extra_feature_columns=None,
               num_timesteps=10,
               loss=ar_model.ARModel.NORMAL_LIKELIHOOD_LOSS,
               num_units=128,
               optimizer="Adam",
               config=None):
    """Initialize the Estimator.

    Args:
      periodicities: periodicities of the input data, in the same units as the
        time feature (for example 24 if feeding hourly data with a daily
        periodicity, or 60 * 24 if feeding minute-level data with daily
        periodicity). Note this can be a single value or a list of values for
        multiple periodicities.
      input_window_size: Number of past time steps of data to look at when doing
        the regression.
      output_window_size: Number of future time steps to predict. Note that
        setting this value to > 1 empirically seems to give a better fit.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      num_features: The dimensionality of the time series (default value is one
        for univariate, more than one for multivariate).
      extra_feature_columns: A list of `tf.feature_column`s (for example
        `tf.feature_column.embedding_column`) corresponding to features which
        provide extra information to the model but are not part of the series to
        be predicted.
      num_timesteps: Number of buckets into which to divide (time %
        periodicity). This value multiplied by the number of periodicities is
        the number of time features added to the model.
      loss: Loss function to use for training. Currently supported values are
        SQUARED_LOSS and NORMAL_LIKELIHOOD_LOSS. Note that for
        NORMAL_LIKELIHOOD_LOSS, we train the covariance term as well. For
        SQUARED_LOSS, the evaluation loss is reported based on un-scaled
        observations and predictions, while the training loss is computed on
        normalized data.
      num_units: The size of the hidden state in the encoder and decoder LSTM
        cells.
      optimizer: string, `tf.compat.v1.train.Optimizer` object, or callable that
        defines the optimizer algorithm to use for training. Defaults to the
        Adam optimizer with a learning rate of 0.01.
      config: Optional `estimator.RunConfig` object to configure the runtime
        settings.
    """
    optimizer = optimizers.get_optimizer_instance(optimizer, learning_rate=0.01)
    model = ar_model.ARModel(
        periodicities=periodicities,
        input_window_size=input_window_size,
        output_window_size=output_window_size,
        num_features=num_features,
        exogenous_feature_columns=extra_feature_columns,
        num_time_buckets=num_timesteps,
        loss=loss,
        prediction_model_factory=functools.partial(
            ar_model.LSTMPredictionModel, num_units=num_units))
    state_manager = state_management.FilteringOnlyStateManager()
    super(LSTMAutoRegressor, self).__init__(
        model=model,
        state_manager=state_manager,
        optimizer=optimizer,
        model_dir=model_dir,
        config=config,
        head_type=ts_head_lib.OneShotPredictionHead)


class StateSpaceRegressor(TimeSeriesRegressor):
  """An Estimator for general state space models."""

  def __init__(self,
               model,
               state_manager=None,
               optimizer=None,
               model_dir=None,
               config=None,
               head_type=ts_head_lib.TimeSeriesRegressionHead):
    """See TimeSeriesRegressor. Uses the ChainingStateManager by default."""
    if not isinstance(model, state_space_model.StateSpaceModel):
      raise ValueError(
          "StateSpaceRegressor only supports state space models (children of "
          "StateSpaceModel) in its `model` argument, got {}.".format(model))
    if state_manager is None:
      state_manager = state_management.ChainingStateManager()
    super(StateSpaceRegressor, self).__init__(
        model=model,
        state_manager=state_manager,
        optimizer=optimizer,
        model_dir=model_dir,
        config=config,
        head_type=head_type)


class StructuralEnsembleRegressor(StateSpaceRegressor):
  """An Estimator for structural time series models.

  "Structural" refers to the fact that this model explicitly accounts for
  structure in the data, such as periodicity and trends.

  `StructuralEnsembleRegressor` is a state space model. It contains components
  for modeling level, local linear trends, periodicity, and mean-reverting
  transients via a moving average component. Multivariate series are fit with
  full covariance matrices for observation and latent state transition noise,
  each feature of the multivariate series having its own latent components.

  Note that unlike `ARRegressor`, `StructuralEnsembleRegressor` is sequential,
  and so accepts variable window sizes with the same model.

  For training, `RandomWindowInputFn` is recommended as an `input_fn`. Model
  state is managed through `ChainingStateManager`: since state space models are
  inherently sequential, we save state from previous iterations to get
  approximate/eventual consistency while achieving good performance through
  batched computation.

  For evaluation, either pass a significant chunk of the series in a single
  window (e.g. set `window_size` to the whole series with
  `WholeDatasetInputFn`), or use enough random evaluation iterations to cover
  several passes through the whole dataset. Either method will ensure that stale
  saved state has been flushed.
  """

  def __init__(self,
               periodicities,
               num_features,
               cycle_num_latent_values=11,
               moving_average_order=4,
               autoregressive_order=0,
               exogenous_feature_columns=None,
               exogenous_update_condition=None,
               dtype=dtypes.float64,
               anomaly_prior_probability=None,
               optimizer=None,
               model_dir=None,
               config=None,
               head_type=ts_head_lib.TimeSeriesRegressionHead):
    """Initialize the Estimator.

    Args:
      periodicities: The expected periodicity of the data (for example 24 if
        feeding hourly data with a daily periodicity, or 60 * 24 if feeding
        minute-level data with daily periodicity). Either a scalar or a list.
        This parameter can be any real value, and does not control the size of
        the model. However, increasing this without increasing
        `num_values_per_cycle` will lead to smoother periodic behavior, as the
        same number of distinct values will be cycled through over a longer
        period of time.
      num_features: The dimensionality of the time series (one for univariate,
        more than one for multivariate).
      cycle_num_latent_values: Along with `moving_average_order` and
        `num_features`, controls the latent state size of the model. Square
        matrices of size `num_features * (moving_average_order +
        cycle_num_latent_values + 3)` are created and multiplied, so larger
          values may be slow. The trade-off is with resolution: cycling between
            a smaller number of latent values means that only smoother functions
            can be modeled.
      moving_average_order: Controls model size (along with
        `cycle_num_latent_values` and `autoregressive_order`) and the number of
        steps before transient deviations revert to the mean defined by the
        period and level/trend components.
      autoregressive_order: Each contribution from this component is a linear
        combination of this many previous contributions. Also helps to determine
        the model size. Learning autoregressive coefficients typically requires
        more steps and a smaller step size than other components.
      exogenous_feature_columns: A list of `tf.feature_column`s (for example
        `tf.feature_column.embedding_column`) corresponding to exogenous
        features which provide extra information to the model but are not part
        of the series to be predicted. Passed to
        `tf.compat.v1.feature_column.input_layer`.
      exogenous_update_condition: A function taking two Tensor arguments,
        `times` (shape [batch size]) and `features` (a dictionary mapping
        exogenous feature keys to Tensors with shapes [batch size, ...]), and
        returning a boolean Tensor with shape [batch size] indicating whether
        state should be updated using exogenous features for each part of the
        batch. Where it is False, no exogenous update is performed. If None
        (default), exogenous updates are always performed. Useful for avoiding
        "leaky" frequent exogenous updates when sparse updates are desired.
        Called only during graph construction. See the "known anomaly" example
        for example usage.
      dtype: The floating point data type to compute with. float32 may be
        faster, but can be problematic for larger models and longer time series.
      anomaly_prior_probability: If not None, the model attempts to
        automatically detect and ignore anomalies during training. This
        parameter then controls the prior probability of an anomaly. Values
        closer to 0 mean that points will be discarded less frequently. The
        default value (None) means that anomalies are not discarded, which may
        be slightly faster.
      optimizer: The optimization algorithm to use when training, inheriting
        from tf.train.Optimizer. Defaults to Adam with step size 0.02.
      model_dir: See `Estimator`.
      config: See `Estimator`.
      head_type: The kind of head to use for the model (inheriting from
        `TimeSeriesRegressionHead`).
    """
    if anomaly_prior_probability is not None:
      filtering_postprocessor = StateInterpolatingAnomalyDetector(
          anomaly_prior_probability=anomaly_prior_probability)
    else:
      filtering_postprocessor = None
    state_space_model_configuration = (
        state_space_model.StateSpaceModelConfiguration(
            num_features=num_features,
            dtype=dtype,
            filtering_postprocessor=filtering_postprocessor,
            exogenous_feature_columns=exogenous_feature_columns,
            exogenous_update_condition=exogenous_update_condition))
    model = structural_ensemble.MultiResolutionStructuralEnsemble(
        cycle_num_latent_values=cycle_num_latent_values,
        moving_average_order=moving_average_order,
        autoregressive_order=autoregressive_order,
        periodicities=periodicities,
        configuration=state_space_model_configuration)
    super(StructuralEnsembleRegressor, self).__init__(
        model=model,
        optimizer=optimizer,
        model_dir=model_dir,
        config=config,
        head_type=head_type)
