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

from tensorflow.contrib.timeseries.python.timeseries import ar_model
from tensorflow.contrib.timeseries.python.timeseries import feature_keys
from tensorflow.contrib.timeseries.python.timeseries import head as ts_head_lib
from tensorflow.contrib.timeseries.python.timeseries import math_utils
from tensorflow.contrib.timeseries.python.timeseries import state_management
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import state_space_model
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import structural_ensemble
from tensorflow.contrib.timeseries.python.timeseries.state_space_models.filtering_postprocessor import StateInterpolatingAnomalyDetector

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.estimator.export import export_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.training import training as train


class TimeSeriesRegressor(estimator_lib.Estimator):
  """An Estimator to fit and evaluate a time series model."""

  def __init__(self, model, state_manager=None, optimizer=None, model_dir=None,
               config=None):
    """Initialize the Estimator.

    Args:
      model: The time series model to wrap (inheriting from TimeSeriesModel).
      state_manager: The state manager to use, or (by default)
          PassthroughStateManager if none is needed.
      optimizer: The optimization algorithm to use when training, inheriting
          from tf.train.Optimizer. Defaults to Adam with step size 0.02.
      model_dir: See `Estimator`.
      config: See `Estimator`.
    """
    input_statistics_generator = math_utils.InputStatisticsFromMiniBatch(
        dtype=model.dtype, num_features=model.num_features)
    if state_manager is None:
      state_manager = state_management.PassthroughStateManager()
    if optimizer is None:
      optimizer = train.AdamOptimizer(0.02)
    self._model = model
    ts_regression_head = ts_head_lib.time_series_regression_head(
        model, state_manager, optimizer,
        input_statistics_generator=input_statistics_generator)
    model_fn = ts_regression_head.create_estimator_spec
    super(TimeSeriesRegressor, self).__init__(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config)

  # TODO(allenl): A parsing input receiver function, which takes a serialized
  # tf.Example containing all features (times, values, any exogenous features)
  # and serialized model state (possibly also as a tf.Example).
  def build_raw_serving_input_receiver_fn(self,
                                          exogenous_features=None,
                                          default_batch_size=None,
                                          default_series_length=None):
    """Build an input_receiver_fn for export_savedmodel which accepts arrays.

    Args:
      exogenous_features: A dictionary mapping feature keys to exogenous
        features (either Numpy arrays or Tensors). Used to determine the shapes
        of placeholders for these features.
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
    if exogenous_features is None:
      exogenous_features = {}

    def _serving_input_receiver_fn():
      """A receiver function to be passed to export_savedmodel."""
      placeholders = {}
      placeholders[feature_keys.TrainEvalFeatures.TIMES] = (
          array_ops.placeholder(
              name=feature_keys.TrainEvalFeatures.TIMES,
              dtype=dtypes.int64,
              shape=[default_batch_size, default_series_length]))
      # Values are only necessary when filtering. For prediction the default
      # value will be ignored.
      placeholders[feature_keys.TrainEvalFeatures.VALUES] = (
          array_ops.placeholder_with_default(
              name=feature_keys.TrainEvalFeatures.VALUES,
              input=array_ops.zeros(
                  shape=[
                      default_batch_size
                      if default_batch_size else 0, default_series_length
                      if default_series_length else 0, self._model.num_features
                  ],
                  dtype=self._model.dtype),
              shape=(default_batch_size, default_series_length,
                     self._model.num_features)))
      for feature_key, feature_value in exogenous_features.items():
        value_tensor = ops.convert_to_tensor(feature_value)
        value_tensor.get_shape().with_rank_at_least(2)
        feature_shape = value_tensor.get_shape().as_list()
        feature_shape[0] = default_batch_size
        feature_shape[1] = default_series_length
        placeholders[feature_key] = array_ops.placeholder(
            dtype=value_tensor.dtype, name=feature_key, shape=feature_shape)
      # Models may not know the shape of their state without creating some
      # variables/ops. Avoid polluting the default graph by making a new one. We
      # use only static metadata from the returned Tensors.
      with ops.Graph().as_default():
        self._model.initialize_graph()
        model_start_state = self._model.get_start_state()
      for prefixed_state_name, state_tensor in ts_head_lib.state_to_dictionary(
          model_start_state).items():
        state_shape_with_batch = tensor_shape.TensorShape(
            (default_batch_size,)).concatenate(state_tensor.get_shape())
        placeholders[prefixed_state_name] = array_ops.placeholder(
            name=prefixed_state_name,
            shape=state_shape_with_batch,
            dtype=state_tensor.dtype)
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

  def __init__(
      self, periodicities, input_window_size, output_window_size,
      num_features, num_time_buckets=10,
      loss=ar_model.ARModel.NORMAL_LIKELIHOOD_LOSS, hidden_layer_sizes=None,
      anomaly_prior_probability=None, anomaly_distribution=None,
      optimizer=None, model_dir=None, config=None):
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
          periodicities=periodicities, num_features=num_features,
          num_time_buckets=num_time_buckets,
          input_window_size=input_window_size,
          output_window_size=output_window_size, loss=loss,
          hidden_layer_sizes=hidden_layer_sizes)
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
          num_time_buckets=num_time_buckets,
          hidden_layer_sizes=hidden_layer_sizes,
          anomaly_prior_probability=anomaly_prior_probability,
          anomaly_distribution=anomaly_distribution)
    state_manager = state_management.FilteringOnlyStateManager()
    super(ARRegressor, self).__init__(
        model=model,
        state_manager=state_manager,
        optimizer=optimizer,
        model_dir=model_dir,
        config=config)


class StateSpaceRegressor(TimeSeriesRegressor):
  """An Estimator for general state space models."""

  def __init__(self, model, state_manager=None, optimizer=None, model_dir=None,
               config=None):
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
        config=config)


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
               config=None):
    """Initialize the Estimator.

    Args:
      periodicities: The expected periodicity of the data (for example 24 if
          feeding hourly data with a daily periodicity, or 60 * 24 if feeding
          minute-level data with daily periodicity). Either a scalar or a
          list. This parameter can be any real value, and does not control the
          size of the model. However, increasing this without increasing
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
          `cycle_num_latent_values` and `autoregressive_order`) and the number
          of steps before transient deviations revert to the mean defined by the
          period and level/trend components.
      autoregressive_order: Each contribution from this component is a linear
          combination of this many previous contributions. Also helps to
          determine the model size. Learning autoregressive coefficients
          typically requires more steps and a smaller step size than other
          components.
      exogenous_feature_columns: A list of tf.contrib.layers.FeatureColumn
          objects (for example tf.contrib.layers.embedding_column) corresponding
          to exogenous features which provide extra information to the model but
          are not part of the series to be predicted. Passed to
          tf.contrib.layers.input_from_feature_columns.
      exogenous_update_condition: A function taking two Tensor arguments,
          `times` (shape [batch size]) and `features` (a dictionary mapping
          exogenous feature keys to Tensors with shapes [batch size, ...]), and
          returning a boolean Tensor with shape [batch size] indicating whether
          state should be updated using exogenous features for each part of the
          batch. Where it is False, no exogenous update is performed. If None
          (default), exogenous updates are always performed. Useful for avoiding
          "leaky" frequent exogenous updates when sparse updates are
          desired. Called only during graph construction. See the "known
          anomaly" example for example usage.
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
        config=config)
