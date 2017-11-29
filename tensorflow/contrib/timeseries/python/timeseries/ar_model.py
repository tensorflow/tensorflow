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
"""Auto-Regressive models for time series data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import distributions

from tensorflow.contrib.timeseries.python.timeseries import model
from tensorflow.contrib.timeseries.python.timeseries import model_utils
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import PredictionFeatures
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import TrainEvalFeatures

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope


class ARModel(model.TimeSeriesModel):
  """Auto-regressive model, both linear and non-linear.

  Features to the model include time and values of input_window_size timesteps,
  and times for output_window_size timesteps. These are passed through zero or
  more hidden layers, and then fed to a loss function (e.g. squared loss).

  Note that this class can also be used to regress against time only by setting
  the input_window_size to zero.
  """
  SQUARED_LOSS = "squared_loss"
  NORMAL_LIKELIHOOD_LOSS = "normal_likelihood_loss"

  def __init__(self,
               periodicities,
               input_window_size,
               output_window_size,
               num_features,
               num_time_buckets=10,
               loss=NORMAL_LIKELIHOOD_LOSS,
               hidden_layer_sizes=None):
    """Constructs an auto-regressive model.

    Args:
      periodicities: periodicities of the input data, in the same units as the
        time feature. Note this can be a single value or a list of values for
        multiple periodicities.
      input_window_size: Number of past time steps of data to look at when doing
        the regression.
      output_window_size: Number of future time steps to predict. Note that
        setting it to > 1 empiricaly seems to give a better fit.
      num_features: number of input features per time step.
      num_time_buckets: Number of buckets into which to divide (time %
        periodicity) for generating time based features.
      loss: Loss function to use for training. Currently supported values are
        SQUARED_LOSS and NORMAL_LIKELIHOOD_LOSS. Note that for
        NORMAL_LIKELIHOOD_LOSS, we train the covariance term as well. For
        SQUARED_LOSS, the evaluation loss is reported based on un-scaled
        observations and predictions, while the training loss is computed on
        normalized data (if input statistics are available).
      hidden_layer_sizes: list of sizes of hidden layers.
    """
    self.input_window_size = input_window_size
    self.output_window_size = output_window_size
    if hidden_layer_sizes is None:
      hidden_layer_sizes = []
    self.hidden_layer_sizes = hidden_layer_sizes
    self.window_size = self.input_window_size + self.output_window_size
    self.loss = loss
    self.stats_means = None
    self.stats_sigmas = None
    super(ARModel, self).__init__(
        num_features=num_features)
    assert num_time_buckets > 0
    self._buckets = int(num_time_buckets)
    if periodicities is None or not periodicities:
      periodicities = []
    elif (not isinstance(periodicities, list) and
          not isinstance(periodicities, tuple)):
      periodicities = [periodicities]
    self._periods = [int(p) for p in periodicities]
    for p in self._periods:
      assert p > 0
    assert len(self._periods) or self.input_window_size
    assert output_window_size > 0

  def scale_data(self, data):
    """Scale data according to stats."""
    if self._input_statistics is not None:
      return (data - self.stats_means) / self.stats_sigmas
    else:
      return data

  def scale_back_data(self, data):
    if self._input_statistics is not None:
      return (data * self.stats_sigmas) + self.stats_means
    else:
      return data

  def scale_back_variance(self, var):
    if self._input_statistics is not None:
      return var * self.stats_sigmas * self.stats_sigmas
    else:
      return var

  def initialize_graph(self, input_statistics=None):
    super(ARModel, self).initialize_graph(input_statistics=input_statistics)
    if self._input_statistics:
      self.stats_means, variances = (
          self._input_statistics.overall_feature_moments)
      self.stats_sigmas = math_ops.sqrt(variances)

  def get_start_state(self):
    # State which matches the format we'll return later. Typically this will not
    # be used by the model directly, but the shapes and dtypes should match so
    # that the serving input_receiver_fn gets placeholder shapes correct.
    return (array_ops.zeros([self.input_window_size], dtype=dtypes.int64),
            array_ops.zeros(
                [self.input_window_size, self.num_features], dtype=self.dtype))

  # TODO(allenl,agarwal): Support sampling for AR.
  def random_model_parameters(self, seed=None):
    pass

  def generate(self, number_of_series, series_length,
               model_parameters=None, seed=None):
    pass

  def _predicted_covariance_op(self, activations, num_values):
    activation, activation_size = activations[-1]
    if self.loss == ARModel.NORMAL_LIKELIHOOD_LOSS:
      log_sigma_square = model_utils.fully_connected(
          activation,
          activation_size,
          self.output_window_size * num_values,
          name="log_sigma_square",
          activation=None)
      predicted_covariance = gen_math_ops.exp(log_sigma_square)
      predicted_covariance = array_ops.reshape(
          predicted_covariance, [-1, self.output_window_size, num_values])
    else:
      shape = array_ops.stack([
          array_ops.shape(activation)[0],
          constant_op.constant(self.output_window_size),
          constant_op.constant(num_values)
      ])
      predicted_covariance = array_ops.ones(shape=shape, dtype=activation.dtype)
    return predicted_covariance

  def _predicted_mean_op(self, activations):
    activation, activation_size = activations[-1]
    predicted_mean = model_utils.fully_connected(
        activation,
        activation_size,
        self.output_window_size * self.num_features,
        name="predicted_mean",
        activation=None)
    return array_ops.reshape(predicted_mean,
                             [-1, self.output_window_size, self.num_features])

  def _create_hidden_stack(self, activation, activation_size):
    activations = []
    for layer_number, layer_size in enumerate(self.hidden_layer_sizes):
      # TODO(agarwal): Migrate to fully_connected in tf slim
      activation = model_utils.fully_connected(
          activation, activation_size, layer_size,
          name="layer_{}".format(layer_number))
      activation_size = layer_size
      activations.append((activation, activation_size))
    return activations

  def prediction_ops(self, times, values):
    """Compute model predictions given input data.

    Args:
      times: A [batch size, self.window_size] integer Tensor, the first
          self.input_window_size times in each part of the batch indicating
          input features, and the last self.output_window_size times indicating
          prediction times.
      values: A [batch size, self.input_window_size, self.num_features] Tensor
          with input features.
    Returns:
      Tuple (predicted_mean, predicted_covariance), where each element is a
      Tensor with shape [batch size, self.output_window_size,
      self.num_features].
    """
    times.get_shape().assert_is_compatible_with([None, self.window_size])
    activations = []
    if self.input_window_size:
      values.get_shape().assert_is_compatible_with(
          [None, self.input_window_size, self.num_features])
    # Create input features.
    if self._periods:
      _, time_features = self._compute_time_features(times)
      activation_size = self.window_size * self._buckets * len(self._periods)
      activation = array_ops.reshape(time_features, [-1, activation_size])
    else:
      activation_size = 0
      activation = None

    if self.input_window_size:
      inp = array_ops.slice(values, [0, 0, 0], [-1, self.input_window_size, -1])
      inp_size = self.input_window_size * self.num_features
      inp = array_ops.reshape(inp, [-1, inp_size])
      if activation is not None:
        activation = array_ops.concat([inp, activation], 1)
      else:
        activation = inp
      activation_size += inp_size
    assert activation_size
    activations.append((activation, activation_size))
    # Create hidden layers.
    activations += self._create_hidden_stack(activation, activation_size)
    # Create mean and convariance ops.
    predicted_mean = self._predicted_mean_op(activations)
    predicted_covariance = self._predicted_covariance_op(activations,
                                                         self.num_features)
    return {"activations": activations,
            "mean": predicted_mean,
            "covariance": predicted_covariance}

  def loss_op(self, targets, prediction_ops):
    """Create loss_op."""
    prediction = prediction_ops["mean"]
    if self.loss == ARModel.NORMAL_LIKELIHOOD_LOSS:
      covariance = prediction_ops["covariance"]
      sigma = math_ops.sqrt(gen_math_ops.maximum(covariance, 1e-5))
      normal = distributions.Normal(loc=targets, scale=sigma)
      loss_op = -math_ops.reduce_sum(normal.log_prob(prediction))
    else:
      assert self.loss == ARModel.SQUARED_LOSS, self.loss
      loss_op = math_ops.reduce_sum(math_ops.square(prediction - targets))
    loss_op /= math_ops.cast(
        math_ops.reduce_prod(array_ops.shape(targets)), loss_op.dtype)
    return loss_op

  # TODO(allenl, agarwal): Consider better ways of warm-starting predictions.
  def predict(self, features):
    """Computes predictions multiple steps into the future.

    Args:
      features: A dictionary with the following key/value pairs:
        PredictionFeatures.TIMES: A [batch size, predict window size]
          integer Tensor of times, after the window of data indicated by
          `STATE_TUPLE`, to make predictions for.
        PredictionFeatures.STATE_TUPLE: A tuple of (times, values), times with
          shape [batch size, self.input_window_size], values with shape [batch
          size, self.input_window_size, self.num_features] representing a
          segment of the time series before `TIMES`. This data is used
          to start of the autoregressive computation. This should have data for
          at least self.input_window_size timesteps.
    Returns:
      A dictionary with keys, "mean", "covariance". The
      values are Tensors of shape [batch_size, predict window size,
      num_features] and correspond to the values passed in `TIMES`.
    """
    predict_times = math_ops.cast(
        ops.convert_to_tensor(features[PredictionFeatures.TIMES]), dtypes.int32)
    batch_size = array_ops.shape(predict_times)[0]
    num_predict_values = array_ops.shape(predict_times)[1]
    prediction_iterations = ((num_predict_values + self.output_window_size - 1)
                             // self.output_window_size)
    # Pad predict_times so as to have exact multiple of self.output_window_size
    # values per example.
    padding_size = (prediction_iterations * self.output_window_size -
                    num_predict_values)
    padding = array_ops.zeros([batch_size, padding_size], predict_times.dtype)
    predict_times = control_flow_ops.cond(
        padding_size > 0, lambda: array_ops.concat([predict_times, padding], 1),
        lambda: predict_times)
    state = features[PredictionFeatures.STATE_TUPLE]
    (state_times, state_values) = state
    state_times = math_ops.cast(
        ops.convert_to_tensor(state_times), dtypes.int32)
    state_values = ops.convert_to_tensor(state_values, dtype=self.dtype)

    initial_input_times = predict_times[:, :self.output_window_size]
    if self.input_window_size > 0:
      initial_input_times = array_ops.concat(
          [state_times[:, -self.input_window_size:], initial_input_times], 1)
      values_size = array_ops.shape(state_values)[1]
      times_size = array_ops.shape(state_times)[1]
      with ops.control_dependencies([
          check_ops.assert_greater_equal(values_size, self.input_window_size),
          check_ops.assert_equal(values_size, times_size)
      ]):
        initial_input_values = state_values[:, -self.input_window_size:, :]
    else:
      initial_input_values = 0

    # Iterate over the predict_times, predicting self.output_window_size values
    # in each iteration.
    def _while_condition(iteration_number, *unused_args):
      return math_ops.less(iteration_number, prediction_iterations)

    def _while_body(iteration_number, input_times, input_values,
                    mean_ta, covariance_ta):
      """Predict self.output_window_size values."""
      prediction_ops = self.prediction_ops(input_times, input_values)
      predicted_mean = prediction_ops["mean"]
      predicted_covariance = prediction_ops["covariance"]
      offset = self.output_window_size * gen_math_ops.minimum(
          iteration_number + 1, prediction_iterations - 1)
      if self.input_window_size > 0:
        if self.output_window_size < self.input_window_size:
          new_input_values = array_ops.concat(
              [input_values[:, self.output_window_size:, :], predicted_mean], 1)
          new_input_times = array_ops.concat([
              input_times[:, self.output_window_size:],
              predict_times[:, offset:offset + self.output_window_size]
          ], 1)
        else:
          new_input_values = predicted_mean[:, -self.input_window_size:, :]
          new_input_times = predict_times[
              :,
              offset - self.input_window_size:offset + self.output_window_size]
      else:
        new_input_values = input_values
        new_input_times = predict_times[:,
                                        offset:offset + self.output_window_size]
      new_input_times.set_shape(initial_input_times.get_shape())
      new_mean_ta = mean_ta.write(iteration_number, predicted_mean)
      if isinstance(covariance_ta, tensor_array_ops.TensorArray):
        new_covariance_ta = covariance_ta.write(iteration_number,
                                                predicted_covariance)
      else:
        new_covariance_ta = covariance_ta
      return (iteration_number + 1,
              new_input_times,
              new_input_values,
              new_mean_ta,
              new_covariance_ta)

    # Note that control_flow_ops.while_loop doesn't seem happy with None. Hence
    # using 0 for cases where we don't want to predict covariance.
    covariance_ta_init = (tensor_array_ops.TensorArray(
        dtype=self.dtype, size=prediction_iterations)
                          if self.loss != ARModel.SQUARED_LOSS else 0.)
    mean_ta_init = tensor_array_ops.TensorArray(
        dtype=self.dtype, size=prediction_iterations)
    _, _, _, mean_ta, covariance_ta = control_flow_ops.while_loop(
        _while_condition, _while_body, [
            0, initial_input_times, initial_input_values, mean_ta_init,
            covariance_ta_init
        ])

    def _parse_ta(values_ta):
      """Helper function to parse the returned TensorArrays."""

      if not isinstance(values_ta, tensor_array_ops.TensorArray):
        return None
      predictions_length = prediction_iterations * self.output_window_size
      # Shape [prediction_iterations, batch_size, self.output_window_size,
      #        self.num_features]
      values_packed = values_ta.stack()
      # Transpose to move batch dimension outside.
      output_values = array_ops.reshape(
          array_ops.transpose(values_packed, [1, 0, 2, 3]),
          array_ops.stack([batch_size, predictions_length, -1]))
      # Clip to desired size
      return output_values[:, :num_predict_values, :]

    predicted_mean = _parse_ta(mean_ta)
    predicted_covariance = _parse_ta(covariance_ta)
    if predicted_covariance is None:
      predicted_covariance = array_ops.ones_like(predicted_mean)

    # Transform and scale the mean and covariance appropriately.
    predicted_mean = self.scale_back_data(predicted_mean)
    predicted_covariance = self.scale_back_variance(predicted_covariance)

    return {"mean": predicted_mean,
            "covariance": predicted_covariance}

  def _process_window(self, features, mode):
    """Compute model outputs on a single window of data."""
    # TODO(agarwal): Use exogenous features
    times = math_ops.cast(features[TrainEvalFeatures.TIMES], dtypes.int64)
    values = math_ops.cast(features[TrainEvalFeatures.VALUES], dtype=self.dtype)
    original_values = values

    # Extra shape checking for the window size (above that in
    # model_utils.make_model_fn).
    expected_times_shape = [None, self.window_size]
    if not times.get_shape().is_compatible_with(expected_times_shape):
      raise ValueError(
          ("ARModel with input_window_size={input_window_size} "
           "and output_window_size={output_window_size} expects "
           "feature '{times_feature}' to have shape (batch_size, "
           "{window_size}) (for any batch_size), but got shape {times_shape}. "
           "If you are using RandomWindowInputFn, set "
           "window_size={window_size} or adjust the input_window_size and "
           "output_window_size arguments to ARModel.").format(
               input_window_size=self.input_window_size,
               output_window_size=self.output_window_size,
               times_feature=TrainEvalFeatures.TIMES,
               window_size=self.window_size,
               times_shape=times.get_shape()))
    values = self.scale_data(values)
    if self.input_window_size > 0:
      input_values = values[:, :self.input_window_size, :]
    else:
      input_values = None
    prediction_ops = self.prediction_ops(times, input_values)
    prediction = prediction_ops["mean"]
    covariance = prediction_ops["covariance"]
    targets = array_ops.slice(values, [0, self.input_window_size, 0],
                              [-1, -1, -1])
    targets.get_shape().assert_is_compatible_with(prediction.get_shape())
    if (mode == estimator_lib.ModeKeys.EVAL
        and self.loss == ARModel.SQUARED_LOSS):
      # Report an evaluation loss which matches the expected
      #  (observed - predicted) ** 2.
      # Note that this affects only evaluation; the training loss is unaffected.
      loss = self.loss_op(
          self.scale_back_data(targets),
          {"mean": self.scale_back_data(prediction_ops["mean"])})
    else:
      loss = self.loss_op(targets, prediction_ops)

    # Scale back the prediction.
    prediction = self.scale_back_data(prediction)
    covariance = self.scale_back_variance(covariance)

    return model.ModelOutputs(
        loss=loss,
        end_state=(times[:, -self.input_window_size:],
                   values[:, -self.input_window_size:, :]),
        predictions={"mean": prediction, "covariance": covariance,
                     "observed": original_values[:, -self.output_window_size:]},
        prediction_times=times[:, -self.output_window_size:])

  def get_batch_loss(self, features, mode, state):
    """Computes predictions and a loss.

    Args:
      features: A dictionary (such as is produced by a chunker) with the
        following key/value pairs (shapes are given as required for training):
          TrainEvalFeatures.TIMES: A [batch size, self.window_size] integer
            Tensor with times for each observation. To train on longer
            sequences, the data should first be chunked.
          TrainEvalFeatures.VALUES: A [batch size, self.window_size,
            self.num_features] Tensor with values for each observation.
        When evaluating, `TIMES` and `VALUES` must have a window size of at
        least self.window_size, but it may be longer, in which case the last
        window_size - self.input_window_size times (or fewer if this is not
        divisible by self.output_window_size) will be evaluated on with
        non-overlapping output windows (and will have associated
        predictions). This is primarily to support qualitative
        evaluation/plotting, and is not a recommended way to compute evaluation
        losses (since there is no overlap in the output windows, which for
        window-based models is an undesirable bias).
      mode: The tf.estimator.ModeKeys mode to use (TRAIN or EVAL).
      state: Unused
    Returns:
      A model.ModelOutputs object.
    Raises:
      ValueError: If `mode` is not TRAIN or EVAL, or if static shape information
      is incorrect.
    """
    features = {feature_name: ops.convert_to_tensor(feature_value)
                for feature_name, feature_value in features.items()}
    if mode == estimator_lib.ModeKeys.TRAIN:
      # For training, we require the window size to be self.window_size as
      # iterating sequentially on larger windows could introduce a bias.
      return self._process_window(features, mode=mode)
    elif mode == estimator_lib.ModeKeys.EVAL:
      # For evaluation, we allow the user to pass in a larger window, in which
      # case we try to cover as much of the window as possible without
      # overlap. Quantitative evaluation is more efficient/correct with fixed
      # windows matching self.window_size (as with training), but this looping
      # allows easy plotting of "in-sample" predictions.
      times = features[TrainEvalFeatures.TIMES]
      times.get_shape().assert_has_rank(2)
      static_window_size = times.get_shape()[1].value
      if (static_window_size is not None
          and static_window_size < self.window_size):
        raise ValueError(
            ("ARModel requires a window of at least input_window_size + "
             "output_window_size to evaluate on (input_window_size={}, "
             "output_window_size={}, and got shape {} for feature '{}' (batch "
             "size, window size)).").format(
                 self.input_window_size, self.output_window_size,
                 times.get_shape(), TrainEvalFeatures.TIMES))
      num_iterations = ((array_ops.shape(times)[1] -  self.input_window_size)
                        // self.output_window_size)
      output_size = num_iterations * self.output_window_size
      # Rather than dealing with overlapping windows of output, discard a bit at
      # the beginning if output windows don't cover evenly.
      crop_length = output_size + self.input_window_size
      features = {feature_name: feature_value[:, -crop_length:]
                  for feature_name, feature_value in features.items()}
      # Note that, unlike the ARModel's predict() while_loop and the
      # SequentialTimeSeriesModel while_loop, each iteration here can run in
      # parallel, since we are not feeding predictions or state from previous
      # iterations.
      def _while_condition(iteration_number, loss_ta, mean_ta, covariance_ta):
        del loss_ta, mean_ta, covariance_ta  # unused
        return iteration_number < num_iterations

      def _while_body(iteration_number, loss_ta, mean_ta, covariance_ta):
        """Perform a processing step on a single window of data."""
        base_offset = iteration_number * self.output_window_size
        model_outputs = self._process_window(
            features={
                feature_name:
                feature_value[:, base_offset:base_offset + self.window_size]
                for feature_name, feature_value in features.items()},
            mode=mode)
        # This code needs to be updated if new predictions are added in
        # self._process_window
        assert len(model_outputs.predictions) == 3
        assert "mean" in model_outputs.predictions
        assert "covariance" in model_outputs.predictions
        assert "observed" in model_outputs.predictions
        return (iteration_number + 1,
                loss_ta.write(
                    iteration_number, model_outputs.loss),
                mean_ta.write(
                    iteration_number, model_outputs.predictions["mean"]),
                covariance_ta.write(
                    iteration_number, model_outputs.predictions["covariance"]))
      _, loss_ta, mean_ta, covariance_ta = control_flow_ops.while_loop(
          _while_condition, _while_body,
          [0,
           tensor_array_ops.TensorArray(dtype=self.dtype, size=num_iterations),
           tensor_array_ops.TensorArray(dtype=self.dtype, size=num_iterations),
           tensor_array_ops.TensorArray(dtype=self.dtype, size=num_iterations)])
      values = math_ops.cast(features[TrainEvalFeatures.VALUES],
                             dtype=self.dtype)
      batch_size = array_ops.shape(times)[0]
      prediction_shape = [batch_size, self.output_window_size * num_iterations,
                          self.num_features]
      previous_state_times, previous_state_values = state
      # Make sure returned state always has windows of self.input_window_size,
      # even if we were passed fewer than self.input_window_size points this
      # time.
      if self.input_window_size > 0:
        new_state_times = array_ops.concat(
            [previous_state_times,
             math_ops.cast(times, dtype=dtypes.int64)],
            axis=1)[:, -self.input_window_size:]
        new_state_times.set_shape((None, self.input_window_size))
        new_state_values = array_ops.concat(
            [previous_state_values,
             self.scale_data(values)], axis=1)[:, -self.input_window_size:, :]
        new_state_values.set_shape((None, self.input_window_size,
                                    self.num_features))
      else:
        # There is no state to keep, and the strided slices above do not handle
        # input_window_size=0.
        new_state_times = previous_state_times
        new_state_values = previous_state_values
      return model.ModelOutputs(
          loss=math_ops.reduce_mean(loss_ta.stack(), axis=0),
          end_state=(new_state_times, new_state_values),
          predictions={
              "mean": array_ops.reshape(
                  array_ops.transpose(mean_ta.stack(), [1, 0, 2, 3]),
                  prediction_shape),
              "covariance": array_ops.reshape(
                  array_ops.transpose(covariance_ta.stack(), [1, 0, 2, 3]),
                  prediction_shape),
              "observed": values[:, -output_size:]},
          prediction_times=times[:, -output_size:])
    else:
      raise ValueError(
          "Unknown mode '{}' passed to get_batch_loss.".format(mode))

  def _compute_time_features(self, time):
    """Compute some features on the time value."""
    batch_size = array_ops.shape(time)[0]
    num_periods = len(self._periods)
    # Reshape to 3D.
    periods = constant_op.constant(
        self._periods, shape=[1, 1, num_periods, 1], dtype=time.dtype)
    time = array_ops.reshape(time, [batch_size, -1, 1, 1])
    window_offset = time / self._periods
    # Cast to appropriate type and scale to [0, 1) range
    mod = (math_ops.cast(time % periods, self.dtype) * self._buckets /
           math_ops.cast(periods, self.dtype))
    # Bucketize based on some fixed width intervals. For a value t and interval
    # [a, b), we return (t - a) if a <= t < b, else 0.
    intervals = array_ops.reshape(
        math_ops.range(self._buckets, dtype=self.dtype),
        [1, 1, 1, self._buckets])
    mod = nn_ops.relu(mod - intervals)
    mod = array_ops.where(mod < 1.0, mod, array_ops.zeros_like(mod))
    return window_offset, mod


class AnomalyMixtureARModel(ARModel):
  """Model data as a mixture of normal and anomaly distributions.

  Note that this model works by changing the loss function to reduce the penalty
  when predicting an anomalous target. However the predictions are still based
  on anomalous input features, and this may affect the quality of fit. One
  possible solution is to downweight/filter anomalous inputs, but that requires
  more sequential processing instead of completely random windows.
  """

  GAUSSIAN_ANOMALY = "gaussian"
  CAUCHY_ANOMALY = "cauchy"

  def __init__(self,
               periodicities,
               anomaly_prior_probability,
               input_window_size,
               output_window_size,
               num_features,
               anomaly_distribution=GAUSSIAN_ANOMALY,
               num_time_buckets=10,
               hidden_layer_sizes=None):
    assert (anomaly_prior_probability < 1.0 and
            anomaly_prior_probability > 0.0)
    self._anomaly_prior_probability = anomaly_prior_probability
    assert anomaly_distribution in [
        AnomalyMixtureARModel.GAUSSIAN_ANOMALY,
        AnomalyMixtureARModel.CAUCHY_ANOMALY]
    self._anomaly_distribution = anomaly_distribution
    super(AnomalyMixtureARModel, self).__init__(
        periodicities=periodicities,
        num_features=num_features,
        num_time_buckets=num_time_buckets,
        input_window_size=input_window_size,
        output_window_size=output_window_size,
        loss=ARModel.NORMAL_LIKELIHOOD_LOSS,
        hidden_layer_sizes=hidden_layer_sizes)

  def _create_anomaly_ops(self, times, values, prediction_ops_dict):
    anomaly_log_param = variable_scope.get_variable(
        "anomaly_log_param",
        shape=[],
        dtype=self.dtype,
        initializer=init_ops.zeros_initializer())
    # Anomaly param is the variance for Gaussian and scale for Cauchy
    # distribution.
    prediction_ops_dict["anomaly_params"] = gen_math_ops.exp(anomaly_log_param)

  def prediction_ops(self, times, values):
    prediction_ops_dict = super(AnomalyMixtureARModel, self).prediction_ops(
        times, values)
    self._create_anomaly_ops(times, values, prediction_ops_dict)
    return prediction_ops_dict

  def _anomaly_log_prob(self, targets, prediction_ops):
    prediction = prediction_ops["mean"]
    if self._anomaly_distribution == AnomalyMixtureARModel.GAUSSIAN_ANOMALY:
      anomaly_variance = prediction_ops["anomaly_params"]
      anomaly_sigma = math_ops.sqrt(
          gen_math_ops.maximum(anomaly_variance, 1e-5))
      normal = distributions.Normal(loc=targets, scale=anomaly_sigma)
      log_prob = normal.log_prob(prediction)
    else:
      assert self._anomaly_distribution == AnomalyMixtureARModel.CAUCHY_ANOMALY
      anomaly_scale = prediction_ops["anomaly_params"]
      cauchy = distributions.StudentT(
          df=array_ops.ones([], dtype=anomaly_scale.dtype),
          loc=targets,
          scale=anomaly_scale)
      log_prob = cauchy.log_prob(prediction)
    return log_prob

  def loss_op(self, targets, prediction_ops):
    """Create loss_op."""
    prediction = prediction_ops["mean"]
    covariance = prediction_ops["covariance"]
    # Normal data log probability.
    sigma = math_ops.sqrt(gen_math_ops.maximum(covariance, 1e-5))
    normal1 = distributions.Normal(loc=targets, scale=sigma)
    log_prob1 = normal1.log_prob(prediction)
    log_prob1 += math_ops.log(1 - self._anomaly_prior_probability)
    # Anomaly log probability.
    log_prob2 = self._anomaly_log_prob(targets, prediction_ops)
    log_prob2 += math_ops.log(self._anomaly_prior_probability)
    # We need to compute log(exp(log_prob1) + exp(log_prob2). For numerical
    # stability, we rewrite the expression as below.
    p1 = gen_math_ops.minimum(log_prob1, log_prob2)
    p2 = gen_math_ops.maximum(log_prob1, log_prob2)
    mixed_log_prob = p2 + math_ops.log(1 + gen_math_ops.exp(p1 - p2))
    loss_op = -math_ops.reduce_sum(mixed_log_prob)
    loss_op /= math_ops.cast(
        math_ops.reduce_prod(array_ops.shape(targets)), self.dtype)
    return loss_op
