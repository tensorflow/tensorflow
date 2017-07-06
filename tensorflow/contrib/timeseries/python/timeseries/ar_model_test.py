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
"""Tests for ar_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.timeseries.python.timeseries import input_pipeline
from tensorflow.contrib.timeseries.python.timeseries import test_utils
from tensorflow.contrib.timeseries.python.timeseries.ar_model import AnomalyMixtureARModel
from tensorflow.contrib.timeseries.python.timeseries.ar_model import ARModel
from tensorflow.contrib.timeseries.python.timeseries.estimators import ARRegressor
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import PredictionFeatures
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import TrainEvalFeatures

from tensorflow.python.client import session
from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator as coordinator_lib
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import training


class ARModelTest(test.TestCase):

  def create_data(self,
                  noise_stddev,
                  anomaly_prob,
                  multiple_periods=False,
                  anomaly_stddev_scale=20):
    self.period = 25
    num_samples = 200
    time = 1 + 3 * np.arange(num_samples).astype(np.int64)
    time_offset = (2 * np.pi * (time % self.period).astype(np.float) /
                   self.period).reshape([-1, 1])
    if multiple_periods:
      period2 = 55
      self.period = [self.period, period2]
      time_offset2 = ((time % period2).astype(np.float) / period2).reshape(
          [-1, 1])
      data1 = np.sin(time_offset / 2.0) ** 2 * (1 + time_offset2)
    else:
      data1 = np.sin(2 * time_offset) + np.cos(3 * time_offset)
    data1 += noise_stddev / 4. * np.random.randn(num_samples, 1)
    data2 = (np.sin(3 * time_offset) + np.cos(5 * time_offset) +
             noise_stddev / 3. * np.random.randn(num_samples, 1))
    # Add some anomalies to data1
    if anomaly_prob > 0.:
      num_anomalies = int(anomaly_prob * num_samples)
      anomaly_values = (anomaly_stddev_scale * noise_stddev / 4 *
                        np.random.randn(num_anomalies))
      indices = np.random.randint(0, num_samples, num_anomalies)
      for index, val in zip(indices, anomaly_values):
        data1[index] += val

    data = np.concatenate((4 * data1, 3 * data2), axis=1)
    split = int(num_samples * 0.8)
    train_data = {TrainEvalFeatures.TIMES: time[0:split],
                  TrainEvalFeatures.VALUES: data[0:split]}
    test_data = {TrainEvalFeatures.TIMES: time[split:],
                 TrainEvalFeatures.VALUES: data[split:]}
    return (train_data, test_data)

  # Note that most models will require many more steps to fully converge. We
  # have used a small number of steps here to keep the running time small.
  def train_helper(self, input_window_size, loss,
                   max_loss=None, train_steps=200,
                   anomaly_prob=0.01,
                   anomaly_distribution=None,
                   multiple_periods=False):
    np.random.seed(3)
    data_noise_stddev = 0.2
    if max_loss is None:
      if loss == ARModel.NORMAL_LIKELIHOOD_LOSS:
        max_loss = 1.0
      else:
        max_loss = 0.05 / (data_noise_stddev ** 2)
    train_data, test_data = self.create_data(
        noise_stddev=data_noise_stddev,
        anomaly_prob=anomaly_prob,
        multiple_periods=multiple_periods)
    output_window_size = 10
    window_size = input_window_size + output_window_size

    class _RunConfig(estimator_lib.RunConfig):

      @property
      def tf_random_seed(self):
        return 3

    estimator = ARRegressor(
        periodicities=self.period,
        anomaly_prior_probability=0.01 if anomaly_distribution else None,
        anomaly_distribution=anomaly_distribution,
        num_features=2,
        output_window_size=output_window_size,
        num_time_buckets=20,
        input_window_size=input_window_size,
        hidden_layer_sizes=[16],
        loss=loss,
        config=_RunConfig())
    train_input_fn = input_pipeline.RandomWindowInputFn(
        time_series_reader=input_pipeline.NumpyReader(train_data),
        window_size=window_size,
        batch_size=64,
        num_threads=1,
        shuffle_seed=2)
    test_input_fn = test_utils.AllWindowInputFn(
        time_series_reader=input_pipeline.NumpyReader(test_data),
        window_size=window_size)

    # Test training
    estimator.train(
        input_fn=train_input_fn,
        steps=train_steps)
    test_evaluation = estimator.evaluate(input_fn=test_input_fn, steps=1)
    test_loss = test_evaluation["loss"]
    logging.info("Final test loss: %f", test_loss)
    self.assertLess(test_loss, max_loss)
    if loss == ARModel.SQUARED_LOSS:
      # Test that the evaluation loss is reported without input scaling.
      self.assertAllClose(
          test_loss,
          np.mean((test_evaluation["mean"] - test_evaluation["observed"]) ** 2))

    # Test predict
    train_data_times = train_data[TrainEvalFeatures.TIMES]
    train_data_values = train_data[TrainEvalFeatures.VALUES]
    test_data_times = test_data[TrainEvalFeatures.TIMES]
    test_data_values = test_data[TrainEvalFeatures.VALUES]
    predict_times = np.expand_dims(np.concatenate(
        [train_data_times[input_window_size:], test_data_times]), 0)
    predict_true_values = np.expand_dims(np.concatenate(
        [train_data_values[input_window_size:], test_data_values]), 0)
    state_times = np.expand_dims(train_data_times[:input_window_size], 0)
    state_values = np.expand_dims(
        train_data_values[:input_window_size, :], 0)

    def prediction_input_fn():
      return ({
          PredictionFeatures.TIMES: training.limit_epochs(
              predict_times, num_epochs=1),
          PredictionFeatures.STATE_TUPLE: (state_times, state_values)
      }, {})
    (predictions,) = tuple(estimator.predict(input_fn=prediction_input_fn))
    predicted_mean = predictions["mean"][:, 0]
    true_values = predict_true_values[0, :, 0]

    if loss == ARModel.NORMAL_LIKELIHOOD_LOSS:
      variances = predictions["covariance"][:, 0]
      standard_deviations = np.sqrt(variances)
      # Note that we may get tighter bounds with more training steps.
      errors = np.abs(predicted_mean - true_values) > 4 * standard_deviations
      fraction_errors = np.mean(errors)
      logging.info("Fraction errors: %f", fraction_errors)

  def test_time_regression_squared(self):
    self.train_helper(input_window_size=0,
                      train_steps=350,
                      loss=ARModel.SQUARED_LOSS)

  def test_autoregression_squared(self):
    self.train_helper(input_window_size=15,
                      loss=ARModel.SQUARED_LOSS)

  def test_autoregression_short_input_window(self):
    self.train_helper(input_window_size=8,
                      loss=ARModel.SQUARED_LOSS)

  def test_autoregression_normal(self):
    self.train_helper(input_window_size=10,
                      loss=ARModel.NORMAL_LIKELIHOOD_LOSS,
                      train_steps=300,
                      max_loss=1.5,
                      anomaly_distribution=None)

  def test_autoregression_normal_multiple_periods(self):
    self.train_helper(input_window_size=10,
                      loss=ARModel.NORMAL_LIKELIHOOD_LOSS,
                      max_loss=2.0,
                      multiple_periods=True,
                      anomaly_distribution=None)

  def test_autoregression_normal_anomalies_normal(self):
    self.train_helper(
        input_window_size=10,
        loss=ARModel.NORMAL_LIKELIHOOD_LOSS,
        anomaly_distribution=AnomalyMixtureARModel.GAUSSIAN_ANOMALY)

  def test_autoregression_normal_anomalies_cauchy(self):
    self.train_helper(
        input_window_size=10,
        max_loss=1.5,
        loss=ARModel.NORMAL_LIKELIHOOD_LOSS,
        anomaly_distribution=AnomalyMixtureARModel.CAUCHY_ANOMALY)

  def test_wrong_window_size(self):
    estimator = ARRegressor(
        periodicities=10, num_features=1,
        input_window_size=10, output_window_size=6)
    def _bad_window_size_input_fn():
      return ({TrainEvalFeatures.TIMES: [[1]],
               TrainEvalFeatures.VALUES: [[[1.]]]},
              None)
    def _good_data():
      return ({TrainEvalFeatures.TIMES: np.arange(16)[None, :],
               TrainEvalFeatures.VALUES: array_ops.reshape(
                   np.arange(16), [1, 16, 1])},
              None)
    with self.assertRaisesRegexp(ValueError, "set window_size=16"):
      estimator.train(input_fn=_bad_window_size_input_fn, steps=1)
    # Get a checkpoint for evaluation
    estimator.train(input_fn=_good_data, steps=1)
    with self.assertRaisesRegexp(ValueError, "requires a window of at least"):
      estimator.evaluate(input_fn=_bad_window_size_input_fn, steps=1)

  def test_predictions_direct(self):
    g = ops.Graph()
    with g.as_default():
      model = ARModel(periodicities=2,
                      num_features=1,
                      num_time_buckets=10,
                      input_window_size=2,
                      output_window_size=2,
                      hidden_layer_sizes=[40, 10])
      with session.Session():
        predicted_values = model.predict({
            PredictionFeatures.TIMES: [[4, 6, 10]],
            PredictionFeatures.STATE_TUPLE: ([[1, 2]], [[[1.], [2.]]])
        })
        variables.global_variables_initializer().run()
        self.assertAllEqual(predicted_values["mean"].eval().shape,
                            [1, 3, 1])

  def test_long_eval(self):
    g = ops.Graph()
    with g.as_default():
      model = ARModel(periodicities=2,
                      num_features=1,
                      num_time_buckets=10,
                      input_window_size=2,
                      output_window_size=1)
      raw_features = {
          TrainEvalFeatures.TIMES: [[1, 3, 5, 7, 11]],
          TrainEvalFeatures.VALUES: [[[1.], [2.], [3.], [4.], [5.]]]}
      chunked_features, _ = test_utils.AllWindowInputFn(
          time_series_reader=input_pipeline.NumpyReader(raw_features),
          window_size=3)()
      model.initialize_graph()
      with variable_scope.variable_scope("armodel") as scope:
        raw_evaluation = model.define_loss(
            raw_features, mode=estimator_lib.ModeKeys.EVAL)
      with variable_scope.variable_scope(scope, reuse=True):
        chunked_evaluation = model.define_loss(
            chunked_features, mode=estimator_lib.ModeKeys.EVAL)
      with session.Session() as sess:
        coordinator = coordinator_lib.Coordinator()
        queue_runner_impl.start_queue_runners(sess, coord=coordinator)
        variables.global_variables_initializer().run()
        raw_evaluation_evaled, chunked_evaluation_evaled = sess.run(
            [raw_evaluation, chunked_evaluation])
        self.assertAllEqual(chunked_evaluation_evaled.loss,
                            raw_evaluation_evaled.loss)
        last_chunk_evaluation_state = [
            state[-1, None] for state in
            chunked_evaluation_evaled.end_state]
        for last_chunk_state_member, raw_state_member in zip(
            last_chunk_evaluation_state, raw_evaluation_evaled.end_state):
          self.assertAllEqual(last_chunk_state_member, raw_state_member)
        self.assertAllEqual([[5, 7, 11]],
                            raw_evaluation_evaled.prediction_times)
        for feature_name in raw_evaluation.predictions:
          self.assertAllEqual(
              [1, 3, 1],  # batch, window, num_features. The window size has 2
                          # cut off for the first input_window.
              raw_evaluation_evaled.predictions[feature_name].shape)
          self.assertAllEqual(
              np.reshape(chunked_evaluation_evaled.predictions[feature_name],
                         [-1]),
              np.reshape(raw_evaluation_evaled.predictions[feature_name],
                         [-1]))
        coordinator.request_stop()
        coordinator.join()

  def test_long_eval_discard_indivisible(self):
    g = ops.Graph()
    with g.as_default():
      model = ARModel(periodicities=2,
                      num_features=1,
                      num_time_buckets=10,
                      input_window_size=2,
                      output_window_size=2)
      raw_features = {
          TrainEvalFeatures.TIMES: [[1, 3, 5, 7, 11]],
          TrainEvalFeatures.VALUES: [[[1.], [2.], [3.], [4.], [5.]]]}
      model.initialize_graph()
      raw_evaluation = model.define_loss(
          raw_features, mode=estimator_lib.ModeKeys.EVAL)
      with session.Session() as sess:
        coordinator = coordinator_lib.Coordinator()
        queue_runner_impl.start_queue_runners(sess, coord=coordinator)
        variables.global_variables_initializer().run()
        raw_evaluation_evaled = sess.run(raw_evaluation)
        self.assertAllEqual([[7, 11]],
                            raw_evaluation_evaled.prediction_times)
        for feature_name in raw_evaluation.predictions:
          self.assertAllEqual(
              [1, 2, 1],  # batch, window, num_features. The window has two cut
                          # off for the first input window and one discarded so
                          # that the remainder is divisible into output windows.
              raw_evaluation_evaled.predictions[feature_name].shape)
        coordinator.request_stop()
        coordinator.join()


if __name__ == "__main__":
  test.main()
