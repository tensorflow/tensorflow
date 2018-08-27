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
"""Example of using an exogenous feature to ignore a known anomaly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
from os import path

import numpy as np
import tensorflow as tf


try:
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use("TkAgg")  # Need Tk for interactive plots.
  from matplotlib import pyplot  # pylint: disable=g-import-not-at-top
  HAS_MATPLOTLIB = True
except ImportError:
  # Plotting requires matplotlib, but the unit test running this code may
  # execute in an environment without it (i.e. matplotlib is not a build
  # dependency). We'd still like to test the TensorFlow-dependent parts of this
  # example, namely train_and_predict.
  HAS_MATPLOTLIB = False

_MODULE_PATH = path.dirname(__file__)
_DATA_FILE = path.join(_MODULE_PATH, "data/changepoints.csv")


def state_space_esitmator(exogenous_feature_columns):
  """Constructs a StructuralEnsembleRegressor."""

  def _exogenous_update_condition(times, features):
    del times  # unused
    # Make exogenous updates sparse by setting an update condition. This in
    # effect allows missing exogenous features: if the condition evaluates to
    # False, no update is performed. Otherwise we sometimes end up with "leaky"
    # updates which add unnecessary uncertainty to the model even when there is
    # no changepoint.
    return tf.equal(tf.squeeze(features["is_changepoint"], axis=-1), "yes")

  return (
      tf.contrib.timeseries.StructuralEnsembleRegressor(
          periodicities=12,
          # Extract a smooth period by constraining the number of latent values
          # being cycled between.
          cycle_num_latent_values=3,
          num_features=1,
          exogenous_feature_columns=exogenous_feature_columns,
          exogenous_update_condition=_exogenous_update_condition),
      # Use truncated backpropagation with a window size of 64, batching
      # together 4 of these windows (random offsets) per training step. Training
      # with exogenous features often requires somewhat larger windows.
      4, 64)


def autoregressive_esitmator(exogenous_feature_columns):
  input_window_size = 8
  output_window_size = 2
  return (
      tf.contrib.timeseries.ARRegressor(
          periodicities=12,
          num_features=1,
          input_window_size=input_window_size,
          output_window_size=output_window_size,
          exogenous_feature_columns=exogenous_feature_columns),
      64, input_window_size + output_window_size)


def train_and_evaluate_exogenous(
    estimator_fn, csv_file_name=_DATA_FILE, train_steps=300):
  """Training, evaluating, and predicting on a series with changepoints."""
  # Indicate the format of our exogenous feature, in this case a string
  # representing a boolean value.
  string_feature = tf.feature_column.categorical_column_with_vocabulary_list(
      key="is_changepoint", vocabulary_list=["no", "yes"])
  # Specify the way this feature is presented to the model, here using a one-hot
  # encoding.
  one_hot_feature = tf.feature_column.indicator_column(
      categorical_column=string_feature)

  estimator, batch_size, window_size = estimator_fn(
      exogenous_feature_columns=[one_hot_feature])
  reader = tf.contrib.timeseries.CSVReader(
      csv_file_name,
      # Indicate the format of our CSV file. First we have two standard columns,
      # one for times and one for values. The third column is a custom exogenous
      # feature indicating whether each timestep is a changepoint. The
      # changepoint feature name must match the string_feature column name
      # above.
      column_names=(tf.contrib.timeseries.TrainEvalFeatures.TIMES,
                    tf.contrib.timeseries.TrainEvalFeatures.VALUES,
                    "is_changepoint"),
      # Indicate dtypes for our features.
      column_dtypes=(tf.int64, tf.float32, tf.string),
      # This CSV has a header line; here we just ignore it.
      skip_header_lines=1)
  train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
      reader, batch_size=batch_size, window_size=window_size)
  estimator.train(input_fn=train_input_fn, steps=train_steps)
  evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
  evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
  # Create an input_fn for prediction, with a simulated changepoint. Since all
  # of the anomalies in the training data are explained by the exogenous
  # feature, we should get relatively confident predictions before the indicated
  # changepoint (since we are telling the model that no changepoint exists at
  # those times) and relatively uncertain predictions after.
  (predictions,) = tuple(estimator.predict(
      input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
          evaluation, steps=100,
          exogenous_features={
              "is_changepoint": [["no"] * 49 + ["yes"] + ["no"] * 50]})))
  times = evaluation["times"][0]
  observed = evaluation["observed"][0, :, 0]
  mean = np.squeeze(np.concatenate(
      [evaluation["mean"][0], predictions["mean"]], axis=0))
  variance = np.squeeze(np.concatenate(
      [evaluation["covariance"][0], predictions["covariance"]], axis=0))
  all_times = np.concatenate([times, predictions["times"]], axis=0)
  upper_limit = mean + np.sqrt(variance)
  lower_limit = mean - np.sqrt(variance)
  # Indicate the locations of the changepoints for plotting vertical lines.
  anomaly_locations = []
  with open(csv_file_name, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
      if row["is_changepoint"] == "yes":
        anomaly_locations.append(int(row["time"]))
  anomaly_locations.append(predictions["times"][49])
  return (times, observed, all_times, mean, upper_limit, lower_limit,
          anomaly_locations)


def make_plot(name, training_times, observed, all_times, mean,
              upper_limit, lower_limit, anomaly_locations):
  """Plot the time series and anomalies in a new figure."""
  pyplot.figure()
  pyplot.plot(training_times, observed, "b", label="training series")
  pyplot.plot(all_times, mean, "r", label="forecast")
  pyplot.axvline(anomaly_locations[0], linestyle="dotted", label="changepoints")
  for anomaly_location in anomaly_locations[1:]:
    pyplot.axvline(anomaly_location, linestyle="dotted")
  pyplot.fill_between(all_times, lower_limit, upper_limit, color="grey",
                      alpha="0.2")
  pyplot.axvline(training_times[-1], color="k", linestyle="--")
  pyplot.xlabel("time")
  pyplot.ylabel("observations")
  pyplot.legend(loc=0)
  pyplot.title(name)


def main(unused_argv):
  if not HAS_MATPLOTLIB:
    raise ImportError(
        "Please install matplotlib to generate a plot from this example.")
  make_plot("Ignoring a known anomaly (state space)",
            *train_and_evaluate_exogenous(
                estimator_fn=state_space_esitmator))
  make_plot("Ignoring a known anomaly (autoregressive)",
            *train_and_evaluate_exogenous(
                estimator_fn=autoregressive_esitmator, train_steps=3000))
  pyplot.show()


if __name__ == "__main__":
  tf.app.run(main=main)
