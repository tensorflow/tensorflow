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
"""An example of training and predicting with a TFTS estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

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

FLAGS = None


_MODULE_PATH = os.path.dirname(__file__)
_DEFAULT_DATA_FILE = os.path.join(_MODULE_PATH, "data/period_trend.csv")


def structural_ensemble_train_and_predict(csv_file_name):
  # Cycle between 5 latent values over a period of 100. This leads to a very
  # smooth periodic component (and a small model), which is a good fit for our
  # example data. Modeling high-frequency periodic variations will require a
  # higher cycle_num_latent_values.
  structural = tf.contrib.timeseries.StructuralEnsembleRegressor(
      periodicities=100, num_features=1, cycle_num_latent_values=5)
  return train_and_predict(structural, csv_file_name, training_steps=150)


def ar_train_and_predict(csv_file_name):
  # An autoregressive model, with periodicity handled as a time-based
  # regression. Note that this requires windows of size 16 (input_window_size +
  # output_window_size) for training.
  ar = tf.contrib.timeseries.ARRegressor(
      periodicities=100, input_window_size=10, output_window_size=6,
      num_features=1,
      # Use the (default) normal likelihood loss to adaptively fit the
      # variance. SQUARED_LOSS overestimates variance when there are trends in
      # the series.
      loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)
  return train_and_predict(ar, csv_file_name, training_steps=600)


def train_and_predict(estimator, csv_file_name, training_steps):
  """A simple example of training and predicting."""
  # Read data in the default "time,value" CSV format with no header
  reader = tf.contrib.timeseries.CSVReader(csv_file_name)
  # Set up windowing and batching for training
  train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
      reader, batch_size=16, window_size=16)
  # Fit model parameters to data
  estimator.train(input_fn=train_input_fn, steps=training_steps)
  # Evaluate on the full dataset sequentially, collecting in-sample predictions
  # for a qualitative evaluation. Note that this loads the whole dataset into
  # memory. For quantitative evaluation, use RandomWindowChunker.
  evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
  evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
  # Predict starting after the evaluation
  (predictions,) = tuple(estimator.predict(
      input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
          evaluation, steps=200)))
  times = evaluation["times"][0]
  observed = evaluation["observed"][0, :, 0]
  mean = np.squeeze(np.concatenate(
      [evaluation["mean"][0], predictions["mean"]], axis=0))
  variance = np.squeeze(np.concatenate(
      [evaluation["covariance"][0], predictions["covariance"]], axis=0))
  all_times = np.concatenate([times, predictions["times"]], axis=0)
  upper_limit = mean + np.sqrt(variance)
  lower_limit = mean - np.sqrt(variance)
  return times, observed, all_times, mean, upper_limit, lower_limit


def make_plot(name, training_times, observed, all_times, mean,
              upper_limit, lower_limit):
  """Plot a time series in a new figure."""
  pyplot.figure()
  pyplot.plot(training_times, observed, "b", label="training series")
  pyplot.plot(all_times, mean, "r", label="forecast")
  pyplot.plot(all_times, upper_limit, "g", label="forecast upper bound")
  pyplot.plot(all_times, lower_limit, "g", label="forecast lower bound")
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
  input_filename = FLAGS.input_filename
  if input_filename is None:
    input_filename = _DEFAULT_DATA_FILE
  make_plot("Structural ensemble",
            *structural_ensemble_train_and_predict(input_filename))
  make_plot("AR", *ar_train_and_predict(input_filename))
  pyplot.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_filename",
      type=str,
      required=False,
      help="Input csv file (omit to use the data/period_trend.csv).")
  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
