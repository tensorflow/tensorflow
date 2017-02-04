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

"""Distributed training and evaluation of a wide and deep model."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys

from six.moves import urllib
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config


# Constants: Data download URLs
TRAIN_DATA_URL = "http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data"
TEST_DATA_URL = "http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test"


# Define features for the model
def census_model_config():
  """Configuration for the census Wide & Deep model.

  Returns:
    columns: Column names to retrieve from the data source
    label_column: Name of the label column
    wide_columns: List of wide columns
    deep_columns: List of deep columns
    categorical_column_names: Names of the categorical columns
    continuous_column_names: Names of the continuous columns
  """
  # 1. Categorical base columns.
  gender = tf.contrib.layers.sparse_column_with_keys(
      column_name="gender", keys=["female", "male"])
  race = tf.contrib.layers.sparse_column_with_keys(
      column_name="race",
      keys=["Amer-Indian-Eskimo",
            "Asian-Pac-Islander",
            "Black",
            "Other",
            "White"])
  education = tf.contrib.layers.sparse_column_with_hash_bucket(
      "education", hash_bucket_size=1000)
  marital_status = tf.contrib.layers.sparse_column_with_hash_bucket(
      "marital_status", hash_bucket_size=100)
  relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
      "relationship", hash_bucket_size=100)
  workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
      "workclass", hash_bucket_size=100)
  occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
      "occupation", hash_bucket_size=1000)
  native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
      "native_country", hash_bucket_size=1000)

  # 2. Continuous base columns.
  age = tf.contrib.layers.real_valued_column("age")
  age_buckets = tf.contrib.layers.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
  education_num = tf.contrib.layers.real_valued_column("education_num")
  capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
  capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
  hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

  wide_columns = [
      gender, native_country, education, occupation, workclass,
      marital_status, relationship, age_buckets,
      tf.contrib.layers.crossed_column([education, occupation],
                                       hash_bucket_size=int(1e4)),
      tf.contrib.layers.crossed_column([native_country, occupation],
                                       hash_bucket_size=int(1e4)),
      tf.contrib.layers.crossed_column([age_buckets, race, occupation],
                                       hash_bucket_size=int(1e6))]

  deep_columns = [
      tf.contrib.layers.embedding_column(workclass, dimension=8),
      tf.contrib.layers.embedding_column(education, dimension=8),
      tf.contrib.layers.embedding_column(marital_status, dimension=8),
      tf.contrib.layers.embedding_column(gender, dimension=8),
      tf.contrib.layers.embedding_column(relationship, dimension=8),
      tf.contrib.layers.embedding_column(race, dimension=8),
      tf.contrib.layers.embedding_column(native_country, dimension=8),
      tf.contrib.layers.embedding_column(occupation, dimension=8),
      age, education_num, capital_gain, capital_loss, hours_per_week]

  # Define the column names for the data sets.
  columns = ["age", "workclass", "fnlwgt", "education", "education_num",
             "marital_status", "occupation", "relationship", "race", "gender",
             "capital_gain", "capital_loss", "hours_per_week",
             "native_country", "income_bracket"]
  label_column = "label"
  categorical_columns = ["workclass", "education", "marital_status",
                         "occupation", "relationship", "race", "gender",
                         "native_country"]
  continuous_columns = ["age", "education_num", "capital_gain",
                        "capital_loss", "hours_per_week"]

  return (columns, label_column, wide_columns, deep_columns,
          categorical_columns, continuous_columns)


class CensusDataSource(object):
  """Source of census data."""

  def __init__(self, data_dir, train_data_url, test_data_url,
               columns, label_column,
               categorical_columns, continuous_columns):
    """Constructor of CensusDataSource.

    Args:
      data_dir: Directory to save/load the data files
      train_data_url: URL from which the training data can be downloaded
      test_data_url: URL from which the test data can be downloaded
      columns: Columns to retrieve from the data files (A list of strings)
      label_column: Name of the label column
      categorical_columns: Names of the categorical columns (A list of strings)
      continuous_columns: Names of the continuous columsn (A list of strings)
    """

    # Retrieve data from disk (if available) or download from the web.
    train_file_path = os.path.join(data_dir, "adult.data")
    if os.path.isfile(train_file_path):
      print("Loading training data from file: %s" % train_file_path)
      train_file = open(train_file_path)
    else:
      urllib.urlretrieve(train_data_url, train_file_path)

    test_file_path = os.path.join(data_dir, "adult.test")
    if os.path.isfile(test_file_path):
      print("Loading test data from file: %s" % test_file_path)
      test_file = open(test_file_path)
    else:
      test_file = open(test_file_path)
      urllib.urlretrieve(test_data_url, test_file_path)

    # Read the training and testing data sets into Pandas DataFrame.
    import pandas  # pylint: disable=g-import-not-at-top
    self._df_train = pandas.read_csv(train_file, names=columns,
                                     skipinitialspace=True)
    self._df_test = pandas.read_csv(test_file, names=columns,
                                    skipinitialspace=True, skiprows=1)

    # Remove the NaN values in the last rows of the tables
    self._df_train = self._df_train[:-1]
    self._df_test = self._df_test[:-1]

    # Apply the threshold to get the labels.
    income_thresh = lambda x: ">50K" in x
    self._df_train[label_column] = (
        self._df_train["income_bracket"].apply(income_thresh)).astype(int)
    self._df_test[label_column] = (
        self._df_test["income_bracket"].apply(income_thresh)).astype(int)

    self.label_column = label_column
    self.categorical_columns = categorical_columns
    self.continuous_columns = continuous_columns

  def input_train_fn(self):
    return self._input_fn(self._df_train)

  def input_test_fn(self):
    return self._input_fn(self._df_test)

  # TODO(cais): Turn into minibatch feeder
  def _input_fn(self, df):
    """Input data function.

    Creates a dictionary mapping from each continuous feature column name
    (k) to the values of that column stored in a constant Tensor.

    Args:
      df: data feed

    Returns:
      feature columns and labels
    """
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in self.continuous_columns}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in self.categorical_columns}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[self.label_column].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def _create_experiment_fn(output_dir):  # pylint: disable=unused-argument
  """Experiment creation function."""
  (columns, label_column, wide_columns, deep_columns, categorical_columns,
   continuous_columns) = census_model_config()

  census_data_source = CensusDataSource(FLAGS.data_dir,
                                        TRAIN_DATA_URL, TEST_DATA_URL,
                                        columns, label_column,
                                        categorical_columns,
                                        continuous_columns)

  os.environ["TF_CONFIG"] = json.dumps({
      "cluster": {
          tf.contrib.learn.TaskType.PS: ["fake_ps"] *
                                        FLAGS.num_parameter_servers
      },
      "task": {
          "index": FLAGS.worker_index
      }
  })
  config = run_config.RunConfig(master=FLAGS.master_grpc_url)

  estimator = tf.contrib.learn.DNNLinearCombinedClassifier(
      model_dir=FLAGS.model_dir,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=[5],
      config=config)

  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=census_data_source.input_train_fn,
      eval_input_fn=census_data_source.input_test_fn,
      train_steps=FLAGS.train_steps,
      eval_steps=FLAGS.eval_steps
  )


def main(unused_argv):
  print("Worker index: %d" % FLAGS.worker_index)
  learn_runner.run(experiment_fn=_create_experiment_fn,
                   output_dir=FLAGS.output_dir,
                   schedule=FLAGS.schedule)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--data_dir",
      type=str,
      default="/tmp/census-data",
      help="Directory for storing the cesnsus data"
  )
  parser.add_argument(
      "--model_dir",
      type=str,
      default="/tmp/census_wide_and_deep_model",
      help="Directory for storing the model"
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="",
      help="Base output directory."
  )
  parser.add_argument(
      "--schedule",
      type=str,
      default="local_run",
      help="Schedule to run for this experiment."
  )
  parser.add_argument(
      "--master_grpc_url",
      type=str,
      default="",
      help="URL to master GRPC tensorflow server, e.g.,grpc://127.0.0.1:2222"
  )
  parser.add_argument(
      "--num_parameter_servers",
      type=int,
      default=0,
      help="Number of parameter servers"
  )
  parser.add_argument(
      "--worker_index",
      type=int,
      default=0,
      help="Worker index (>=0)"
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=1000,
      help="Number of training steps"
  )
  parser.add_argument(
      "--eval_steps",
      type=int,
      default=1,
      help="Number of evaluation steps"
  )
  global FLAGS  # pylint:disable=global-at-module-level
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
