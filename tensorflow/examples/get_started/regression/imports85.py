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
"""A dataset loader for imports85.data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import pandas as pd
import tensorflow as tf

header = collections.OrderedDict([
    ("symboling", np.int32),
    ("normalized-losses", np.float32),
    ("make", str),
    ("fuel-type", str),
    ("aspiration", str),
    ("num-of-doors", str),
    ("body-style", str),
    ("drive-wheels", str),
    ("engine-location", str),
    ("wheel-base", np.float32),
    ("length", np.float32),
    ("width", np.float32),
    ("height", np.float32),
    ("curb-weight", np.float32),
    ("engine-type", str),
    ("num-of-cylinders", str),
    ("engine-size", np.float32),
    ("fuel-system", str),
    ("bore", np.float32),
    ("stroke", np.float32),
    ("compression-ratio", np.float32),
    ("horsepower", np.float32),
    ("peak-rpm", np.float32),
    ("city-mpg", np.float32),
    ("highway-mpg", np.float32),
    ("price", np.float32)
])  # pyformat: disable


def raw():
  """Get the imports85 data and load it as a pd.DataFrame."""
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"  # pylint: disable=line-too-long
  # Download and cache the data.
  path = tf.contrib.keras.utils.get_file(url.split("/")[-1], url)

  # Load the CSV data into a pandas dataframe.
  df = pd.read_csv(path, names=header.keys(), dtype=header, na_values="?")

  return df


def load_data(y_name="price", train_fraction=0.7, seed=None):
  """Returns the imports85 shuffled and split into train and test subsets.

  A description of the data is available at:
    https://archive.ics.uci.edu/ml/datasets/automobile

  The data itself can be found at:
    https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

  Args:
    y_name: the column to return as the label.
    train_fraction: the fraction of the dataset to use for training.
    seed: The random seed to use when shuffling the data. `None` generates a
      unique shuffle every run.
  Returns:
    a pair of pairs where the first pair is the training data, and the second
    is the test data:
    `(x_train, y_train), (x_test, y_test) = get_imports85_dataset(...)`
    `x` contains a pandas DataFrame of features, while `y` contains the label
    array.
  """
  # Load the raw data columns.
  data = raw()

  # Delete rows with unknowns
  data = data.dropna()

  # Shuffle the data
  np.random.seed(seed)

  # Split the data into train/test subsets.
  x_train = data.sample(frac=train_fraction, random_state=seed)
  x_test = data.drop(x_train.index)

  # Extract the label from the features dataframe.
  y_train = x_train.pop(y_name)
  y_test = x_test.pop(y_name)

  return (x_train, y_train), (x_test, y_test)
