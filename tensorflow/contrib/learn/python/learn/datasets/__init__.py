"""Module inclues reference datasets and utilities to load datasets."""
#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import collections
from os import path

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.contrib.learn.python.learn.datasets import text_datasets

# Export load_iris and load_boston.
load_iris = base.load_iris
load_boston = base.load_boston

# List of all available datasets.
# Note, currently they may return different types.
DATASETS = {
    # Returns base.Dataset.
    'iris': base.load_iris,
    'boston': base.load_boston,
    # Returns base.Datasets (train/validation/test sets).
    'mnist': mnist.load_mnist,
    'dbpedia': text_datasets.load_dbpedia,
}


def load_dataset(name):
  """Loads dataset by name.

  Args:
    name: Name of the dataset to load.

  Returns:
    Features and targets for given dataset. Can be numpy or iterator.
  """
  if name not in DATASETS:
    raise ValueError('Name of dataset is not found: %s' % name)
  return DATASETS[name]()
