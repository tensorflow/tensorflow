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

"""Dataset utilities and synthetic/reference datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
from os import path

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.contrib.learn.python.learn.datasets import synthetic
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

# List of all synthetic datasets
SYNTHETIC = {
  # All of these will return ['data', 'target'] -> base.Dataset
  'circles': synthetic.circles,
  'spirals': synthetic.spirals
}

def load_dataset(name, size='small', test_with_fake_data=False):
  """Loads dataset by name.

  Args:
    name: Name of the dataset to load.
    size: Size of the dataset to load.
    test_with_fake_data: If true, load with fake dataset.

  Returns:
    Features and labels for given dataset. Can be numpy or iterator.

  Raises:
    ValueError: if `name` is not found.
  """
  if name not in DATASETS:
    raise ValueError('Name of dataset is not found: %s' % name)
  if name == 'dbpedia':
    return DATASETS[name](size, test_with_fake_data)
  else:
    return DATASETS[name]()


def make_dataset(name, n_samples=100, noise=None, seed=42, *args, **kwargs):
  """Creates binary synthetic datasets

  Args:
    name: str, name of the dataset to generate
    n_samples: int, number of datapoints to generate
    noise: float or None, standard deviation of the Gaussian noise added
    seed: int or None, seed for noise

  Returns:
    Shuffled features and labels for given synthetic dataset of type `base.Dataset`

  Raises:
    ValueError: Raised if `name` not found

  Note:
    - This is a generic synthetic data generator - individual generators might have more parameters!
      See documentation for individual parameters
    - Note that the `noise` parameter uses `numpy.random.normal` and depends on `numpy`'s seed

  TODO:
    - Support multiclass datasets
    - Need shuffling routine. Currently synthetic datasets are reshuffled to avoid train/test correlation,
      but that hurts reprodusability
  """
  # seed = kwargs.pop('seed', None)
  if name not in SYNTHETIC:
    raise ValueError('Synthetic dataset not found or not implemeted: %s' % name)
  else:
    return SYNTHETIC[name](n_samples=n_samples, noise=noise, seed=seed, *args, **kwargs)
