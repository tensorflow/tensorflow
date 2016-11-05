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

"""Base utilities for loading datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from os import path
import tempfile

import numpy as np
from six.moves import urllib

from tensorflow.contrib.framework import deprecated
from tensorflow.python.platform import gfile

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_csv_with_header(filename,
                         target_dtype,
                         features_dtype,
                         target_column=-1):
  """Load dataset from CSV file with a header row."""
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    header = next(data_file)
    n_samples = int(header[0])
    n_features = int(header[1])
    data = np.zeros((n_samples, n_features), dtype=features_dtype)
    target = np.zeros((n_samples,), dtype=target_dtype)
    for i, row in enumerate(data_file):
      target[i] = np.asarray(row.pop(target_column), dtype=target_dtype)
      data[i] = np.asarray(row, dtype=features_dtype)

  return Dataset(data=data, target=target)


def load_csv_without_header(filename,
                            target_dtype,
                            features_dtype,
                            target_column=-1):
  """Load dataset from CSV file without a header row."""
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    data, target = [], []
    for row in data_file:
      target.append(row.pop(target_column))
      data.append(np.asarray(row, dtype=features_dtype))

  target = np.array(target, dtype=target_dtype)
  data = np.array(data)
  return Dataset(data=data, target=target)


def shrink_csv(filename, ratio):
  """Create a smaller dataset of only 1/ratio of original data."""
  filename_small = filename.replace('.', '_small.')
  with gfile.Open(filename_small, 'w') as csv_file_small:
    writer = csv.writer(csv_file_small)
    with gfile.Open(filename) as csv_file:
      reader = csv.reader(csv_file)
      i = 0
      for row in reader:
        if i % ratio == 0:
          writer.writerow(row)
        i += 1


def load_iris(data_path=None):
  """Load Iris dataset.

  Args:
      data_path: string, path to iris dataset (optional)

  Returns:
    Dataset object containing data in-memory.
  """
  if data_path is None:
    module_path = path.dirname(__file__)
    data_path = path.join(module_path, 'data', 'iris.csv')
  return load_csv_with_header(
      data_path,
      target_dtype=np.int,
      features_dtype=np.float)


def load_boston(data_path=None):
  """Load Boston housing dataset.

  Args:
      data_path: string, path to boston dataset (optional)

  Returns:
    Dataset object containing data in-memory.
  """
  if data_path is None:
    module_path = path.dirname(__file__)
    data_path = path.join(module_path, 'data', 'boston_house_prices.csv')
  return load_csv_with_header(
      data_path,
      target_dtype=np.float,
      features_dtype=np.float)


def maybe_download(filename, work_directory, source_url):
  """Download the data from source url, unless it's already here.

  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.

  Returns:
      Path to resulting file.
  """
  if not gfile.Exists(work_directory):
    gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not gfile.Exists(filepath):
    with tempfile.NamedTemporaryFile() as tmpfile:
      temp_file_name = tmpfile.name
      urllib.request.urlretrieve(source_url, temp_file_name)
      gfile.Copy(temp_file_name, filepath)
      with gfile.GFile(filepath) as f:
        size = f.size()
      print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def train_test_split(dataset, *args, **kwargs):
  """Split the dataset into train and test datasets

  The method splits the provided dataset into two parts: train and test
  
  Args:
    dataset: Dataset, dataset to split 
    test_ratio or train_ratio: float, The ratio of the test subset
    shuffle: boolean, shuffle the data before splitting

  Returns:
    train, test: Tuple of datasets of the same type as input

  Raises:
    ValueError: if either `test_ratio` or `train_ratio` out of [0.,1.] bounds
    TypeError: if `dataset` is not of type `Dataset`

  TODO:
    - Consider refering the `_sklearn._train_test_split` to the current routine.
        Currently, the arguments and return values are of incompatible types,
        and might cause issues!
  """
  if not isinstance(dataset, Dataset):
    raise TypeError('`dataset` type should be `Dataset`: %s' % type(dataset))
  test_ratio = kwargs.pop('test_ratio', None)
  train_ratio = kwargs.pop('train_ratio', None)
  shuffle = kwargs.pop('shuffle', None)
  if test_ratio is None and train_ratio is None:
    train_ratio = 0.7
  elif train_ratio is None:
    train_ratio = 1. - test_ratio
  if not 0. <= test_ratio <= 1.:
    raise ValueError('The test_ratio value is out of bounds: %s' % test_ratio)
  if not 0. <= train_ratio <= 1.:
    raise ValueError('The trainratio value is out of bounds: %s' % train_ratio)
  len_train = int(dataset.data.shape[0]*train_ratio)
  # We don't want to modify the original dataset, so make a copy
  X = dataset.data[:]
  y = dataset.target[:]
  if shuffle:
    indices = np.random.permutation(range(X.shape[0]))
    X = X[indices]
    y = y[indices]
  return  Dataset(data=X[:len_train], target=y[:len_train]), \
          Dataset(data=X[len_train:], target=y[len_train:])
