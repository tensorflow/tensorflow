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

from tensorflow.python.platform import gfile

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_csv(filename, target_dtype, target_column=-1, has_header=True):
  """Load dataset from CSV file."""
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    if has_header:
      header = next(data_file)
      n_samples = int(header[0])
      n_features = int(header[1])
      data = np.empty((n_samples, n_features))
      target = np.empty((n_samples,), dtype=np.int)
      for i, ir in enumerate(data_file):
        target[i] = np.asarray(ir.pop(target_column), dtype=target_dtype)
        data[i] = np.asarray(ir, dtype=np.float64)
    else:
      data, target = [], []
      for ir in data_file:
        target.append(ir.pop(target_column))
        data.append(ir)
  return Dataset(data=data, target=target)


def load_iris():
  """Load Iris dataset.

  Returns:
    Dataset object containing data in-memory.
  """
  module_path = path.dirname(__file__)
  return load_csv(
      path.join(module_path, 'data', 'iris.csv'),
      target_dtype=np.int)


def load_boston():
  """Load Boston housing dataset.

  Returns:
    Dataset object containing data in-memory.
  """
  module_path = path.dirname(__file__)
  return load_csv(
      path.join(module_path, 'data', 'boston_house_prices.csv'),
      target_dtype=np.float)


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
        size = f.Size()
      print('Successfully downloaded', filename, size, 'bytes.')
  return filepath
