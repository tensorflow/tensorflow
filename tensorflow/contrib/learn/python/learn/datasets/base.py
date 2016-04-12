"""Base utilities for loading datasets."""
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
import os
from os import path
import tempfile
from six.moves import urllib

import numpy as np
from tensorflow.python.platform.default import _gfile as gfile

    
Dataset = collections.namedtuple('Dataset', ['data', 'target'])


def load_csv(filename, target_dtype):
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        header = next(data_file)
        n_samples = int(header[0])
        n_features = int(header[1])
        target_names = np.array(header[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=target_dtype)

    return Dataset(data=data, target=target)


def load_iris():
    """Load Iris dataset.

    Returns:
        Dataset object containing data in-memory.
    """
    module_path = path.dirname(__file__)
    return load_csv(path.join(module_path, 'data', 'iris.csv'),
                    target_dtype=np.int)


def load_boston():
    """Load Boston housing dataset.
    
    Returns:
        Dataset object containing data in-memory.
    """
    module_path = path.dirname(__file__)
    return load_csv(path.join(module_path, 'data', 'boston_house_prices.csv'),
                    target_dtype=np.float)


def maybe_download(filename, work_directory, source_url):
  """Download the data from source url, unless it's already here."""
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

