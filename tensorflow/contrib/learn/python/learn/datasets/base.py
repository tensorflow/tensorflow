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

"""Base utilities for loading datasets (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from os import path
import random
import time

import numpy as np
from six.moves import urllib

from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated


Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


@deprecated(None, 'Use tf.data instead.')
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


@deprecated(None, 'Use tf.data instead.')
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


@deprecated(None, 'Use tf.data instead.')
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


@deprecated(None, 'Use scikits.learn.datasets.')
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
      data_path, target_dtype=np.int, features_dtype=np.float)


@deprecated(None, 'Use scikits.learn.datasets.')
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
      data_path, target_dtype=np.float, features_dtype=np.float)


@deprecated(None, 'Use the retry module or similar alternatives.')
def retry(initial_delay,
          max_delay,
          factor=2.0,
          jitter=0.25,
          is_retriable=None):
  """Simple decorator for wrapping retriable functions.

  Args:
    initial_delay: the initial delay.
    max_delay: the maximum delay allowed (actual max is
        max_delay * (1 + jitter).
    factor: each subsequent retry, the delay is multiplied by this value.
        (must be >= 1).
    jitter: to avoid lockstep, the returned delay is multiplied by a random
        number between (1-jitter) and (1+jitter). To add a 20% jitter, set
        jitter = 0.2. Must be < 1.
    is_retriable: (optional) a function that takes an Exception as an argument
        and returns true if retry should be applied.

  Returns:
    A function that wraps another function to automatically retry it.
  """
  return _internal_retry(
      initial_delay=initial_delay,
      max_delay=max_delay,
      factor=factor,
      jitter=jitter,
      is_retriable=is_retriable)


def _internal_retry(initial_delay,
                    max_delay,
                    factor=2.0,
                    jitter=0.25,
                    is_retriable=None):
  """Simple decorator for wrapping retriable functions, for internal use only.

  Args:
    initial_delay: the initial delay.
    max_delay: the maximum delay allowed (actual max is
        max_delay * (1 + jitter).
    factor: each subsequent retry, the delay is multiplied by this value.
        (must be >= 1).
    jitter: to avoid lockstep, the returned delay is multiplied by a random
        number between (1-jitter) and (1+jitter). To add a 20% jitter, set
        jitter = 0.2. Must be < 1.
    is_retriable: (optional) a function that takes an Exception as an argument
        and returns true if retry should be applied.

  Returns:
    A function that wraps another function to automatically retry it.
  """
  if factor < 1:
    raise ValueError('factor must be >= 1; was %f' % (factor,))

  if jitter >= 1:
    raise ValueError('jitter must be < 1; was %f' % (jitter,))

  # Generator to compute the individual delays
  def delays():
    delay = initial_delay
    while delay <= max_delay:
      yield delay * random.uniform(1 - jitter, 1 + jitter)
      delay *= factor

  def wrap(fn):
    """Wrapper function factory invoked by decorator magic."""

    def wrapped_fn(*args, **kwargs):
      """The actual wrapper function that applies the retry logic."""
      for delay in delays():
        try:
          return fn(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
          if is_retriable is None:
            continue

          if is_retriable(e):
            time.sleep(delay)
          else:
            raise
      return fn(*args, **kwargs)

    return wrapped_fn

  return wrap


_RETRIABLE_ERRNOS = {
    110,  # Connection timed out [socket.py]
}


def _is_retriable(e):
  return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS


@deprecated(None, 'Please use urllib or similar directly.')
@_internal_retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def urlretrieve_with_retry(url, filename=None):
  return urllib.request.urlretrieve(url, filename)


@deprecated(None, 'Please write your own downloading logic.')
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
    temp_file_name, _ = urlretrieve_with_retry(source_url)
    gfile.Copy(temp_file_name, filepath)
    with gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath
