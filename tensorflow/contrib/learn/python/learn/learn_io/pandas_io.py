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

"""Methods to allow pandas.DataFrame."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.learn.python.learn.dataframe.queues import feeding_functions

try:
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  HAS_PANDAS = True
except ImportError:
  HAS_PANDAS = False

PANDAS_DTYPES = {
    'int8': 'int',
    'int16': 'int',
    'int32': 'int',
    'int64': 'int',
    'uint8': 'int',
    'uint16': 'int',
    'uint32': 'int',
    'uint64': 'int',
    'float16': 'float',
    'float32': 'float',
    'float64': 'float',
    'bool': 'i'
}


def extract_pandas_data(data):
  """Extract data from pandas.DataFrame for predictors.

  Given a DataFrame, will extract the values and cast them to float. The
  DataFrame is expected to contain values of type int, float or bool.

  Args:
    data: `pandas.DataFrame` containing the data to be extracted.

  Returns:
    A numpy `ndarray` of the DataFrame's values as floats.

  Raises:
    ValueError: if data contains types other than int, float or bool.
  """
  if not isinstance(data, pd.DataFrame):
    return data

  bad_data = [column for column in data
              if data[column].dtype.name not in PANDAS_DTYPES]

  if not bad_data:
    return data.values.astype('float')
  else:
    error_report = [("'" + str(column) + "' type='" +
                     data[column].dtype.name + "'") for column in bad_data]
    raise ValueError('Data types for extracting pandas data must be int, '
                     'float, or bool. Found: ' + ', '.join(error_report))


def extract_pandas_matrix(data):
  """Extracts numpy matrix from pandas DataFrame.

  Args:
    data: `pandas.DataFrame` containing the data to be extracted.

  Returns:
    A numpy `ndarray` of the DataFrame's values.
  """
  if not isinstance(data, pd.DataFrame):
    return data

  return data.as_matrix()


def extract_pandas_labels(labels):
  """Extract data from pandas.DataFrame for labels.

  Args:
    labels: `pandas.DataFrame` or `pandas.Series` containing one column of
      labels to be extracted.

  Returns:
    A numpy `ndarray` of labels from the DataFrame.

  Raises:
    ValueError: if more than one column is found or type is not int, float or
      bool.
  """
  if isinstance(labels,
                pd.DataFrame):  # pandas.Series also belongs to DataFrame
    if len(labels.columns) > 1:
      raise ValueError('Only one column for labels is allowed.')

    bad_data = [column for column in labels
                if labels[column].dtype.name not in PANDAS_DTYPES]
    if not bad_data:
      return labels.values
    else:
      error_report = ["'" + str(column) + "' type="
                      + str(labels[column].dtype.name) for column in bad_data]
      raise ValueError('Data types for extracting labels must be int, '
                       'float, or bool. Found: ' + ', '.join(error_report))
  else:
    return labels


def pandas_input_fn(x, y=None, batch_size=128, num_epochs=1, shuffle=True,
                    queue_capacity=1000, num_threads=1, target_column='target',
                    index_column='index'):
  """Returns input function that would feed pandas DataFrame into the model.

  Note: If y's index doesn't match x's index exception will be raised.

  Args:
    x: pandas `DataFrame` object.
    y: pandas `Series` object.
    batch_size: int, size of batches to return.
    num_epochs: int, number of epochs to iterate over data. If `None` will
      run forever.
    shuffle: bool, if shuffle the queue. Please make sure you don't shuffle at
      prediction time.
    queue_capacity: int, size of queue to accumulate.
    num_threads: int, number of threads used for reading and enqueueing.
    target_column: str, used to pack `y` into `x` DataFrame under this column.
    index_column: str, name of the feature return with index.

  Returns:
    Function, that has signature of ()->(dict of `features`, `target`)

  Raises:
    ValueError: if `target_column` column is already in `x` DataFrame.
  """
  def input_fn():
    """Pandas input function."""
    if y is not None:
      if target_column in x:
        raise ValueError('Found already column \'%s\' in x, please change '
                         'target_column to something else. Current columns '
                         'in x: %s', target_column, x.columns)
      if not np.array_equal(x.index, y.index):
        raise ValueError('Index for x and y are mismatch, this will lead '
                         'to missing values. Please make sure they match or '
                         'use .reset_index() method.\n'
                         'Index for x: %s\n'
                         'Index for y: %s\n', x.index, y.index)
      x[target_column] = y
    queue = feeding_functions.enqueue_data(
        x, queue_capacity, shuffle=shuffle, num_threads=num_threads,
        enqueue_size=batch_size, num_epochs=num_epochs)
    if num_epochs is None:
      features = queue.dequeue_many(batch_size)
    else:
      features = queue.dequeue_up_to(batch_size)
    features = dict(zip([index_column] + list(x.columns), features))
    if y is not None:
      target = features.pop(target_column)
      return features, target
    return features
  return input_fn
