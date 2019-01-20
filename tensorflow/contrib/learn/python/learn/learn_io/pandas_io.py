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

"""Methods to allow pandas.DataFrame (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator.inputs.pandas_io import pandas_input_fn as core_pandas_input_fn
from tensorflow.python.util.deprecation import deprecated

try:
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  HAS_PANDAS = True
except IOError:
  # Pandas writes a temporary file during import. If it fails, don't use pandas.
  HAS_PANDAS = False
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


@deprecated(None, 'Please use tf.compat.v1.estimator.inputs.pandas_input_fn')
def pandas_input_fn(x,
                    y=None,
                    batch_size=128,
                    num_epochs=1,
                    shuffle=True,
                    queue_capacity=1000,
                    num_threads=1,
                    target_column='target'):
  """This input_fn diffs from the core version with default `shuffle`."""
  return core_pandas_input_fn(x=x,
                              y=y,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_epochs=num_epochs,
                              queue_capacity=queue_capacity,
                              num_threads=num_threads,
                              target_column=target_column)


@deprecated(None, 'Please access pandas data directly.')
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


@deprecated(None, 'Please access pandas data directly.')
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


@deprecated(None, 'Please access pandas data directly.')
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
