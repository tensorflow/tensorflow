# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
import uuid

import numpy as np
from tensorflow.python.estimator.inputs.queues import feeding_functions
from tensorflow.python.util.tf_export import tf_export

try:
  # pylint: disable=g-import-not-at-top
  # pylint: disable=unused-import
  import pandas as pd
  HAS_PANDAS = True
except IOError:
  # Pandas writes a temporary file during import. If it fails, don't use pandas.
  HAS_PANDAS = False
except ImportError:
  HAS_PANDAS = False


def _get_unique_target_key(features, target_column_name):
  """Returns a key that does not exist in the input DataFrame `features`.

  Args:
    features: DataFrame
    target_column_name: Name of the target column as a `str`

  Returns:
    A unique key that can be used to insert the target into
      features.
  """
  if target_column_name in features:
    target_column_name += '_' + str(uuid.uuid4())
  return target_column_name


@tf_export('estimator.inputs.pandas_input_fn')
def pandas_input_fn(x,
                    y=None,
                    batch_size=128,
                    num_epochs=1,
                    shuffle=None,
                    queue_capacity=1000,
                    num_threads=1,
                    target_column='target'):
  """Returns input function that would feed Pandas DataFrame into the model.

  Note: `y`'s index must match `x`'s index.

  Args:
    x: pandas `DataFrame` object.
    y: pandas `Series` object or `DataFrame`. `None` if absent.
    batch_size: int, size of batches to return.
    num_epochs: int, number of epochs to iterate over data. If not `None`,
      read attempts that would exceed this value will raise `OutOfRangeError`.
    shuffle: bool, whether to read the records in random order.
    queue_capacity: int, size of the read queue. If `None`, it will be set
      roughly to the size of `x`.
    num_threads: Integer, number of threads used for reading and enqueueing. In
      order to have predicted and repeatable order of reading and enqueueing,
      such as in prediction and evaluation mode, `num_threads` should be 1.
    target_column: str, name to give the target column `y`. This parameter
      is not used when `y` is a `DataFrame`.

  Returns:
    Function, that has signature of ()->(dict of `features`, `target`)

  Raises:
    ValueError: if `x` already contains a column with the same name as `y`, or
      if the indexes of `x` and `y` don't match.
    TypeError: `shuffle` is not bool.
  """
  if not HAS_PANDAS:
    raise TypeError(
        'pandas_input_fn should not be called without pandas installed')

  if not isinstance(shuffle, bool):
    raise TypeError('shuffle must be explicitly set as boolean; '
                    'got {}'.format(shuffle))

  if not isinstance(target_column, six.string_types):
    raise TypeError('target_column must be a string type')

  x = x.copy()
  if y is not None:
    if target_column in x:
      raise ValueError(
          'Cannot use name %s for target column: DataFrame already has a '
          'column with that name: %s' % (target_column, x.columns))
    if not np.array_equal(x.index, y.index):
      raise ValueError('Index for x and y are mismatched.\nIndex for x: %s\n'
                       'Index for y: %s\n' % (x.index, y.index))
    if isinstance(y, pd.DataFrame):
      y_columns = [(column, _get_unique_target_key(x, column)) for column in list(y)]
      target_column = [v for _, v in y_columns]
      x[target_column] = y
    else:
      x[target_column] = y

  # TODO(mdan): These are memory copies. We probably don't need 4x slack space.
  # The sizes below are consistent with what I've seen elsewhere.
  if queue_capacity is None:
    if shuffle:
      queue_capacity = 4 * len(x)
    else:
      queue_capacity = len(x)
  min_after_dequeue = max(queue_capacity / 4, 1)

  def input_fn():
    """Pandas input function."""
    queue = feeding_functions._enqueue_data(  # pylint: disable=protected-access
        x,
        queue_capacity,
        shuffle=shuffle,
        min_after_dequeue=min_after_dequeue,
        num_threads=num_threads,
        enqueue_size=batch_size,
        num_epochs=num_epochs)
    if num_epochs is None:
      features = queue.dequeue_many(batch_size)
    else:
      features = queue.dequeue_up_to(batch_size)
    assert len(features) == len(x.columns) + 1, ('Features should have one '
                                                 'extra element for the index.')
    features = features[1:]
    features = dict(zip(list(x.columns), features))
    if y is not None:
      if isinstance(target_column, list):
        keys = [k for k, _ in y_columns]
        values = [features.pop(column) for column in target_column]
        target = {k: v for k, v in zip(keys, values)}
      else:
        target = features.pop(target_column)
      return features, target
    return features
  return input_fn
