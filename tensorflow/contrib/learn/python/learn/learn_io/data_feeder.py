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

"""Implementations of different data feeders to provide data for TF trainer."""

# TODO(ipolosukhin): Replace this module with feed-dict queue runners & queues.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging

# pylint: disable=g-multiple-import,g-bad-import-order
from .pandas_io import HAS_PANDAS, extract_pandas_data, extract_pandas_matrix, extract_pandas_labels
from .dask_io import HAS_DASK, extract_dask_data, extract_dask_labels
# pylint: enable=g-multiple-import,g-bad-import-order


def _get_in_out_shape(x_shape, y_shape, n_classes, batch_size=None):
  """Returns shape for input and output of the data feeder."""
  if batch_size is None:
    batch_size = x_shape[0]
  elif batch_size <= 0:
    raise ValueError('Invalid batch_size %d.' % batch_size)
  x_shape = list(x_shape[1:]) if len(x_shape) > 1 else [1]
  input_shape = [batch_size] + x_shape
  if y_shape is None:
    return input_shape, None, batch_size
  y_shape = list(y_shape[1:]) if len(y_shape) > 1 else []
  # Skip first dimension if it is 1.
  if y_shape and y_shape[0] == 1:
    y_shape = y_shape[1:]
  if n_classes is not None and n_classes > 1:
    output_shape = [batch_size] + y_shape + [n_classes]
  else:
    output_shape = [batch_size] + y_shape
  return input_shape, output_shape, batch_size


def _data_type_filter(x, y):
  """Filter data types into acceptable format."""
  if HAS_DASK:
    x = extract_dask_data(x)
    if y is not None:
      y = extract_dask_labels(y)
  if HAS_PANDAS:
    x = extract_pandas_data(x)
    if y is not None:
      y = extract_pandas_labels(y)
  return x, y


def _is_iterable(x):
  return hasattr(x, 'next') or hasattr(x, '__next__')


def setup_train_data_feeder(
    x, y, n_classes, batch_size=None, shuffle=True, epochs=None):
  """Create data feeder, to sample inputs from dataset.

  If `x` and `y` are iterators, use `StreamingDataFeeder`.

  Args:
    x: numpy, pandas or Dask matrix or iterable.
    y: numpy, pandas or Dask array or iterable.
    n_classes: number of classes.
    batch_size: size to split data into parts. Must be >= 1.
    shuffle: Whether to shuffle the inputs.
    epochs: Number of epochs to run.

  Returns:
    DataFeeder object that returns training data.

  Raises:
    ValueError: if one of `x` and `y` is iterable and the other is not.
  """
  x, y = _data_type_filter(x, y)
  if HAS_DASK:
    # pylint: disable=g-import-not-at-top
    import dask.dataframe as dd
    if (isinstance(x, (dd.Series, dd.DataFrame)) and
        (y is None or isinstance(y, (dd.Series, dd.DataFrame)))):
      data_feeder_cls = DaskDataFeeder
    else:
      data_feeder_cls = DataFeeder
  else:
    data_feeder_cls = DataFeeder

  if _is_iterable(x):
    if y is not None and not _is_iterable(y):
      raise ValueError('Both x and y should be iterators for '
                       'streaming learning to work.')
    return StreamingDataFeeder(x, y, n_classes, batch_size)
  return data_feeder_cls(
      x, y, n_classes, batch_size, shuffle=shuffle, epochs=epochs)


def _batch_data(x, batch_size=None):
  if (batch_size is not None) and (batch_size <= 0):
    raise ValueError('Invalid batch_size %d.' % batch_size)
  chunk = []
  for data in x:
    chunk.append(data)
    if (batch_size is not None) and (len(chunk) >= batch_size):
      yield np.matrix(chunk)
      chunk = []
  yield np.matrix(chunk)


def setup_predict_data_feeder(x, batch_size=None):
  """Returns an iterable for feeding into predict step.

  Args:
    x: numpy, pandas, Dask array or iterable.
    batch_size: Size of batches to split data into.
      If `None`, returns one batch of full size.

  Returns:
    List or iterator of parts of data to predict on.

  Raises:
    ValueError: if `batch_size` <= 0.
  """
  if HAS_DASK:
    x = extract_dask_data(x)
  if HAS_PANDAS:
    x = extract_pandas_data(x)
  if _is_iterable(x):
    return _batch_data(x, batch_size)
  if len(x.shape) == 1:
    x = np.reshape(x, (-1, 1))
  if batch_size is not None:
    if batch_size <= 0:
      raise ValueError('Invalid batch_size %d.' % batch_size)
    n_batches = int(math.ceil(float(len(x)) / batch_size))
    return [x[i * batch_size:(i + 1) * batch_size] for i in xrange(n_batches)]
  return [x]


def setup_processor_data_feeder(x):
  """Sets up processor iterable.

  Args:
    x: numpy, pandas or iterable.

  Returns:
    Iterable of data to process.
  """
  if HAS_PANDAS:
    x = extract_pandas_matrix(x)
  return x


def check_array(array, dtype):
  """Checks array on dtype and converts it if different.

  Args:
    array: Input array.
    dtype: Expected dtype.

  Returns:
    Original array or converted.
  """
  # skip check if array is instance of other classes, e.g. h5py.Dataset
  # to avoid copying array and loading whole data into memory
  if isinstance(array, (np.ndarray, list)):
    array = np.array(array, dtype=dtype, order=None, copy=False)
  return array


def _access(data, iloc):
  """Accesses an element from collection, using integer location based indexing.

  Args:
    data: array-like. The collection to access
    iloc: `int` or `list` of `int`s. Location(s) to access in `collection`

  Returns:
    The element of `a` found at location(s) `iloc`.
  """
  if HAS_PANDAS:
    import pandas as pd  # pylint: disable=g-import-not-at-top
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
      return data.iloc[iloc]
  return data[iloc]


def _check_dtype(dtype):
  if dtypes.as_dtype(dtype) == dtypes.float64:
    logging.warn(
        'float64 is not supported by many models, consider casting to float32.')
  return dtype


class DataFeeder(object):
  """Data feeder is an example class to sample data for TF trainer."""

  def __init__(
      self, x, y, n_classes, batch_size=None, shuffle=True, random_state=None,
      epochs=None):
    """Initializes a DataFeeder instance.

    Args:
      x: Feature Nd numpy matrix of shape `[n_samples, n_features, ...]`.
      y: Target vector, either floats for regression or class id for
        classification. If matrix, will consider as a sequence
        of targets. Can be `None` for unsupervised setting.
      n_classes: Number of classes, 0 and 1 are considered regression, `None`
        will pass through the input labels without one-hot conversion.
      batch_size: Mini-batch size to accumulate.
      shuffle: Whether to shuffle `x`.
      random_state: Numpy `RandomState` object to reproduce sampling.
      epochs: Number of times to iterate over input data before raising
        `StopIteration` exception.

    Attributes:
      x: Input features.
      y: Input target.
      n_classes: Number of classes (if `None`, pass through indices without
        one-hot conversion).
      batch_size: Mini-batch size to accumulate.
      input_shape: Shape of the input.
      output_shape: Shape of the output.
      input_dtype: DType of input.
      output_dtype: DType of output.
    """
    self._x = check_array(x, dtype=x.dtype)
    # self.n_classes is None means we're passing in raw target indices.
    y_dtype = (
        np.int64 if n_classes is not None and n_classes > 1 else np.float32)
    if n_classes is not None:
      self._y = (None if y is None else check_array(y, dtype=y_dtype))
    elif isinstance(y, list):
      self._y = np.array(y)
    else:
      self._y = y
    self.n_classes = n_classes
    self.max_epochs = epochs
    self.input_shape, self.output_shape, self._batch_size = _get_in_out_shape(
        self._x.shape, None if self._y is None else self._y.shape, n_classes,
        batch_size)
    # Input dtype matches dtype of x.
    self._input_dtype = _check_dtype(self._x.dtype)
    # self.n_classes is None means we're passing in raw target indices
    if n_classes is not None or self._y is None:
      self._output_dtype = np.float32
    else:
      self._output_dtype = _check_dtype(self._y.dtype)
    self._shuffle = shuffle
    self.random_state = np.random.RandomState(
        42) if random_state is None else random_state
    if self._shuffle:
      self.indices = self.random_state.permutation(self._x.shape[0])
    else:
      self.indices = np.array(range(self._x.shape[0]))
    self.offset = 0
    self.epoch = 0
    self._epoch_placeholder = None

  @property
  def x(self):
    return self._x

  @property
  def y(self):
    return self._y

  @property
  def shuffle(self):
    return self._shuffle

  @property
  def input_dtype(self):
    return self._input_dtype

  @property
  def output_dtype(self):
    return self._output_dtype

  @property
  def batch_size(self):
    return self._batch_size

  def make_epoch_variable(self):
    """Adds a placeholder variable for the epoch to the graph.

    Returns:
      The epoch placeholder.
    """
    self._epoch_placeholder = array_ops.placeholder(dtypes.int32, [1],
                                                    name='epoch')
    return self._epoch_placeholder

  def input_builder(self):
    """Builds inputs in the graph.

    Returns:
      Two placeholders for inputs and outputs.
    """
    input_shape = [None] + self.input_shape[1:]
    self._input_placeholder = array_ops.placeholder(
        dtypes.as_dtype(self._input_dtype),
        input_shape,
        name='input')
    if self.output_shape is None:
      self._output_placeholder = None
    else:
      output_shape = [None] + self.output_shape[1:]
      self._output_placeholder = array_ops.placeholder(
          dtypes.as_dtype(self._output_dtype),
          output_shape,
          name='output')
    return self._input_placeholder, self._output_placeholder

  def set_placeholders(self, input_placeholder, output_placeholder):
    """Sets placeholders for this data feeder.

    Args:
      input_placeholder: Placeholder for `x` variable. Should match shape
        of the examples in the x dataset.
      output_placeholder: Placeholder for `y` variable. Should match
        shape of the examples in the y dataset. Can be None.
    """
    self._input_placeholder = input_placeholder
    self._output_placeholder = output_placeholder

  def get_feed_params(self):
    """Function returns a dict with data feed params while training.

    Returns:
      A dict with data feed params while training.
    """
    return {
        'epoch': self.epoch,
        'offset': self.offset,
        'batch_size': self._batch_size
    }

  def get_feed_dict_fn(self):
    """Returns a function that samples data into given placeholders.

    Returns:
      A function that when called samples a random subset of batch size
      from x and y.
    """
    def _feed_dict_fn():
      """Function that samples data into given placeholders."""
      if self.max_epochs is not None and self.epoch + 1 > self.max_epochs:
        raise StopIteration
      assert self._input_placeholder is not None
      feed_dict = {}
      if self._epoch_placeholder is not None:
        feed_dict[self._epoch_placeholder.name] = [self.epoch]

      # Take next batch of indices.
      end = min(self._x.shape[0], self.offset + self._batch_size)
      batch_indices = self.indices[self.offset:end]

      # Assign input features from random indices.
      inp = (
          np.array(_access(self._x, batch_indices)).reshape(
              (batch_indices.shape[0], 1))
          if len(self._x.shape) == 1 else _access(self._x, batch_indices))
      feed_dict[self._input_placeholder.name] = inp

      # move offset and reset it if necessary
      self.offset += self._batch_size
      if self.offset >= self._x.shape[0]:
        self.indices = self.random_state.permutation(self._x.shape[0])
        self.offset = 0
        self.epoch += 1

      # return early if there are no labels
      if self._output_placeholder is None:
        return feed_dict

      # assign labels from random indices
      self.output_shape[0] = batch_indices.shape[0]
      out = np.zeros(self.output_shape, dtype=self._output_dtype)
      for i in xrange(out.shape[0]):
        sample = batch_indices[i]
        # self.n_classes is None means we're passing in raw target indices
        if self.n_classes is None:
          out[i] = _access(self._y, sample)
        else:
          if self.n_classes > 1:
            if len(self.output_shape) == 2:
              out.itemset((i, int(_access(self._y, sample))), 1.0)
            else:
              for idx, value in enumerate(_access(self._y, sample)):
                out.itemset(tuple([i, idx, value]), 1.0)
          else:
            out[i] = _access(self._y, sample)
      feed_dict[self._output_placeholder.name] = out

      return feed_dict

    return _feed_dict_fn


class StreamingDataFeeder(DataFeeder):
  """Data feeder for TF trainer that reads data from iterator.

  Streaming data feeder allows to read data as it comes it from disk or
  somewhere else. It's custom to have this iterators rotate infinetly over
  the dataset, to allow control of how much to learn on the trainer side.
  """

  def __init__(self, x, y, n_classes, batch_size):
    """Initializes a StreamingDataFeeder instance.

    Args:
      x: iterator that returns for each element, returns features.
      y: iterator that returns for each element, returns 1 or many classes /
         regression values.
      n_classes: indicator of how many classes the target has.
      batch_size: Mini batch size to accumulate.

    Attributes:
      x: input features.
      y: input target.
      n_classes: number of classes.
      batch_size: mini batch size to accumulate.
      input_shape: shape of the input.
      output_shape: shape of the output.
      input_dtype: dtype of input.
      output_dtype: dtype of output.
    """
    # pylint: disable=invalid-name,super-init-not-called
    x_first_el = six.next(x)
    self._x = itertools.chain([x_first_el], x)
    if y is not None:
      y_first_el = six.next(y)
      self._y = itertools.chain([y_first_el], y)
    else:
      y_first_el = None
      self._y = None
    self.n_classes = n_classes
    self.input_shape, self.output_shape, self._batch_size = _get_in_out_shape(
        [1] + list(x_first_el.shape),
        [1] + list(y_first_el.shape) if y is not None else None,
        n_classes,
        batch_size)
    self._input_dtype = _check_dtype(x_first_el.dtype)
    # Output types are floats, due to both softmaxes and regression req.
    if n_classes is not None and n_classes > 0:
      self._output_dtype = np.float32
    elif y is not None:
      if isinstance(y_first_el, list) or isinstance(y_first_el, np.ndarray):
        self._output_dtype = _check_dtype(np.dtype(type(y_first_el[0])))
      else:
        self._output_dtype = _check_dtype(np.dtype(type(y_first_el)))

  def get_feed_params(self):
    """Function returns a dict with data feed params while training.

    Returns:
      A dict with data feed params while training.
    """
    return {'batch_size': self._batch_size}

  def get_feed_dict_fn(self):
    """Returns a function, that will sample data and provide it to placeholders.

    Returns:
      A function that when called samples a random subset of batch size
      from x and y.
    """
    self.stopped = False

    def _feed_dict_fn():
      """Samples data and provides it to placeholders.

      Returns:
        Dict of input and output tensors.
      """
      if self.stopped:
        raise StopIteration
      inp = np.zeros(self.input_shape, dtype=self._input_dtype)
      if self._y is not None:
        out = np.zeros(self.output_shape, dtype=self._output_dtype)
      for i in xrange(self._batch_size):
        # Add handling when queue ends.
        try:
          inp[i, :] = six.next(self._x)
        except StopIteration:
          self.stopped = True
          inp = inp[:i, :]
          if self._y is not None:
            out = out[:i]
          break

        if self._y is not None:
          y = six.next(self._y)
          if self.n_classes is not None and self.n_classes > 1:
            if len(self.output_shape) == 2:
              out.itemset((i, y), 1.0)
            else:
              for idx, value in enumerate(y):
                out.itemset(tuple([i, idx, value]), 1.0)
          else:
            out[i] = y
      if self._y is None:
        return {self._input_placeholder.name: inp}
      return {self._input_placeholder.name: inp,
              self._output_placeholder.name: out}

    return _feed_dict_fn


class DaskDataFeeder(object):
  """Data feeder for that reads data from dask.Series and dask.DataFrame.

  Numpy arrays can be serialized to disk and it's possible to do random seeks
  into them. DaskDataFeeder will remove requirement to have full dataset in the
  memory and still do random seeks for sampling of batches.
  """

  def __init__(self, x, y, n_classes, batch_size, shuffle=True,
               random_state=None, epochs=None):
    """Initializes a DaskDataFeeder instance.

    Args:
      x: iterator that returns for each element, returns features.
      y: iterator that returns for each element, returns 1 or many classes /
        regression values.
      n_classes: indicator of how many classes the target has.
      batch_size: Mini batch size to accumulate.
      shuffle: Whether to shuffle the inputs.
      random_state: random state for RNG. Note that it will mutate so use a
        int value for this if you want consistent sized batches.
      epochs: Number of epochs to run.

    Attributes:
      x: input features.
      y: input target.
      n_classes: number of classes.
      batch_size: mini batch size to accumulate.
      input_shape: shape of the input.
      output_shape: shape of the output.
      input_dtype: dtype of input.
      output_dtype: dtype of output.
    """
    # pylint: disable=invalid-name,super-init-not-called
    import dask.dataframe as dd  # pylint: disable=g-import-not-at-top
    # TODO(terrytangyuan): check x and y dtypes in dask_io like pandas
    self._x = x
    self._y = y
    # save column names
    self._x_columns = list(x.columns)
    if isinstance(y.columns[0], str):
      self._y_columns = list(y.columns)
    else:
      # deal with cases where two DFs have overlapped default numeric colnames
      self._y_columns = len(self._x_columns) + 1
      self._y = self._y.rename(columns={y.columns[0]: self._y_columns})

    # TODO(terrytangyuan): deal with unsupervised cases
    # combine into a data frame
    self.df = dd.multi.concat([self._x, self._y], axis=1)
    self.n_classes = n_classes

    x_count = x.count().compute()[0]
    x_shape = (x_count, len(self._x.columns))
    y_shape = (x_count, len(self._y.columns))
    # TODO(terrytangyuan): Add support for shuffle and epochs.
    self._shuffle = shuffle
    self.epochs = epochs
    self.input_shape, self.output_shape, self._batch_size = _get_in_out_shape(
        x_shape, y_shape, n_classes, batch_size)
    self.sample_fraction = self._batch_size / float(x_count)
    self._input_dtype = _check_dtype(self._x.dtypes[0])
    self._output_dtype = _check_dtype(self._y.dtypes[self._y_columns])
    if random_state is None:
      self.random_state = 66
    else:
      self.random_state = random_state

  def get_feed_params(self):
    """Function returns a dict with data feed params while training.

    Returns:
      A dict with data feed params while training.
    """
    return {'batch_size': self._batch_size}

  def get_feed_dict_fn(self, input_placeholder, output_placeholder):
    """Returns a function, that will sample data and provide it to placeholders.

    Args:
      input_placeholder: tf.Placeholder for input features mini batch.
      output_placeholder: tf.Placeholder for output targets.

    Returns:
      A function that when called samples a random subset of batch size
      from x and y.
    """
    def _feed_dict_fn():
      """Samples data and provides it to placeholders."""
      # TODO(ipolosukhin): option for with/without replacement (dev version of
      # dask)
      sample = self.df.random_split(
          [self.sample_fraction, 1 - self.sample_fraction],
          random_state=self.random_state)
      inp = extract_pandas_matrix(sample[0][self._x_columns].compute()).tolist()
      out = extract_pandas_matrix(sample[0][self._y_columns].compute())
      # convert to correct dtype
      inp = np.array(inp, dtype=self._input_dtype)
      # one-hot encode out for each class for cross entropy loss
      if HAS_PANDAS:
        import pandas as pd  # pylint: disable=g-import-not-at-top
        if not isinstance(out, pd.Series):
          out = out.flatten()
      out_max = self._y.max().compute().values[0]
      encoded_out = np.zeros((out.size, out_max + 1), dtype=self._output_dtype)
      encoded_out[np.arange(out.size), out] = 1
      return {input_placeholder.name: inp,
              output_placeholder.name: encoded_out}
    return _feed_dict_fn
