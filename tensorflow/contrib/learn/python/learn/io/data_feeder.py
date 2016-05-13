"""Implementations of different data feeders to provide data for TF trainer."""

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

import itertools
import math

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

from .pandas_io import HAS_PANDAS, extract_pandas_data, extract_pandas_matrix, extract_pandas_labels
from .dask_io import HAS_DASK, extract_dask_data, extract_dask_labels


def _get_in_out_shape(x_shape, y_shape, n_classes, batch_size):
  """Returns shape for input and output of the data feeder."""
  x_shape = list(x_shape[1:]) if len(x_shape) > 1 else [1]
  input_shape = [batch_size] + x_shape
  if y_shape is None:
    return input_shape, None
  y_shape = list(y_shape[1:]) if len(y_shape) > 1 else []
  # Skip first dimension if it is 1.
  if y_shape and y_shape[0] == 1:
    y_shape = y_shape[1:]
  if n_classes > 1:
    output_shape = [batch_size] + y_shape + [n_classes]
  else:
    output_shape = [batch_size] + y_shape
  return input_shape, output_shape


def _data_type_filter(X, y):
  """Filter data types into acceptable format"""
  if HAS_DASK:
    X = extract_dask_data(X)
    if y is not None:
      y = extract_dask_labels(y)
  if HAS_PANDAS:
    X = extract_pandas_data(X)
    if y is not None:
      y = extract_pandas_labels(y)
  return X, y


def _is_iterable(X):
  return hasattr(X, 'next') or hasattr(X, '__next__')


def setup_train_data_feeder(X, y, n_classes, batch_size):
  """Create data feeder, to sample inputs from dataset.

  If X and y are iterators, use StreamingDataFeeder.

  Args:
    X: numpy, pandas or Dask matrix or iterable.
    y: numpy, pandas or Dask array or iterable.
    n_classes: number of classes.
    batch_size: size to split data into parts.

  Returns:
    DataFeeder object that returns training data.
  """
  X, y = _data_type_filter(X, y)
  if HAS_DASK:
    import dask.dataframe as dd
    if (isinstance(X, (dd.Series, dd.DataFrame)) and
        (y is None or isinstance(y, (dd.Series, dd.DataFrame)))):
      data_feeder_cls = DaskDataFeeder
    else:
      data_feeder_cls = DataFeeder
  else:
    data_feeder_cls = DataFeeder

  if _is_iterable(X):
    if y is not None and not _is_iterable(y):
      raise ValueError('Both X and y should be iterators for '
                       'streaming learning to work.')
    data_feeder_cls = StreamingDataFeeder
  return data_feeder_cls(X, y, n_classes, batch_size)


def _batch_data(X, batch_size):
  chunk = []
  for data in X:
    chunk.append(data)
    if batch_size > 0 and len(chunk) >= batch_size:
      yield np.matrix(chunk)
      chunk = []
  yield np.matrix(chunk)


def setup_predict_data_feeder(X, batch_size=-1):
  """Returns an iterable for feeding into predict step.

  Args:
    X: numpy, pandas, Dask array or iterable.
    batch_size: Size of batches to split data into.
      If negative, returns one batch of full size.

  Returns:
    List or iterator of parts of data to predict on.
  """
  if HAS_DASK:
    X = extract_dask_data(X)
  if HAS_PANDAS:
    X = extract_pandas_data(X)
  if _is_iterable(X):
    return _batch_data(X, batch_size)
  if len(X.shape) == 1:
    X = np.reshape(X, (-1, 1))
  if batch_size > 0:
    n_batches = int(math.ceil(float(len(X)) / batch_size))
    return [X[i * batch_size:(i + 1) * batch_size] for i in xrange(n_batches)]
  return [X]


def setup_processor_data_feeder(X):
  """Sets up processor iterable.

  Args:
    X: numpy, pandas or iterable.

  Returns:
    Iterable of data to process.
  """
  if HAS_PANDAS:
    X = extract_pandas_matrix(X)
  return X


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


class DataFeeder(object):
  """Data feeder is an example class to sample data for TF trainer.

  Parameters:
    X: feature Nd numpy matrix of shape [n_samples, n_features, ...].
    y: target vector, either floats for regression or class id for
      classification. If matrix, will consider as a sequence
      of targets. Can be None for unsupervised setting.
    n_classes: number of classes, 0 and 1 are considered regression, None will
      pass through the input labels without one-hot conversion.
    batch_size: mini batch size to accumulate.
    random_state: numpy RandomState object to reproduce sampling.

  Attributes:
    X: input features.
    y: input target.
    n_classes: number of classes (if None, pass through indices without
      one-hot conversion).
    batch_size: mini batch size to accumulate.
    input_shape: shape of the input.
    output_shape: shape of the output.
    input_dtype: dtype of input.
    output_dtype: dtype of output.
  """

  def __init__(self, X, y, n_classes, batch_size, random_state=None):
    x_dtype = np.int64 if X.dtype == np.int64 else np.float32
    y_dtype = (
        np.int64 if n_classes is not None and n_classes > 1 else np.float32)
    self.X = check_array(X, dtype=x_dtype)
    # self.n_classes is None means we're passing in raw target indices
    if n_classes is not None:
      self.y = (None if y is None else check_array(y, dtype=y_dtype))
    else:
      self.y = y
    self.n_classes = n_classes
    self.batch_size = batch_size
    self.input_shape, self.output_shape = _get_in_out_shape(self.X.shape, None
                                                            if self.y is None
                                                            else self.y.shape,
                                                            n_classes,
                                                            batch_size)
    # Input dtype matches dtype of X.
    self.input_dtype = self.X.dtype
    # self.n_classes is None means we're passing in raw target indices
    if n_classes is not None or y is None:
      self.output_dtype = np.float32
    else:
      self.output_dtype = y.dtype
    self.random_state = np.random.RandomState(
        42) if random_state is None else random_state
    self.indices = self.random_state.permutation(self.X.shape[0])
    self.offset = 0
    self.epoch = 0
    self._epoch_placeholder = None

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
        dtypes.as_dtype(self.input_dtype),
        input_shape,
        name='input')
    if self.output_shape is None:
      self._output_placeholder = None
    else:
      output_shape = [None] + self.output_shape[1:]
      self._output_placeholder = array_ops.placeholder(
          dtypes.as_dtype(self.output_dtype),
          output_shape,
          name='output')
    return self._input_placeholder, self._output_placeholder

  def set_placeholders(self, input_placeholder, output_placeholder):
    """Sets placeholders for this data feeder.

    Args:
      input_placeholder: Placeholder for `X` variable. Should match shape
        of the examples in the X dataset.
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
        'batch_size': self.batch_size
    }

  def get_feed_dict_fn(self):
    """Returns a function that samples data into given placeholders.

    Returns:
      A function that when called samples a random subset of batch size
      from X and y.
    """
    def _feed_dict_fn():
      assert self._input_placeholder != None
      feed_dict = {}
      if self._epoch_placeholder is not None:
        feed_dict[self._epoch_placeholder.name] = [self.epoch]

      # take random indices
      if self.batch_size < 0:
        batch_indices = self.indices
      else:
        batch_indices = self.indices[self.offset:self.offset + self.batch_size]

      # assign input features from random indices
      inp = np.array(self.X[batch_indices]).reshape((batch_indices.shape[0], 1)) \
          if len(self.X.shape) == 1 else self.X[batch_indices]
      feed_dict[self._input_placeholder.name] = inp

      # move offset and reset it if necessary
      self.offset += self.batch_size
      if self.offset >= self.X.shape[0]:
        self.indices = self.random_state.permutation(self.X.shape[0])
        self.offset = 0
        self.epoch += 1

      # return early if there are no labels
      if self._output_placeholder is None:
        return feed_dict

      # assign labels from random indices
      self.output_shape[0] = batch_indices.shape[0]
      out = np.zeros(self.output_shape, dtype=self.output_dtype)
      for i in xrange(out.shape[0]):
        sample = batch_indices[i]
        # self.n_classes is None means we're passing in raw target indices
        if self.n_classes is None:
          out[i] = self.y[sample]
        else:
          if self.n_classes > 1:
            if len(self.output_shape) == 2:
              out.itemset((i, self.y[sample]), 1.0)
            else:
              for idx, value in enumerate(self.y[sample]):
                out.itemset(tuple([i, idx, value]), 1.0)
          else:
            out[i] = self.y[sample]
      feed_dict[self._output_placeholder.name] = out

      return feed_dict

    return _feed_dict_fn


class StreamingDataFeeder(DataFeeder):
  """Data feeder for TF trainer that reads data from iterator.

  Streaming data feeder allows to read data as it comes it from disk or
  somewhere else. It's custom to have this iterators rotate infinetly over
  the dataset, to allow control of how much to learn on the trainer side.

  Parameters:
    X: iterator that returns for each element, returns features.
    y: iterator that returns for each element, returns 1 or many classes /
       regression values.
    n_classes: indicator of how many classes the target has.
    batch_size: Mini batch size to accumulate.

  Attributes:
    X: input features.
    y: input target.
    n_classes: number of classes.
    batch_size: mini batch size to accumulate.
    input_shape: shape of the input.
    output_shape: shape of the output.
    input_dtype: dtype of input.
    output_dtype: dtype of output.
  """

  def __init__(self, X, y, n_classes, batch_size):
    X_first_el = six.next(X)
    y_first_el = six.next(y)
    self.X = itertools.chain([X_first_el], X)
    self.y = itertools.chain([y_first_el], y)
    self.n_classes = n_classes
    self.batch_size = batch_size
    self.input_shape, self.output_shape = _get_in_out_shape(
        [1] + list(X_first_el.shape), [1] + list(y_first_el.shape), n_classes,
        batch_size)
    self.input_dtype = X_first_el.dtype
    # Convert float64 to float32, as all the parameters in the model are
    # floats32 and there is a lot of benefits in using it in NNs.
    if self.input_dtype == np.float64:
      self.input_dtype = np.float32
    # Output types are floats, due to both softmaxes and regression req.
    self.output_dtype = np.float32

  def get_feed_params(self):
    """Function returns a dict with data feed params while training.

    Returns:
      A dict with data feed params while training.
    """
    return {'batch_size': self.batch_size}

  def get_feed_dict_fn(self):
    """Returns a function, that will sample data and provide it to placeholders.

    Args:
      input_placeholder: tf.Placeholder for input features mini batch.
      output_placeholder: tf.Placeholder for output targets.

    Returns:
      A function that when called samples a random subset of batch size
      from X and y.
    """

    def _feed_dict_fn():
      inp = np.zeros(self.input_shape, dtype=self.input_dtype)
      out = np.zeros(self.output_shape, dtype=self.output_dtype)
      for i in xrange(self.batch_size):
        inp[i, :] = six.next(self.X)
        y = six.next(self.y)
        if self.n_classes > 1:
          if len(self.output_shape) == 2:
            out.itemset((i, y), 1.0)
          else:
            for idx, value in enumerate(y):
              out.itemset(tuple([i, idx, value]), 1.0)
        else:
          out[i] = y
      return {self._input_placeholder.name: inp,
              self._output_placeholder.name: out}

    return _feed_dict_fn


class DaskDataFeeder(object):
  """Data feeder for TF trainer that reads data from dask.Series and dask.DataFrame.

  Numpy arrays can be serialized to disk and it's possible to do random seeks
  into them. DaskDataFeeder will remove requirement to have full dataset in the 
  memory and still do random seeks for sampling of batches.

  Parameters:
    X: iterator that returns for each element, returns features.
    y: iterator that returns for each element, returns 1 or many classes /
      regression values.
    n_classes: indicator of how many classes the target has.
    batch_size: Mini batch size to accumulate.
    random_state: random state for RNG. Note that it will mutate so use a
      int value for this if you want consistent sized batches.

  Attributes:
    X: input features.
    y: input target.
    n_classes: number of classes.
    batch_size: mini batch size to accumulate.
    input_shape: shape of the input.
    output_shape: shape of the output.
    input_dtype: dtype of input.
    output_dtype: dtype of output.
  """
  def __init__(self, X, y, n_classes, batch_size, random_state=None):
    import dask.dataframe as dd
    # TODO(terrytangyuan): check X and y dtypes in dask_io like pandas
    self.X = X
    self.y = y
    # save column names
    self.X_columns = list(X.columns)
    if isinstance(y.columns[0], str):
      self.y_columns = list(y.columns)
    else:
      # deal with cases where two DFs have overlapped default numeric colnames
      self.y_columns = len(self.X_columns) + 1
      self.y = self.y.rename(columns={y.columns[0]: self.y_columns})

    # TODO(terrytangyuan): deal with unsupervised cases
    # combine into a data frame
    self.df = dd.multi.concat([self.X, self.y], axis=1)
    self.n_classes = n_classes

    X_count = X.count().compute()[0]
    X_shape = (X_count, len(self.X.columns))
    y_shape = (X_count, len(self.y.columns))
    self.sample_fraction = batch_size / float(X_count)
    self.input_shape, self.output_shape = _get_in_out_shape(X_shape, y_shape,
                                                            n_classes,
                                                            batch_size)
    # self.X.dtypes[0], self.y.dtypes[self.y_columns]
    self.input_dtype, self.output_dtype = np.float32, np.float32
    if random_state is None:
      self.random_state = 66
    else:
      self.random_state = random_state
    self.batch_size = batch_size

  def get_feed_params(self):
    """Function returns a dict with data feed params while training.

    Returns:
      A dict with data feed params while training.
    """
    return {'batch_size': self.batch_size}

  def get_feed_dict_fn(self, input_placeholder, output_placeholder):
    """Returns a function, that will sample data and provide it to placeholders.

    Args:
      input_placeholder: tf.Placeholder for input features mini batch.
      output_placeholder: tf.Placeholder for output targets.

    Returns:
      A function that when called samples a random subset of batch size
      from X and y.
    """
    def _feed_dict_fn():
      # TODO: option for with/without replacement (dev version of dask)
      sample = self.df.random_split(
          [self.sample_fraction, 1 - self.sample_fraction],
          random_state=self.random_state)
      inp = extract_pandas_matrix(sample[0][self.X_columns].compute()).tolist()
      out = extract_pandas_matrix(sample[0][self.y_columns].compute())
      # convert to correct dtype
      inp = np.array(inp, dtype=self.input_dtype)
      # one-hot encode out for each class for cross entropy loss
      if HAS_PANDAS:
        import pandas as pd
        if not isinstance(out, pd.Series):
          out = out.flatten()
      out_max = self.y.max().compute().values[0]
      encoded_out = np.zeros((out.size, out_max + 1), dtype=self.output_dtype)
      encoded_out[np.arange(out.size), out] = 1
      return {input_placeholder.name: inp,
              output_placeholder.name: encoded_out}
    return _feed_dict_fn
