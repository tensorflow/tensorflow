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
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging

# pylint: disable=g-multiple-import,g-bad-import-order
from .pandas_io import HAS_PANDAS, extract_pandas_data, extract_pandas_matrix, extract_pandas_labels
from .dask_io import HAS_DASK, extract_dask_data, extract_dask_labels
# pylint: enable=g-multiple-import,g-bad-import-order


def _get_in_out_shape(x_shape, y_shape, n_classes, batch_size=None):
  """Returns shape for input and output of the data feeder."""
  x_is_dict, y_is_dict = isinstance(
      x_shape, dict), y_shape is not None and isinstance(y_shape, dict)
  if y_is_dict and n_classes is not None:
    assert (isinstance(n_classes, dict))

  if batch_size is None:
    batch_size = list(x_shape.values())[0][0] if x_is_dict else x_shape[0]
  elif batch_size <= 0:
    raise ValueError('Invalid batch_size %d.' % batch_size)

  if x_is_dict:
    input_shape = {}
    for k, v in list(x_shape.items()):
      input_shape[k] = [batch_size] + (list(v[1:]) if len(v) > 1 else [1])
  else:
    x_shape = list(x_shape[1:]) if len(x_shape) > 1 else [1]
    input_shape = [batch_size] + x_shape

  if y_shape is None:
    return input_shape, None, batch_size

  def out_el_shape(out_shape, num_classes):
    out_shape = list(out_shape[1:]) if len(out_shape) > 1 else []
    # Skip first dimension if it is 1.
    if out_shape and out_shape[0] == 1:
      out_shape = out_shape[1:]
    if num_classes is not None and num_classes > 1:
      return [batch_size] + out_shape + [num_classes]
    else:
      return [batch_size] + out_shape

  if not y_is_dict:
    output_shape = out_el_shape(y_shape, n_classes)
  else:
    output_shape = dict([
        (k, out_el_shape(v, n_classes[k]
                         if n_classes is not None and k in n_classes else None))
        for k, v in list(y_shape.items())
    ])

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


def setup_train_data_feeder(x,
                            y,
                            n_classes,
                            batch_size=None,
                            shuffle=True,
                            epochs=None):
  """Create data feeder, to sample inputs from dataset.

  If `x` and `y` are iterators, use `StreamingDataFeeder`.

  Args:
    x: numpy, pandas or Dask matrix or dictionary of aforementioned. Also
      supports iterables.
    y: numpy, pandas or Dask array or dictionary of aforementioned. Also
      supports
      iterables.
    n_classes: number of classes. Must be None or same type as y. In case, `y`
      is `dict`
      (or iterable which returns dict) such that `n_classes[key] = n_classes for
        y[key]`
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

  x_first_el = six.next(x)
  x = itertools.chain([x_first_el], x)

  chunk = dict([(k, []) for k in list(x_first_el.keys())]) if isinstance(
      x_first_el, dict) else []
  chunk_filled = False
  for data in x:
    if isinstance(data, dict):
      for k, v in list(data.items()):
        chunk[k].append(v)
        if (batch_size is not None) and (len(chunk[k]) >= batch_size):
          chunk[k] = np.matrix(chunk[k])
          chunk_filled = True
      if chunk_filled:
        yield chunk
        chunk = dict([(k, []) for k in list(x_first_el.keys())]) if isinstance(
            x_first_el, dict) else []
        chunk_filled = False
    else:
      chunk.append(data)
      if (batch_size is not None) and (len(chunk) >= batch_size):
        yield np.matrix(chunk)
        chunk = []

  if isinstance(x_first_el, dict):
    for k, v in list(data.items()):
      chunk[k] = np.matrix(chunk[k])
    yield chunk
  else:
    yield np.matrix(chunk)


def setup_predict_data_feeder(x, batch_size=None):
  """Returns an iterable for feeding into predict step.

  Args:
    x: numpy, pandas, Dask array or dictionary of aforementioned. Also supports
      iterable.
    batch_size: Size of batches to split data into. If `None`, returns one
      batch of full size.

  Returns:
    List or iterator (or dictionary thereof) of parts of data to predict on.

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

  def __init__(self,
               x,
               y,
               n_classes,
               batch_size=None,
               shuffle=True,
               random_state=None,
               epochs=None):
    """Initializes a DataFeeder instance.

    Args:
      x: One feature sample which can either Nd numpy matrix of shape
        `[n_samples, n_features, ...]` or dictionary of Nd numpy matrix.
      y: label vector, either floats for regression or class id for
        classification. If matrix, will consider as a sequence of labels.
        Can be `None` for unsupervised setting. Also supports dictionary of
        labels.
      n_classes: Number of classes, 0 and 1 are considered regression, `None`
        will pass through the input labels without one-hot conversion. Also, if
        `y` is `dict`, then `n_classes` must be `dict` such that
        `n_classes[key] = n_classes for label y[key]`, `None` otherwise.
      batch_size: Mini-batch size to accumulate samples in one mini batch.
      shuffle: Whether to shuffle `x`.
      random_state: Numpy `RandomState` object to reproduce sampling.
      epochs: Number of times to iterate over input data before raising
        `StopIteration` exception.

    Attributes:
      x: Input features (ndarray or dictionary of ndarrays).
      y: Input label (ndarray or dictionary of ndarrays).
      n_classes: Number of classes (if `None`, pass through indices without
        one-hot conversion).
      batch_size: Mini-batch size to accumulate.
      input_shape: Shape of the input (or dictionary of shapes).
      output_shape: Shape of the output (or dictionary of shapes).
      input_dtype: DType of input (or dictionary of shapes).
      output_dtype: DType of output (or dictionary of shapes.
    """
    x_is_dict, y_is_dict = isinstance(x, dict), y is not None and isinstance(
        y, dict)
    if isinstance(y, list):
      y = np.array(y)

    self._x = dict([(k, check_array(v, v.dtype)) for k, v in list(x.items())
                   ]) if x_is_dict else check_array(x, x.dtype)
    self._y = None if y is None else \
      dict([(k, check_array(v, v.dtype)) for k, v in list(y.items())]) if x_is_dict else check_array(y, y.dtype)

    # self.n_classes is not None means we're converting raw target indices to one-hot.
    if n_classes is not None:
      if not y_is_dict:
        y_dtype = (np.int64
                   if n_classes is not None and n_classes > 1 else np.float32)
        self._y = (None if y is None else check_array(y, dtype=y_dtype))

    self.n_classes = n_classes
    self.max_epochs = epochs

    x_shape = dict([(k, v.shape) for k, v in list(self._x.items())
                   ]) if x_is_dict else self._x.shape
    y_shape = dict([(k, v.shape) for k, v in list(self._y.items())
                   ]) if y_is_dict else None if y is None else self._y.shape

    self.input_shape, self.output_shape, self._batch_size = _get_in_out_shape(
        x_shape, y_shape, n_classes, batch_size)

    # Input dtype matches dtype of x.
    self._input_dtype = dict([(k, _check_dtype(v.dtype)) for k, v in list(self._x.items())]) if x_is_dict \
      else _check_dtype(self._x.dtype)

    # note: self._output_dtype = np.float32 when y is None
    self._output_dtype = dict([(k, _check_dtype(v.dtype)) for k, v in list(self._y.items())]) if y_is_dict \
      else _check_dtype(self._y.dtype) if y is not None else np.float32

    # self.n_classes is None means we're passing in raw target indices
    if n_classes is not None and y_is_dict:
      for key in list(n_classes.keys()):
        if key in self._output_dtype:
          self._output_dtype[key] = np.float32

    self._shuffle = shuffle
    self.random_state = np.random.RandomState(
        42) if random_state is None else random_state

    if x_is_dict:
      num_samples = list(self._x.values())[0].shape[0]
    elif tensor_util.is_tensor(self._x):
      num_samples = self._x.shape[0].value  # shape will be a Dimension, extract an int
    else:
      num_samples = self._x.shape[0]
      
    if self._shuffle:
      self.indices = self.random_state.permutation(num_samples)
    else:
      self.indices = np.array(range(num_samples))
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
    self._epoch_placeholder = array_ops.placeholder(
        dtypes.int32, [1], name='epoch')
    return self._epoch_placeholder

  def input_builder(self):
    """Builds inputs in the graph.

    Returns:
      Two placeholders for inputs and outputs.
    """

    def get_placeholder(shape, dtype, name_prepend):
      if shape is None:
        return None
      if isinstance(shape, dict):
        placeholder = {}
        for key in list(shape.keys()):
          placeholder[key] = array_ops.placeholder(
              dtypes.as_dtype(dtype[key]), [None] + shape[key][1:],
              name=name_prepend + '_' + key)
      else:
        placeholder = array_ops.placeholder(
            dtypes.as_dtype(dtype), [None] + shape[1:], name=name_prepend)
      return placeholder

    self._input_placeholder = get_placeholder(self.input_shape,
                                              self._input_dtype, 'input')
    self._output_placeholder = get_placeholder(self.output_shape,
                                               self._output_dtype, 'output')
    return self._input_placeholder, self._output_placeholder

  def set_placeholders(self, input_placeholder, output_placeholder):
    """Sets placeholders for this data feeder.

    Args:
      input_placeholder: Placeholder for `x` variable. Should match shape
        of the examples in the x dataset.
      output_placeholder: Placeholder for `y` variable. Should match
        shape of the examples in the y dataset. Can be `None`.
    """
    self._input_placeholder = input_placeholder
    self._output_placeholder = output_placeholder

  def get_feed_params(self):
    """Function returns a `dict` with data feed params while training.

    Returns:
      A `dict` with data feed params while training.
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
      from `x` and `y`.
    """
    x_is_dict, y_is_dict = isinstance(
        self._x, dict), self._y is not None and isinstance(self._y, dict)

    # Assign input features from random indices.
    def extract(data, indices):
      return (np.array(_access(data, indices)).reshape((indices.shape[0], 1)) if
              len(data.shape) == 1 else _access(data, indices))

    # assign labels from random indices
    def assign_label(data, shape, dtype, n_classes, indices):
      shape[0] = indices.shape[0]
      out = np.zeros(shape, dtype=dtype)
      for i in xrange(out.shape[0]):
        sample = indices[i]
        # self.n_classes is None means we're passing in raw target indices
        if n_classes is None:
          out[i] = _access(data, sample)
        else:
          if n_classes > 1:
            if len(shape) == 2:
              out.itemset((i, int(_access(data, sample))), 1.0)
            else:
              for idx, value in enumerate(_access(data, sample)):
                out.itemset(tuple([i, idx, value]), 1.0)
          else:
            out[i] = _access(data, sample)
      return out

    def _feed_dict_fn():
      """Function that samples data into given placeholders."""
      if self.max_epochs is not None and self.epoch + 1 > self.max_epochs:
        raise StopIteration
      assert self._input_placeholder is not None
      feed_dict = {}
      if self._epoch_placeholder is not None:
        feed_dict[self._epoch_placeholder.name] = [self.epoch]

      # Take next batch of indices.
      x_len = list(self._x.values())[0].shape[
          0] if x_is_dict else self._x.shape[0]
      end = min(x_len, self.offset + self._batch_size)
      batch_indices = self.indices[self.offset:end]

      # adding input placeholder
      feed_dict.update(
          dict([(self._input_placeholder[k].name, extract(v, batch_indices))
                for k, v in list(self._x.items())]) if x_is_dict else
          {self._input_placeholder.name: extract(self._x, batch_indices)})

      # move offset and reset it if necessary
      self.offset += self._batch_size
      if self.offset >= x_len:
        self.indices = self.random_state.permutation(
            x_len) if self._shuffle else np.array(range(x_len))
        self.offset = 0
        self.epoch += 1

      # return early if there are no labels
      if self._output_placeholder is None:
        return feed_dict

      # adding output placeholders
      if y_is_dict:
        for k, v in list(self._y.items()):
          n_classes = (self.n_classes[k] if k in self.n_classes else
                       None) if self.n_classes is not None else None
          shape, dtype = self.output_shape[k], self._output_dtype[k]
          feed_dict.update({
              self._output_placeholder[k].name:
                  assign_label(v, shape, dtype, n_classes, batch_indices)
          })
      else:
        shape, dtype, n_classes = self.output_shape, self._output_dtype, self.n_classes
        feed_dict.update({
            self._output_placeholder.name:
                assign_label(self._y, shape, dtype, n_classes, batch_indices)
        })

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
      x: iterator each element of which returns one feature sample. Sample can
        be a Nd numpy matrix or dictionary of Nd numpy matrices.
      y: iterator each element of which returns one label sample. Sample can be
        a Nd numpy matrix or dictionary of Nd numpy matrices with 1 or many
        classes regression values.
      n_classes: indicator of how many classes the corresponding label sample
        has for the purposes of one-hot conversion of label. In case where `y`
        is a dictionary, `n_classes` must be dictionary (with same keys as `y`)
        of how many classes there are in each label in `y`. If key is
        present in `y` and missing in `n_classes`, the value is assumed `None`
        and no one-hot conversion will be applied to the label with that key.
      batch_size: Mini batch size to accumulate samples in one batch. If set
        `None`, then assumes that iterator to return already batched element.

    Attributes:
      x: input features (or dictionary of input features).
      y: input label (or dictionary of output features).
      n_classes: number of classes.
      batch_size: mini batch size to accumulate.
      input_shape: shape of the input (can be dictionary depending on `x`).
      output_shape: shape of the output (can be dictionary depending on `y`).
      input_dtype: dtype of input (can be dictionary depending on `x`).
      output_dtype: dtype of output (can be dictionary depending on `y`).
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

    x_is_dict = isinstance(x_first_el, dict)
    y_is_dict = y is not None and isinstance(y_first_el, dict)
    if y_is_dict and n_classes is not None:
      assert isinstance(n_classes, dict)

    # extract shapes for first_elements
    if x_is_dict:
      x_first_el_shape = dict(
          [(k, [1] + list(v.shape)) for k, v in list(x_first_el.items())])
    else:
      x_first_el_shape = [1] + list(x_first_el.shape)

    if y_is_dict:
      y_first_el_shape = dict(
          [(k, [1] + list(v.shape)) for k, v in list(y_first_el.items())])
    elif y is None:
      y_first_el_shape = None
    else:
      y_first_el_shape = ([1] + list(y_first_el[0].shape if isinstance(
          y_first_el, list) else y_first_el.shape))

    self.input_shape, self.output_shape, self._batch_size = _get_in_out_shape(
        x_first_el_shape, y_first_el_shape, n_classes, batch_size)

    # Input dtype of x_first_el.
    if x_is_dict:
      self._input_dtype = dict(
          [(k, _check_dtype(v.dtype)) for k, v in list(x_first_el.items())])
    else:
      self._input_dtype = _check_dtype(x_first_el.dtype)

    # Output dtype of y_first_el.
    def check_y_dtype(el):
      if isinstance(el, np.ndarray):
        return el.dtype
      elif isinstance(el, list):
        return check_y_dtype(el[0])
      else:
        return _check_dtype(np.dtype(type(el)))

    # Output types are floats, due to both softmaxes and regression req.
    if n_classes is not None and (y is None or not y_is_dict) and n_classes > 0:
      self._output_dtype = np.float32
    elif y_is_dict:
      self._output_dtype = dict(
          [(k, check_y_dtype(v)) for k, v in list(y_first_el.items())])
    elif y is None:
      self._output_dtype = None
    else:
      self._output_dtype = check_y_dtype(y_first_el)

  def get_feed_params(self):
    """Function returns a `dict` with data feed params while training.

    Returns:
      A `dict` with data feed params while training.
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
        `dict` of input and output tensors.
      """

      def init_array(shape, dtype):
        """Initialize array of given shape or dict of shapes and dtype."""
        if shape is None:
          return None
        elif isinstance(shape, dict):
          return dict([(k, np.zeros(shape[k], dtype[k]))
                       for k in list(shape.keys())])
        else:
          return np.zeros(shape, dtype=dtype)

      def put_data_array(dest, index, source=None, n_classes=None):
        """Puts data array into container."""
        if source is None:
          dest = dest[:index]
        elif n_classes is not None and n_classes > 1:
          if len(self.output_shape) == 2:
            dest.itemset((index, source), 1.0)
          else:
            for idx, value in enumerate(source):
              dest.itemset(tuple([index, idx, value]), 1.0)
        else:
          if len(dest.shape) > 1:
            dest[index, :] = source
          else:
            dest[index] = source[0] if isinstance(source, list) else source
        return dest

      def put_data_array_or_dict(holder, index, data=None, n_classes=None):
        """Puts data array or data dictionary into container."""
        if holder is None:
          return None
        if isinstance(holder, dict):
          if data is None:
            data = {k: None for k in holder.keys()}
          assert isinstance(data, dict)
          for k in holder.keys():
            num_classes = n_classes[k] if (n_classes is not None and
                                           k in n_classes) else None
            holder[k] = put_data_array(holder[k], index, data[k], num_classes)
        else:
          holder = put_data_array(holder, index, data, n_classes)
        return holder

      if self.stopped:
        raise StopIteration

      inp = init_array(self.input_shape, self._input_dtype)
      out = init_array(self.output_shape, self._output_dtype)

      for i in xrange(self._batch_size):
        # Add handling when queue ends.
        try:
          next_inp = six.next(self._x)
          inp = put_data_array_or_dict(inp, i, next_inp, None)
        except StopIteration:
          self.stopped = True
          if i == 0:
            raise
          inp = put_data_array_or_dict(inp, i, None, None)
          out = put_data_array_or_dict(out, i, None, None)
          break

        if self._y is not None:
          next_out = six.next(self._y)
          out = put_data_array_or_dict(out, i, next_out, self.n_classes)

      # creating feed_dict
      if isinstance(inp, dict):
        feed_dict = dict([(self._input_placeholder[k].name, inp[k])
                          for k in list(self._input_placeholder.keys())])
      else:
        feed_dict = {self._input_placeholder.name: inp}
      if self._y is not None:
        if isinstance(out, dict):
          feed_dict.update(
              dict([(self._output_placeholder[k].name, out[k])
                    for k in list(self._output_placeholder.keys())]))
        else:
          feed_dict.update({self._output_placeholder.name: out})

      return feed_dict

    return _feed_dict_fn


class DaskDataFeeder(object):
  """Data feeder for that reads data from dask.Series and dask.DataFrame.

  Numpy arrays can be serialized to disk and it's possible to do random seeks
  into them. DaskDataFeeder will remove requirement to have full dataset in the
  memory and still do random seeks for sampling of batches.
  """

  def __init__(self,
               x,
               y,
               n_classes,
               batch_size,
               shuffle=True,
               random_state=None,
               epochs=None):
    """Initializes a DaskDataFeeder instance.

    Args:
      x: iterator that returns for each element, returns features.
      y: iterator that returns for each element, returns 1 or many classes /
        regression values.
      n_classes: indicator of how many classes the label has.
      batch_size: Mini batch size to accumulate.
      shuffle: Whether to shuffle the inputs.
      random_state: random state for RNG. Note that it will mutate so use a
        int value for this if you want consistent sized batches.
      epochs: Number of epochs to run.

    Attributes:
      x: input features.
      y: input label.
      n_classes: number of classes.
      batch_size: mini batch size to accumulate.
      input_shape: shape of the input.
      output_shape: shape of the output.
      input_dtype: dtype of input.
      output_dtype: dtype of output.

    Raises:
      ValueError: if `x` or `y` are `dict`, as they are not supported currently.
    """

    if isinstance(x, dict) or isinstance(y, dict):
      raise ValueError(
          'DaskDataFeeder does not support dictionaries at the moment.')

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
    """Function returns a `dict` with data feed params while training.

    Returns:
      A `dict` with data feed params while training.
    """
    return {'batch_size': self._batch_size}

  def get_feed_dict_fn(self, input_placeholder, output_placeholder):
    """Returns a function, that will sample data and provide it to placeholders.

    Args:
      input_placeholder: tf.Placeholder for input features mini batch.
      output_placeholder: tf.Placeholder for output labels.

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
      return {input_placeholder.name: inp, output_placeholder.name: encoded_out}

    return _feed_dict_fn
