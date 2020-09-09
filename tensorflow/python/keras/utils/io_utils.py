# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-import-not-at-top
"""Utilities related to disk I/O."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import numpy as np
import six
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export

try:
  import h5py
except ImportError:
  h5py = None


if sys.version_info >= (3, 6):

  def _path_to_string(path):
    if isinstance(path, os.PathLike):
      return os.fspath(path)
    return path
elif sys.version_info >= (3, 4):

  def _path_to_string(path):
    import pathlib
    if isinstance(path, pathlib.Path):
      return str(path)
    return path
else:

  def _path_to_string(path):
    return path


def path_to_string(path):
  """Convert `PathLike` objects to their string representation.

  If given a non-string typed path object, converts it to its string
  representation. Depending on the python version used, this function
  can handle the following arguments:
  python >= 3.6: Everything supporting the fs path protocol
    https://www.python.org/dev/peps/pep-0519
  python >= 3.4: Only `pathlib.Path` objects

  If the object passed to `path` is not among the above, then it is
  returned unchanged. This allows e.g. passthrough of file objects
  through this function.

  Args:
    path: `PathLike` object that represents a path

  Returns:
    A string representation of the path argument, if Python support exists.
  """
  return _path_to_string(path)


@keras_export('keras.utils.HDF5Matrix')
class HDF5Matrix(object):
  """Representation of HDF5 dataset to be used instead of a Numpy array.

  THIS CLASS IS DEPRECATED.
  Training with HDF5Matrix may not be optimized for performance, and might
  not work with every distribution strategy.

  We recommend using https://github.com/tensorflow/io to load your
  HDF5 data into a tf.data Dataset and passing that dataset to Keras.
  """
  refs = collections.defaultdict(int)

  @deprecation.deprecated('2020-05-30', 'Training with '
                          'HDF5Matrix is not optimized for performance. '
                          'Instead, we recommend using '
                          'https://github.com/tensorflow/io to load your '
                          'HDF5 data into a tf.data Dataset and passing '
                          'that dataset to Keras.')
  def __init__(self, datapath, dataset, start=0, end=None, normalizer=None):
    """Representation of HDF5 dataset to be used instead of a Numpy array.

    Example:

    ```python
        x_data = HDF5Matrix('input/file.hdf5', 'data')
        model.predict(x_data)
    ```

    Providing `start` and `end` allows use of a slice of the dataset.

    Optionally, a normalizer function (or lambda) can be given. This will
    be called on every slice of data retrieved.

    Arguments:
        datapath: string, path to a HDF5 file
        dataset: string, name of the HDF5 dataset in the file specified
            in datapath
        start: int, start of desired slice of the specified dataset
        end: int, end of desired slice of the specified dataset
        normalizer: function to be called on data when retrieved

    Returns:
        An array-like HDF5 dataset.

    Raises:
      ImportError if HDF5 & h5py are not installed
    """
    if h5py is None:
      raise ImportError('The use of HDF5Matrix requires '
                        'HDF5 and h5py installed.')

    if datapath not in list(self.refs.keys()):
      f = h5py.File(datapath)
      self.refs[datapath] = f
    else:
      f = self.refs[datapath]
    self.data = f[dataset]
    self.start = start
    if end is None:
      self.end = self.data.shape[0]
    else:
      self.end = end
    self.normalizer = normalizer

  def __len__(self):
    return self.end - self.start

  def __getitem__(self, key):
    if isinstance(key, slice):
      start, stop = key.start, key.stop
      if start is None:
        start = 0
      if stop is None:
        stop = self.shape[0]
      if stop + self.start <= self.end:
        idx = slice(start + self.start, stop + self.start)
      else:
        raise IndexError
    elif isinstance(key, (int, np.integer)):
      if key + self.start < self.end:
        idx = key + self.start
      else:
        raise IndexError
    elif isinstance(key, np.ndarray):
      if np.max(key) + self.start < self.end:
        idx = (self.start + key).tolist()
      else:
        raise IndexError
    else:
      # Assume list/iterable
      if max(key) + self.start < self.end:
        idx = [x + self.start for x in key]
      else:
        raise IndexError
    if self.normalizer is not None:
      return self.normalizer(self.data[idx])
    else:
      return self.data[idx]

  @property
  def shape(self):
    """Gets a numpy-style shape tuple giving the dataset dimensions.

    Returns:
        A numpy-style shape tuple.
    """
    return (self.end - self.start,) + self.data.shape[1:]

  @property
  def dtype(self):
    """Gets the datatype of the dataset.

    Returns:
        A numpy dtype string.
    """
    return self.data.dtype

  @property
  def ndim(self):
    """Gets the number of dimensions (rank) of the dataset.

    Returns:
        An integer denoting the number of dimensions (rank) of the dataset.
    """
    return self.data.ndim

  @property
  def size(self):
    """Gets the total dataset size (number of elements).

    Returns:
        An integer denoting the number of elements in the dataset.
    """
    return np.prod(self.shape)

  @staticmethod
  def _to_type_spec(value):
    """Gets the Tensorflow TypeSpec corresponding to the passed dataset.

    Args:
      value: A HDF5Matrix object.

    Returns:
      A tf.TensorSpec.
    """
    if not isinstance(value, HDF5Matrix):
      raise TypeError('Expected value to be a HDF5Matrix, but saw: {}'.format(
          type(value)))
    return tensor_spec.TensorSpec(shape=value.shape, dtype=value.dtype)


type_spec.register_type_spec_from_value_converter(HDF5Matrix,
                                                  HDF5Matrix._to_type_spec)  # pylint: disable=protected-access


def ask_to_proceed_with_overwrite(filepath):
  """Produces a prompt asking about overwriting a file.

  Arguments:
      filepath: the path to the file to be overwritten.

  Returns:
      True if we can proceed with overwrite, False otherwise.
  """
  overwrite = six.moves.input('[WARNING] %s already exists - overwrite? '
                              '[y/n]' % (filepath)).strip().lower()
  while overwrite not in ('y', 'n'):
    overwrite = six.moves.input('Enter "y" (overwrite) or "n" '
                                '(cancel).').strip().lower()
  if overwrite == 'n':
    return False
  print('[TIP] Next time specify overwrite=True!')
  return True
