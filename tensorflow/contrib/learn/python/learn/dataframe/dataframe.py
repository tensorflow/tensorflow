"""A DataFrame is a container for ingesting and preprocessing data."""
# Copyright 2016 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
import collections

from .column import Column
from .transform import Transform


class DataFrame(object):
  """A DataFrame is a container for ingesting and preprocessing data."""
  __metaclass__ = ABCMeta

  def __init__(self):
    self._columns = {}

  def columns(self):
    """Set of the column names."""
    return frozenset(self._columns.keys())

  def __len__(self):
    """The number of columns in the DataFrame."""
    return len(self._columns)

  def assign(self, **kwargs):
    """Adds columns to DataFrame.

    Args:
      **kwargs: assignments of the form key=value where key is a string
      and value is an `inflow.Series`, a `pandas.Series` or a numpy array.

    Raises:
      TypeError: keys are not strings.
      TypeError: values are not `inflow.Series`, `pandas.Series` or
      `numpy.ndarray`.

    TODO(jamieas): pandas assign method returns a new DataFrame. Consider
    switching to this behavior, changing the name or adding in_place as an
    argument.
    """
    for k, v in kwargs.items():
      if not isinstance(k, str):
        raise TypeError("The only supported type for keys is string; got %s" %
                        type(k))
      if isinstance(v, Column):
        s = v
      elif isinstance(v, Transform) and v.input_valency() == 0:
        s = v()
      # TODO(jamieas): hook up these special cases again
      # TODO(soergel): can these special cases be generalized?
      # elif isinstance(v, pd.Series):
      #   s = series.NumpySeries(v.values)
      # elif isinstance(v, np.ndarray):
      #   s = series.NumpySeries(v)
      else:
        raise TypeError(
            "Column in assignment must be an inflow.Column, pandas.Series or a"
            " numpy array; got type '%s'." % type(v).__name__)
      self._columns[k] = s

  def select(self, keys):
    """Returns a new DataFrame with a subset of columns.

    Args:
      keys: A list of strings. Each should be the name of a column in the
        DataFrame.
    Returns:
      A new DataFrame containing only the specified columns.
    """
    result = type(self)()
    for key in keys:
      result[key] = self._columns[key]
    return result

  def __getitem__(self, key):
    """Indexing functionality for DataFrames.

    Args:
      key: a string or an iterable of strings.

    Returns:
      A Series or list of Series corresponding to the given keys.
    """
    if isinstance(key, str):
      return self._columns[key]
    elif isinstance(key, collections.Iterable):
      for i in key:
        if not isinstance(i, str):
          raise TypeError("Expected a String; entry %s has type %s." %
                          (i, type(i).__name__))
      return [self.__getitem__(i) for i in key]
    raise TypeError(
        "Invalid index: %s of type %s. Only strings or lists of strings are "
        "supported." % (key, type(key)))

  def __setitem__(self, key, value):
    if isinstance(key, str):
      key = [key]
    if isinstance(value, Column):
      value = [value]
    self.assign(**dict(zip(key, value)))

  def build(self, cache=None):
    if cache is None:
      cache = {}
    tensors = {name: c.build(cache) for name, c in self._columns.items()}
    return tensors
