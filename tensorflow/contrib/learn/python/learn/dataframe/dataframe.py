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

"""A DataFrame is a container for ingesting and preprocessing data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from .series import Series
from .transform import Transform


class DataFrame(object):
  """A DataFrame is a container for ingesting and preprocessing data."""

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
      if v is None:
        del self._columns[k]
      elif isinstance(v, Series):
        self._columns[k] = v
      elif isinstance(v, Transform) and v.input_valency() == 0:
        self._columns[k] = v()
      else:
        raise TypeError(
            "Column in assignment must be an inflow.Series, inflow.Transform,"
            " or None; got type '%s'." % type(v).__name__)

  def select_columns(self, keys):
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

  def exclude_columns(self, exclude_keys):
    """Returns a new DataFrame with all columns not excluded via exclude_keys.

    Args:
      exclude_keys: A list of strings. Each should be the name of a column in
        the DataFrame.  These columns will be excluded from the result.
    Returns:
      A new DataFrame containing all columns except those specified.
    """
    result = type(self)()
    for key, value in self._columns.items():
      if key not in exclude_keys:
        result[key] = value
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
    if isinstance(value, Series):
      value = [value]
    self.assign(**dict(zip(key, value)))

  def __delitem__(self, key):
    if isinstance(key, str):
      key = [key]
    value = [None for _ in key]
    self.assign(**dict(zip(key, value)))

  def build(self, **kwargs):
    # We do not allow passing a cache here, because that would encourage
    # working around the rule that DataFrames cannot be expected to be
    # synced with each other (e.g., they shuffle independently).
    cache = {}
    tensors = {name: c.build(cache, **kwargs)
               for name, c in self._columns.items()}
    return tensors
