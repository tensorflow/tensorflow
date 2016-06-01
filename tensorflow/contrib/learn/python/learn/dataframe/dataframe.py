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

from abc import ABCMeta
import collections

from .series import Series
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
      if isinstance(v, Series):
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
            "Column in assignment must be an inflow.Series, pandas.Series or a"
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
    if isinstance(value, Series):
      value = [value]
    self.assign(**dict(zip(key, value)))

  def build(self):
    # We do not allow passing a cache here, because that would encourage
    # working around the rule that DataFrames cannot be expected to be
    # synced with each other (e.g., they shuffle independently).
    cache = {}
    tensors = {name: c.build(cache) for name, c in self._columns.items()}
    return tensors

  def to_input_fn(self, feature_keys=None, target_keys=None):
    """Build an input_fn suitable for use with Estimator.

    Args:
      feature_keys: the names of columns to be used as features.  If None, all
        columns except those in target_keys are used.
      target_keys: the names of columns to be used as targets.  None is
        acceptable for unsupervised learning.

    Returns:
      A function that returns a pair of dicts (features, targets), each mapping
        string names to Tensors.

    Raises:
      ValueError: when the feature and target key sets are non-disjoint
    """
    if target_keys is None:
      target_keys = []

    if feature_keys is None:
      feature_keys = self.columns() - set(target_keys)
    else:
      in_both = set(feature_keys) & set(target_keys)
      if in_both:
        raise ValueError(
            "Columns cannot be used for both features and targets: %s" %
            ", ".join(in_both))

    def input_fn():
      # It's important to build all the tensors together in one DataFrame.
      # If we did df.select() for both key sets and then build those, the two
      # resulting DataFrames would be shuffled independently.
      tensors = self.build()

      # Note that (for now at least) we provide our columns to Estimator keyed
      # by strings, so they are base features as far as Estimator is concerned.
      # TODO(soergel): reconcile with FeatureColumn keys, Transformer etc.
      features = {key: tensors[key] for key in feature_keys}
      targets = {key: tensors[key] for key in target_keys}
      return features, targets

    return input_fn
