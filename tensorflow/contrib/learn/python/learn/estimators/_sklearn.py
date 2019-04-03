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

"""sklearn cross-support (deprecated)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import six


def _pprint(d):
  return ', '.join(['%s=%s' % (key, str(value)) for key, value in d.items()])


class _BaseEstimator(object):
  """This is a cross-import when sklearn is not available.

  Adopted from sklearn.BaseEstimator implementation.
  https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py
  """

  def get_params(self, deep=True):
    """Get parameters for this estimator.

    Args:
      deep: boolean, optional

        If `True`, will return the parameters for this estimator and
        contained subobjects that are estimators.

    Returns:
      params : mapping of string to any
      Parameter names mapped to their values.
    """
    out = {}
    param_names = [name for name in self.__dict__ if not name.startswith('_')]
    for key in param_names:
      value = getattr(self, key, None)

      if isinstance(value, collections.Callable):
        continue

      # XXX: should we rather test if instance of estimator?
      if deep and hasattr(value, 'get_params'):
        deep_items = value.get_params().items()
        out.update((key + '__' + k, val) for k, val in deep_items)
      out[key] = value
    return out

  def set_params(self, **params):
    """Set the parameters of this estimator.

    The method works on simple estimators as well as on nested objects
    (such as pipelines). The former have parameters of the form
    ``<component>__<parameter>`` so that it's possible to update each
    component of a nested object.

    Args:
      **params: Parameters.

    Returns:
      self

    Raises:
      ValueError: If params contain invalid names.
    """
    if not params:
      # Simple optimisation to gain speed (inspect is slow)
      return self
    valid_params = self.get_params(deep=True)
    for key, value in six.iteritems(params):
      split = key.split('__', 1)
      if len(split) > 1:
        # nested objects case
        name, sub_name = split
        if name not in valid_params:
          raise ValueError('Invalid parameter %s for estimator %s. '
                           'Check the list of available parameters '
                           'with `estimator.get_params().keys()`.' %
                           (name, self))
        sub_object = valid_params[name]
        sub_object.set_params(**{sub_name: value})
      else:
        # simple objects case
        if key not in valid_params:
          raise ValueError('Invalid parameter %s for estimator %s. '
                           'Check the list of available parameters '
                           'with `estimator.get_params().keys()`.' %
                           (key, self.__class__.__name__))
        setattr(self, key, value)
    return self

  def __repr__(self):
    class_name = self.__class__.__name__
    return '%s(%s)' % (class_name,
                       _pprint(self.get_params(deep=False)),)


# pylint: disable=old-style-class
class _ClassifierMixin():
  """Mixin class for all classifiers."""
  pass


class _RegressorMixin():
  """Mixin class for all regression estimators."""
  pass


class _TransformerMixin():
  """Mixin class for all transformer estimators."""


class NotFittedError(ValueError, AttributeError):
  """Exception class to raise if estimator is used before fitting.

  USE OF THIS EXCEPTION IS DEPRECATED.

  This class inherits from both ValueError and AttributeError to help with
  exception handling and backward compatibility.

  Examples:
  >>> from sklearn.svm import LinearSVC
  >>> from sklearn.exceptions import NotFittedError
  >>> try:
  ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
  ... except NotFittedError as e:
  ...     print(repr(e))
  ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  NotFittedError('This LinearSVC instance is not fitted yet',)

  Copied from
  https://github.com/scikit-learn/scikit-learn/master/sklearn/exceptions.py
  """

# pylint: enable=old-style-class


def _accuracy_score(y_true, y_pred):
  score = y_true == y_pred
  return np.average(score)


def _mean_squared_error(y_true, y_pred):
  if len(y_true.shape) > 1:
    y_true = np.squeeze(y_true)
  if len(y_pred.shape) > 1:
    y_pred = np.squeeze(y_pred)
  return np.average((y_true - y_pred)**2)


def _train_test_split(*args, **options):
  # pylint: disable=missing-docstring
  test_size = options.pop('test_size', None)
  train_size = options.pop('train_size', None)
  random_state = options.pop('random_state', None)

  if test_size is None and train_size is None:
    train_size = 0.75
  elif train_size is None:
    train_size = 1 - test_size
  train_size = int(train_size * args[0].shape[0])

  np.random.seed(random_state)
  indices = np.random.permutation(args[0].shape[0])
  train_idx, test_idx = indices[:train_size], indices[train_size:]
  result = []
  for x in args:
    result += [x.take(train_idx, axis=0), x.take(test_idx, axis=0)]
  return tuple(result)


# If "TENSORFLOW_SKLEARN" flag is defined then try to import from sklearn.
TRY_IMPORT_SKLEARN = os.environ.get('TENSORFLOW_SKLEARN', False)
if TRY_IMPORT_SKLEARN:
  # pylint: disable=g-import-not-at-top,g-multiple-import,unused-import
  from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
  from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
  from sklearn.model_selection import train_test_split
  try:
    from sklearn.exceptions import NotFittedError
  except ImportError:
    try:
      from sklearn.utils.validation import NotFittedError
    except ImportError:
      pass
else:
  # Naive implementations of sklearn classes and functions.
  BaseEstimator = _BaseEstimator
  ClassifierMixin = _ClassifierMixin
  RegressorMixin = _RegressorMixin
  TransformerMixin = _TransformerMixin
  accuracy_score = _accuracy_score
  log_loss = None
  mean_squared_error = _mean_squared_error
  train_test_split = _train_test_split
