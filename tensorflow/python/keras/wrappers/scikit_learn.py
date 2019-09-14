# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Wrapper for using the Scikit-Learn API with Keras models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import types

import numpy as np

from tensorflow.python.keras import losses
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.utils.generic_utils import has_arg
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.util.tf_export import keras_export


class BaseWrapper(object):
  """Base class for the Keras scikit-learn wrapper.

  Warning: This class should not be used directly.
  Use descendant classes instead.

  Arguments:
      build_fn: callable function or class instance
      **sk_params: model parameters & fitting parameters

  The `build_fn` should construct, compile and return a Keras model, which
  will then be used to fit/predict. It must accept `input_shape` and
  `output_shape` as arguments, both of which are tuples of integers. One of
  the following three values could be passed to `build_fn`:
  1. A function
  2. An instance of a class that implements the `__call__` method
  3. None. This means you implement a class that inherits from either
  `KerasClassifier` or `KerasRegressor`. The `__call__` method of the
  present class will then be treated as the default `build_fn`.

  `sk_params` takes both model parameters and fitting parameters. Legal model
  parameters are the arguments of `build_fn`. Note that like all other
  estimators in scikit-learn, `build_fn` should provide default values for
  its arguments, so that you could create the estimator without passing any
  values to `sk_params`.

  `sk_params` could also accept parameters for calling `fit`, `predict`,
  `predict_proba`, and `score` methods (e.g., `epochs`, `batch_size`).
  fitting (predicting) parameters are selected in the following order:

  1. Values passed to the dictionary arguments of
  `fit`, `predict`, `predict_proba`, and `score` methods
  2. Values passed to `sk_params`
  3. The default values of the `keras.models.Sequential`
  `fit`, `predict`, `predict_proba` and `score` methods

  When using scikit-learn's `grid_search` API, legal tunable parameters are
  those you could pass to `sk_params`, including fitting parameters.
  In other words, you could use `grid_search` to search for the best
  `batch_size` or `epochs` as well as the model parameters.
  """

  def __init__(self, build_fn=None, **sk_params):
    self.build_fn = build_fn
    self.sk_params = sk_params
    self.check_params(sk_params)

  def check_params(self, params):
    """Checks for user typos in `params`.

    Arguments:
        params: dictionary; the parameters to be checked

    Raises:
        ValueError: if any member of `params` is not a valid argument.
    """
    legal_params_fns = [
        Sequential.evaluate, Sequential.fit, Sequential.predict,
        Sequential.predict_classes, Model.evaluate, Model.fit, Model.predict
    ]
    if self.build_fn is None:
      legal_params_fns.append(self.__call__)
    elif (not isinstance(self.build_fn, types.FunctionType) and
          not isinstance(self.build_fn, types.MethodType)):
      legal_params_fns.append(self.build_fn.__call__)
    else:
      legal_params_fns.append(self.build_fn)

    for params_name in params:
      for fn in legal_params_fns:
        if has_arg(fn, params_name):
          break
      else:
        if params_name != 'nb_epoch':
          raise ValueError('{} is not a legal parameter'.format(params_name))

  def get_params(self, **params):  # pylint: disable=unused-argument
    """Gets parameters for this estimator.

    Arguments:
        **params: ignored (exists for API compatibility).

    Returns:
        Dictionary of parameter names mapped to their values.
    """
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

  def set_params(self, **params):
    """Sets the parameters of this estimator.

    Arguments:
        **params: Dictionary of parameter names mapped to their values.

    Returns:
        self
    """
    self.check_params(params)
    self.sk_params.update(params)
    return self

  def fit(self, x, y, **kwargs):
    """Constructs a new model with `build_fn` & fit the model to `(x, y)`.

    Arguments:
        x: Input data. It could be:
            - A Numpy array (or array-like), or a list of arrays
              (in case the model has multiple inputs).
            - A dict mapping input names to the corresponding
              array/tensors, if the model has named inputs.
            - None (default) if feeding from framework-native
              tensors (e.g. TensorFlow data tensors).
        y: Target data. Like the input data `x`,
            it could be either Numpy array(s), framework-native tensor(s),
            list of Numpy arrays (if the model has multiple outputs) or
            None (default) if feeding from framework-native tensors
            (e.g. TensorFlow data tensors).
            If output layers in the model are named, you can also pass a
            dictionary mapping output names to Numpy arrays.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.fit`

    Returns:
        history : object
            details about the training history at each epoch.
    """
    if isinstance(x, dict):
      input_shape = {k: v.shape[1:] for k, v in x.items()}
    elif isinstance(x, list) or isinstance(x, tuple):
      input_shape = [i.shape[1:] for i in x]
    else:
      input_shape = x.shape[1:]
    if isinstance(y, dict):
      output_shape = {k: v.shape[1:] for k, v in y.items()}
    elif isinstance(y, list) or isinstance(y, tuple):
      output_shape = [i.shape[1:] for i in y]
    else:
      output_shape = y.shape[1:]
    if self.build_fn is None:
      self.model = self.__call__(input_shape, output_shape,
                                 **self.filter_sk_params(self.__call__))
    elif (not isinstance(self.build_fn, types.FunctionType) and
          not isinstance(self.build_fn, types.MethodType)):
      self.model = self.build_fn(input_shape, output_shape,
                                 **self.filter_sk_params(
                                   self.build_fn.__call__))
    else:
      self.model = self.build_fn(input_shape, output_shape,
                                 **self.filter_sk_params(self.build_fn))

    fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
    fit_args.update(kwargs)

    history = self.model.fit(x, y, **fit_args)

    return history

  def filter_sk_params(self, fn, override=None):
    """Filters `sk_params` and returns those in `fn`'s arguments.

    Arguments:
        fn : arbitrary function
        override: dictionary, values to override `sk_params`

    Returns:
        res : dictionary containing variables
            in both `sk_params` and `fn`'s arguments.
    """
    override = override or {}
    res = {}
    for name, value in self.sk_params.items():
      if has_arg(fn, name):
        res.update({name: value})
    res.update(override)
    return res


  def predict(self, x, **kwargs):
    """Returns predictions for the given test data.

    Arguments:
        x : Input data. It could be:
            - A Numpy array (or array-like), or a list of arrays
              (in case the model has multiple inputs).
            - A dict mapping input names to the corresponding
              array/tensors, if the model has named inputs.
            - None (default) if feeding from framework-native
              tensors (e.g. TensorFlow data tensors).
        **kwargs : dictionary arguments
            Legal arguments are the arguments of `Model.predict`.

    Returns:
        Numpy array(s) of predictions.
    """
    kwargs = self.filter_sk_params(Model.predict, kwargs)
    return self.model.predict(x, **kwargs)


  def score(self, x, y, **kwargs):
    """Returns the mean loss on the given test data and labels.

    Arguments:
        x : Input data. It could be:
            - A Numpy array (or array-like), or a list of arrays
              (in case the model has multiple inputs).
            - A dict mapping input names to the corresponding
              array/tensors, if the model has named inputs.
            - None (default) if feeding from framework-native
              tensors (e.g. TensorFlow data tensors).
        y : Target data. Like the input data `x`,
            it could be either Numpy array(s), framework-native tensor(s),
            list of Numpy arrays (if the model has multiple outputs) or
            None (default) if feeding from framework-native tensors
            (e.g. TensorFlow data tensors).
            If output layers in the model are named, you can also pass a
            dictionary mapping output names to Numpy arrays.
        **kwargs : dictionary arguments
            Legal arguments are the arguments of `Model.evaluate`.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """
    kwargs = self.filter_sk_params(Model.evaluate, kwargs)
    return self.model.evaluate(x, y, **kwargs)


@keras_export('keras.wrappers.scikit_learn.KerasClassifier')
class KerasClassifier(BaseWrapper):
  """Implementation of the scikit-learn classifier API for Keras.
  """

  _estimator_type = "classifier"

  def fit(self, x, y, **kwargs):
    """Constructs a new model with `build_fn` & fit the model to `(x, y)`.

    Arguments:
        x : array-like, shape `(n_samples, n_features)`
            Training samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
            True labels for `x`.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.fit`

    Returns:
        history : object
            details about the training history at each epoch.

    Raises:
        ValueError: In case of invalid shape for `y` argument.
    """
    y = np.array(y)
    if len(y.shape) == 2 and y.shape[1] > 1:
      self.classes_ = np.arange(y.shape[1])
    elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
      self.classes_ = np.unique(y)
      y = np.searchsorted(self.classes_, y)
    else:
      raise ValueError('Invalid shape for y: ' + str(y.shape))
    self.n_classes_ = len(self.classes_)
    if len(y.shape) != 2:
      y = to_categorical(y)
    return super(KerasClassifier, self).fit(x, y, **kwargs)

  def predict(self, x, **kwargs):
    """Returns the class predictions for the given test data.

    Arguments:
        x: array-like, shape `(n_samples, n_features)`
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        **kwargs: dictionary arguments
            Legal arguments are the arguments
            of `Sequential.predict_classes`.

    Returns:
        preds: array-like, shape `(n_samples,)`
            Class predictions.
    """
    kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
    classes = self.model.predict_classes(x, **kwargs)
    return self.classes_[classes]

  def predict_proba(self, x, **kwargs):
    """Returns class probability estimates for the given test data.

    Arguments:
        x: array-like, shape `(n_samples, n_features)`
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        **kwargs: dictionary arguments
            Legal arguments are the arguments
            of `Sequential.predict_classes`.

    Returns:
        proba: array-like, shape `(n_samples, n_outputs)`
            Class probability estimates.
            In the case of binary classification,
            to match the scikit-learn API,
            will return an array of shape `(n_samples, 2)`
            (instead of `(n_sample, 1)` as in Keras).
    """
    kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
    probs = self.model.predict_proba(x, **kwargs)

    # check if binary classification
    if probs.shape[1] == 1:
      # first column is probability of class 0 and second is of class 1
      probs = np.hstack([1 - probs, probs])
    return probs

  def score(self, x, y, **kwargs):
    """Returns the mean accuracy on the given test data and labels.

    Arguments:
        x: array-like, shape `(n_samples, n_features)`
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
            True labels for `x`.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.evaluate`.

    Returns:
        score: float
            Mean accuracy of predictions on `x` wrt. `y`.

    Raises:
        ValueError: If the underlying model isn't configured to
            compute accuracy. You should pass `metrics=["accuracy"]` to
            the `.compile()` method of the model.
    """
    y = np.searchsorted(self.classes_, y)
    kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

    loss_name = self.model.loss
    if hasattr(loss_name, '__name__'):
      loss_name = loss_name.__name__
    if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
      y = to_categorical(y)

    outputs = self.model.evaluate(x, y, **kwargs)
    if not isinstance(outputs, list):
      outputs = [outputs]
    for name, output in zip(self.model.metrics_names, outputs):
      if name in ['accuracy', 'acc']:
        return output
    raise ValueError('The model is not configured to compute accuracy. '
                     'You should pass `metrics=["accuracy"]` to '
                     'the `model.compile()` method.')


@keras_export('keras.wrappers.scikit_learn.KerasRegressor')
class KerasRegressor(BaseWrapper):
  """Implementation of the scikit-learn regressor API for Keras.
  """

  _estimator_type = "regressor"

  def predict(self, x, **kwargs):
    """Returns predictions for the given test data.

    Arguments:
        x: array-like, shape `(n_samples, n_features)`
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.predict`.

    Returns:
        preds: array-like, shape `(n_samples,)`
            Predictions.
    """
    kwargs = self.filter_sk_params(Sequential.predict, kwargs)
    return np.squeeze(self.model.predict(x, **kwargs))

  def score(self, x, y, **kwargs):
    """Returns the mean loss on the given test data and labels.

    Arguments:
        x: array-like, shape `(n_samples, n_features)`
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y: array-like, shape `(n_samples,)`
            True labels for `x`.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.evaluate`.

    Returns:
        score: float
            Mean accuracy of predictions on `x` wrt. `y`.
    """
    kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
    loss = self.model.evaluate(x, y, **kwargs)
    if isinstance(loss, list):
      return -loss[0]
    return -loss
