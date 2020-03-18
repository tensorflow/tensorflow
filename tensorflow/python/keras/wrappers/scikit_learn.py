# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
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

import warnings
from collections import defaultdict, namedtuple
import copy

import numpy as np

from tensorflow.python.keras.losses import (
    is_categorical_crossentropy,
)
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential, clone_model
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.layers import serialize, deserialize
from tensorflow.python.util import tf_inspect
from tensorflow.python.keras.utils.generic_utils import has_arg
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.util.tf_export import keras_export


# namedtuple used for pickling Model instances
SavedKerasModel = namedtuple(
    "SavedKerasModel", "cls model training_config weights",
)


# known keras function names that will be added to _legal_params_fns if they
# exist in the generated model
KNOWN_KERAS_FN_NAMES = (
    "fit",
    "evaluate",
    "predict_classes",
    "predict",
)

def _merge_dicts(*args):
  """Utility to merge multiple dictionaries last one wins style.
  """
  res = dict()
  for arg in args:
    res.update(arg)
  return res

def _check_array_sizes(y_true, y_pred, sample_weight):
  """Array size checks for scoring functions.
  """
  if y_true.ndim == 1:
    y_true = y_true.reshape(-1, 1)
  if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

  if y_true.shape != y_pred.shape:
    raise ValueError(
        "Shape of `y_true` and `y_pred` did not match: %s != %s"
        % (y_true.shape, y_pred.shape)
    )
  if sample_weight is not None:
    if y_true.shape[0] != sample_weight.shape[0]:
      raise ValueError(
          "Shape of `y_true` and `sample_weight` did not match: %s != %s"
          % (y_true.shape, sample_weight.shape)
      )
    if sample_weight.ndim != 1:
      raise ValueError(
          "`sample_weight` must be an array of shape (n_samples,)"
      )

  return y_true, y_pred, sample_weight

def _accuracy_score(y_true, y_pred, sample_weight=None, **kwargs):
  """Accuracy score for classification functions.

  Implementation borrowed from scikit-learn:
  scikit-learn/scikit-learn/sklearn/metrics/_classification.py
  """
  y_true, y_pred, sample_weight = _check_array_sizes(
      y_true, y_pred, sample_weight
  )
  # default cls_type is "multiclass"
  cls_type = kwargs.get("cls_type", "multiclass")

  if cls_type in ("multiclass", "binary"):
    score = y_true == y_pred
  if cls_type == "multilabel-indicator":
    # need to manually compute this one
    # pass kwargs since many arguments are shared
    # classes_ should have shape (n_samples, n_outputs)
    # compute accuracy like scikit-learn does,
    # each row (categories each sample matched)
    # must EXACTLY match between the prediction and truth ys
    n_samples = y_true.shape[0]
    if sample_weight is None:
      sample_weight = np.ones(dtype=int, shape=(n_samples,))
    differing_labels = np.count_nonzero(y_true - y_pred, axis=1)
    score = differing_labels == 0
  elif cls_type == "multiclass-multioutput":
    # return the mean of the scores
    # this is what scikit-learn does by default
    score = np.mean(np.all(y_true == y_pred, axis=1))

  return np.average(np.squeeze(score), weights=sample_weight)


def _r2_score(y_true, y_pred, sample_weight=None, **kwargs): # pylint: disable=unused-argument
  """R^2 (coefficient of determination) regression score function.

  Implementation borrowed from scikit-learn:
  scikit-learn/scikit-learn/sklearn/metrics/_regression.py
  """
  y_true, y_pred, sample_weight = _check_array_sizes(
      y_true, y_pred, sample_weight
  )

  if y_pred.shape[0] < 2:
    msg = "R^2 score is not well-defined with less than two samples."
    warnings.warn(msg)
    return float("nan")

  if sample_weight is not None:
    weight = sample_weight[:, np.newaxis]
  else:
    weight = 1.0

  numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
  denominator = (
      weight * (y_true - np.average(y_true, axis=0, weights=sample_weight)) ** 2
  ).sum(axis=0, dtype=np.float64)
  nonzero_denominator = denominator != 0
  nonzero_numerator = numerator != 0
  valid_score = nonzero_denominator & nonzero_numerator
  output_scores = np.ones([y_true.shape[1]])
  output_scores[valid_score] = 1 - \
    (numerator[valid_score] / denominator[valid_score])
  # arbitrary set to zero to avoid -inf scores, having a constant
  # y_true is not interesting for scoring a regression anyway
  output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

  return np.average(output_scores)

def _clone_prebuilt_model(build_fn):
  """Clones and compiles a pre-built model when build_fn is an existing
      Keras model instance.

  Arguments:
    build_fn : instance of Keras Model.

  Returns: copy of the input model with no training.
  """
  model = clone_model(build_fn)
  # clone_model does not compy over compilation parameters, do those manually
  model_metadata = saving_utils.model_metadata(build_fn)
  if "training_config" in model_metadata:
    training_config = model_metadata["training_config"]
  else:
    raise ValueError(
        "To use %s as `build_fn`, you must compile it first." % build_fn
    )

  model.compile(
      **saving_utils.compile_args_from_training_config(training_config)
  )

  return model

class BaseWrapper(object):  # pylint:disable=useless-object-inheritance
  """Base class for the Keras scikit-learn wrapper.

  Warning: This class should not be used directly.
  Use descendant classes instead.

  Arguments:
    build_fn: callable function or class instance
    **sk_params: model parameters & fitting parameters

  The `build_fn` should construct, compile and return a Keras model, which
  will then be used to fit/predict. One of the following
  three values could be passed to `build_fn`:
  1. A function
  2. An instance of a class that implements the `__call__` method
  3. An instance of a Keras Model. A copy of this instance will be made
  4. None. This means you implement a class that inherits from `BaseWrapper`,
  `KerasClassifier` or `KerasRegressor`. The `__call__` method of the
  present class will then be treated as the default `build_fn`.
  If `build_fn` has parameters X or y, these will be passed automatically.

  `sk_params` takes both model parameters and fitting parameters. Legal model
  parameters are the arguments of `build_fn`. Note that like all other
  estimators in scikit-learn, `build_fn` or your child class should provide
  default values for its arguments, so that you could create the estimator
  without passing any values to `sk_params`.

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

  # basic legal parameter set, based on functions that will normally be called
  # the model building function will be dynamically added
  _legal_params_fns = [
      Sequential.evaluate,
      Sequential.fit,
      Sequential.predict,
      Sequential.predict_classes,
      Model.evaluate,
      Model.fit,
      Model.predict,
  ]

  model = None

  def __init__(self, build_fn=None, **sk_params):
    self.build_fn = build_fn

    # the sklearn API requires that all __init__ parameters be saved as an instance
    # attribute of the same name
    for name, val in sk_params.items():
      setattr(self, name, val)

    # collect all __init__ params for this base class as well as
    # all child classes
    init_params = []
    # reverse the MRO, we want the 1st one to overwrite the nth
    for clss_ in reversed(tf_inspect.getmro(self.__class__)):
      if clss_ == object:
        continue
      argspec = tf_inspect.getfullargspec(clss_.__init__)
      for p in argspec.args+argspec.kwonlyargs:
        if p != 'self':
          init_params.append(p)

    # add parameters from sk_params
    self._init_params = set(init_params + list(sk_params.keys()))

    # check that all __init__ parameters were assigned (as per sklearn API)
    for param in self._init_params:
      if not hasattr(self, param):
        raise RuntimeError("Parameter %s was not assigned")

    # determine what type of build_fn to use
    self._check_build_fn(build_fn)

    # check that all parameters correspond to a fit or model param
    self._check_params(self.get_params(deep=False))

  def _check_build_fn(self, build_fn):
    """Checks `build_fn`.

    Arguments:
      build_fn : method or callable class as defined in __init__

    Raises:
      ValueError: if `build_fn` is not valid.
    """
    if build_fn is None:
      # no build_fn, use this class' __call__method
      if not hasattr(self, "__call__"):
        raise ValueError(
            "If not using the `build_fn` param, "
            "you must implement `__call__`"
        )
    elif isinstance(build_fn, Model):
      # pre-built Keras model
      self.__call__ = _clone_prebuilt_model # pylint:disable=method-hidden
    elif tf_inspect.isfunction(build_fn):
      if hasattr(self, "__call__"):
        raise ValueError(
            "This class cannot implement `__call__` if"
            "using the `build_fn` parameter"
        )
      # a callable method/function
      self.__call__ = build_fn
    elif (
        callable(build_fn)
        and hasattr(build_fn, "__class__")
        and hasattr(build_fn.__class__, "__call__")
    ):
      if hasattr(self, "__call__"):
        raise ValueError(
            "This class cannot implement `__call__` if"
            "using the `build_fn` parameter"
        )
      # an instance of a class implementing __call__
      self.__call__ = build_fn.__call__
    else:
      raise ValueError("`build_fn` must be a callable or None")
    # append legal parameters
    self._legal_params_fns.append(self.__call__)

  def _check_params(self, params):
    """Checks for user typos in `params`.

    To disable, override this method in child class.

    Arguments:
      params: dictionary; the parameters to be checked

    Raises:
      ValueError: if any member of `params` is not a valid argument.
    """
    for param_name in params:
      for fn in self._legal_params_fns:
        if has_arg(fn, param_name) or param_name in self._init_params:
          break
      else:
        raise ValueError(
            "{} is not a legal parameter".format(param_name)
        )

  def _build_keras_model(self, X, y, sample_weight, **kwargs):  # pylint:disable=invalid-name
    """Build the Keras model.

    This method will process all arguments and call the model building
    function with appropriate arguments.

    Arguments:
      X : array-like, shape `(n_samples, n_features)`
        Training samples where `n_samples` is the number of samples
        and `n_features` is the number of features.
      y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
        True labels for `X`.
      sample_weight : array-like of shape (n_samples,)
        Sample weights. The Keras Model must support this.
      **kwargs: dictionary arguments
        Legal arguments are the arguments `build_fn`.
    Returns:
      self : object
        a reference to the instance that can be chain called
        (ex: instance.fit(X,y).transform(X) )
    Raises:
      ValuError : In case sample_weight != None and the Keras model's `fit`
            method does not support that parameter.
    """
    # dynamically build model, i.e. self.__call__ builds a Keras model

    # get model arguments
    model_args = self._filter_params(self.__call__)

    # add `sample_weight` param
    # while it is not usually needed to build the model, some Keras models
    # require knowledge of the type of sample_weight to be built.
    sample_weight_arg = self._filter_params(
        self.__call__, params_to_check={"sample_weight": sample_weight}
    )

    # check if the model building function requires X and/or y to be passed
    X_y_args = self._filter_params(  # pylint:disable=invalid-name
        self.__call__, params_to_check={"X": X, "y": y}
    )

    # filter kwargs
    kwargs = self._filter_params(self.__call__, params_to_check=kwargs)

    # combine all arguments
    build_args = _merge_dicts(model_args, X_y_args, sample_weight_arg, kwargs)

    # build model
    model = self.__call__(**build_args)

    # append legal parameter names from model
    for known_keras_fn in KNOWN_KERAS_FN_NAMES:
      if hasattr(model, known_keras_fn):
        self._legal_params_fns.append(getattr(model, known_keras_fn))

    return model

  def _fit_keras_model(self, X, y, sample_weight, **kwargs):  # pylint:disable=invalid-name
    """Fits the Keras model.

    This method will process all arguments and call the Keras
    model's `fit` method with approriate arguments.

    Arguments:
      X : array-like, shape `(n_samples, n_features)`
        Training samples where `n_samples` is the number of samples
        and `n_features` is the number of features.
      y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
        True labels for `X`.
      sample_weight : array-like of shape (n_samples,)
        Sample weights. The Keras Model must support this.
      **kwargs: dictionary arguments
        Legal arguments are the arguments of the keras model's `fit` method.
    Returns:
      self : object
        a reference to the instance that can be chain called
        (ex: instance.fit(X,y).transform(X) )
    Raises:
      ValuError : In case sample_weight != None and the Keras model's `fit`
            method does not support that parameter.
    """
    # add `sample_weight` param, required to be explicit by some sklearn functions
    # that use inspect.signature on the `score` method
    if sample_weight is not None:
      # avoid pesky Keras warnings if sample_weight is not used
      kwargs.update({"sample_weight": sample_weight})

    # filter kwargs down to those accepted by self.model.fit
    kwargs = self._filter_params(self.model.fit, params_to_check=kwargs)

    if sample_weight is not None and "sample_weight" not in kwargs:
      raise ValueError(
          "Parameter `sample_weight` is unsupported by Keras model %s"
          % self.model
      )

    # get model.fit's arguments (allows arbitrary model use)
    fit_args = self._filter_params(self.model.fit)

    # fit model and save history
    # order implies kwargs overwrites fit_args
    fit_args = _merge_dicts(fit_args, kwargs)

    self.history = self.model.fit(x=X, y=y, **fit_args)

    # return self to allow fit_transform and such to work
    return self

  def _check_output_model_compatibility(self, y):
    """Checks that the model output number and y shape match, reshape as needed.
    """
    # check if this is a multi-output model
    n_outputs = len(self.model.outputs)
    if n_outputs != len(y):
      raise RuntimeError(
          "Detected a model with %s ouputs, but y has incompatible"
          " shape %s" % (n_outputs, len(y))
      )

    # format y into the exact format that Keras expects
    # this is the only format compatible with tf v1 and v2
    if len(y) == 1:
      y = y[0]
    else:
      y = tuple(np.squeeze(y_) for y_ in y)
    return y

  @staticmethod
  def _pre_process_y(y):
    """Handles manipulation of y inputs to fit or score.

    By default, this just makes sure y is a numpy arrray.
    Subclass and override this method to customize processing.

    Arguments:
      y : 1D or 2D numpy array, or iterable

    Returns:
      y : numpy array of shape (n_samples, n_ouputs)
      extra_args : dictionary of output attributes, ex: n_outputs_
          These parameters are added to `self` by `fit` and consumed (but not
          re-set) by `score`.
    """
    y = np.atleast_1d(y)
    if y.ndim == 1:
      y = np.reshape(y, (-1, 1))

    extra_args = dict()

    return y, extra_args

  @staticmethod
  def _post_process_y(y):
    """Handles manipulation of predicted `y` values.

    By default, it joins lists of predictions for multi-ouput models
    into a single numpy array.
    Subclass and override this method to customize processing.

    Arguments:
      y : 2D numpy array or list of numpy arrays
        (the latter is for multi-ouput models)

    Returns:
      y : 2D numpy array with singular dimensions stripped
      extra_args : attributes of output `y` such as probabilites.
          Currently unused by KerasRegressor but kept for flexibility.
    """
    y = np.column_stack(y)

    extra_args = dict()
    return np.squeeze(y), extra_args

  @staticmethod
  def _pre_process_X(X):  # pylint:disable=invalid-name
    """Handles manipulation of X before fitting.

       Subclass and override this method to process X, for example
       accomodate a multi-input model.

    Arguments:
      X : 2D numpy array

    Returns:
      X : unchanged 2D numpy array
      extra_args : attributes of output `y` such as probabilites.
          Currently unused by KerasRegressor but kept for flexibility.
    """
    extra_args = dict()
    return X, extra_args

  def fit(self, X, y, sample_weight=None, **kwargs):  # pylint:disable=invalid-name
    """Constructs a new model with `build_fn` & fit the model to `(X, y)`.

    Arguments:
      X : array-like, shape `(n_samples, n_features)`
        Training samples where `n_samples` is the number of samples
        and `n_features` is the number of features.
      y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
        True labels for `X`.
      sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. The Keras Model must support this.
      **kwargs: dictionary arguments
        Legal arguments are the arguments of the keras model's `fit` method.
    Returns:
      self : object
        a reference to the instance that can be chain called
        (ex: instance.fit(X,y).transform(X) )
    Raises:
      ValueError : In case of invalid shape for `y` argument.
      ValuError : In case sample_weight != None and the Keras model's `fit`
            method does not support that parameter.
    """
    # pre process X, y
    X, _ = self._pre_process_X(X)
    y, extra_args = self._pre_process_y(y)
    # update self.classes_, self.n_outputs_, self.n_classes_, self.cls_type
    for attr_name, attr_val in extra_args.items():
      setattr(self, attr_name, attr_val)

    # build model
    self.model = self._build_keras_model(
        X, y, sample_weight=sample_weight, **kwargs
    )

    y = self._check_output_model_compatibility(y)

    # fit model
    return self._fit_keras_model(X, y, sample_weight=sample_weight, **kwargs)

  def predict(self, X, **kwargs):  # pylint:disable=invalid-name
    """Returns predictions for the given test data.

    Arguments:
      X: array-like, shape `(n_samples, n_features)`
        Test samples where `n_samples` is the number of samples
        and `n_features` is the number of features.
      sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. The Keras Model must support this.
      **kwargs: dictionary arguments
        Legal arguments are the arguments of `self.model.predict`.

    Returns:
      preds: array-like, shape `(n_samples,)`
        Predictions.
    """
    # pre process X
    X, _ = self._pre_process_X(X)

    # filter kwargs and get attributes for predict
    kwargs = self._filter_params(self.model.predict, params_to_check=kwargs)
    predict_args = self._filter_params(self.model.predict)

    # predict with Keras model
    pred_args = _merge_dicts(predict_args, kwargs)
    y_pred = self.model.predict(X, **pred_args)

    # post process y
    y, _ = self._post_process_y(y_pred)
    return y

  def score(self, X, y, sample_weight=None, **kwargs):  # pylint:disable=invalid-name
    """Returns the mean accuracy on the given test data and labels.

    Arguments:
      X: array-like, shape `(n_samples, n_features)`
        Test samples where `n_samples` is the number of samples
        and `n_features` is the number of features.
      y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
        True labels for `X`.
      sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. The Keras Model must support this.
      **kwargs: dictionary arguments
        Legal arguments are those of self.model.evaluate.

    Returns:
      score: float
        Mean accuracy of predictions on `X` wrt. `y`.

    Raises:
      ValueError: If the underlying model isn't configured to
        compute accuracy. You should pass `metrics=["accuracy"]` to
        the `.compile()` method of the model.
    """
    # pre process X, y
    _, extra_args = self._pre_process_y(y)

    # compute Keras model score
    y_pred = self.predict(X, **kwargs)

    return self._scorer(y, y_pred, sample_weight, **extra_args)

  def _filter_params(self, fn, params_to_check=None):
    """Filters all instance attributes (parameters) and
       returns those in `fn`'s arguments.

    Arguments:
      fn : arbitrary function
      params_to_check : dictionary, parameters to check.
        Defaults to checking all attributes of this estimator.

    Returns:
      res : dictionary containing variables
        in both self and `fn`'s arguments.
    """
    res = {}
    for name, value in (params_to_check or self.__dict__).items():
      if has_arg(fn, name):
        res.update({name: value})
    return res

  def get_params(self, deep=True):
    """Get parameters for this estimator.

    This method mimics sklearn.base.BaseEstimator.get_params

    Arguments:
      deep : bool, default=True
        If True, will return the parameters for this estimator and
        contained subobjects that are estimators.

    Returns:
      params : mapping of string to any
        Parameter names mapped to their values.
    """
    out = dict()
    for key in self._init_params:
      value = getattr(self, key)
      if deep and hasattr(value, "get_params"):
        deep_items = value.get_params().items()
        out.update((key + "__" + k, val) for k, val in deep_items)
      out[key] = value
    return out

  def set_params(self, **params):
    """Set the parameters of this estimator.

    The method works on simple estimators as well as on nested objects
    (such as in sklearn Pipelines). The latter have parameters of the form
    ``<component>__<parameter>`` so that it's possible to update each
    component of a nested object.

    This method mimics sklearn.base.BaseEstimator.set_params

    Arguments:
      **params : dict
        Estimator parameters.
    Returns:
      self : object
        Estimator instance.
    """
    if not params:
      # Simple optimization to gain speed
      return self
    valid_params = self.get_params(deep=True)

    nested_params = defaultdict(dict)  # grouped by prefix
    for key, value in params.items():
      key, delim, sub_key = key.partition("__")
      if key not in valid_params:
        raise ValueError(
            "Invalid parameter %s for estimator %s. "
            "Check the list of available parameters "
            "with `estimator.get_params().keys()`." % (key, self)
        )
      if delim:
        nested_params[key][sub_key] = value
      else:
        setattr(self, key, value)
        valid_params[key] = value

    for key, sub_params in nested_params.items():
      valid_params[key].set_params(**sub_params)

    return self

  def __getstate__(self):
    """Get state of instance as a picklable/copyable dict.

       Used by various scikit-learn methods to clone estimators. Also used
       for pickling.
       Because some objects (mainly Keras `Model` instances) are not pickleable,
       it is necessary to iterate through all attributes and clone the
       unpicklables manually.

    Returns:
      state : dictionary containing a copy of all attributes of this
          estimator with Keras Model instances being saved as
          HDF5 binary objects.
    """

    def _pack_obj(obj):
      """Recursively packs objects.
      """
      try:
        return copy.deepcopy(obj)
      except TypeError:
        pass  # is this a Keras serializable?
      try:
        model_metadata = saving_utils.model_metadata(obj)
        training_config = model_metadata["training_config"]
        model = serialize(obj)
        weights = obj.get_weights()
        return SavedKerasModel(
            cls=obj.__class__,
            model=model,
            weights=weights,
            training_config=training_config,
        )
      except (TypeError, AttributeError):
        pass  # try manually packing the object
      if hasattr(obj, "__dict__"):
        for key, val in obj.__dict__.items():
          obj.__dict__[key] = _pack_obj(val)
        return obj
      if isinstance(obj, (list, tuple)):
        obj_type = type(obj)
        new_obj = obj_type([_pack_obj(o) for o in obj])
        return new_obj

      return obj

    state = self.__dict__.copy()
    for key, val in self.__dict__.items():
      state[key] = _pack_obj(val)
    return state

  def __setstate__(self, state):
    """Set state of live object from state saved via __getstate__.

       Because some objects (mainly Keras `Model` instances) are not pickleable,
       it is necessary to iterate through all attributes and clone the
       unpicklables manually.

    Arguments:
      state : dict
        dictionary from a previous call to `get_state` that will be
        unpacked to this instance's `__dict__`.
    """

    def _unpack_obj(obj):
      """Recursively unpacks objects.
      """
      if isinstance(obj, SavedKerasModel):
        restored_model = deserialize(obj.model)
        training_config = obj.training_config
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(training_config)
        )
        restored_model.set_weights(obj.weights)
        return restored_model
      if hasattr(obj, "__dict__"):
        for key, val in obj.__dict__.items():
          obj.__dict__[key] = _unpack_obj(val)
        return obj
      if isinstance(obj, (list, tuple)):
        obj_type = type(obj)
        new_obj = obj_type([_unpack_obj(o) for o in obj])
        return new_obj

      return obj  # not much we can do at this point, cross fingers

    for key, val in state.items():
      setattr(self, key, _unpack_obj(val))


@keras_export("keras.wrappers.scikit_learn.KerasClassifier")
class KerasClassifier(BaseWrapper):
  """Implementation of the scikit-learn classifier API for Keras.
  """

  _estimator_type = "classifier"
  _scorer = staticmethod(_accuracy_score)

  @staticmethod
  def _pre_process_y(y):
    """Handles manipulation of y inputs to fit or score.

       For KerasClassifier, this handles interpreting classes from `y`.

    Arguments:
      y : 1D or 2D numpy array

    Returns:
      y : modified 2D numpy array with 0 indexed integer class labels.
      classes_ : list of original class labels.
      n_classes_ : number of classes.
      one_hot_encoded : True if input y was one-hot-encoded.
    """
    y, _ = super(KerasClassifier, KerasClassifier)._pre_process_y(y)

    if y.shape[1] == 1:
      # single output
      if np.unique(y).size == 2:
        # y = array([1, 0, 1, 0]) or y = array(['used', 'new', 'used'])
        cls_type = "binary"
        # single task, single label, binary classification
        classes_ = np.unique(y)
        # convert to 0 indexed classes
        y = np.searchsorted(classes_, y)
        classes_ = [classes_]
        y = [y]
      elif np.unique(y).size > 2:
        # y = array([1, 5, 2]) or y = array(['apple', 'orange', 'banana'])
        cls_type = "multiclass"
        classes_ = np.unique(y)
        # convert to 0 indexed classes
        y = np.searchsorted(classes_, y)
        classes_ = [classes_]
        y = [y]
    elif y.shape[1] > 1:
      # multi-output / multi-task
      if np.unique(y).size == 2:
        # y = array([1, 1, 1, 0], [0, 0, 1, 1])
        cls_type = "multilabel-indicator"
        # split into lists for multi-output Keras
        # will be processed as multiple binary classifications
        classes_ = [np.array([0, 1])] * y.shape[1]
        y = np.split(y, y.shape[1], axis=1)
      else:
        # y = array(['apple', 0, 5], ['orange', 1, 3])
        cls_type = "multiclass-multioutput"
        # split into lists for multi-output Keras
        # each will be processesed as a seperate multiclass problem
        y = np.split(y, y.shape[1], axis=1)
        classes_ = [np.unique(y_) for y_ in y]

    if not classes_:
      # other situations, such as 3 dimensional arrays, are not supported
      raise ValueError("Invalid shape for y: " + str(y.shape))

    # self.classes_ is kept as an array when n_outputs==1 for compatibility
    # with ensembles and other meta estimators which do not support multioutput
    if len(classes_) == 1:
      n_classes_ = classes_[0].size
      classes_ = classes_[0]
      n_outputs_ = 1
    else:
      n_classes_ = [class_.shape[0] for class_ in classes_]
      n_outputs_ = len(classes_)

    extra_args = {
        "classes_": classes_,
        "n_outputs_": n_outputs_,
        "n_classes_": n_classes_,
        "cls_type": cls_type,
    }

    return y, extra_args

  def _post_process_y(self, y):
    """Reverts _pre_process_inputs to return predicted probabilites
       in formats sklearn likes as well as retrieving the original classes.
    """
    if not isinstance(y, list):
      # convert single-target y to a list for easier processing
      y = [y]

    # self.classes_ is kept as an array when n_outputs==1 for compatibility
    # with meta estimators
    if self.n_outputs_ == 1:
      cls_ = [self.classes_]
    else:
      cls_ = self.classes_

    y = copy.deepcopy(y)
    cls_type = self.cls_type

    class_predictions = []
    for i, (y_, classes_) in enumerate(zip(y, cls_)):
      if cls_type == "binary":
        if y_.shape[1] == 1:
          class_predictions.append(classes_[np.where(y_ > 0.5, 1, 0)])
          # first column is probability of class 0 and second is of class 1
          y[i] = np.stack([1 - y_, y_], axis=1)
        else:
          # array([0.9, 0.1], [.2, .8]) -> array(['yes', 'no'])
          class_predictions.append(
              classes_[np.argmax(np.where(y_ > 0.5, 1, 0), axis=1)]
          )
      elif cls_type == "multiclass":
        # array([0.8, 0.1, 0.1], [.1, .8, .1]) -> array(['apple', 'orange'])
        class_predictions.append(classes_[np.argmax(y_, axis=1)])
      elif cls_type == "multilabel-indicator":
        class_predictions.append(np.where(y_ > 0.5, 1, 0))
      elif cls_type == "multiclass-multioutput":
        # array([0.9, 0.1], [.2, .8]) -> array(['apple', 'fruit'])
        class_predictions.append(classes_[np.argmax(y_, axis=1)])
      else:
        raise ValueError("Unknown classification task type '%s'" % cls_)

    class_probabilities = np.squeeze(np.column_stack(y))

    y = np.squeeze(np.column_stack(class_predictions))

    extra_args = {"class_probabilities": class_probabilities}

    return y, extra_args

  def _check_output_model_compatibility(self, y):
    """Checks that the model output number and loss functions match y.
    """
    # check loss function to adjust the encoding of the input
    # we need to do this to mimick scikit-learn behavior
    if isinstance(self.model.loss, list):
      losses = self.model.loss
    else:
      losses = [self.model.loss] * self.n_outputs_
    for i, (loss, y_) in enumerate(zip(losses, y)):
      if is_categorical_crossentropy(loss) and \
        (y_.ndim == 1 or y_.shape[1] == 1):
        y[i] = to_categorical(y_)

    return super(KerasClassifier, self)._check_output_model_compatibility(y)

  def predict_proba(self, X, **kwargs):  # pylint:disable=invalid-name
    """Returns class probability estimates for the given test data.

    Arguments:
      X: array-like, shape `(n_samples, n_features)`
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
    # pre process X
    X, _ = self._pre_process_X(X)

    # filter kwargs and get attributes that are inputs to model.predict
    kwargs = self._filter_params(self.model.predict, params_to_check=kwargs)
    predict_args = self._filter_params(self.model.predict)

    # call the Keras model
    predict_args = _merge_dicts(predict_args, kwargs)
    outputs = self.model.predict(X, **predict_args)

    # join list of outputs into single output array
    _, extra_args = self._post_process_y(outputs)

    class_probabilities = extra_args["class_probabilities"]

    return class_probabilities


@keras_export("keras.wrappers.scikit_learn.KerasRegressor")
class KerasRegressor(BaseWrapper):
  """Implementation of the scikit-learn regressor API for Keras.
  """

  _estimator_type = "regressor"
  _scorer = staticmethod(_r2_score)

  def _pre_process_y(self, y):
    """Split y for multi-output tasks.
    """
    y, _ = super(KerasRegressor, self)._pre_process_y(y)
    y = np.split(y, y.shape[1], axis=1)

    n_outputs_ = len(y)

    extra_args = {"n_outputs_": n_outputs_}

    return y, extra_args

  def score(self, X, y, sample_weight=None, **kwargs):  # pylint:disable=invalid-name
    """Returns the mean loss on the given test data and labels.

    Arguments:
      X: array-like, shape `(n_samples, n_features)`
        Test samples where `n_samples` is the number of samples
        and `n_features` is the number of features.
      y: array-like, shape `(n_samples,)`
        True labels for `X`.
      **kwargs: dictionary arguments
        Legal arguments are the arguments of `Sequential.evaluate`.

    Returns:
      score: float
        Mean accuracy of predictions on `X` wrt. `y`.
    """
    res = super(KerasRegressor, self).score(X, y, sample_weight, **kwargs)

    # check loss function and warn if it is not the same as score function
    if self.model.loss not in (
        "mean_squared_error", self.root_mean_squared_error
    ):
      warnings.warn(
          "R^2 is used to compute the score, it is advisable to use a"
          " compatible loss function. This class provides an R^2"
          " implementation in `KerasRegressor.root_mean_squared_error`."
      )

    return res

  @staticmethod
  def root_mean_squared_error(y_true, y_pred):
    """A simple Keras implementation of R^2 that can be used as a Keras
       loss function.

       Since `score` uses R^2, it is advisable to use the same loss/metric
       when optimizing the model.
    """
    ss_res = K.sum(K.square(y_true - y_pred), axis=0)
    ss_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)
    return K.mean(1 - ss_res / (ss_tot + K.epsilon()), axis=-1)
