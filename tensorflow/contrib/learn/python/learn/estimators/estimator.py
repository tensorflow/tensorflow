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

"""Base Estimator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy
import inspect
import itertools
import os
import tempfile
import time

import numpy as np
import six

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.framework import deprecated_args
from tensorflow.contrib.framework import list_variables
from tensorflow.contrib.framework import load_variable
from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.framework.python.ops import ops as contrib_ops
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import graph_actions
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators import _sklearn as sklearn
from tensorflow.contrib.learn.python.learn.estimators import metric_key
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.estimators import tensor_signature
from tensorflow.contrib.learn.python.learn.estimators._sklearn import NotFittedError
from tensorflow.contrib.learn.python.learn.learn_io import data_feeder
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import device_setter
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import summary_io
from tensorflow.python.util import compat


AS_ITERABLE_DATE = '2016-09-15'
AS_ITERABLE_INSTRUCTIONS = (
    'The default behavior of predict() is changing. The default value for\n'
    'as_iterable will change to True, and then the flag will be removed\n'
    'altogether. The behavior of this flag is described below.')
SCIKIT_DECOUPLE_DATE = '2016-12-01'
SCIKIT_DECOUPLE_INSTRUCTIONS = (
    'Estimator is decoupled from Scikit Learn interface by moving into\n'
    'separate class SKCompat. Arguments x, y and batch_size are only\n'
    'available in the SKCompat class, Estimator will only accept input_fn.\n'
    'Example conversion:\n'
    '  est = Estimator(...) -> est = SKCompat(Estimator(...))')


def _get_input_fn(x, y, input_fn, feed_fn, batch_size, shuffle=False, epochs=1):
  """Make inputs into input and feed functions.

  Args:
    x: Numpy, Pandas or Dask matrix or iterable.
    y: Numpy, Pandas or Dask matrix or iterable.
    input_fn: Pre-defined input function for training data.
    feed_fn: Pre-defined data feeder function.
    batch_size: Size to split data into parts. Must be >= 1.
    shuffle: Whether to shuffle the inputs.
    epochs: Number of epochs to run.

  Returns:
    Data input and feeder function based on training data.

  Raises:
    ValueError: Only one of `(x & y)` or `input_fn` must be provided.
  """
  if input_fn is None:
    if x is None:
      raise ValueError('Either x or input_fn must be provided.')

    if contrib_framework.is_tensor(x) or (y is not None and
                                          contrib_framework.is_tensor(y)):
      raise ValueError('Inputs cannot be tensors. Please provide input_fn.')

    if feed_fn is not None:
      raise ValueError('Can not provide both feed_fn and x or y.')

    df = data_feeder.setup_train_data_feeder(x, y, n_classes=None,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             epochs=epochs)
    return df.input_builder, df.get_feed_dict_fn()

  if (x is not None) or (y is not None):
    raise ValueError('Can not provide both input_fn and x or y.')
  if batch_size is not None:
    raise ValueError('Can not provide both input_fn and batch_size.')

  return input_fn, feed_fn


def infer_real_valued_columns_from_input_fn(input_fn):
  """Creates `FeatureColumn` objects for inputs defined by `input_fn`.

  This interprets all inputs as dense, fixed-length float values. This creates
  a local graph in which it calls `input_fn` to build the tensors, then discards
  it.

  Args:
    input_fn: Input function returning a tuple of:
        features - Dictionary of string feature name to `Tensor` or `Tensor`.
        labels - `Tensor` of label values.

  Returns:
    List of `FeatureColumn` objects.
  """
  with ops.Graph().as_default():
    features, _ = input_fn()
    return layers.infer_real_valued_columns(features)


def infer_real_valued_columns_from_input(x):
  """Creates `FeatureColumn` objects for inputs defined by input `x`.

  This interprets all inputs as dense, fixed-length float values.

  Args:
    x: Real-valued matrix of shape [n_samples, n_features...]. Can be
       iterator that returns arrays of features.

  Returns:
    List of `FeatureColumn` objects.
  """
  input_fn, _ = _get_input_fn(
      x=x, y=None, input_fn=None, feed_fn=None, batch_size=None)
  return infer_real_valued_columns_from_input_fn(input_fn)


def _get_arguments(func):
  """Returns list of arguments this function has."""
  if hasattr(func, '__code__'):
    # Regular function.
    return inspect.getargspec(func).args
  elif hasattr(func, '__call__'):
    # Callable object.
    return _get_arguments(func.__call__)
  elif hasattr(func, 'func'):
    # Partial function.
    return _get_arguments(func.func)


def _get_replica_device_setter(config):
  """Creates a replica device setter if required.

  Args:
    config: A RunConfig instance.

  Returns:
    A replica device setter, or None.
  """
  ps_ops = [
      'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
      'MutableHashTableOfTensors', 'MutableDenseHashTable'
  ]

  if config.task_type:
    worker_device = '/job:%s/task:%d' % (config.task_type, config.task_id)
  else:
    worker_device = '/job:worker'

  if config.num_ps_replicas > 0:
    return device_setter.replica_device_setter(
        ps_tasks=config.num_ps_replicas, worker_device=worker_device,
        merge_devices=True, ps_ops=ps_ops, cluster=config.cluster_spec)
  else:
    return None


def _make_metrics_ops(metrics, features, labels, predictions):
  """Add metrics based on `features`, `labels`, and `predictions`.

  `metrics` contains a specification for how to run metrics. It is a dict
  mapping friendly names to either `MetricSpec` objects, or directly to a metric
  function (assuming that `predictions` and `labels` are single tensors), or to
  `(pred_name, metric)` `tuple`, which passes `predictions[pred_name]` and
  `labels` to `metric` (assuming `labels` is a single tensor).

  Users are encouraged to use `MetricSpec` objects, which are more flexible and
  cleaner. They also lead to clearer errors.

  Args:
    metrics: A dict mapping names to metrics specification, for example
      `MetricSpec` objects.
    features: A dict of tensors returned from an input_fn as features/inputs.
    labels: A single tensor or a dict of tensors returned from an input_fn as
      labels.
    predictions: A single tensor or a dict of tensors output from a model as
      predictions.

  Returns:
    A dict mapping the friendly given in `metrics` to the result of calling the
    given metric function.

  Raises:
    ValueError: If metrics specifications do not work with the type of
      `features`, `labels`, or `predictions` provided. Mostly, a dict is given
      but no pred_name specified.
  """
  metrics = metrics or {}

  # If labels is a dict with a single key, unpack into a single tensor.
  labels_tensor_or_dict = labels
  if isinstance(labels, dict) and len(labels) == 1:
    labels_tensor_or_dict = labels[list(labels.keys())[0]]

  result = {}
  # Iterate in lexicographic order, so the graph is identical among runs.
  for name, metric in sorted(six.iteritems(metrics)):
    if isinstance(metric, metric_spec.MetricSpec):
      result[name] = metric.create_metric_ops(features, labels, predictions)
      continue

    # TODO(b/31229024): Remove the rest of this loop
    logging.warning('Please specify metrics using MetricSpec. Using bare '
                    'functions or (key, fn) tuples is deprecated and support '
                    'for it will be removed on Oct 1, 2016.')

    if isinstance(name, tuple):
      # Multi-head metrics.
      if len(name) != 2:
        raise ValueError('Invalid metric for {}. It returned a tuple with '
                         'len {}, expected 2.'.format(name, len(name)))
      if not isinstance(predictions, dict):
        raise ValueError(
            'Metrics passed provide (name, prediction), '
            'but predictions are not dict. '
            'Metrics: %s, Predictions: %s.' % (metrics, predictions))
      # Here are two options: labels are single Tensor or a dict.
      if isinstance(labels, dict) and name[1] in labels:
        # If labels are dict and the prediction name is in it, apply metric.
        result[name[0]] = metric(predictions[name[1]], labels[name[1]])
      else:
        # Otherwise pass the labels to the metric.
        result[name[0]] = metric(predictions[name[1]], labels_tensor_or_dict)
    else:
      # Single head metrics.
      if isinstance(predictions, dict):
        raise ValueError(
            'Metrics passed provide only name, no prediction, '
            'but predictions are dict. '
            'Metrics: %s, Labels: %s.' % (metrics, labels_tensor_or_dict))
      result[name] = metric(predictions, labels_tensor_or_dict)
  return result


class BaseEstimator(
    sklearn.BaseEstimator, evaluable.Evaluable, trainable.Trainable):
  """Abstract BaseEstimator class to train and evaluate TensorFlow models.

  Concrete implementation of this class should provide the following functions:

    * _get_train_ops
    * _get_eval_ops
    * _get_predict_ops

  `Estimator` implemented below is a good example of how to use this class.
  """
  __metaclass__ = abc.ABCMeta

  # Note that for Google users, this is overriden with
  # learn_runner.EstimatorConfig.
  # TODO(wicke): Remove this once launcher takes over config functionality
  _Config = run_config.RunConfig  # pylint: disable=invalid-name

  def __init__(self, model_dir=None, config=None):
    """Initializes a BaseEstimator instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      config: A RunConfig instance.
    """
    # Model directory.
    self._model_dir = model_dir
    if self._model_dir is None:
      self._model_dir = tempfile.mkdtemp()
      logging.warning('Using temporary folder as model directory: %s',
                      self._model_dir)

    # Create a run configuration.
    if config is None:
      self._config = BaseEstimator._Config()
      logging.info('Using default config.')
    else:
      self._config = config
    logging.info('Using config: %s', str(vars(self._config)))

    # Set device function depending if there are replicas or not.
    self._device_fn = _get_replica_device_setter(self._config)

    # Features and labels TensorSignature objects.
    # TODO(wicke): Rename these to something more descriptive
    self._features_info = None
    self._labels_info = None

    self._graph = None

  @property
  def config(self):
    # TODO(wicke): make RunConfig immutable, and then return it without a copy.
    return copy.deepcopy(self._config)

  @deprecated_args(
      SCIKIT_DECOUPLE_DATE, SCIKIT_DECOUPLE_INSTRUCTIONS, ('x', None),
      ('y', None), ('batch_size', None)
  )
  def fit(self, x=None, y=None, input_fn=None, steps=None, batch_size=None,
          monitors=None, max_steps=None):
    # pylint: disable=g-doc-args,g-doc-return-or-yield
    """See `Trainable`.

    Raises:
      ValueError: If `x` or `y` are not `None` while `input_fn` is not `None`.
      ValueError: If both `steps` and `max_steps` are not `None`.
    """
    if (steps is not None) and (max_steps is not None):
      raise ValueError('Can not provide both steps and max_steps.')

    input_fn, feed_fn = _get_input_fn(x, y, input_fn, feed_fn=None,
                                      batch_size=batch_size, shuffle=True,
                                      epochs=None)
    loss = self._train_model(input_fn=input_fn,
                             feed_fn=feed_fn,
                             steps=steps,
                             monitors=monitors,
                             max_steps=max_steps)
    logging.info('Loss for final step: %s.', loss)
    return self

  @deprecated_args(
      SCIKIT_DECOUPLE_DATE, SCIKIT_DECOUPLE_INSTRUCTIONS, ('x', None),
      ('y', None), ('batch_size', None)
  )
  def partial_fit(
      self, x=None, y=None, input_fn=None, steps=1, batch_size=None,
      monitors=None):
    """Incremental fit on a batch of samples.

    This method is expected to be called several times consecutively
    on different or the same chunks of the dataset. This either can
    implement iterative training or out-of-core/online training.

    This is especially useful when the whole dataset is too big to
    fit in memory at the same time. Or when model is taking long time
    to converge, and you want to split up training into subparts.

    Args:
      x: Matrix of shape [n_samples, n_features...]. Can be iterator that
         returns arrays of features. The training input samples for fitting the
         model. If set, `input_fn` must be `None`.
      y: Vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
         iterator that returns array of labels. The training label values
         (class labels in classification, real numbers in regression). If set,
         `input_fn` must be `None`.
      input_fn: Input function. If set, `x`, `y`, and `batch_size` must be
        `None`.
      steps: Number of steps for which to train model. If `None`, train forever.
      batch_size: minibatch size to use on the input, defaults to first
        dimension of `x`. Must be `None` if `input_fn` is provided.
      monitors: List of `BaseMonitor` subclass instances. Used for callbacks
        inside the training loop.

    Returns:
      `self`, for chaining.

    Raises:
      ValueError: If at least one of `x` and `y` is provided, and `input_fn` is
          provided.
    """
    logging.warning('The current implementation of partial_fit is not optimized'
                    ' for use in a loop. Consider using fit() instead.')
    return self.fit(x=x, y=y, input_fn=input_fn, steps=steps,
                    batch_size=batch_size, monitors=monitors)

  @deprecated_args(
      SCIKIT_DECOUPLE_DATE, SCIKIT_DECOUPLE_INSTRUCTIONS, ('x', None),
      ('y', None), ('batch_size', None)
  )
  def evaluate(
      self, x=None, y=None, input_fn=None, feed_fn=None, batch_size=None,
      steps=None, metrics=None, name=None, checkpoint_path=None):
    # pylint: disable=g-doc-args,g-doc-return-or-yield
    """See `Evaluable`.

    Raises:
      ValueError: If at least one of `x` or `y` is provided, and at least one of
          `input_fn` or `feed_fn` is provided.
          Or if `metrics` is not `None` or `dict`.
    """
    input_fn, feed_fn = _get_input_fn(x, y, input_fn=input_fn,
                                      feed_fn=feed_fn, batch_size=batch_size,
                                      shuffle=False, epochs=1)
    if metrics is not None and not isinstance(metrics, dict):
      raise ValueError('Metrics argument should be None or dict. '
                       'Got %s.' % metrics)
    eval_results, global_step = self._evaluate_model(
        input_fn=input_fn,
        feed_fn=feed_fn,
        steps=steps,
        metrics=metrics,
        name=name,
        checkpoint_path=checkpoint_path)
    if eval_results is not None:
      eval_results.update({'global_step': global_step})
    return eval_results

  @deprecated_args(
      SCIKIT_DECOUPLE_DATE, SCIKIT_DECOUPLE_INSTRUCTIONS, ('x', None),
      ('batch_size', None), ('as_iterable', True)
  )
  def predict(
      self, x=None, input_fn=None, batch_size=None, outputs=None,
      as_iterable=True):
    """Returns predictions for given features.

    Args:
      x: Matrix of shape [n_samples, n_features...]. Can be iterator that
         returns arrays of features. The training input samples for fitting the
         model. If set, `input_fn` must be `None`.
      input_fn: Input function. If set, `x` and 'batch_size' must be `None`.
      batch_size: Override default batch size. If set, 'input_fn' must be
        'None'.
      outputs: list of `str`, name of the output to predict.
        If `None`, returns all.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      A numpy array of predicted classes or regression values if the
      constructor's `model_fn` returns a `Tensor` for `predictions` or a `dict`
      of numpy arrays if `model_fn` returns a `dict`. Returns an iterable of
      predictions if as_iterable is True.

    Raises:
      ValueError: If x and input_fn are both provided or both `None`.
    """
    input_fn, feed_fn = _get_input_fn(
        x, None, input_fn=input_fn, feed_fn=None, batch_size=batch_size,
        shuffle=False, epochs=1)
    return self._infer_model(
        input_fn=input_fn, feed_fn=feed_fn, outputs=outputs,
        as_iterable=as_iterable)

  def get_variable_value(self, name):
    """Returns value of the variable given by name.

    Args:
      name: string, name of the tensor.

    Returns:
      Numpy array - value of the tensor.
    """
    return load_variable(self.model_dir, name)

  def get_variable_names(self):
    """Returns list of all variable names in this model.

    Returns:
      List of names.
    """
    return [name for name, _ in list_variables(self.model_dir)]

  @property
  def model_dir(self):
    return self._model_dir

  @deprecated_arg_values(
      '2016-09-23',
      'The signature of the input_fn accepted by export is changing to be '
      'consistent with what\'s used by tf.Learn Estimator\'s train/evaluate. '
      'input_fn (and in most cases, input_feature_key) will become required '
      'args, and use_deprecated_input_fn will default to False and be removed '
      'altogether.',
      use_deprecated_input_fn=True,
      input_fn=None)
  def export(self,
             export_dir,
             input_fn=export._default_input_fn,  # pylint: disable=protected-access
             input_feature_key=None,
             use_deprecated_input_fn=True,
             signature_fn=None,
             prediction_key=None,
             default_batch_size=1,
             exports_to_keep=None):
    """Exports inference graph into given dir.

    Args:
      export_dir: A string containing a directory to write the exported graph
        and checkpoints.
      input_fn: If `use_deprecated_input_fn` is true, then a function that given
        `Tensor` of `Example` strings, parses it into features that are then
        passed to the model. Otherwise, a function that takes no argument and
        returns a tuple of (features, labels), where features is a dict of
        string key to `Tensor` and labels is a `Tensor` that's currently not
        used (and so can be `None`).
      input_feature_key: Only used if `use_deprecated_input_fn` is false. String
        key into the features dict returned by `input_fn` that corresponds to a
        the raw `Example` strings `Tensor` that the exported model will take as
        input. Can only be `None` if you're using a custom `signature_fn` that
        does not use the first arg (examples).
      use_deprecated_input_fn: Determines the signature format of `input_fn`.
      signature_fn: Function that returns a default signature and a named
        signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
        for features and `Tensor` or `dict` of `Tensor`s for predictions.
      prediction_key: The key for a tensor in the `predictions` dict (output
        from the `model_fn`) to use as the `predictions` input to the
        `signature_fn`. Optional. If `None`, predictions will pass to
        `signature_fn` without filtering.
      default_batch_size: Default batch size of the `Example` placeholder.
      exports_to_keep: Number of exports to keep.

    Returns:
      The string path to the exported directory. NB: this functionality was
      added ca. 2016/09/25; clients that depend on the return value may need
      to handle the case where this function returns None because subclasses
      are not returning a value.
    """
    # pylint: disable=protected-access
    return export._export_estimator(
        estimator=self,
        export_dir=export_dir,
        signature_fn=signature_fn,
        prediction_key=prediction_key,
        input_fn=input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=use_deprecated_input_fn,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)

  @abc.abstractproperty
  def _get_train_ops(self, features, labels):
    """Method that builds model graph and returns trainer ops.

    Expected to be overridden by sub-classes that require custom support.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      labels: `Tensor` or `dict` of `Tensor` objects.

    Returns:
      A `ModelFnOps` object.
    """
    pass

  @abc.abstractproperty
  def _get_predict_ops(self, features):
    """Method that builds model graph and returns prediction ops.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.

    Returns:
      A `ModelFnOps` object.
    """
    pass

  def _get_eval_ops(self, features, labels, metrics):
    """Method that builds model graph and returns evaluation ops.

    Expected to be overriden by sub-classes that require custom support.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      labels: `Tensor` or `dict` of `Tensor` objects.
      metrics: Dict of metrics to run. If None, the default metric functions
        are used; if {}, no metrics are used. Otherwise, `metrics` should map
        friendly names for the metric to a `MetricSpec` object defining which
        model outputs to evaluate against which labels with which metric
        function. Metric ops should support streaming, e.g., returning
        update_op and value tensors. See more details in
        `../../../../metrics/python/metrics/ops/streaming_metrics.py` and
        `../metric_spec.py`.

    Returns:
      A `ModelFnOps` object.
    """
    raise NotImplementedError('_get_eval_ops not implemented in BaseEstimator')

  @deprecated(
      '2016-09-23',
      'The signature of the input_fn accepted by export is changing to be '
      'consistent with what\'s used by tf.Learn Estimator\'s train/evaluate, '
      'which makes this function useless. This will be removed after the '
      'deprecation date.')
  def _get_feature_ops_from_example(self, examples_batch):
    """Returns feature parser for given example batch using features info.

    This function requires `fit()` has been called.

    Args:
      examples_batch: batch of tf.Example

    Returns:
      features: `Tensor` or `dict` of `Tensor` objects.

    Raises:
      ValueError: If `_features_info` attribute is not available (usually
      because `fit()` has not been called).
    """
    if self._features_info is None:
      raise ValueError('Features information missing, was fit() ever called?')
    return tensor_signature.create_example_parser_from_signatures(
        self._features_info, examples_batch)

  def _check_inputs(self, features, labels):
    if self._features_info is not None:
      logging.debug('Given features: %s, required signatures: %s.',
                    str(features), str(self._features_info))
      if not tensor_signature.tensors_compatible(features, self._features_info):
        raise ValueError('Features are incompatible with given information. '
                         'Given features: %s, required signatures: %s.' %
                         (str(features), str(self._features_info)))
    else:
      self._features_info = tensor_signature.create_signatures(features)
      logging.debug('Setting feature info to %s.', str(self._features_info))
    if labels is not None:
      if self._labels_info is not None:
        logging.debug('Given labels: %s, required signatures: %s.',
                      str(labels), str(self._labels_info))
        if not tensor_signature.tensors_compatible(labels, self._labels_info):
          raise ValueError('Labels are incompatible with given information. '
                           'Given labels: %s, required signatures: %s.' %
                           (str(labels), str(self._labels_info)))
      else:
        self._labels_info = tensor_signature.create_signatures(labels)
        logging.debug('Setting labels info to %s', str(self._labels_info))

  def _train_model(self,
                   input_fn,
                   steps,
                   feed_fn=None,
                   init_op=None,
                   init_feed_fn=None,
                   init_fn=None,
                   device_fn=None,
                   monitors=None,
                   log_every_steps=100,
                   fail_on_nan_loss=True,
                   max_steps=None):
    # TODO(wicke): Remove this once Model and associated code are gone.
    if hasattr(self._config, 'execution_mode'):
      if self._config.execution_mode not in ('all', 'train'):
        return

      # Stagger startup of worker sessions based on task id.
      sleep_secs = min(
          self._config.training_worker_max_startup_secs,
          self._config.task_id *
          self._config.training_worker_session_startup_stagger_secs)
      if sleep_secs:
        logging.info('Waiting %d secs before starting task %d.', sleep_secs,
                     self._config.task_id)
        time.sleep(sleep_secs)

    # Device allocation
    device_fn = device_fn or self._device_fn

    self._graph = ops.Graph()
    with self._graph.as_default() as g, g.device(device_fn):
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step = contrib_framework.create_global_step(g)
      features, labels = input_fn()
      self._check_inputs(features, labels)

      # The default return type of _get_train_ops is ModelFnOps. But there are
      # some subclasses of tf.contrib.learn.Estimator which override this
      # method and use the legacy signature, namely _get_train_ops returns a
      # (train_op, loss) tuple. The following else-statement code covers these
      # cases, but will soon be deleted after the subclasses are updated.
      # TODO(b/32664904): Update subclasses and delete the else-statement.
      train_ops = self._get_train_ops(features, labels)
      if isinstance(train_ops, model_fn_lib.ModelFnOps):  # Default signature
        train_op = train_ops.train_op
        loss_op = train_ops.loss
        if self.config.is_chief:
          hooks = train_ops.training_chief_hooks + train_ops.training_hooks
        else:
          hooks = train_ops.training_hooks
      else:  # Legacy signature
        if len(train_ops) != 2:
          raise ValueError('Expected a tuple of train_op and loss, got {}'.
                           format(train_ops))
        train_op = train_ops[0]
        loss_op = train_ops[1]
        hooks = []

      hooks += monitor_lib.replace_monitors_with_hooks(monitors, self)

      ops.add_to_collection(ops.GraphKeys.LOSSES, loss_op)
      return graph_actions._monitored_train(  # pylint: disable=protected-access
          graph=g,
          output_dir=self._model_dir,
          train_op=train_op,
          loss_op=loss_op,
          global_step_tensor=global_step,
          init_op=init_op,
          init_feed_dict=init_feed_fn() if init_feed_fn is not None else None,
          init_fn=init_fn,
          log_every_steps=log_every_steps,
          supervisor_is_chief=self.config.is_chief,
          supervisor_master=self._config.master,
          supervisor_save_model_secs=self._config.save_checkpoints_secs,
          supervisor_save_model_steps=self._config.save_checkpoints_steps,
          supervisor_save_summaries_steps=self._config.save_summary_steps,
          keep_checkpoint_max=self._config.keep_checkpoint_max,
          feed_fn=feed_fn,
          steps=steps,
          fail_on_nan_loss=fail_on_nan_loss,
          hooks=hooks,
          max_steps=max_steps)

  def _extract_metric_update_ops(self, eval_dict):
    """Separate update operations from metric value operations."""
    update_ops = []
    value_ops = {}
    for name, metric_ops in six.iteritems(eval_dict):
      if isinstance(metric_ops, (list, tuple)):
        if len(metric_ops) == 2:
          value_ops[name] = metric_ops[0]
          update_ops.append(metric_ops[1])
        else:
          logging.warning(
              'Ignoring metric {}. It returned a list|tuple with len {}, '
              'expected 2'.format(name, len(metric_ops)))
          value_ops[name] = metric_ops
      else:
        value_ops[name] = metric_ops

    if update_ops:
      update_ops = control_flow_ops.group(*update_ops)
    else:
      update_ops = None

    return update_ops, value_ops

  def _evaluate_model(self,
                      input_fn,
                      steps,
                      feed_fn=None,
                      metrics=None,
                      name='',
                      checkpoint_path=None):
    # TODO(wicke): Remove this once Model and associated code are gone.
    if (hasattr(self._config, 'execution_mode') and
        self._config.execution_mode not in ('all', 'evaluate', 'eval_evalset')):
      return None, None

    # Check that model has been trained (if nothing has been set explicitly).
    if not checkpoint_path:
      latest_path = saver.latest_checkpoint(self._model_dir)
      if not latest_path:
        raise NotFittedError("Couldn't find trained model at %s."
                             % self._model_dir)
      checkpoint_path = self._model_dir

    # Setup output directory.
    eval_dir = os.path.join(self._model_dir, 'eval' if not name else
                            'eval_' + name)

    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step = contrib_framework.create_global_step(g)
      features, labels = input_fn()
      self._check_inputs(features, labels)

      # The default return type of _get_eval_ops is ModelFnOps. But there are
      # some subclasses of tf.contrib.learn.Estimator which override this
      # method and use the legacy signature, namely _get_eval_ops returns an
      # `eval_dict` dictionary of Tensors. The following else-statement code
      # covers these cases, but will soon be deleted after the subclasses are
      # updated.
      # TODO(b/32664904): Update subclasses and delete the else-statement.
      eval_ops = self._get_eval_ops(features, labels, metrics)
      if isinstance(eval_ops, model_fn_lib.ModelFnOps):  # Default signature
        eval_dict = eval_ops.eval_metric_ops
      else:  # Legacy signature
        eval_dict = eval_ops

      update_op, eval_dict = self._extract_metric_update_ops(eval_dict)
      eval_results, current_global_step = graph_actions.evaluate(
          graph=g,
          output_dir=eval_dir,
          checkpoint_path=checkpoint_path,
          eval_dict=eval_dict,
          update_op=update_op,
          global_step_tensor=global_step,
          supervisor_master=self._config.evaluation_master,
          feed_fn=feed_fn,
          max_steps=steps)

      return eval_results, current_global_step

  def _get_features_from_input_fn(self, input_fn):
    result = input_fn()
    if isinstance(result, (list, tuple)):
      return result[0]
    return result

  def _infer_model(
      self, input_fn, feed_fn=None, outputs=None, as_iterable=True):
    # Check that model has been trained.
    checkpoint_path = saver.latest_checkpoint(self._model_dir)
    if not checkpoint_path:
      raise NotFittedError("Couldn't find trained model at %s."
                           % self._model_dir)

    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(self._config.tf_random_seed)
      contrib_framework.create_global_step(g)
      features = self._get_features_from_input_fn(input_fn)

      # The default return type of _get_predict_ops is ModelFnOps. But there are
      # some subclasses of tf.contrib.learn.Estimator which override this
      # method and use the legacy signature, namely _get_predict_ops returns a
      # `predictions` Tensor or dict or Tensors. The following else-statement
      # code covers these cases, but will soon be deleted after the subclasses
      # are updated.
      # TODO(b/32664904): Update subclasses and delete the else-statement.
      infer_ops = self._get_predict_ops(features)
      if isinstance(infer_ops, model_fn_lib.ModelFnOps):  # Default signature
        predictions = infer_ops.predictions
      else:  # Legacy signature
        predictions = infer_ops

      # If predictions is single output - wrap it into dict, and remember to
      # return not a dict.
      return_dict = isinstance(predictions, dict)
      if not return_dict:
        predictions = {'predictions': predictions}

      # Filter what to run predictions on, if outputs provided.
      if outputs:
        existing_keys = predictions.keys()
        predictions = {
            key: value
            for key, value in six.iteritems(predictions) if key in outputs
        }
        if not predictions:
          raise ValueError('Expected to run at least one output from %s, '
                           'provided %s.' % (existing_keys, outputs))

      if as_iterable:
        return self._infer_model_as_iterable(
            checkpoint_path, predictions, feed_fn, return_dict)
      else:
        return self._infer_model_single(
            checkpoint_path, predictions, feed_fn, return_dict)

  def _infer_model_single(
      self, checkpoint_path, predictions, feed_fn, return_dict):
    if feed_fn is None:
      preds = graph_actions.infer(checkpoint_path, predictions)
    else:
      def _feed_fn():
        while True:
          yield feed_fn()

      outputs = graph_actions.run_feeds(
          output_dict=predictions,
          feed_dicts=_feed_fn(),
          restore_checkpoint_path=checkpoint_path)
      preds = {
          key: np.concatenate([output[key] for output in outputs], axis=0)
          for key in predictions}

    return preds if return_dict else preds['predictions']

  def _infer_model_as_iterable(
      self, checkpoint_path, predictions, feed_fn, return_dict):
    if feed_fn is None:
      # If there are no queue_runners, the input `predictions` is a
      # constant, and we should stop after the first epoch.  If,
      # instead, there are queue_runners, eventually they should throw
      # an `OutOfRangeError`.
      graph = contrib_ops.get_graph_from_inputs(predictions.values())
      if graph.get_collection(ops.GraphKeys.QUEUE_RUNNERS):
        feed_dicts = itertools.repeat(None)
      else:
        feed_dicts = [None]
    else:
      def _feed_fn():
        while True:
          yield feed_fn()
      feed_dicts = _feed_fn()

    try:
      for output_batch in graph_actions.run_feeds_iter(
          output_dict=predictions,
          feed_dicts=feed_dicts,
          restore_checkpoint_path=checkpoint_path):
        # Unpack batches into individual predictions
        if return_dict:
          batch_length = list(output_batch.values())[0].shape[0]
          for i in range(batch_length):
            yield {key: value[i] for key, value in six.iteritems(output_batch)}
        else:
          for pred in output_batch['predictions']:
            yield pred

    except errors.OutOfRangeError:
      # We fall out of the above loop naturally if feed_fn raises StopIteration,
      # or we catch an OutOfRangeError if we've reached the end of inputs.
      logging.info('Reached end of inputs for predict_iter.')


def _identity_feature_engineering_fn(features, labels):
  return features, labels


class Estimator(BaseEstimator):
  """Estimator class is the basic TensorFlow model trainer/evaluator.
  """

  def __init__(self,
               model_fn=None,
               model_dir=None,
               config=None,
               params=None,
               feature_engineering_fn=None):
    """Constructs an `Estimator` instance.

    Args:
      model_fn: Model function. Follows the signature:
        * Args:
          * `features` are single `Tensor` or `dict` of `Tensor`s
                 (depending on data passed to `fit`),
          * `labels` are `Tensor` or `dict` of `Tensor`s (for multi-head
                 models). If mode is `ModeKeys.INFER`, `labels=None` will be
                 passed. If the `model_fn`'s signature does not accept
                 `mode`, the `model_fn` must still be able to handle
                 `labels=None`.
          * `mode` specifies if this training, evaluation or
                 prediction. See `ModeKeys`.
          * `params` is a `dict` of hyperparameters.  Will receive what
                 is passed to Estimator in `params` parameter. This allows
                 to configure Estimators from hyper parameter tuning.
          * `config` is a Configuration object. Will receive what is passed to
                 Estimator in `config` parameter. This allows updating things in
                 your model_fn based on configuration such as num_ps_replicas.

        * Returns:
          `ModelFnOps`

        Also supports a legacy signature which returns tuple of:

          * predictions: `Tensor`, `SparseTensor` or dictionary of same.
              Can also be any type that is convertible to a `Tensor` or
              `SparseTensor`, or dictionary of same.
          * loss: Scalar loss `Tensor`.
          * train_op: Training update `Tensor` or `Operation`.

        Supports next three signatures for the function:

          * `(features, labels) -> (predictions, loss, train_op)`
          * `(features, labels, mode) -> (predictions, loss, train_op)`
          * `(features, labels, mode, params) -> (predictions, loss, train_op)`
          * `(features, labels, mode, params, config) ->
             (predictions, loss, train_op)`

      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      config: Configuration object.
      params: `dict` of hyper parameters that will be passed into `model_fn`.
              Keys are names of parameters, values are basic python types.
      feature_engineering_fn: Feature engineering function. Takes features and
                              labels which are the output of `input_fn` and
                              returns features and labels which will be fed
                              into `model_fn`. Please check `model_fn` for
                              a definition of features and labels.

    Raises:
      ValueError: parameters of `model_fn` don't match `params`.
    """
    super(Estimator, self).__init__(model_dir=model_dir, config=config)
    if model_fn is not None:
      # Check number of arguments of the given function matches requirements.
      model_fn_args = _get_arguments(model_fn)
      if params is not None and 'params' not in model_fn_args:
        raise ValueError('Estimator\'s model_fn (%s) has less than 4 '
                         'arguments, but not None params (%s) are passed.' %
                         (model_fn, params))
      if params is None and 'params' in model_fn_args:
        logging.warning('Estimator\'s model_fn (%s) includes params '
                        'argument, but params are not passed to Estimator.',
                        model_fn)
    self._model_fn = model_fn
    self.params = params
    self._feature_engineering_fn = (
        feature_engineering_fn or _identity_feature_engineering_fn)

  def _call_model_fn(self, features, labels, mode):
    """Calls model function with support of 2, 3 or 4 arguments.

    Args:
      features: features dict.
      labels: labels dict.
      mode: ModeKeys

    Returns:
      A `ModelFnOps` object. If model_fn returns a tuple, wraps them up in a
      `ModelFnOps` object.

    Raises:
      ValueError: if model_fn returns invalid objects.
    """
    features, labels = self._feature_engineering_fn(features, labels)
    model_fn_args = _get_arguments(self._model_fn)
    kwargs = {}
    if 'mode' in model_fn_args:
      kwargs['mode'] = mode
    if 'params' in model_fn_args:
      kwargs['params'] = self.params
    if 'config' in model_fn_args:
      kwargs['config'] = self.config
    model_fn_results = self._model_fn(features, labels, **kwargs)

    if isinstance(model_fn_results, model_fn_lib.ModelFnOps):
      return model_fn_results

    # Here model_fn_ops should be a tuple with 3 elements.
    if len(model_fn_results) != 3:
      raise ValueError('Unrecognized value returned by model_fn, '
                       'please return ModelFnOps.')
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=model_fn_results[0],
        loss=model_fn_results[1],
        train_op=model_fn_results[2])

  def _get_train_ops(self, features, labels):
    """Method that builds model graph and returns trainer ops.

    Expected to be overriden by sub-classes that require custom support.
    This implementation uses `model_fn` passed as parameter to constructor to
    build model.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      labels: `Tensor` or `dict` of `Tensor` objects.

    Returns:
      `ModelFnOps` object.
    """
    return self._call_model_fn(features, labels, model_fn_lib.ModeKeys.TRAIN)

  # TODO(ispir): delete this function after converting all legacy usages.
  def _call_legacy_get_train_ops(self, features, labels):
    train_ops = self._get_train_ops(features, labels)
    if isinstance(train_ops, model_fn_lib.ModelFnOps):  # Default signature
      return train_ops
    return model_fn_lib.ModelFnOps(
        mode=model_fn_lib.ModeKeys.TRAIN,
        predictions=None,
        loss=train_ops[1],
        train_op=train_ops[0])

  def _get_eval_ops(self, features, labels, metrics):
    """Method that builds model graph and returns evaluation ops.

    Expected to be overriden by sub-classes that require custom support.
    This implementation uses `model_fn` passed as parameter to constructor to
    build model.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      labels: `Tensor` or `dict` of `Tensor` objects.
      metrics: Dict of metrics to run. If None, the default metric functions
        are used; if {}, no metrics are used. Otherwise, `metrics` should map
        friendly names for the metric to a `MetricSpec` object defining which
        model outputs to evaluate against which labels with which metric
        function. Metric ops should support streaming, e.g., returning
        update_op and value tensors. See more details in
        `../../../../metrics/python/metrics/ops/streaming_metrics.py` and
        `../metric_spec.py`.

    Returns:
      `ModelFnOps` object.

    Raises:
      ValueError: if `metrics` don't match `labels`.
    """
    model_fn_ops = self._call_model_fn(
        features, labels, model_fn_lib.ModeKeys.EVAL)

    # Custom metrics should overwrite defaults.
    if metrics:
      model_fn_ops.eval_metric_ops.update(_make_metrics_ops(
          metrics, features, labels, model_fn_ops.predictions))

    if metric_key.MetricKey.LOSS not in model_fn_ops.eval_metric_ops:
      model_fn_ops.eval_metric_ops[metric_key.MetricKey.LOSS] = (
          metrics_lib.streaming_mean(model_fn_ops.loss))
    return model_fn_ops

  def _get_predict_ops(self, features):
    """Method that builds model graph and returns prediction ops.

    Expected to be overriden by sub-classes that require custom support.
    This implementation uses `model_fn` passed as parameter to constructor to
    build model.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.

    Returns:
      `ModelFnOps` object.
    """
    labels = tensor_signature.create_placeholders_from_signatures(
        self._labels_info)
    return self._call_model_fn(features, labels, model_fn_lib.ModeKeys.INFER)

  @experimental
  def export_savedmodel(
      self, export_dir_base, input_fn,
      default_output_alternative_key=None,
      assets_extra=None,
      as_text=False,
      exports_to_keep=None):
    """Exports inference graph as a SavedModel into given dir.

    Args:
      export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
      input_fn: A function that takes no argument and
        returns an `InputFnOps`.
      default_output_alternative_key: the name of the head to serve when none is
        specified.
      assets_extra: A dict specifying how to populate the assets.extra directory
        within the exported SavedModel.  Each key should give the destination
        path (including the filename) relative to the assets.extra directory.
        The corresponding value gives the full path of the source file to be
        copied.  For example, the simple case of copying a single file without
        renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format.
      exports_to_keep: Number of exports to keep.

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: if an unrecognized export_type is requested.
    """
    if input_fn is None:
      raise ValueError('input_fn must be defined.')

    with ops.Graph().as_default() as g:
      contrib_variables.create_global_step(g)

      # Call the input_fn and collect the input alternatives.
      input_ops = input_fn()
      input_alternatives, features = (
          saved_model_export_utils.get_input_alternatives(input_ops))

      # Call the model_fn and collect the output alternatives.
      model_fn_ops = self._call_model_fn(features, None,
                                         model_fn_lib.ModeKeys.INFER)
      output_alternatives, actual_default_output_alternative_key = (
          saved_model_export_utils.get_output_alternatives(
              model_fn_ops, default_output_alternative_key))

      # Build the SignatureDefs from all pairs of input and output signatures
      signature_def_map = saved_model_export_utils.build_all_signature_defs(
          input_alternatives, output_alternatives,
          actual_default_output_alternative_key)

      # Locate the latest checkpoint
      # TODO(soergel): does it help that we know we have one from this step?
      checkpoint_path = saver.latest_checkpoint(self._model_dir)
      if not checkpoint_path:
        raise NotFittedError("Couldn't find trained model at %s."
                             % self._model_dir)

      export_dir = saved_model_export_utils.get_timestamped_export_dir(
          export_dir_base)

      with tf_session.Session('') as session:
        variables.initialize_local_variables()
        data_flow_ops.initialize_all_tables()
        saver_for_restore = saver.Saver(
            variables.global_variables(),
            sharded=True)
        saver_for_restore.restore(session, checkpoint_path)

        init_op = control_flow_ops.group(
            variables.local_variables_initializer(),
            data_flow_ops.initialize_all_tables())

        # Perform the export
        builder = saved_model_builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            signature_def_map=signature_def_map,
            assets_collection=ops.get_collection(
                ops.GraphKeys.ASSET_FILEPATHS),
            legacy_init_op=init_op)
        builder.save(as_text)

      # Add the extra assets
      if assets_extra:
        assets_extra_path = os.path.join(compat.as_bytes(export_dir),
                                         compat.as_bytes('assets.extra'))
        for dest_relative, source in assets_extra.items():
          dest_absolute = os.path.join(compat.as_bytes(assets_extra_path),
                                       compat.as_bytes(dest_relative))
          dest_path = os.path.dirname(dest_absolute)
          gfile.MakeDirs(dest_path)
          gfile.Copy(source, dest_absolute)

      return export_dir

  @deprecated_args(SCIKIT_DECOUPLE_DATE, SCIKIT_DECOUPLE_INSTRUCTIONS, 'x', 'y',
                   'batch_size')
  def fit(self,
          x=None,
          y=None,
          input_fn=None,
          steps=None,
          batch_size=None,
          monitors=None,
          max_steps=None):
    # pylint: disable=g-doc-args,g-doc-return-or-yield
    """See `Trainable`.

    Raises:
      ValueError: If `x` or `y` are not `None` while `input_fn` is not `None`.
      ValueError: If both `steps` and `max_steps` are not `None`.
    """
    if (steps is not None) and (max_steps is not None):
      raise ValueError('Can not provide both steps and max_steps.')
    if max_steps is not None:
      try:
        start_step = load_variable(self._model_dir, ops.GraphKeys.GLOBAL_STEP)
        if max_steps <= start_step:
          logging.info('Skipping training since max_steps has already saved.')
          return None
      except:  # pylint: disable=bare-except
        pass

    hooks = monitor_lib.replace_monitors_with_hooks(monitors, self)
    if steps is not None or max_steps is not None:
      hooks.append(basic_session_run_hooks.StopAtStepHook(steps, max_steps))

    input_fn, feed_fn = _get_input_fn(
        x,
        y,
        input_fn,
        feed_fn=None,
        batch_size=batch_size,
        shuffle=True,
        epochs=None)
    if feed_fn:
      hooks.append(_FeedFnHook(feed_fn))
    loss = self._train_model_v2(input_fn=input_fn, hooks=hooks)
    logging.info('Loss for final step: %s.', loss)
    return self

  def _train_model_v2(self, input_fn, hooks):
    all_hooks = []
    self._graph = ops.Graph()
    with self._graph.as_default() as g, g.device(self._device_fn):
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step = contrib_framework.create_global_step(g)
      features, labels = input_fn()
      self._check_inputs(features, labels)
      model_fn_ops = self._call_legacy_get_train_ops(features, labels)
      ops.add_to_collection(ops.GraphKeys.LOSSES, model_fn_ops.loss)
      all_hooks.extend([
          basic_session_run_hooks.NanTensorHook(model_fn_ops.loss),
          basic_session_run_hooks.LoggingTensorHook(
              {
                  'loss': model_fn_ops.loss,
                  'step': global_step
              },
              every_n_iter=100)
      ])
      all_hooks.extend(hooks)

      scaffold = model_fn_ops.training_scaffold or monitored_session.Scaffold()
      if not (scaffold.saver or ops.get_collection(ops.GraphKeys.SAVERS)):
        ops.add_to_collection(
            ops.GraphKeys.SAVERS,
            saver.Saver(
                sharded=True,
                max_to_keep=self._config.keep_checkpoint_max,
                defer_build=True))

      chief_hooks = []
      if (self._config.save_checkpoints_secs or
          self._config.save_checkpoints_steps):
        saver_hook_exists = any([
            isinstance(h, basic_session_run_hooks.CheckpointSaverHook)
            for h in (all_hooks + model_fn_ops.training_hooks + chief_hooks +
                      model_fn_ops.training_chief_hooks)
        ])
        if not saver_hook_exists:
          chief_hooks = [
              basic_session_run_hooks.CheckpointSaverHook(
                  self._model_dir,
                  save_secs=self._config.save_checkpoints_secs,
                  save_steps=self._config.save_checkpoints_steps,
                  scaffold=scaffold)
          ]
      with monitored_session.MonitoredTrainingSession(
          master=self._config.master,
          is_chief=self._config.is_chief,
          checkpoint_dir=self._model_dir,
          scaffold=scaffold,
          hooks=all_hooks + model_fn_ops.training_hooks,
          chief_only_hooks=chief_hooks + model_fn_ops.training_chief_hooks,
          save_checkpoint_secs=0,  # Saving is handled by a hook.
          save_summaries_steps=self._config.save_summary_steps,
          config=None) as mon_sess:
        loss = None
        while not mon_sess.should_stop():
          _, loss = mon_sess.run([model_fn_ops.train_op, model_fn_ops.loss])
      summary_io.SummaryWriterCache.clear()
      return loss


class _FeedFnHook(session_run_hook.SessionRunHook):
  """Runs feed_fn and sets the feed_dict accordingly."""

  def __init__(self, feed_fn):
    self.feed_fn = feed_fn

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return session_run_hook.SessionRunArgs(
        fetches=None, feed_dict=self.feed_fn())


# For time of deprecation x,y from Estimator allow direct access.
# pylint: disable=protected-access
class SKCompat(sklearn.BaseEstimator):
  """Scikit learn wrapper for TensorFlow Learn Estimator."""

  def __init__(self, estimator):
    self._estimator = estimator

  def fit(self, x, y, batch_size=128, steps=None, max_steps=None,
          monitors=None):
    input_fn, feed_fn = _get_input_fn(x, y, input_fn=None, feed_fn=None,
                                      batch_size=batch_size, shuffle=True,
                                      epochs=None)
    all_monitors = []
    if feed_fn:
      all_monitors = [_FeedFnHook(feed_fn)]
    if monitors:
      all_monitors.extend(monitors)

    self._estimator.fit(input_fn=input_fn,
                        steps=steps,
                        max_steps=max_steps,
                        monitors=all_monitors)
    return self

  def score(self, x, y, batch_size=128, steps=None, metrics=None):
    input_fn, feed_fn = _get_input_fn(x, y, input_fn=None,
                                      feed_fn=None, batch_size=batch_size,
                                      shuffle=False, epochs=1)
    if metrics is not None and not isinstance(metrics, dict):
      raise ValueError('Metrics argument should be None or dict. '
                       'Got %s.' % metrics)
    eval_results, global_step = self._estimator._evaluate_model(
        input_fn=input_fn,
        feed_fn=feed_fn,
        steps=steps,
        metrics=metrics,
        name='score')
    if eval_results is not None:
      eval_results.update({'global_step': global_step})
    return eval_results

  def predict(self, x, batch_size=128, outputs=None):
    input_fn, feed_fn = _get_input_fn(
        x, None, input_fn=None, feed_fn=None, batch_size=batch_size,
        shuffle=False, epochs=1)
    return self._estimator._infer_model(
        input_fn=input_fn, feed_fn=feed_fn, outputs=outputs,
        as_iterable=False)
