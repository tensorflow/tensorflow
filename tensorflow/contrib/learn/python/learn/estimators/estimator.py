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
"""Base Estimator class (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import copy
import os
import tempfile

import numpy as np
import six

from google.protobuf import message
from tensorflow.contrib import layers
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_args
from tensorflow.contrib.framework import list_variables
from tensorflow.contrib.framework import load_variable
from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators import _sklearn as sklearn
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import metric_key
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.estimators import tensor_signature
from tensorflow.contrib.learn.python.learn.estimators._sklearn import NotFittedError
from tensorflow.contrib.learn.python.learn.learn_io import data_feeder
from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.contrib.meta_graph_transform import meta_graph_transform
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.summary import summary as core_summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import device_setter
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import training_util
from tensorflow.python.util import compat
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

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


def _verify_input_args(x, y, input_fn, feed_fn, batch_size):
  """Verifies validity of co-existence of input arguments."""
  if input_fn is None:
    if x is None:
      raise ValueError('Either x or input_fn must be provided.')

    if tensor_util.is_tensor(x) or y is not None and tensor_util.is_tensor(y):
      raise ValueError('Inputs cannot be tensors. Please provide input_fn.')

    if feed_fn is not None:
      raise ValueError('Can not provide both feed_fn and x or y.')
  else:
    if (x is not None) or (y is not None):
      raise ValueError('Can not provide both input_fn and x or y.')
    if batch_size is not None:
      raise ValueError('Can not provide both input_fn and batch_size.')


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
  _verify_input_args(x, y, input_fn, feed_fn, batch_size)
  if input_fn is not None:
    return input_fn, feed_fn
  df = data_feeder.setup_train_data_feeder(
      x,
      y,
      n_classes=None,
      batch_size=batch_size,
      shuffle=shuffle,
      epochs=epochs)
  return df.input_builder, df.get_feed_dict_fn()


@deprecated(None, 'Please specify feature columns explicitly.')
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


@deprecated(None, 'Please specify feature columns explicitly.')
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


def _model_fn_args(fn):
  """Get argument names for function-like object.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `tuple` of string argument names.

  Raises:
    ValueError: if partial function has positionally bound arguments
  """
  _, fn = tf_decorator.unwrap(fn)
  if hasattr(fn, 'func') and hasattr(fn, 'keywords') and hasattr(fn, 'args'):
    # Handle functools.partial and similar objects.
    return tuple([
        arg for arg in tf_inspect.getargspec(fn.func).args[len(fn.args):]
        if arg not in set(fn.keywords.keys())
    ])
  # Handle function.
  return tuple(tf_inspect.getargspec(fn).args)


def _get_replica_device_setter(config):
  """Creates a replica device setter if required.

  Args:
    config: A RunConfig instance.

  Returns:
    A replica device setter, or None.
  """
  ps_ops = [
      'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
      'MutableHashTableV2', 'MutableHashTableOfTensors',
      'MutableHashTableOfTensorsV2', 'MutableDenseHashTable',
      'MutableDenseHashTableV2', 'VarHandleOp'
  ]

  if config.task_type:
    worker_device = '/job:%s/task:%d' % (config.task_type, config.task_id)
  else:
    worker_device = '/job:worker'

  if config.num_ps_replicas > 0:
    return device_setter.replica_device_setter(
        ps_tasks=config.num_ps_replicas,
        worker_device=worker_device,
        merge_devices=True,
        ps_ops=ps_ops,
        cluster=config.cluster_spec)
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
        raise ValueError('Metrics passed provide (name, prediction), '
                         'but predictions are not dict. '
                         'Metrics: %s, Predictions: %s.' % (metrics,
                                                            predictions))
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
        raise ValueError('Metrics passed provide only name, no prediction, '
                         'but predictions are dict. '
                         'Metrics: %s, Labels: %s.' % (metrics,
                                                       labels_tensor_or_dict))
      result[name] = metric(predictions, labels_tensor_or_dict)
  return result


def _dict_to_str(dictionary):
  """Get a `str` representation of a `dict`.

  Args:
    dictionary: The `dict` to be represented as `str`.

  Returns:
    A `str` representing the `dictionary`.
  """
  results = []
  for k, v in sorted(dictionary.items()):
    if isinstance(v, float) or isinstance(v, np.float32) or isinstance(
        v, int) or isinstance(v, np.int64) or isinstance(v, np.int32):
      results.append('%s = %s' % (k, v))
    else:
      results.append('Type of %s = %s' % (k, type(v)))

  return ', '.join(results)


def _write_dict_to_summary(output_dir, dictionary, current_global_step):
  """Writes a `dict` into summary file in given output directory.

  Args:
    output_dir: `str`, directory to write the summary file in.
    dictionary: the `dict` to be written to summary file.
    current_global_step: `int`, the current global step.
  """
  logging.info('Saving dict for global step %d: %s', current_global_step,
               _dict_to_str(dictionary))
  summary_writer = core_summary.FileWriterCache.get(output_dir)
  summary_proto = summary_pb2.Summary()
  for key in dictionary:
    if dictionary[key] is None:
      continue
    if key == 'global_step':
      continue
    if (isinstance(dictionary[key], np.float32) or
        isinstance(dictionary[key], float)):
      summary_proto.value.add(tag=key, simple_value=float(dictionary[key]))
    elif (isinstance(dictionary[key], np.int64) or
          isinstance(dictionary[key], np.int32) or
          isinstance(dictionary[key], int)):
      summary_proto.value.add(tag=key, simple_value=int(dictionary[key]))
    elif isinstance(dictionary[key], six.string_types):
      try:
        summ = summary_pb2.Summary.FromString(dictionary[key])
        for i, _ in enumerate(summ.value):
          summ.value[i].tag = key
        summary_proto.value.extend(summ.value)
      except message.DecodeError:
        logging.warn('Skipping summary for %s, cannot parse string to Summary.',
                     key)
        continue
    elif isinstance(dictionary[key], np.ndarray):
      value = summary_proto.value.add()
      value.tag = key
      value.node_name = key
      tensor_proto = tensor_util.make_tensor_proto(dictionary[key])
      value.tensor.CopyFrom(tensor_proto)
      logging.info(
          'Summary for np.ndarray is not visible in Tensorboard by default. '
          'Consider using a Tensorboard plugin for visualization (see '
          'https://github.com/tensorflow/tensorboard-plugin-example/blob/master/README.md'
          ' for more information).')
    else:
      logging.warn(
          'Skipping summary for %s, must be a float, np.float32, np.int64, '
          'np.int32 or int or np.ndarray or a serialized string of Summary.',
          key)
  summary_writer.add_summary(summary_proto, current_global_step)
  summary_writer.flush()


GraphRewriteSpec = collections.namedtuple('GraphRewriteSpec',
                                          ['tags', 'transforms'])


class BaseEstimator(sklearn.BaseEstimator, evaluable.Evaluable,
                    trainable.Trainable):
  """Abstract BaseEstimator class to train and evaluate TensorFlow models.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Users should not instantiate or subclass this class. Instead, use an
  `Estimator`.
  """
  __metaclass__ = abc.ABCMeta

  # Note that for Google users, this is overridden with
  # learn_runner.EstimatorConfig.
  # TODO(wicke): Remove this once launcher takes over config functionality
  _Config = run_config.RunConfig  # pylint: disable=invalid-name

  @deprecated(None, 'Please replace uses of any Estimator from tf.contrib.learn'
              ' with an Estimator from tf.estimator.*')
  def __init__(self, model_dir=None, config=None):
    """Initializes a BaseEstimator instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model. If `None`, the model_dir in
        `config` will be used if set. If both are set, they must be same.
      config: A RunConfig instance.
    """
    # Create a run configuration.
    if config is None:
      self._config = BaseEstimator._Config()
      logging.info('Using default config.')
    else:
      self._config = config

    if self._config.session_config is None:
      self._session_config = config_pb2.ConfigProto(allow_soft_placement=True)
    else:
      self._session_config = self._config.session_config

    # Model directory.
    if (model_dir is not None) and (self._config.model_dir is not None):
      if model_dir != self._config.model_dir:
        # TODO(b/9965722): remove this suppression after it is no longer
        #                  necessary.
        # pylint: disable=g-doc-exception
        raise ValueError(
            'model_dir are set both in constructor and RunConfig, but with '
            "different values. In constructor: '{}', in RunConfig: "
            "'{}' ".format(model_dir, self._config.model_dir))
        # pylint: enable=g-doc-exception

    self._model_dir = model_dir or self._config.model_dir
    if self._model_dir is None:
      self._model_dir = tempfile.mkdtemp()
      logging.warning('Using temporary folder as model directory: %s',
                      self._model_dir)
    if self._config.model_dir is None:
      self._config = self._config.replace(model_dir=self._model_dir)
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

  @property
  def model_fn(self):
    """Returns the model_fn which is bound to self.params.

    Returns:
      The model_fn with the following signature:
        `def model_fn(features, labels, mode, metrics)`
    """

    def public_model_fn(features, labels, mode, config):
      return self._call_model_fn(features, labels, mode, config=config)

    return public_model_fn

  @deprecated_args(SCIKIT_DECOUPLE_DATE, SCIKIT_DECOUPLE_INSTRUCTIONS,
                   ('x', None), ('y', None), ('batch_size', None))
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
    _verify_input_args(x, y, input_fn, None, batch_size)
    if x is not None:
      SKCompat(self).fit(x, y, batch_size, steps, max_steps, monitors)
      return self

    if max_steps is not None:
      try:
        start_step = load_variable(self._model_dir, ops.GraphKeys.GLOBAL_STEP)
        if max_steps <= start_step:
          logging.info('Skipping training since max_steps has already saved.')
          return self
      except:  # pylint: disable=bare-except
        pass

    hooks = monitor_lib.replace_monitors_with_hooks(monitors, self)
    if steps is not None or max_steps is not None:
      hooks.append(basic_session_run_hooks.StopAtStepHook(steps, max_steps))

    loss = self._train_model(input_fn=input_fn, hooks=hooks)
    logging.info('Loss for final step: %s.', loss)
    return self

  @deprecated_args(SCIKIT_DECOUPLE_DATE, SCIKIT_DECOUPLE_INSTRUCTIONS,
                   ('x', None), ('y', None), ('batch_size', None))
  def partial_fit(self,
                  x=None,
                  y=None,
                  input_fn=None,
                  steps=1,
                  batch_size=None,
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
    return self.fit(
        x=x,
        y=y,
        input_fn=input_fn,
        steps=steps,
        batch_size=batch_size,
        monitors=monitors)

  @deprecated_args(SCIKIT_DECOUPLE_DATE, SCIKIT_DECOUPLE_INSTRUCTIONS,
                   ('x', None), ('y', None), ('batch_size', None))
  def evaluate(self,
               x=None,
               y=None,
               input_fn=None,
               feed_fn=None,
               batch_size=None,
               steps=None,
               metrics=None,
               name=None,
               checkpoint_path=None,
               hooks=None,
               log_progress=True):
    # pylint: disable=g-doc-args,g-doc-return-or-yield
    """See `Evaluable`.

    Raises:
      ValueError: If at least one of `x` or `y` is provided, and at least one of
          `input_fn` or `feed_fn` is provided.
          Or if `metrics` is not `None` or `dict`.
    """
    _verify_input_args(x, y, input_fn, feed_fn, batch_size)
    if x is not None:
      return SKCompat(self).score(x, y, batch_size, steps, metrics, name)

    if metrics is not None and not isinstance(metrics, dict):
      raise ValueError('Metrics argument should be None or dict. '
                       'Got %s.' % metrics)
    eval_results, global_step = self._evaluate_model(
        input_fn=input_fn,
        feed_fn=feed_fn,
        steps=steps,
        metrics=metrics,
        name=name,
        checkpoint_path=checkpoint_path,
        hooks=hooks,
        log_progress=log_progress)

    if eval_results is not None:
      eval_results.update({'global_step': global_step})
    return eval_results

  @deprecated_args(SCIKIT_DECOUPLE_DATE, SCIKIT_DECOUPLE_INSTRUCTIONS,
                   ('x', None), ('batch_size', None), ('as_iterable', True))
  def predict(self,
              x=None,
              input_fn=None,
              batch_size=None,
              outputs=None,
              as_iterable=True,
              iterate_batches=False):
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
      iterate_batches: If True, yield the whole batch at once instead of
        decomposing the batch into individual samples. Only relevant when
        as_iterable is True.

    Returns:
      A numpy array of predicted classes or regression values if the
      constructor's `model_fn` returns a `Tensor` for `predictions` or a `dict`
      of numpy arrays if `model_fn` returns a `dict`. Returns an iterable of
      predictions if as_iterable is True.

    Raises:
      ValueError: If x and input_fn are both provided or both `None`.
    """
    _verify_input_args(x, None, input_fn, None, batch_size)
    if x is not None and not as_iterable:
      return SKCompat(self).predict(x, batch_size)

    input_fn, feed_fn = _get_input_fn(x, None, input_fn, None, batch_size)
    return self._infer_model(
        input_fn=input_fn,
        feed_fn=feed_fn,
        outputs=outputs,
        as_iterable=as_iterable,
        iterate_batches=iterate_batches)

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

  @deprecated('2017-03-25', 'Please use Estimator.export_savedmodel() instead.')
  def export(
      self,
      export_dir,
      input_fn=export._default_input_fn,  # pylint: disable=protected-access
      input_feature_key=None,
      use_deprecated_input_fn=True,
      signature_fn=None,
      prediction_key=None,
      default_batch_size=1,
      exports_to_keep=None,
      checkpoint_path=None):
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
      checkpoint_path: the checkpoint path of the model to be exported. If it is
          `None` (which is default), will use the latest checkpoint in
          export_dir.

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
        exports_to_keep=exports_to_keep,
        checkpoint_path=checkpoint_path)

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

    Expected to be overridden by sub-classes that require custom support.

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
        logging.debug('Given labels: %s, required signatures: %s.', str(labels),
                      str(self._labels_info))
        if not tensor_signature.tensors_compatible(labels, self._labels_info):
          raise ValueError('Labels are incompatible with given information. '
                           'Given labels: %s, required signatures: %s.' %
                           (str(labels), str(self._labels_info)))
      else:
        self._labels_info = tensor_signature.create_signatures(labels)
        logging.debug('Setting labels info to %s', str(self._labels_info))

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
                      checkpoint_path=None,
                      hooks=None,
                      log_progress=True):
    # TODO(wicke): Remove this once Model and associated code are gone.
    if (hasattr(self._config, 'execution_mode') and
        self._config.execution_mode not in ('all', 'evaluate', 'eval_evalset')):
      return None, None

    # Check that model has been trained (if nothing has been set explicitly).
    if not checkpoint_path:
      latest_path = saver.latest_checkpoint(self._model_dir)
      if not latest_path:
        raise NotFittedError(
            "Couldn't find trained model at %s." % self._model_dir)
      checkpoint_path = latest_path

    # Setup output directory.
    eval_dir = os.path.join(self._model_dir, 'eval'
                            if not name else 'eval_' + name)

    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step = training_util.create_global_step(g)
      features, labels = input_fn()
      self._check_inputs(features, labels)

      model_fn_results = self._get_eval_ops(features, labels, metrics)
      eval_dict = model_fn_results.eval_metric_ops

      update_op, eval_dict = self._extract_metric_update_ops(eval_dict)

      # We need to copy the hook array as we modify it, thus [:].
      hooks = hooks[:] if hooks else []
      if feed_fn:
        hooks.append(basic_session_run_hooks.FeedFnHook(feed_fn))
      if steps == 0:
        logging.warning('evaluation steps are 0. If `input_fn` does not raise '
                        '`OutOfRangeError`, the evaluation will never stop. '
                        'Use steps=None if intended.')
      if steps:
        hooks.append(
            evaluation.StopAfterNEvalsHook(steps, log_progress=log_progress))

      global_step_key = 'global_step'
      while global_step_key in eval_dict:
        global_step_key = '_' + global_step_key
      eval_dict[global_step_key] = global_step

      eval_results = evaluation.evaluate_once(
          checkpoint_path=checkpoint_path,
          master=self._config.evaluation_master,
          scaffold=model_fn_results.scaffold,
          eval_ops=update_op,
          final_ops=eval_dict,
          hooks=hooks,
          config=self._session_config)
      current_global_step = eval_results[global_step_key]

      _write_dict_to_summary(eval_dir, eval_results, current_global_step)

    return eval_results, current_global_step

  def _get_features_from_input_fn(self, input_fn):
    result = input_fn()
    if isinstance(result, (list, tuple)):
      return result[0]
    return result

  def _infer_model(self,
                   input_fn,
                   feed_fn=None,
                   outputs=None,
                   as_iterable=True,
                   iterate_batches=False):
    # Check that model has been trained.
    checkpoint_path = saver.latest_checkpoint(self._model_dir)
    if not checkpoint_path:
      raise NotFittedError(
          "Couldn't find trained model at %s." % self._model_dir)

    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(self._config.tf_random_seed)
      training_util.create_global_step(g)
      features = self._get_features_from_input_fn(input_fn)
      infer_ops = self._get_predict_ops(features)
      predictions = self._filter_predictions(infer_ops.predictions, outputs)
      mon_sess = monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              checkpoint_filename_with_path=checkpoint_path,
              scaffold=infer_ops.scaffold,
              config=self._session_config))
      if not as_iterable:
        with mon_sess:
          if not mon_sess.should_stop():
            return mon_sess.run(predictions, feed_fn() if feed_fn else None)
      else:
        return self._predict_generator(mon_sess, predictions, feed_fn,
                                       iterate_batches)

  def _predict_generator(self, mon_sess, predictions, feed_fn, iterate_batches):
    with mon_sess:
      while not mon_sess.should_stop():
        preds = mon_sess.run(predictions, feed_fn() if feed_fn else None)
        if iterate_batches:
          yield preds
        elif not isinstance(predictions, dict):
          for pred in preds:
            yield pred
        else:
          first_tensor = list(preds.values())[0]
          if isinstance(first_tensor, sparse_tensor.SparseTensorValue):
            batch_length = first_tensor.dense_shape[0]
          else:
            batch_length = first_tensor.shape[0]
          for i in range(batch_length):
            yield {key: value[i] for key, value in six.iteritems(preds)}
        if self._is_input_constant(feed_fn, mon_sess.graph):
          return

  def _is_input_constant(self, feed_fn, graph):
    # If there are no queue_runners, the input `predictions` is a
    # constant, and we should stop after the first epoch.  If,
    # instead, there are queue_runners, eventually they should throw
    # an `OutOfRangeError`.
    if graph.get_collection(ops.GraphKeys.QUEUE_RUNNERS):
      return False
    # data_feeder uses feed_fn to generate `OutOfRangeError`.
    if feed_fn is not None:
      return False
    return True

  def _filter_predictions(self, predictions, outputs):
    if not outputs:
      return predictions
    if not isinstance(predictions, dict):
      raise ValueError(
          'outputs argument is not valid in case of non-dict predictions.')
    existing_keys = predictions.keys()
    predictions = {
        key: value
        for key, value in six.iteritems(predictions)
        if key in outputs
    }
    if not predictions:
      raise ValueError('Expected to run at least one output from %s, '
                       'provided %s.' % (existing_keys, outputs))
    return predictions

  def _train_model(self, input_fn, hooks):
    all_hooks = []
    self._graph = ops.Graph()
    with self._graph.as_default() as g, g.device(self._device_fn):
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step = training_util.create_global_step(g)
      features, labels = input_fn()
      self._check_inputs(features, labels)
      training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
      model_fn_ops = self._get_train_ops(features, labels)
      ops.add_to_collection(ops.GraphKeys.LOSSES, model_fn_ops.loss)
      all_hooks.extend(hooks)
      all_hooks.extend([
          basic_session_run_hooks.NanTensorHook(model_fn_ops.loss),
          basic_session_run_hooks.LoggingTensorHook(
              {
                  'loss': model_fn_ops.loss,
                  'step': global_step
              },
              every_n_iter=100)
      ])

      scaffold = model_fn_ops.scaffold or monitored_session.Scaffold()
      if not (scaffold.saver or ops.get_collection(ops.GraphKeys.SAVERS)):
        ops.add_to_collection(
            ops.GraphKeys.SAVERS,
            saver.Saver(
                sharded=True,
                max_to_keep=self._config.keep_checkpoint_max,
                keep_checkpoint_every_n_hours=(
                    self._config.keep_checkpoint_every_n_hours),
                defer_build=True,
                save_relative_paths=True))

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
          config=self._session_config) as mon_sess:
        loss = None
        while not mon_sess.should_stop():
          _, loss = mon_sess.run([model_fn_ops.train_op, model_fn_ops.loss])
      return loss


def _identity_feature_engineering_fn(features, labels):
  return features, labels


class Estimator(BaseEstimator):
  """Estimator class is the basic TensorFlow model trainer/evaluator.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
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
          * `features`: single `Tensor` or `dict` of `Tensor`s
                 (depending on data passed to `fit`),
          * `labels`: `Tensor` or `dict` of `Tensor`s (for multi-head
                 models). If mode is `ModeKeys.INFER`, `labels=None` will be
                 passed. If the `model_fn`'s signature does not accept
                 `mode`, the `model_fn` must still be able to handle
                 `labels=None`.
          * `mode`: Optional. Specifies if this training, evaluation or
                 prediction. See `ModeKeys`.
          * `params`: Optional `dict` of hyperparameters.  Will receive what
                 is passed to Estimator in `params` parameter. This allows
                 to configure Estimators from hyper parameter tuning.
          * `config`: Optional configuration object. Will receive what is passed
                 to Estimator in `config` parameter, or the default `config`.
                 Allows updating things in your model_fn based on configuration
                 such as `num_ps_replicas`.
          * `model_dir`: Optional directory where model parameters, graph etc
                 are saved. Will receive what is passed to Estimator in
                 `model_dir` parameter, or the default `model_dir`. Allows
                 updating things in your model_fn that expect model_dir, such as
                 training hooks.

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
          * `(features, labels, mode, params, config, model_dir) ->
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
      model_fn_args = _model_fn_args(model_fn)
      if params is not None and 'params' not in model_fn_args:
        raise ValueError('Estimator\'s model_fn (%s) does not have a params '
                         'argument, but params (%s) were passed to the '
                         'Estimator\'s constructor.' % (model_fn, params))
      if params is None and 'params' in model_fn_args:
        logging.warning('Estimator\'s model_fn (%s) includes params '
                        'argument, but params are not passed to Estimator.',
                        model_fn)
    self._model_fn = model_fn
    self.params = params
    self._feature_engineering_fn = (
        feature_engineering_fn or _identity_feature_engineering_fn)

  def _call_model_fn(self, features, labels, mode, metrics=None, config=None):
    """Calls model function with support of 2, 3 or 4 arguments.

    Args:
      features: features dict.
      labels: labels dict.
      mode: ModeKeys
      metrics: Dict of metrics.
      config: RunConfig.

    Returns:
      A `ModelFnOps` object. If model_fn returns a tuple, wraps them up in a
      `ModelFnOps` object.

    Raises:
      ValueError: if model_fn returns invalid objects.
    """
    features, labels = self._feature_engineering_fn(features, labels)
    model_fn_args = _model_fn_args(self._model_fn)
    kwargs = {}
    if 'mode' in model_fn_args:
      kwargs['mode'] = mode
    if 'params' in model_fn_args:
      kwargs['params'] = self.params
    if 'config' in model_fn_args:
      if config:
        kwargs['config'] = config
      else:
        kwargs['config'] = self.config
    if 'model_dir' in model_fn_args:
      kwargs['model_dir'] = self.model_dir
    model_fn_results = self._model_fn(features, labels, **kwargs)

    if isinstance(model_fn_results, model_fn_lib.ModelFnOps):
      model_fn_ops = model_fn_results
    else:
      # Here model_fn_results should be a tuple with 3 elements.
      if len(model_fn_results) != 3:
        raise ValueError('Unrecognized value returned by model_fn, '
                         'please return ModelFnOps.')
      model_fn_ops = model_fn_lib.ModelFnOps(
          mode=mode,
          predictions=model_fn_results[0],
          loss=model_fn_results[1],
          train_op=model_fn_results[2])

    # Custom metrics should overwrite defaults.
    if metrics:
      model_fn_ops.eval_metric_ops.update(
          _make_metrics_ops(metrics, features, labels,
                            model_fn_ops.predictions))

    return model_fn_ops

  def _get_train_ops(self, features, labels):
    """Method that builds model graph and returns trainer ops.

    Expected to be overridden by sub-classes that require custom support.
    This implementation uses `model_fn` passed as parameter to constructor to
    build model.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      labels: `Tensor` or `dict` of `Tensor` objects.

    Returns:
      `ModelFnOps` object.
    """
    return self._call_model_fn(features, labels, model_fn_lib.ModeKeys.TRAIN)

  def _get_eval_ops(self, features, labels, metrics):
    """Method that builds model graph and returns evaluation ops.

    Expected to be overridden by sub-classes that require custom support.
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
    model_fn_ops = self._call_model_fn(features, labels,
                                       model_fn_lib.ModeKeys.EVAL, metrics)

    if metric_key.MetricKey.LOSS not in model_fn_ops.eval_metric_ops:
      model_fn_ops.eval_metric_ops[metric_key.MetricKey.LOSS] = (
          metrics_lib.mean(model_fn_ops.loss))
    return model_fn_ops

  def _get_predict_ops(self, features):
    """Method that builds model graph and returns prediction ops.

    Expected to be overridden by sub-classes that require custom support.
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

  def export_savedmodel(self,
                        export_dir_base,
                        serving_input_fn,
                        default_output_alternative_key=None,
                        assets_extra=None,
                        as_text=False,
                        checkpoint_path=None,
                        graph_rewrite_specs=(GraphRewriteSpec(
                            (tag_constants.SERVING,), ()),),
                        strip_default_attrs=False):
    # pylint: disable=line-too-long
    """Exports inference graph as a SavedModel into given dir.

    Args:
      export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
      serving_input_fn: A function that takes no argument and
        returns an `InputFnOps`.
      default_output_alternative_key: the name of the head to serve when none is
        specified.  Not needed for single-headed models.
      assets_extra: A dict specifying how to populate the assets.extra directory
        within the exported SavedModel.  Each key should give the destination
        path (including the filename) relative to the assets.extra directory.
        The corresponding value gives the full path of the source file to be
        copied.  For example, the simple case of copying a single file without
        renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format.
      checkpoint_path: The checkpoint path to export.  If None (the default),
        the most recent checkpoint found within the model directory is chosen.
      graph_rewrite_specs: an iterable of `GraphRewriteSpec`.  Each element will
        produce a separate MetaGraphDef within the exported SavedModel, tagged
        and rewritten as specified.  Defaults to a single entry using the
        default serving tag ("serve") and no rewriting.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued
          Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: if an unrecognized export_type is requested.
    """
    # pylint: enable=line-too-long
    if serving_input_fn is None:
      raise ValueError('serving_input_fn must be defined.')

    if not checkpoint_path:
      # Locate the latest checkpoint
      checkpoint_path = saver.latest_checkpoint(self._model_dir)
    if not checkpoint_path:
      raise NotFittedError(
          "Couldn't find trained model at %s." % self._model_dir)

    export_dir = saved_model_export_utils.get_timestamped_export_dir(
        export_dir_base)
    # We'll write the SavedModel to a temporary directory and then atomically
    # rename it at the end.  This helps to avoid corrupt / incomplete outputs,
    # which could otherwise occur if the job is preempted or otherwise fails
    # in the middle of SavedModel creation.
    temp_export_dir = saved_model_export_utils.get_temp_export_dir(export_dir)
    builder = saved_model_builder.SavedModelBuilder(temp_export_dir)

    # Build the base graph
    with ops.Graph().as_default() as g:
      training_util.create_global_step(g)

      # Call the serving_input_fn and collect the input alternatives.
      input_ops = serving_input_fn()
      input_alternatives, features = (
          saved_model_export_utils.get_input_alternatives(input_ops))

      # TODO(b/34388557) This is a stopgap, pending recording model provenance.
      # Record which features are expected at serving time.  It is assumed that
      # these are the features that were used in training.
      for feature_key in input_ops.features.keys():
        ops.add_to_collection(
            constants.COLLECTION_DEF_KEY_FOR_INPUT_FEATURE_KEYS, feature_key)

      # Call the model_fn and collect the output alternatives.
      model_fn_ops = self._call_model_fn(features, None,
                                         model_fn_lib.ModeKeys.INFER)
      output_alternatives, actual_default_output_alternative_key = (
          saved_model_export_utils.get_output_alternatives(
              model_fn_ops, default_output_alternative_key))

      init_op = control_flow_ops.group(variables.local_variables_initializer(),
                                       resources.initialize_resources(
                                           resources.shared_resources()),
                                       lookup_ops.tables_initializer())

      # Build the SignatureDefs from all pairs of input and output alternatives
      signature_def_map = saved_model_export_utils.build_all_signature_defs(
          input_alternatives, output_alternatives,
          actual_default_output_alternative_key)

      # Export the first MetaGraphDef with variables, assets etc.
      with tf_session.Session('') as session:

        # pylint: disable=protected-access
        saveables = variables._all_saveable_objects()
        # pylint: enable=protected-access

        if (model_fn_ops.scaffold is not None and
            model_fn_ops.scaffold.saver is not None):
          saver_for_restore = model_fn_ops.scaffold.saver
        elif saveables:
          saver_for_restore = saver.Saver(saveables, sharded=True)

        saver_for_restore.restore(session, checkpoint_path)

        # Perform the export
        if not graph_rewrite_specs or graph_rewrite_specs[0].transforms:
          raise ValueError('The first element of graph_rewrite_specs '
                           'must specify no transforms.')
        untransformed_tags = graph_rewrite_specs[0].tags

        # TODO(soergel): switch to main_op or otherwise update when dust settles
        builder.add_meta_graph_and_variables(
            session,
            untransformed_tags,
            signature_def_map=signature_def_map,
            assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
            legacy_init_op=init_op,
            strip_default_attrs=strip_default_attrs)

    # pylint: disable=protected-access
    base_meta_graph_def = builder._saved_model.meta_graphs[0]
    # pylint: enable=protected-access

    if graph_rewrite_specs[1:]:
      # Prepare the input_names and output_names needed for the
      # meta_graph_transform call below.
      input_names = [
          tensor.name
          for input_dict in input_alternatives.values()
          for tensor in input_dict.values()
      ]
      output_names = [
          tensor.name
          for output_alternative in output_alternatives.values()
          for tensor in output_alternative[1].values()
      ]

    # Write the additional MetaGraphDefs
    for graph_rewrite_spec in graph_rewrite_specs[1:]:

      # TODO(soergel) consider moving most of this to saved_model.builder_impl
      # as e.g. builder.add_rewritten_meta_graph(rewritten_graph_def, tags)

      transformed_meta_graph_def = meta_graph_transform.meta_graph_transform(
          base_meta_graph_def, input_names, output_names,
          graph_rewrite_spec.transforms, graph_rewrite_spec.tags)

      # pylint: disable=protected-access
      meta_graph_def = builder._saved_model.meta_graphs.add()
      # pylint: enable=protected-access
      meta_graph_def.CopyFrom(transformed_meta_graph_def)

    # Add the extra assets
    if assets_extra:
      assets_extra_path = os.path.join(
          compat.as_bytes(temp_export_dir), compat.as_bytes('assets.extra'))
      for dest_relative, source in assets_extra.items():
        dest_absolute = os.path.join(
            compat.as_bytes(assets_extra_path), compat.as_bytes(dest_relative))
        dest_path = os.path.dirname(dest_absolute)
        gfile.MakeDirs(dest_path)
        gfile.Copy(source, dest_absolute)

    builder.save(as_text)
    gfile.Rename(temp_export_dir, export_dir)
    return export_dir


# For time of deprecation x,y from Estimator allow direct access.
# pylint: disable=protected-access
class SKCompat(sklearn.BaseEstimator):
  """Scikit learn wrapper for TensorFlow Learn Estimator.
  
  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  """

  @deprecated(None, 'Please switch to the Estimator interface.')
  def __init__(self, estimator):
    self._estimator = estimator

  def fit(self, x, y, batch_size=128, steps=None, max_steps=None,
          monitors=None):
    input_fn, feed_fn = _get_input_fn(
        x,
        y,
        input_fn=None,
        feed_fn=None,
        batch_size=batch_size,
        shuffle=True,
        epochs=None)
    all_monitors = []
    if feed_fn:
      all_monitors = [basic_session_run_hooks.FeedFnHook(feed_fn)]
    if monitors:
      all_monitors.extend(monitors)

    self._estimator.fit(
        input_fn=input_fn,
        steps=steps,
        max_steps=max_steps,
        monitors=all_monitors)
    return self

  def score(self, x, y, batch_size=128, steps=None, metrics=None, name=None):
    input_fn, feed_fn = _get_input_fn(
        x,
        y,
        input_fn=None,
        feed_fn=None,
        batch_size=batch_size,
        shuffle=False,
        epochs=1)
    if metrics is not None and not isinstance(metrics, dict):
      raise ValueError('Metrics argument should be None or dict. '
                       'Got %s.' % metrics)
    eval_results, global_step = self._estimator._evaluate_model(
        input_fn=input_fn,
        feed_fn=feed_fn,
        steps=steps,
        metrics=metrics,
        name=name)
    if eval_results is not None:
      eval_results.update({'global_step': global_step})
    return eval_results

  def predict(self, x, batch_size=128, outputs=None):
    input_fn, feed_fn = _get_input_fn(
        x,
        None,
        input_fn=None,
        feed_fn=None,
        batch_size=batch_size,
        shuffle=False,
        epochs=1)
    results = list(
        self._estimator._infer_model(
            input_fn=input_fn,
            feed_fn=feed_fn,
            outputs=outputs,
            as_iterable=True,
            iterate_batches=True))
    if not isinstance(results[0], dict):
      return np.concatenate([output for output in results], axis=0)
    return {
        key: np.concatenate([output[key] for output in results], axis=0)
        for key in results[0]
    }
