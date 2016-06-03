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
import inspect
import os
import tempfile
import time

import numpy as np
import six

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib.learn.python.learn import graph_actions
from tensorflow.contrib.learn.python.learn import monitors as monitors_lib
from tensorflow.contrib.learn.python.learn.estimators import _sklearn as sklearn
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.estimators import tensor_signature
from tensorflow.contrib.learn.python.learn.estimators._sklearn import NotFittedError
from tensorflow.contrib.learn.python.learn.io import data_feeder
from tensorflow.contrib.learn.python.learn.utils import checkpoints

from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.training import saver


class ModeKeys(object):
  """Standard names for model modes.

  The following standard keys are defined:

  * `TRAIN`: training mode.
  * `EVAL`: evaluation mode.
  * `INFER`: inference mode.
  """

  TRAIN = 'train'
  EVAL = 'eval'
  INFER = 'infer'


def _get_input_fn(x, y, batch_size):
  df = data_feeder.setup_train_data_feeder(
      x, y, n_classes=None, batch_size=batch_size)
  return df.input_builder, df.get_feed_dict_fn()


def _get_predict_input_fn(x, y, batch_size):
  df = data_feeder.setup_train_data_feeder(
      x, y, n_classes=None, batch_size=batch_size,
      shuffle=False, epochs=1)
  return df.input_builder, df.get_feed_dict_fn()


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


class BaseEstimator(sklearn.BaseEstimator):
  """Abstract BaseEstimator class to train and evaluate TensorFlow models.

  Concrete implementation of this class should provide following functions:
    * _get_train_ops
    * _get_eval_ops
    * _get_predict_ops

  `Estimator` implemented below is a good example of how to use this class.

  Parameters:
    model_dir: Directory to save model parameters, graph and etc.
  """
  __metaclass__ = abc.ABCMeta

  # TODO(wicke): Remove this once launcher takes over config functionality
  _Config = run_config.RunConfig  # pylint: disable=invalid-name

  def __init__(self, model_dir=None, config=None):
    # Model directory.
    self._model_dir = model_dir
    if self._model_dir is None:
      self._model_dir = tempfile.mkdtemp()
      logging.info('Using temporary folder as model directory: %s',
                   self._model_dir)

    # Create a run configuration
    if config is None:
      self._config = BaseEstimator._Config()
    else:
      self._config = config

    # Set device function depending if there are replicas or not.
    if self._config.num_ps_replicas > 0:
      ps_ops = ['Variable', 'AutoReloadVariable']
      self._device_fn = device_setter.replica_device_setter(
          ps_tasks=self._config.num_ps_replicas,
          merge_devices=False, ps_ops=ps_ops)
    else:
      self._device_fn = None

    # Features and targets TensorSingature objects.
    self._features_info = None
    self._targets_info = None

    self._graph = None

  def fit(
      self, x=None, y=None, input_fn=None, steps=None, batch_size=None,
      monitors=None):
    """Trains a model given training data `x` predictions and `y` targets.

    Args:
      x: matrix or tensor of shape [n_samples, n_features...]. Can be
         iterator that returns arrays of features. The training input
         samples for fitting the model. If set, `input_fn` must be `None`.
      y: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
         iterator that returns array of targets. The training target values
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
      ValueError: If `x` or `y` are not `None` while `input_fn` is not `None`.

    Raises:
      ValueError: If at least one of `x` and `y` is provided, and `input_fn` is
          provided.
    """
    feed_fn = None
    if input_fn is None:
      if x is None:
        raise ValueError('Either x or input_fn must be provided.')
      input_fn, feed_fn = _get_input_fn(x, y, batch_size)
    elif (x is not None) or (y is not None):
      raise ValueError('Can not provide both input_fn and either of x and y.')
    elif batch_size is not None:
      raise ValueError('Can not provide both input_fn and batch_size.')

    loss = self._train_model(input_fn=input_fn,
                             feed_fn=feed_fn,
                             steps=steps,
                             monitors=monitors)
    logging.info('Loss for final step: %s.', loss)
    return self

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
      x: matrix or tensor of shape [n_samples, n_features...]. Can be
        iterator that returns arrays of features. The training input
        samples for fitting the model. If set, `input_fn` must be `None`.
      y: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
        iterator that returns array of targets. The training target values
        (class label in classification, real numbers in regression). If set,
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
    feed_fn = None
    if input_fn is None:
      if x is None:
        raise ValueError('Either x or input_fn must be provided.')
      input_fn, feed_fn = _get_input_fn(x, y, batch_size)
    elif (x is not None) or (y is not None):
      raise ValueError('Can not provide both input_fn and either of x and y.')
    loss = self._train_model(input_fn=input_fn,
                             feed_fn=feed_fn,
                             steps=steps,
                             monitors=monitors)
    logging.info('Loss for final step: %s.', loss)
    return self

  def evaluate(self,
               x=None,
               y=None,
               input_fn=None,
               feed_fn=None,
               batch_size=None,
               steps=None,
               metrics=None,
               name=None):
    """Evaluates given model with provided evaluation data.

    Args:
      x: features.
      y: targets.
      input_fn: Input function. If set, `x`, `y`, and `batch_size` must be
        `None`.
      feed_fn: Function creating a feed dict every time it is called. Called
        once per iteration.
      batch_size: minibatch size to use on the input, defaults to first
        dimension of `x`. Must be `None` if `input_fn` is provided.
      steps: Number of steps for which to evaluate model. If `None`, evaluate
        forever.
      metrics: Dict of metric ops to run. If None, the default metric functions
        are used; if {}, no metrics are used.
      name: Name of the evaluation if user needs to run multiple evaluation on
        different data sets, such as evaluate on training data vs test data.

    Returns:
      Returns `dict` with evaluation results.

    Raises:
      ValueError: If at least one of `x` or `y` is provided, and at least one of
          `input_fn` or `feed_fn` is provided.
    """
    if input_fn is None:
      if x is None:
        raise ValueError('Either x or input_fn must be provided.')
      if feed_fn is not None:
        raise ValueError('Cannot provide both x and feed_fn.')
      input_fn, feed_fn = _get_predict_input_fn(x, y, batch_size)
    elif (x is not None) or (y is not None):
      raise ValueError('Can not provide both input_fn and either of x and y.')
    elif batch_size is not None:
      raise ValueError('Can not provide both input_fn and batch_size.')
    return self._evaluate_model(input_fn=input_fn,
                                feed_fn=feed_fn,
                                steps=steps,
                                metrics=metrics,
                                name=name)

  def predict(self, x=None, input_fn=None, batch_size=None, outputs=None):
    """Returns predictions for given features.

    Args:
      x: Features. If set, `input_fn` must be `None`.
      input_fn: Input function. If set, `x` must be `None`.
      batch_size: Override default batch size.
      outputs: list of `str`, name of the output to predict.
               If `None`, returns all.

    Returns:
      Numpy array of predicted classes or regression values.

    Raises:
      ValueError: If x and input_fn are both provided or both `None`.
    """
    if x is None and input_fn is None:
      raise ValueError('Either x or input_fn must be provided.')
    if x is not None and input_fn is not None:
      raise ValueError('Can not provide both input_fn and x.')
    feed_fn = None
    if x is not None:
      input_fn, feed_fn = _get_predict_input_fn(x, None, batch_size)
    return self._infer_model(input_fn=input_fn, feed_fn=feed_fn,
                             outputs=outputs)

  def get_variable_value(self, name):
    """Returns value of the variable given by name.

    Args:
      name: string, name of the tensor.

    Returns:
      Numpy array - value of the tensor.
    """
    if name.endswith(':0'):
      name = name[:-2]
    return checkpoints.load_variable(self.model_dir, name)

  def get_variable_names(self):
    """Returns list of all variable names in this model.

    Returns:
      List of names.
    """
    return [name for name, _ in checkpoints.list_variables(self.model_dir)]

  @property
  def model_dir(self):
    return self._model_dir

  @abc.abstractproperty
  def _get_train_ops(self, features, targets):
    """Method that builds model graph and returns trainer ops.

    Expected to be overriden by sub-classes that require custom support.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      targets: `Tensor` or `dict` of `Tensor` objects.

    Returns:
      Tuple of train `Operation` and loss `Tensor`.
    """
    pass

  @abc.abstractproperty
  def _get_predict_ops(self, features):
    """Method that builds model graph and returns prediction ops.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.

    Returns:
      predictions: `Tensor` or `dict` of `Tensor` objects.
    """
    pass

  def _get_eval_ops(self, features, targets, metrics):
    """Method that builds model graph and returns evaluation ops.

    Expected to be overriden by sub-classes that require custom support.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      targets: `Tensor` or `dict` of `Tensor` objects.
      metrics: `dict` of functions that take predictions and targets.

    Returns:
      metrics: `dict` of `Tensor` objects.
    """
    raise NotImplementedError('_get_eval_ops not implemented in BaseEstimator')

  def _get_feature_ops_from_example(self, examples_batch):
    """Returns feature parser for given example batch using features info.

    Args:
      examples_batch: batch of tf.Example

    Returns:
      features: `Tensor` or `dict` of `Tensor` objects.

    Raises:
      ValueError: If `_features_info` attribute is not available.
    """
    if self._features_info is None:
      raise ValueError('Features information is missing.')
    return tensor_signature.create_example_parser_from_signatures(
        self._features_info, examples_batch)

  def _check_inputs(self, features, targets):
    if self._features_info is not None:
      if not tensor_signature.tensors_compatible(features, self._features_info):
        raise ValueError('Features are incompatible with given information. '
                         'Given features: %s, required signatures: %s.' %
                         (str(features), str(self._features_info)))
    else:
      self._features_info = tensor_signature.create_signatures(features)
    if targets is not None:
      if self._targets_info is not None:
        if not tensor_signature.tensors_compatible(targets, self._targets_info):
          raise ValueError('Targets are incompatible with given information. '
                           'Given targets: %s, required signatures: %s.' %
                           (str(targets), str(self._targets_info)))
      else:
        self._targets_info = tensor_signature.create_signatures(targets)

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
                   fail_on_nan_loss=True):
    # TODO(wicke): This is a hack and needs to go.
    if self._config.execution_mode not in ('all', 'train'):
      return

    if not self._model_dir:
      raise ValueError('Estimator\'s model_dir should be non-empty.')

    # Stagger startup of worker sessions based on task id.
    sleep_secs = min(self._config.training_worker_max_startup_secs,
                     self._config.task *
                     self._config.training_worker_session_startup_stagger_secs)
    if sleep_secs:
      logging.info('Waiting %d secs before starting task %d.', sleep_secs,
                   self._config.task)
      time.sleep(sleep_secs)

    # Device allocation
    device_fn = device_fn or self._device_fn

    self._graph = ops.Graph()
    with self._graph.as_default() as g, g.device(device_fn):
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step = contrib_framework.create_global_step(g)
      features, targets = input_fn()
      self._check_inputs(features, targets)
      train_op, loss_op = self._get_train_ops(features, targets)

      # Add default monitors.
      if monitors is None:
        monitors = []
      monitors += monitors_lib.get_default_monitors(
          loss_op=loss_op,
          summary_op=logging_ops.get_summary_op(),
          save_summary_steps=100,
          summary_writer=graph_actions.get_summary_writer(self._model_dir))

      is_chief = self._config.task == 0
      if not is_chief:
        # Run monitors only on chief.
        monitors = []

      # Setup monitors.
      for monitor in monitors:
        monitor.set_estimator(self)

      return graph_actions.train(
          graph=g,
          output_dir=self._model_dir,
          train_op=train_op,
          loss_op=loss_op,
          global_step_tensor=global_step,
          init_op=init_op,
          init_feed_dict=init_feed_fn() if init_feed_fn is not None else None,
          init_fn=init_fn,
          log_every_steps=log_every_steps,
          supervisor_is_chief=is_chief,
          supervisor_master=self._config.master,
          feed_fn=feed_fn,
          max_steps=steps,
          fail_on_nan_loss=fail_on_nan_loss,
          monitors=monitors)

  def _extract_metric_update_ops(self, eval_dict):
    """Separate update operations from metric value operations."""
    update_ops = []
    value_ops = {}
    for name, metric_ops in eval_dict.items():
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
                      name=''):
    # TODO(wicke): This is a hack and needs to go.
    if self._config.execution_mode not in ('all', 'evaluate', 'eval_evalset'):
      return

    # Check that model has been trained.
    checkpoint_path = self._model_dir
    latest_path = saver.latest_checkpoint(checkpoint_path)
    if not latest_path:
      raise NotFittedError("Couldn't find trained model at %s."
                           % checkpoint_path)
    # Setup output directory.
    eval_dir = os.path.join(self._model_dir, 'eval' if not name else
                            'eval_' + name)

    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step = contrib_framework.create_global_step(g)
      features, targets = input_fn()
      self._check_inputs(features, targets)
      eval_dict = self._get_eval_ops(features, targets, metrics)
      update_op, eval_dict = self._extract_metric_update_ops(eval_dict)
      eval_results, _ = graph_actions.evaluate(
          graph=g,
          output_dir=eval_dir,
          checkpoint_path=checkpoint_path,
          eval_dict=eval_dict,
          update_op=update_op,
          global_step_tensor=global_step,
          supervisor_master=self._config.master,
          feed_fn=feed_fn,
          max_steps=steps)
      return eval_results

  def _get_features_from_input_fn(self, input_fn):
    result = input_fn()
    if isinstance(result, (list, tuple)):
      return result[0]
    return result

  def _infer_model(self, input_fn, feed_fn=None, outputs=None):
    # Check that model has been trained.
    checkpoint_path = saver.latest_checkpoint(self._model_dir)
    if not checkpoint_path:
      raise NotFittedError("Couldn't find trained model at %s."
                           % self._model_dir)

    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(self._config.tf_random_seed)
      contrib_framework.create_global_step(g)
      features = self._get_features_from_input_fn(input_fn)
      predictions = self._get_predict_ops(features)
      # If predictions is single output - wrap it into dict, and remember to
      # return not a dict.
      return_dict = True
      if not isinstance(predictions, dict):
        predictions, return_dict = {'predictions': predictions}, False
      # Filter what to run predictions on, if outputs provided.
      if outputs:
        existing_keys = predictions.keys()
        predictions = {
            key: value for key, value in predictions.items() if key in outputs
        }
        if not predictions:
          raise ValueError('Expected to run at least one output from %s, '
                           'provided %s.' % (existing_keys, outputs))
      if feed_fn is None:
        preds = graph_actions.infer(checkpoint_path, predictions)
      else:
        preds = {}
        def _feed_fn():
          while True:
            yield feed_fn()
        outputs = graph_actions.run_feeds(
            output_dict=predictions,
            feed_dicts=_feed_fn(),
            restore_checkpoint_path=checkpoint_path)
        for key in predictions:
          preds[key] = np.concatenate(
              [output[key] for output in outputs], axis=0)
      if return_dict:
        return preds
      return preds['predictions']


class Estimator(BaseEstimator):
  """Estimator class is the basic TensorFlow model trainer/evaluator.

  Parameters:
    model_fn: Model function, takes features and targets tensors or dicts of
              tensors and returns predictions and loss tensors.
              Supports next three signatures for the function:
                * `(features, targets) -> (predictions, loss, train_op)`
                * `(features, targets, mode) -> (predictions, loss, train_op)`
                * `(features, targets, mode, params) ->
                    (predictions, loss, train_op)`
              Where:
                * `features` are single `Tensor` or `dict` of `Tensor`s
                     (depending on data passed to `fit`),
                * `targets` are `Tensor` or
                    `dict` of `Tensor`s (for multi-head model).
                * `mode` represents if this training, evaluation or prediction.
                    See `ModeKeys` for example keys.
                * `params` is a `dict` of hyperparameters. Will receive what is
                    passed to Estimator in `params` parameter. This allows to
                    configure Estimators from hyper parameter tunning.
    model_dir: Directory to save model parameters, graph and etc.
    config: Configuration object.
    params: `dict` of hyper parameters that will be passed into `model_fn`.
            Keys are names of parameters, values are basic python types.
  """

  def __init__(self,
               model_fn=None,
               model_dir=None,
               config=None,
               params=None):
    super(Estimator, self).__init__(model_dir=model_dir, config=config)
    if model_fn is not None:
      # Check number of arguments of the given function matches requirements.
      model_fn_args = _get_arguments(model_fn)
      if params is not None and 'params' not in model_fn_args:
        raise ValueError('Estimator\'s model_fn (%s) has less then 4 '
                         'arguments, but not None params (%s) are passed.' %
                         (model_fn, params))
      if params is None and 'params' in model_fn_args:
        logging.warning('Estimator\'s model_fn (%s) has includes params '
                        'argument, but params are not passed to Estimator.' %
                        model_fn)
    self._model_fn = model_fn
    self.params = params

  def _call_model_fn(self, features, targets, mode):
    """Calls model function with support of 2, 3 or 4 arguments."""
    model_fn_args = _get_arguments(self._model_fn)
    if 'mode' in model_fn_args:
      if 'params' in model_fn_args:
        return self._model_fn(
            features, targets, mode=mode, params=self.params)
      else:
        return self._model_fn(
            features, targets, mode=mode)
    return self._model_fn(features, targets)

  def _get_train_ops(self, features, targets):
    """Method that builds model graph and returns trainer ops.

    Expected to be overriden by sub-classes that require custom support.
    This implementation uses `model_fn` passed as parameter to constructor to
    build model.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      targets: `Tensor` or `dict` of `Tensor` objects.

    Returns:
      Tuple of train `Operation` and loss `Tensor`.
    """
    _, loss, train_op = self._call_model_fn(features, targets, ModeKeys.TRAIN)
    return train_op, loss

  def _get_eval_ops(self, features, targets, metrics):
    """Method that builds model graph and returns evaluation ops.

    Expected to be overriden by sub-classes that require custom support.
    This implementation uses `model_fn` passed as parameter to constructor to
    build model.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      targets: `Tensor` or `dict` of `Tensor` objects.
      metrics: `dict` of functions that take predictions and targets.

    Returns:
      metrics: `dict` of `Tensor` objects.
    """
    predictions, loss, _ = self._call_model_fn(features, targets, ModeKeys.EVAL)
    result = {'loss': loss}
    metrics = metrics or {}
    if isinstance(targets, dict) and len(targets) == 1:
      # Unpack single target into just tensor.
      targets = targets[list(targets.keys())[0]]
    for name, metric in six.iteritems(metrics):
      if isinstance(name, tuple):
        # Multi-head metrics.
        result[name[0]] = metric(predictions[name[1]], targets)
      else:
        # Single head metrics.
        result[name] = metric(predictions, targets)
    return result

  def _get_predict_ops(self, features):
    """Method that builds model graph and returns prediction ops.

    Expected to be overriden by sub-classes that require custom support.
    This implementation uses `model_fn` passed as parameter to constructor to
    build model.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.

    Returns:
      predictions: `Tensor` or `dict` of `Tensor` objects.
    """
    targets = tensor_signature.create_placeholders_from_signatures(
        self._targets_info)
    predictions, _, _ = self._call_model_fn(features, targets, ModeKeys.INFER)
    return predictions
