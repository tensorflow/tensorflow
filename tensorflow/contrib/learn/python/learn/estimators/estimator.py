#  Copyright 2015 Google Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Base Estimator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import tempfile
import time

import six

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.learn.python.learn.estimators import _sklearn as sklearn
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.estimators import tensor_signature
from tensorflow.contrib.learn.python.learn.graph_actions import evaluate
from tensorflow.contrib.learn.python.learn.graph_actions import infer
from tensorflow.contrib.learn.python.learn.graph_actions import train
from tensorflow.contrib.learn.python.learn.io import data_feeder

from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.training import saver


# Default metrics for evaluation.
_EVAL_METRICS = {
    'regression': {
        'mean_squared_error': losses.sum_of_squares,
    },
    'classification': {
        'logistic': losses.sigmoid_cross_entropy,
    },}


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
  # TODO(ipoloshukin): Remove this when refactor of data_feeder is done
  if hasattr(x, 'create_graph') and hasattr(y, 'create_graph'):
    def input_fn():
      return x.create_graph(), y.create_graph()
    return input_fn, None

  df = data_feeder.setup_train_data_feeder(x, y,
                                           n_classes=None,
                                           batch_size=batch_size)
  return df.input_builder, df.get_feed_dict_fn()


def _get_predict_input_fn(x, batch_size):
  # TODO(ipoloshukin): Remove this when refactor of data_feeder is done
  if hasattr(x, 'create_graph'):
    def input_fn():
      return x.create_graph()
    return input_fn, None

  df = data_feeder.setup_train_data_feeder(x, None,
                                           n_classes=None,
                                           batch_size=batch_size)
  return df.input_builder, df.get_feed_dict_fn()


class BaseEstimator(sklearn.BaseEstimator):
  """Abstract BaseEstimator class to train and evaluate TensorFlow models.

  Concrete implementation of this class should provide following functions:
    * _get_train_ops
    * _get_eval_ops
    * _get_predict_ops
  It may override _get_default_metric_functions.

  `Estimator` implemented below is a good example of how to use this class.

  Parameters:
    model_dir: Directory to save model parameters, graph and etc.
  """
  __metaclass__ = abc.ABCMeta

  # TODO(wicke): Remove this once launcher takes over config functionality
  _Config = run_config.RunConfig  # pylint: disable=invalid-name

  def __init__(self, model_dir=None):
    # Model directory.
    self._model_dir = model_dir
    if self._model_dir is None:
      self._model_dir = tempfile.mkdtemp()
      logging.info('Using temporary folder as model directory: %s',
                   self._model_dir)

    # Create a run configuration
    self._config = BaseEstimator._Config()

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

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      targets: `Tensor` or `dict` of `Tensor` objects.
      metrics: `dict` of functions that take predictions and targets.

    Returns:
      metrics: `dict` of `Tensor` objects.
    """
    predictions = self._get_predict_ops(features)
    result = {}
    for name, metric in six.iteritems(metrics):
      result[name] = metric(predictions, targets)
    return result

  def _get_feature_ops_from_example(self, examples_batch):
    """Method that returns features given the batch of examples.

    This method will be used to export model into a server.

    Args:
      examples_batch: batch of tf.Example

    Returns:
      features: `Tensor` or `dict` of `Tensor` objects.
    """
    raise NotImplementedError('_get_feature_ops_from_example not implemented '
                              'in BaseEstimator')

  def _get_default_metric_functions(self):
    """Method that provides default metric operations.

    This functions is intented to be overridden by sub-classes.
    Returns:
      `dict` of functions that take predictions and targets `Tensor` objects and
      return `Tensor`.
    """
    return {}

  def fit(self, x, y, steps, batch_size=32, monitor=None):
    """Trains a model given training data X and y.

    Args:
      x: matrix or tensor of shape [n_samples, n_features...]. Can be
         iterator that returns arrays of features. The training input
         samples for fitting the model.
      y: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
         iterator that returns array of targets. The training target values
         (class labels in classification, real numbers in regression).
      steps: number of steps to train model for.
      batch_size: minibatch size to use on the input, defaults to 32.
      monitor: monitor object to print training progress and invoke
               early stopping.

    Returns:
      Returns self.
    """
    input_fn, feed_fn = _get_input_fn(x, y, batch_size)
    return self._train_model(input_fn=input_fn,
                             feed_fn=feed_fn,
                             steps=steps,
                             monitor=monitor)

  def train(self, input_fn, steps, monitor=None):
    """Trains a model given input builder function.

    Args:
      input_fn: Input builder function, returns tuple of dicts or
                dict and Tensor.
      steps: number of steps to train model for.
      monitor: monitor object to print training progress and invoke
               early stopping.

    Returns:
      Returns self.
    """
    return self._train_model(input_fn=input_fn, steps=steps, monitor=monitor)

  def partial_fit(self, x, y, steps=1, batch_size=32, monitor=None):
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
        samples for fitting the model.
      y: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
        iterator that returns array of targets. The training target values
        (class label in classification, real numbers in regression).
      steps: number of steps to train model for.
      batch_size: minibatch size to use on the input, defaults to 32.
      monitor: Monitor object to print training progress and invoke
               early stopping.

    Returns:
      Returns self.
    """
    input_fn, feed_fn = _get_input_fn(x, y, batch_size)
    return self._train_model(input_fn=input_fn,
                             feed_fn=feed_fn,
                             steps=steps,
                             monitor=monitor)

  def evaluate(self, x=None, y=None, input_fn=None, feed_fn=None,
               batch_size=32, steps=100, metrics=None):
    """Evaluates given model with provided evaluation data.

    Args:
      x: features.
      y: targets.
      input_fn: Input function. If set, x and y must be None.
      feed_fn: Function creating a feed dict every time it is called. Called
        once per iteration.
      batch_size: minibatch size to use on the input, defaults to 32. Ignored
        if input_fn is set.
      steps: Number of steps to evalute for.
      metrics: Dict of metric ops to run.

    Returns:
      Returns self.

    Raises:
      ValueError: If x or y are not None while input_fn or feed_fn is not None.
    """
    if (x is not None or y is not None) and input_fn is not None:
      raise ValueError('Either x and y or input_fn must be None.')
    if input_fn is None:
      assert x is not None
      input_fn, feed_fn = _get_input_fn(x, y, batch_size)
    return self._evaluate_model(input_fn=input_fn, feed_fn=feed_fn,
                                steps=steps, metrics=metrics)

  def predict(self, x, axis=None, batch_size=None):
    """Returns predictions for given features.

    Args:
      x: features.
      axis: Axis on which to argmax. (for classification).
      batch_size: Override default batch size.

    Returns:
      Numpy array of predicted classes or regression values.
    """
    return self._infer_model(x=x, batch_size=batch_size, axis=axis)

  def predict_proba(self, x, batch_size=None):
    """Returns prediction probabilities for given features (classification).

    Args:
      x: features.
      batch_size: OVerride default batch size.

    Returns:
      Numpy array of predicted probabilities.
    """
    return self._infer_model(x=x, batch_size=batch_size, proba=True)

  def _check_inputs(self, features, targets):
    if self._features_info is not None:
      if not tensor_signature.tensors_compatible(features, self._features_info):
        raise ValueError('Features are incompatible with given information. '
                         'Given features: %s, required signatures: %s.' %
                         (str(features), str(self._features_info)))
    else:
      self._features_info = tensor_signature.create_signatures(features)
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
                   device_fn=None,
                   monitor=None,
                   log_every_steps=100,
                   fail_on_nan_loss=True):
    if self._config.execution_mode not in ('all', 'train'):
      return

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

    with ops.Graph().as_default() as g, g.device(device_fn):
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step = contrib_framework.create_global_step(g)
      features, targets = input_fn()
      self._check_inputs(features, targets)
      train_op, loss_op = self._get_train_ops(features, targets)
      return train(
          graph=g,
          output_dir=self._model_dir,
          train_op=train_op,
          loss_op=loss_op,
          global_step_tensor=global_step,
          log_every_steps=log_every_steps,
          supervisor_is_chief=(self._config.task == 0),
          supervisor_master=self._config.master,
          feed_fn=feed_fn,
          max_steps=steps,
          fail_on_nan_loss=fail_on_nan_loss)

  def _evaluate_model(self, input_fn, steps, feed_fn=None, metrics=None):
    if self._config.execution_mode not in ('all', 'evaluate', 'eval_evalset'):
      return

    checkpoint_path = saver.latest_checkpoint(self._model_dir)
    eval_dir = os.path.join(self._model_dir, 'eval')
    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step = contrib_framework.create_global_step(g)
      features, targets = input_fn()
      self._check_inputs(features, targets)
      eval_dict = self._get_eval_ops(features, targets, metrics or
                                     self._get_default_metric_functions())
      eval_results, _ = evaluate(
          graph=g,
          output_dir=eval_dir,
          checkpoint_path=checkpoint_path,
          eval_dict=eval_dict,
          global_step_tensor=global_step,
          supervisor_master=self._config.master,
          feed_fn=feed_fn,
          max_steps=steps)
      return eval_results

  def _infer_model(self, x, batch_size=None, axis=None, proba=False):
    # Converts inputs into tf.DataFrame / tf.Series.
    batch_size = -1 if batch_size is None else batch_size
    input_fn, feed_fn = _get_predict_input_fn(x, batch_size)

    checkpoint_path = saver.latest_checkpoint(self._model_dir)
    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(self._config.tf_random_seed)
      contrib_framework.create_global_step(g)
      features, _ = input_fn()
      feed_dict = feed_fn() if feed_fn is not None else None
      predictions = self._get_predict_ops(features)
      if not isinstance(predictions, dict):
        predictions = {'predictions': predictions}
      # TODO(ipolosukhin): Support batching
      return infer(checkpoint_path, predictions, feed_dict=feed_dict)


class Estimator(BaseEstimator):
  """Estimator class is the basic TensorFlow model trainer/evaluator.

  Parameters:
    model_fn: Model function, takes features and targets tensors or dicts of
              tensors and returns predictions and loss tensors.
              E.g. `(features, targets) -> (predictions, loss)`.
    model_dir: Directory to save model parameters, graph and etc.
    classification: boolean, true if classification problem.
    learning_rate: learning rate for the model.
    optimizer: optimizer for the model, can be:
               string: name of optimizer, like 'SGD', 'Adam', 'Adagrad', 'Ftl',
                 'Momentum', 'RMSProp', 'Momentum').
                 Full list in contrib/layers/optimizers.py
               class: sub-class of Optimizer
                 (like tf.train.GradientDescentOptimizer).
    clip_gradients: clip_norm value for call to `clip_by_global_norm`. None
                    denotes no gradient clipping.
  """

  def __init__(self,
               model_fn=None,
               model_dir=None,
               classification=True,
               learning_rate=0.01,
               optimizer='SGD',
               clip_gradients=None):
    super(Estimator, self).__init__(model_dir=model_dir)

    self._model_fn = model_fn
    self._classification = classification
    if isinstance(optimizer, six.string_types):
      if optimizer not in layers.OPTIMIZER_CLS_NAMES:
        raise ValueError(
            'Optimizer name should be one of [%s], you provided %s.' %
            (', '.join(layers.OPTIMIZER_CLS_NAMES), optimizer))
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.clip_gradients = clip_gradients

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
    _, loss = self._model_fn(features, targets, ModeKeys.TRAIN)
    train_op = layers.optimize_loss(
        loss,
        contrib_framework.get_global_step(),
        learning_rate=self.learning_rate,
        optimizer=self.optimizer,
        clip_gradients=self.clip_gradients)
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
    predictions, loss = self._model_fn(features, targets, ModeKeys.EVAL)
    result = {'loss': loss}
    if isinstance(targets, dict) and len(targets) == 1:
      # Unpack single target into just tensor.
      targets = targets[targets.keys()[0]]
    for name, metric in six.iteritems(metrics):
      # TODO(ipolosukhin): Add support for multi-head metrics.
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
    predictions, _ = self._model_fn(features, targets, ModeKeys.INFER)
    return predictions

  def _get_default_metric_functions(self):
    """Method that provides default metric operations.

    Returns:
      a dictionary of metric operations.
    """
    return _EVAL_METRICS[
        'classification' if self._classification else 'regression']

  def _get_feature_ops_from_example(self, examples_batch):
    """Unimplemented.

    TODO(vihanjain): We need a way to parse tf.Example into features.

    Args:
      examples_batch: batch of tf.Example

    Returns:
      features: `Tensor` or `dict` of `Tensor` objects.

    Raises:
      Exception: Unimplemented
    """
    raise NotImplementedError('_get_feature_ops_from_example not yet '
                              'implemented')
