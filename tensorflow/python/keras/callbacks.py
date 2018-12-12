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
# pylint: disable=g-import-not-at-top
"""Callbacks: utilities called at certain points during model training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import csv
import io
import json
import os
import time

import numpy as np
import six

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training import saver
from tensorflow.python.util.tf_export import tf_export


try:
  import requests
except ImportError:
  requests = None


# pylint: disable=protected-access
def configure_callbacks(callbacks,
                        model,
                        do_validation=False,
                        batch_size=None,
                        epochs=None,
                        steps_per_epoch=None,
                        samples=None,
                        verbose=1,
                        count_mode='steps',
                        mode='train'):
  """Configures callbacks for use in various training loops.

  Arguments:
      callbacks: List of Callbacks.
      model: Model being trained.
      do_validation: Whether or not validation loop will be run.
      batch_size: Number of samples per batch.
      epochs: Number of epoch to train.
      steps_per_epoch: Number of batches to run per training epoch.
      samples: Number of training samples.
      verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
      count_mode: One of 'steps' or 'samples'. Per-batch or per-sample count.
      mode: String. One of 'train', 'test', or 'predict'. Which loop mode to
        configure callbacks for.

  Returns:
      Instance of CallbackList used to control all Callbacks.
  """
  # Check if callbacks have already been configured.
  if isinstance(callbacks, CallbackList):
    return callbacks

  if not callbacks:
    callbacks = []

  # Add additional callbacks during training.
  if mode == 'train':
    model.history = History()
    stateful_metric_names = None
    if hasattr(model, 'metrics_names'):
      stateful_metric_names = model.metrics_names[1:]  # Exclude `loss`
    callbacks = [BaseLogger(stateful_metrics=stateful_metric_names)
                ] + (callbacks or []) + [model.history]
    if verbose:
      callbacks.append(
          ProgbarLogger(count_mode, stateful_metrics=stateful_metric_names))
  callback_list = CallbackList(callbacks)

  # Set callback model
  callback_model = model._get_callback_model()
  callback_list.set_model(callback_model)

  # Set callback parameters
  callback_metrics = []
  # When we have deferred build scenario with iterator input, we will compile
  # when we standardize first batch of data.
  if mode != 'predict' and hasattr(model, 'metrics_names'):
    callback_metrics = copy.copy(model.metrics_names)
    if do_validation:
      callback_metrics += ['val_' + n for n in model.metrics_names]
  callback_params = {
      'batch_size': batch_size,
      'epochs': epochs,
      'steps': steps_per_epoch,
      'samples': samples,
      'verbose': verbose,
      'do_validation': do_validation,
      'metrics': callback_metrics,
  }
  callback_list.set_params(callback_params)

  if (do_validation and not model._distribution_strategy and
      not model.run_eagerly):
    # Need to create the eval_function before start of the first epoch
    # because TensorBoard callback on_epoch_begin adds summary to the
    # list of fetches of the eval_function
    callback_model._make_eval_function()

  callback_list.model.stop_training = False
  return callback_list
# pylint: enable=protected-access


def _is_generator_like(data):
  """Checks if data is a generator, Sequence, or Iterator."""
  return (hasattr(data, 'next') or hasattr(data, '__next__') or isinstance(
      data, (Sequence, iterator_ops.Iterator, iterator_ops.EagerIterator)))


class CallbackList(object):
  """Container abstracting a list of callbacks.

  Arguments:
      callbacks: List of `Callback` instances.
      queue_length: Queue length for keeping
          running statistics over callback execution time.
  """

  def __init__(self, callbacks=None, queue_length=10):
    callbacks = callbacks or []
    self.callbacks = [c for c in callbacks]
    self.queue_length = queue_length
    self.params = {}
    self.model = None
    self._reset_batch_timing()

  def _reset_batch_timing(self):
    self._delta_t_batch = 0.
    self._delta_ts = collections.defaultdict(
        lambda: collections.deque([], maxlen=self.queue_length))

  def append(self, callback):
    self.callbacks.append(callback)

  def set_params(self, params):
    self.params = params
    for callback in self.callbacks:
      callback.set_params(params)

  def set_model(self, model):
    self.model = model
    for callback in self.callbacks:
      callback.set_model(model)

  def _call_batch_hook(self, mode, hook, batch, logs=None):
    """Helper function for all batch_{begin | end} methods."""
    # TODO(omalleyt): add batch hooks for test/predict.
    if mode != 'train':
      return

    hook_name = 'on_{mode}_batch_{hook}'.format(mode=mode, hook=hook)
    if hook == 'begin':
      self._t_enter_batch = time.time()
    if hook == 'end':
      # Batch is ending, calculate batch time.
      self._delta_t_batch = time.time() - self._t_enter_batch

    logs = logs or {}
    t_before_callbacks = time.time()
    for callback in self.callbacks:
      batch_hook = getattr(callback, hook_name)
      batch_hook(batch, logs)
    self._delta_ts[hook_name].append(time.time() - t_before_callbacks)

    delta_t_median = np.median(self._delta_ts[hook_name])
    if (self._delta_t_batch > 0. and
        delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
      logging.warning(
          'Method (%s) is slow compared '
          'to the batch update (%f). Check your callbacks.', hook_name,
          delta_t_median)

  def _call_begin_hook(self, mode):
    """Helper function for on_{train|test|predict}_begin methods."""
    # TODO(omalleyt): add test/predict methods.
    if mode == 'train':
      self.on_train_begin()

  def _call_end_hook(self, mode):
    """Helper function for on_{train|test|predict}_end methods."""
    # TODO(omalleyt): add test/predict methods.
    if mode == 'train':
      self.on_train_end()

  def on_batch_begin(self, batch, logs=None):
    self._call_batch_hook('train', 'begin', batch, logs=logs)

  def on_batch_end(self, batch, logs=None):
    self._call_batch_hook('train', 'end', batch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None, mode='train'):
    """Called at the start of an epoch.

    Arguments:
        epoch: integer, index of epoch.
        logs: dictionary of logs.
        mode: One of 'train'/'test'/'predict'
    """
    if mode == 'train':
      logs = logs or {}
      for callback in self.callbacks:
        callback.on_epoch_begin(epoch, logs)
    self._reset_batch_timing()

  def on_epoch_end(self, epoch, logs=None, mode='train'):
    """Called at the end of an epoch.

    Arguments:
        epoch: integer, index of epoch.
        logs: dictionary of logs.
        mode: One of 'train'/'test'/'predict'
    """
    if mode == 'train':
      logs = logs or {}
      for callback in self.callbacks:
        callback.on_epoch_end(epoch, logs)

  def on_train_batch_begin(self, batch, logs=None):
    """Called at the beginning of a training batch in `fit` methods.

    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dictionary of logs.
    """
    self._call_batch_hook('train', 'begin', batch, logs=logs)

  def on_train_batch_end(self, batch, logs=None):
    """Called at the end of a training batch in `fit` methods.

    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dictionary of logs.
    """
    self._call_batch_hook('train', 'end', batch, logs=logs)

  def on_train_begin(self, logs=None):
    """Called at the beginning of training.

    Arguments:
        logs: dictionary of logs.
    """
    logs = logs or {}
    for callback in self.callbacks:
      callback.on_train_begin(logs)

  def on_train_end(self, logs=None):
    """Called at the end of training.

    Arguments:
        logs: dictionary of logs.
    """
    logs = logs or {}
    for callback in self.callbacks:
      callback.on_train_end(logs)

  def __iter__(self):
    return iter(self.callbacks)


@tf_export('keras.callbacks.Callback')
class Callback(object):
  """Abstract base class used to build new callbacks.

  Attributes:
      params: dict. Training parameters
          (eg. verbosity, batch size, number of epochs...).
      model: instance of `keras.models.Model`.
          Reference of the model being trained.

  The `logs` dictionary that callback methods
  take as argument will contain keys for quantities relevant to
  the current batch or epoch.

  Currently, the `.fit()` method of the `Sequential` model class
  will include the following quantities in the `logs` that
  it passes to its callbacks:

      on_epoch_end: logs include `acc` and `loss`, and
          optionally include `val_loss`
          (if validation is enabled in `fit`), and `val_acc`
          (if validation and accuracy monitoring are enabled).
      on_batch_begin: logs include `size`,
          the number of samples in the current batch.
      on_batch_end: logs include `loss`, and optionally `acc`
          (if accuracy monitoring is enabled).
  """

  def __init__(self):
    self.validation_data = None
    self.model = None

  def set_params(self, params):
    self.params = params

  def set_model(self, model):
    self.model = model

  def on_epoch_begin(self, epoch, logs=None):
    pass

  def on_epoch_end(self, epoch, logs=None):
    pass

  def on_batch_begin(self, batch, logs=None):
    pass

  def on_batch_end(self, batch, logs=None):
    pass

  def on_train_batch_begin(self, batch, logs=None):
    # For backwards compatibility
    self.on_batch_begin(batch, logs=logs)

  def on_train_batch_end(self, batch, logs=None):
    # For backwards compatibility
    self.on_batch_end(batch, logs=logs)

  def on_train_begin(self, logs=None):
    pass

  def on_train_end(self, logs=None):
    pass


@tf_export('keras.callbacks.BaseLogger')
class BaseLogger(Callback):
  """Callback that accumulates epoch averages of metrics.

  This callback is automatically applied to every Keras model.

  Arguments:
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over an epoch.
          Metrics in this list will be logged as-is in `on_epoch_end`.
          All others will be averaged in `on_epoch_end`.
  """

  def __init__(self, stateful_metrics=None):
    super(BaseLogger, self).__init__()
    self.stateful_metrics = set(stateful_metrics or [])

  def on_epoch_begin(self, epoch, logs=None):
    self.seen = 0
    self.totals = {}

  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    batch_size = logs.get('size', 0)
    # In case of distribution strategy we can potentially run multiple steps
    # at the same time, we should account for that in the `seen` calculation.
    num_steps = logs.get('num_steps', 1)
    self.seen += batch_size * num_steps

    for k, v in logs.items():
      if k in self.stateful_metrics:
        self.totals[k] = v
      else:
        if k in self.totals:
          self.totals[k] += v * batch_size
        else:
          self.totals[k] = v * batch_size

  def on_epoch_end(self, epoch, logs=None):
    if logs is not None:
      for k in self.params['metrics']:
        if k in self.totals:
          # Make value available to next callbacks.
          if k in self.stateful_metrics:
            logs[k] = self.totals[k]
          else:
            logs[k] = self.totals[k] / self.seen


@tf_export('keras.callbacks.TerminateOnNaN')
class TerminateOnNaN(Callback):
  """Callback that terminates training when a NaN loss is encountered.
  """

  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    loss = logs.get('loss')
    if loss is not None:
      if np.isnan(loss) or np.isinf(loss):
        print('Batch %d: Invalid loss, terminating training' % (batch))
        self.model.stop_training = True


@tf_export('keras.callbacks.ProgbarLogger')
class ProgbarLogger(Callback):
  """Callback that prints metrics to stdout.

  Arguments:
      count_mode: One of "steps" or "samples".
          Whether the progress bar should
          count samples seen or steps (batches) seen.
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over an epoch.
          Metrics in this list will be logged as-is.
          All others will be averaged over time (e.g. loss, etc).

  Raises:
      ValueError: In case of invalid `count_mode`.
  """

  def __init__(self, count_mode='samples', stateful_metrics=None):
    super(ProgbarLogger, self).__init__()
    if count_mode == 'samples':
      self.use_steps = False
    elif count_mode == 'steps':
      self.use_steps = True
    else:
      raise ValueError('Unknown `count_mode`: ' + str(count_mode))
    self.stateful_metrics = set(stateful_metrics or [])

  def on_train_begin(self, logs=None):
    self.verbose = self.params['verbose']
    self.epochs = self.params['epochs']

  def on_epoch_begin(self, epoch, logs=None):
    self.seen = 0
    if self.use_steps:
      self.target = self.params['steps']
    else:
      self.target = self.params['samples']

    if self.verbose:
      if self.epochs > 1:
        print('Epoch %d/%d' % (epoch + 1, self.epochs))
      self.progbar = Progbar(
          target=self.target,
          verbose=self.verbose,
          stateful_metrics=self.stateful_metrics,
          unit_name='step' if self.use_steps else 'sample')

  def on_batch_begin(self, batch, logs=None):
    if self.seen < self.target:
      self.log_values = []

  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    batch_size = logs.get('size', 0)
    # In case of distribution strategy we can potentially run multiple steps
    # at the same time, we should account for that in the `seen` calculation.
    num_steps = logs.get('num_steps', 1)
    if self.use_steps:
      self.seen += num_steps
    else:
      self.seen += batch_size * num_steps

    for k in self.params['metrics']:
      if k in logs:
        self.log_values.append((k, logs[k]))

    # Skip progbar update for the last batch;
    # will be handled by on_epoch_end.
    if self.verbose and self.seen < self.target:
      self.progbar.update(self.seen, self.log_values)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    for k in self.params['metrics']:
      if k in logs:
        self.log_values.append((k, logs[k]))
    if self.verbose:
      self.progbar.update(self.seen, self.log_values)


@tf_export('keras.callbacks.History')
class History(Callback):
  """Callback that records events into a `History` object.

  This callback is automatically applied to
  every Keras model. The `History` object
  gets returned by the `fit` method of models.
  """

  def on_train_begin(self, logs=None):
    self.epoch = []
    self.history = {}

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    self.epoch.append(epoch)
    for k, v in logs.items():
      self.history.setdefault(k, []).append(v)


@tf_export('keras.callbacks.ModelCheckpoint')
class ModelCheckpoint(Callback):
  """Save the model after every epoch.

  `filepath` can contain named formatting options,
  which will be filled the value of `epoch` and
  keys in `logs` (passed in `on_epoch_end`).

  For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
  then the model checkpoints will be saved with the epoch number and
  the validation loss in the filename.

  Arguments:
      filepath: string, path to save the model file.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`,
          the latest best model according to
          the quantity monitored will not be overwritten.
      mode: one of {auto, min, max}.
          If `save_best_only=True`, the decision
          to overwrite the current save file is made
          based on either the maximization or the
          minimization of the monitored quantity. For `val_acc`,
          this should be `max`, for `val_loss` this should
          be `min`, etc. In `auto` mode, the direction is
          automatically inferred from the name of the monitored quantity.
      save_weights_only: if True, then only the model's weights will be
          saved (`model.save_weights(filepath)`), else the full model
          is saved (`model.save(filepath)`).
      period: Interval (number of epochs) between checkpoints.
  """

  def __init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               period=1):
    super(ModelCheckpoint, self).__init__()
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = filepath
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.period = period
    self.epochs_since_last_save = 0

    if mode not in ['auto', 'min', 'max']:
      logging.warning('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else:
        self.monitor_op = np.less
        self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    self.epochs_since_last_save += 1
    if self.epochs_since_last_save >= self.period:
      self.epochs_since_last_save = 0
      filepath = self.filepath.format(epoch=epoch + 1, **logs)
      if self.save_best_only:
        current = logs.get(self.monitor)
        if current is None:
          logging.warning('Can save best model only with %s available, '
                          'skipping.', self.monitor)
        else:
          if self.monitor_op(current, self.best):
            if self.verbose > 0:
              print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                    ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                             current, filepath))
            self.best = current
            if self.save_weights_only:
              self.model.save_weights(filepath, overwrite=True)
            else:
              self.model.save(filepath, overwrite=True)
          else:
            if self.verbose > 0:
              print('\nEpoch %05d: %s did not improve from %0.5f' %
                    (epoch + 1, self.monitor, self.best))
      else:
        if self.verbose > 0:
          print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
        if self.save_weights_only:
          self.model.save_weights(filepath, overwrite=True)
        else:
          self.model.save(filepath, overwrite=True)


@tf_export('keras.callbacks.EarlyStopping')
class EarlyStopping(Callback):
  """Stop training when a monitored quantity has stopped improving.

  Arguments:
      monitor: Quantity to be monitored.
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of epochs with no improvement
          after which training will be stopped.
      verbose: verbosity mode.
      mode: One of `{"auto", "min", "max"}`. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `max`
          mode it will stop when the quantity
          monitored has stopped increasing; in `auto`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      baseline: Baseline value for the monitored quantity.
          Training will stop if the model doesn't show improvement over the
          baseline.
      restore_best_weights: Whether to restore model weights from
          the epoch with the best value of the monitored quantity.
          If False, the model weights obtained at the last step of
          training are used.
  """

  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None,
               restore_best_weights=False):
    super(EarlyStopping, self).__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.restore_best_weights = restore_best_weights
    self.best_weights = None

    if mode not in ['auto', 'min', 'max']:
      logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor:
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    if self.baseline is not None:
      self.best = self.baseline
    else:
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        if self.restore_best_weights:
          if self.verbose > 0:
            print('Restoring model weights from the end of the best epoch.')
          self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    if monitor_value is None:
      logging.warning('Early stopping conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))
    return monitor_value


@tf_export('keras.callbacks.RemoteMonitor')
class RemoteMonitor(Callback):
  """Callback used to stream events to a server.

  Requires the `requests` library.
  Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
  HTTP POST, with a `data` argument which is a
  JSON-encoded dictionary of event data.
  If send_as_json is set to True, the content type of the request will be
  application/json. Otherwise the serialized JSON will be sent within a form.

  Arguments:
      root: String; root url of the target server.
      path: String; path relative to `root` to which the events will be sent.
      field: String; JSON field under which the data will be stored.
          The field is used only if the payload is sent within a form
          (i.e. send_as_json is set to False).
      headers: Dictionary; optional custom HTTP headers.
      send_as_json: Boolean; whether the request should be
          sent as application/json.
  """

  def __init__(self,
               root='http://localhost:9000',
               path='/publish/epoch/end/',
               field='data',
               headers=None,
               send_as_json=False):
    super(RemoteMonitor, self).__init__()

    self.root = root
    self.path = path
    self.field = field
    self.headers = headers
    self.send_as_json = send_as_json

  def on_epoch_end(self, epoch, logs=None):
    if requests is None:
      raise ImportError('RemoteMonitor requires the `requests` library.')
    logs = logs or {}
    send = {}
    send['epoch'] = epoch
    for k, v in logs.items():
      send[k] = v
    try:
      if self.send_as_json:
        requests.post(self.root + self.path, json=send, headers=self.headers)
      else:
        requests.post(
            self.root + self.path, {self.field: json.dumps(send)},
            headers=self.headers)
    except requests.exceptions.RequestException:
      logging.warning('Warning: could not reach RemoteMonitor '
                      'root server at ' + str(self.root))


@tf_export('keras.callbacks.LearningRateScheduler')
class LearningRateScheduler(Callback):
  """Learning rate scheduler.

  Arguments:
      schedule: a function that takes an epoch index as input
          (integer, indexed from 0) and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages.
  """

  def __init__(self, schedule, verbose=0):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule
    self.verbose = verbose

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    try:  # new API
      lr = float(K.get_value(self.model.optimizer.lr))
      lr = self.schedule(epoch, lr)
    except TypeError:  # Support for old API for backward compatibility
      lr = self.schedule(epoch)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    K.set_value(self.model.optimizer.lr, lr)
    if self.verbose > 0:
      print('\nEpoch %05d: LearningRateScheduler reducing learning '
            'rate to %s.' % (epoch + 1, lr))

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = K.get_value(self.model.optimizer.lr)


@tf_export('keras.callbacks.TensorBoard')
class TensorBoard(Callback):
  # pylint: disable=line-too-long
  """Tensorboard basic visualizations.

  This callback writes a log for TensorBoard, which allows
  you to visualize dynamic graphs of your training and test
  metrics, as well as activation histograms for the different
  layers in your model.

  TensorBoard is a visualization tool provided with TensorFlow.

  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:

  ```sh
  tensorboard --logdir=/full_path_to_your_logs
  ```

  You can find more information about TensorBoard
  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

  Arguments:
      log_dir: the path of the directory where to save the log
          files to be parsed by TensorBoard.
      histogram_freq: frequency (in epochs) at which to compute activation
          and weight histograms for the layers of the model. If set to 0,
          histograms won't be computed. Validation data (or split) must be
          specified for histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard.
          The log file can become quite large when
          write_graph is set to True.
      write_grads: whether to visualize gradient histograms in TensorBoard.
          `histogram_freq` must be greater than 0.
      batch_size: size of batch of inputs to feed to the network
          for histograms computation.
      write_images: whether to write model weights to visualize as
          image in TensorBoard.
      embeddings_freq: frequency (in epochs) at which selected embedding
          layers will be saved. If set to 0, embeddings won't be computed.
          Data to be visualized in TensorBoard's Embedding tab must be passed
          as `embeddings_data`.
      embeddings_layer_names: a list of names of layers to keep eye on. If
          None or empty list all the embedding layer will be watched.
      embeddings_metadata: a dictionary which maps layer name to a file name
          in which metadata for this embedding layer is saved. See the
          [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
          about metadata files format. In case if the same metadata file is
          used for all embedding layers, string can be passed.
      embeddings_data: data to be embedded at layers specified in
          `embeddings_layer_names`. Numpy array (if the model has a single
          input) or list of Numpy arrays (if the model has multiple inputs).
          Learn [more about embeddings](https://www.tensorflow.org/programmers_guide/embedding)
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
          writes the losses and metrics to TensorBoard after each batch.
          The same applies for `'epoch'`. If using an integer, let's say `1000`,
          the callback will write the metrics and losses to TensorBoard every
          1000 samples. Note that writing too frequently to TensorBoard
          can slow down your training.

  Raises:
      ValueError: If histogram_freq is set and no validation data is provided.

  @compatibility(eager)
  Using `Tensorboard` callback will work while eager execution is enabled,
  however outputting histogram summaries of weights and gradients is not
  supported, and thus `histogram_freq` will be ignored.
  @end_compatibility
  """

  # pylint: enable=line-too-long

  def __init__(self,
               log_dir='./logs',
               histogram_freq=0,
               batch_size=32,
               write_graph=True,
               write_grads=False,
               write_images=False,
               embeddings_freq=0,
               embeddings_layer_names=None,
               embeddings_metadata=None,
               embeddings_data=None,
               update_freq='epoch'):
    super(TensorBoard, self).__init__()
    self.log_dir = log_dir
    self.histogram_freq = histogram_freq
    if self.histogram_freq and context.executing_eagerly():
      logging.warning(
          UserWarning('Weight and gradient histograms not supported for eager'
                      'execution, setting `histogram_freq` to `0`.'))
      self.histogram_freq = 0
    self.merged = None
    self.write_graph = write_graph
    self.write_grads = write_grads
    self.write_images = write_images
    self.batch_size = batch_size
    self._current_batch = 0
    self._total_batches_seen = 0
    self._total_val_batches_seen = 0
    self.embeddings_freq = embeddings_freq
    self.embeddings_layer_names = embeddings_layer_names
    self.embeddings_metadata = embeddings_metadata
    self.embeddings_data = embeddings_data
    if update_freq == 'batch':
      self.update_freq = 1
    else:
      self.update_freq = update_freq
    self._samples_seen = 0
    self._samples_seen_at_last_write = 0

  def _init_writer(self):
    """Sets file writer."""
    if context.executing_eagerly():
      self.writer = summary_ops_v2.create_file_writer(self.log_dir)
    elif self.write_graph:
      self.writer = tf_summary.FileWriter(self.log_dir, K.get_session().graph)
    else:
      self.writer = tf_summary.FileWriter(self.log_dir)

  def _make_histogram_ops(self, model):
    """Defines histogram ops when histogram_freq > 0."""
    # only make histogram summary op if it hasn't already been made
    if self.histogram_freq and self.merged is None:
      for layer in self.model.layers:
        for weight in layer.weights:
          mapped_weight_name = weight.name.replace(':', '_')
          tf_summary.histogram(mapped_weight_name, weight)
          if self.write_images:
            w_img = array_ops.squeeze(weight)
            shape = K.int_shape(w_img)
            if len(shape) == 2:  # dense layer kernel case
              if shape[0] > shape[1]:
                w_img = array_ops.transpose(w_img)
                shape = K.int_shape(w_img)
              w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
            elif len(shape) == 3:  # convnet case
              if K.image_data_format() == 'channels_last':
                # switch to channels_first to display
                # every kernel as a separate image
                w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                shape = K.int_shape(w_img)
              w_img = array_ops.reshape(w_img,
                                        [shape[0], shape[1], shape[2], 1])
            elif len(shape) == 1:  # bias case
              w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
            else:
              # not possible to handle 3D convnets etc.
              continue

            shape = K.int_shape(w_img)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            tf_summary.image(mapped_weight_name, w_img)

        if self.write_grads:
          for weight in layer.trainable_weights:
            mapped_weight_name = weight.name.replace(':', '_')
            grads = model.optimizer.get_gradients(model.total_loss, weight)

            def is_indexed_slices(grad):
              return type(grad).__name__ == 'IndexedSlices'

            grads = [
                grad.values if is_indexed_slices(grad) else grad
                for grad in grads
            ]
            tf_summary.histogram('{}_grad'.format(mapped_weight_name), grads)

        if hasattr(layer, 'output'):
          if isinstance(layer.output, list):
            for i, output in enumerate(layer.output):
              tf_summary.histogram('{}_out_{}'.format(layer.name, i), output)
          else:
            tf_summary.histogram('{}_out'.format(layer.name), layer.output)

  def set_model(self, model):
    """Sets Keras model and creates summary ops."""

    self.model = model
    self._init_writer()
    # histogram summaries only enabled in graph mode
    if not context.executing_eagerly():
      self._make_histogram_ops(model)
      self.merged = tf_summary.merge_all()

    # If both embedding_freq and embeddings_data are available, we will
    # visualize embeddings.
    if self.embeddings_freq and self.embeddings_data is not None:
      # Avoid circular dependency.
      from tensorflow.python.keras.engine import training_utils  # pylint: disable=g-import-not-at-top
      self.embeddings_data = training_utils.standardize_input_data(
          self.embeddings_data, model.input_names)

      # If embedding_layer_names are not provided, get all of the embedding
      # layers from the model.
      embeddings_layer_names = self.embeddings_layer_names
      if not embeddings_layer_names:
        embeddings_layer_names = [
            layer.name
            for layer in self.model.layers
            if type(layer).__name__ == 'Embedding'
        ]

      self.assign_embeddings = []
      embeddings_vars = {}

      self.batch_id = batch_id = array_ops.placeholder(dtypes.int32)
      self.step = step = array_ops.placeholder(dtypes.int32)

      for layer in self.model.layers:
        if layer.name in embeddings_layer_names:
          embedding_input = self.model.get_layer(layer.name).output
          embedding_size = np.prod(embedding_input.shape[1:])
          embedding_input = array_ops.reshape(embedding_input,
                                              (step, int(embedding_size)))
          shape = (self.embeddings_data[0].shape[0], int(embedding_size))
          embedding = variables.Variable(
              array_ops.zeros(shape), name=layer.name + '_embedding')
          embeddings_vars[layer.name] = embedding
          batch = state_ops.assign(embedding[batch_id:batch_id + step],
                                   embedding_input)
          self.assign_embeddings.append(batch)

      self.saver = saver.Saver(list(embeddings_vars.values()))

      # Create embeddings_metadata dictionary
      if isinstance(self.embeddings_metadata, str):
        embeddings_metadata = {
            layer_name: self.embeddings_metadata
            for layer_name in embeddings_vars.keys()
        }
      else:
        # If embedding_metadata is already a dictionary
        embeddings_metadata = self.embeddings_metadata

      try:
        from tensorboard.plugins import projector
      except ImportError:
        raise ImportError('Failed to import TensorBoard. Please make sure that '
                          'TensorBoard integration is complete."')

      # TODO(psv): Add integration tests to test embedding visualization
      # with TensorBoard callback. We are unable to write a unit test for this
      # because TensorBoard dependency assumes TensorFlow package is installed.
      config = projector.ProjectorConfig()
      for layer_name, tensor in embeddings_vars.items():
        embedding = config.embeddings.add()
        embedding.tensor_name = tensor.name

        if (embeddings_metadata is not None and
            layer_name in embeddings_metadata):
          embedding.metadata_path = embeddings_metadata[layer_name]

      projector.visualize_embeddings(self.writer, config)

  def _fetch_callback(self, summary):
    self.writer.add_summary(summary, self._total_val_batches_seen)
    self._total_val_batches_seen += 1

  def _write_custom_summaries(self, step, logs=None):
    """Writes metrics out as custom scalar summaries.

    Arguments:
        step: the global step to use for Tensorboard.
        logs: dict. Keys are scalar summary names, values are
            NumPy scalars.

    """
    logs = logs or {}
    if context.executing_eagerly():
      # use v2 summary ops
      with self.writer.as_default(), summary_ops_v2.always_record_summaries():
        for name, value in logs.items():
          if isinstance(value, np.ndarray):
            value = value.item()
          summary_ops_v2.scalar(name, value, step=step)
    else:
      # use FileWriter from v1 summary
      for name, value in logs.items():
        if isinstance(value, np.ndarray):
          value = value.item()
        summary = tf_summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        self.writer.add_summary(summary, step)
    self.writer.flush()

  def on_batch_end(self, batch, logs=None):
    """Writes scalar summaries for metrics on every training batch."""
    # Don't output batch_size and batch number as Tensorboard summaries
    logs = logs or {}
    self._samples_seen += logs.get('size', 1)
    samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
    if self.update_freq != 'epoch' and samples_seen_since >= self.update_freq:
      batch_logs = {('batch_' + k): v
                    for k, v in logs.items()
                    if k not in ['batch', 'size', 'num_steps']}
      self._write_custom_summaries(self._total_batches_seen, batch_logs)
      self._samples_seen_at_last_write = self._samples_seen
    self._total_batches_seen += 1

  def on_epoch_begin(self, epoch, logs=None):
    """Add histogram op to Model eval_function callbacks, reset batch count."""

    # check if histogram summary should be run for this epoch
    if self.histogram_freq and epoch % self.histogram_freq == 0:
      self._epoch = epoch
      # pylint: disable=protected-access
      # add the histogram summary op if it should run this epoch
      if self.merged not in self.model._eval_function.fetches:
        self.model._eval_function.fetches.append(self.merged)
        self.model._eval_function.fetch_callbacks[
            self.merged] = self._fetch_callback
      # pylint: enable=protected-access

  def on_epoch_end(self, epoch, logs=None):
    """Checks if summary ops should run next epoch, logs scalar summaries."""

    # don't output batch_size and
    # batch number as Tensorboard summaries
    logs = {('epoch_' + k): v
            for k, v in logs.items()
            if k not in ['batch', 'size', 'num_steps']}
    if self.update_freq == 'epoch':
      step = epoch
    else:
      step = self._samples_seen
    self._write_custom_summaries(step, logs)

    # pop the histogram summary op after each epoch
    if self.histogram_freq:
      # pylint: disable=protected-access
      if self.merged in self.model._eval_function.fetches:
        self.model._eval_function.fetches.remove(self.merged)
      if self.merged in self.model._eval_function.fetch_callbacks:
        self.model._eval_function.fetch_callbacks.pop(self.merged)
      # pylint: enable=protected-access

    if self.embeddings_data is None and self.embeddings_freq:
      raise ValueError('To visualize embeddings, embeddings_data must '
                       'be provided.')

    if self.embeddings_freq and self.embeddings_data is not None:
      if epoch % self.embeddings_freq == 0:
        # We need a second forward-pass here because we're passing
        # the `embeddings_data` explicitly. This design allows to pass
        # arbitrary data as `embeddings_data` and results from the fact
        # that we need to know the size of the `tf.Variable`s which
        # hold the embeddings in `set_model`. At this point, however,
        # the `validation_data` is not yet set.

        embeddings_data = self.embeddings_data
        n_samples = embeddings_data[0].shape[0]
        i = 0
        while i < n_samples:
          step = min(self.batch_size, n_samples - i)
          batch = slice(i, i + step)

          if isinstance(self.model.input, list):
            feed_dict = {
                model_input: embeddings_data[idx][batch]
                for idx, model_input in enumerate(self.model.input)
            }
          else:
            feed_dict = {self.model.input: embeddings_data[0][batch]}

          feed_dict.update({self.batch_id: i, self.step: step})

          if not isinstance(K.learning_phase(), int):
            feed_dict[K.learning_phase()] = False

          self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
          self.saver.save(self.sess,
                          os.path.join(self.log_dir, 'keras_embedding.ckpt'),
                          epoch)

          i += self.batch_size

  def on_train_end(self, logs=None):
    self.writer.close()


@tf_export('keras.callbacks.ReduceLROnPlateau')
class ReduceLROnPlateau(Callback):
  """Reduce learning rate when a metric has stopped improving.

  Models often benefit from reducing the learning rate by a factor
  of 2-10 once learning stagnates. This callback monitors a
  quantity and if no improvement is seen for a 'patience' number
  of epochs, the learning rate is reduced.

  Example:

  ```python
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
  model.fit(X_train, Y_train, callbacks=[reduce_lr])
  ```

  Arguments:
      monitor: quantity to be monitored.
      factor: factor by which the learning rate will
          be reduced. new_lr = lr * factor
      patience: number of epochs with no improvement
          after which learning rate will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      mode: one of {auto, min, max}. In `min` mode,
          lr will be reduced when the quantity
          monitored has stopped decreasing; in `max`
          mode it will be reduced when the quantity
          monitored has stopped increasing; in `auto`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      min_delta: threshold for measuring the new optimum,
          to only focus on significant changes.
      cooldown: number of epochs to wait before resuming
          normal operation after lr has been reduced.
      min_lr: lower bound on the learning rate.
  """

  def __init__(self,
               monitor='val_loss',
               factor=0.1,
               patience=10,
               verbose=0,
               mode='auto',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0,
               **kwargs):
    super(ReduceLROnPlateau, self).__init__()

    self.monitor = monitor
    if factor >= 1.0:
      raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
    if 'epsilon' in kwargs:
      min_delta = kwargs.pop('epsilon')
      logging.warning('`epsilon` argument is deprecated and '
                      'will be removed, use `min_delta` instead.')
    self.factor = factor
    self.min_lr = min_lr
    self.min_delta = min_delta
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.best = 0
    self.mode = mode
    self.monitor_op = None
    self._reset()

  def _reset(self):
    """Resets wait counter and cooldown counter.
    """
    if self.mode not in ['auto', 'min', 'max']:
      logging.warning('Learning Rate Plateau Reducing mode %s is unknown, '
                      'fallback to auto mode.', self.mode)
      self.mode = 'auto'
    if (self.mode == 'min' or
        (self.mode == 'auto' and 'acc' not in self.monitor)):
      self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
      self.best = np.Inf
    else:
      self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
      self.best = -np.Inf
    self.cooldown_counter = 0
    self.wait = 0

  def on_train_begin(self, logs=None):
    self._reset()

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = K.get_value(self.model.optimizer.lr)
    current = logs.get(self.monitor)
    if current is None:
      logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))

    else:
      if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0

      if self.monitor_op(current, self.best):
        self.best = current
        self.wait = 0
      elif not self.in_cooldown():
        self.wait += 1
        if self.wait >= self.patience:
          old_lr = float(K.get_value(self.model.optimizer.lr))
          if old_lr > self.min_lr:
            new_lr = old_lr * self.factor
            new_lr = max(new_lr, self.min_lr)
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
              print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                    'rate to %s.' % (epoch + 1, new_lr))
            self.cooldown_counter = self.cooldown
            self.wait = 0

  def in_cooldown(self):
    return self.cooldown_counter > 0


@tf_export('keras.callbacks.CSVLogger')
class CSVLogger(Callback):
  """Callback that streams epoch results to a csv file.

  Supports all values that can be represented as a string,
  including 1D iterables such as np.ndarray.

  Example:

  ```python
  csv_logger = CSVLogger('training.log')
  model.fit(X_train, Y_train, callbacks=[csv_logger])
  ```

  Arguments:
      filename: filename of the csv file, e.g. 'run/log.csv'.
      separator: string used to separate elements in the csv file.
      append: True: append if file exists (useful for continuing
          training). False: overwrite existing file,
  """

  def __init__(self, filename, separator=',', append=False):
    self.sep = separator
    self.filename = filename
    self.append = append
    self.writer = None
    self.keys = None
    self.append_header = True
    if six.PY2:
      self.file_flags = 'b'
      self._open_args = {}
    else:
      self.file_flags = ''
      self._open_args = {'newline': '\n'}
    super(CSVLogger, self).__init__()

  def on_train_begin(self, logs=None):
    if self.append:
      if os.path.exists(self.filename):
        with open(self.filename, 'r' + self.file_flags) as f:
          self.append_header = not bool(len(f.readline()))
      mode = 'a'
    else:
      mode = 'w'
    self.csv_file = io.open(self.filename,
                            mode + self.file_flags,
                            **self._open_args)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}

    def handle_value(k):
      is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
      if isinstance(k, six.string_types):
        return k
      elif isinstance(k, collections.Iterable) and not is_zero_dim_ndarray:
        return '"[%s]"' % (', '.join(map(str, k)))
      else:
        return k

    if self.keys is None:
      self.keys = sorted(logs.keys())

    if self.model.stop_training:
      # We set NA so that csv parsers do not fail for this last epoch.
      logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

    if not self.writer:

      class CustomDialect(csv.excel):
        delimiter = self.sep

      fieldnames = ['epoch'] + self.keys
      if six.PY2:
        fieldnames = [unicode(x) for x in fieldnames]

      self.writer = csv.DictWriter(
          self.csv_file,
          fieldnames=fieldnames,
          dialect=CustomDialect)
      if self.append_header:
        self.writer.writeheader()

    row_dict = collections.OrderedDict({'epoch': epoch})
    row_dict.update((key, handle_value(logs[key])) for key in self.keys)
    self.writer.writerow(row_dict)
    self.csv_file.flush()

  def on_train_end(self, logs=None):
    self.csv_file.close()
    self.writer = None


@tf_export('keras.callbacks.LambdaCallback')
class LambdaCallback(Callback):
  r"""Callback for creating simple, custom callbacks on-the-fly.

  This callback is constructed with anonymous functions that will be called
  at the appropriate time. Note that the callbacks expects positional
  arguments, as:

   - `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
      `epoch`, `logs`
   - `on_batch_begin` and `on_batch_end` expect two positional arguments:
      `batch`, `logs`
   - `on_train_begin` and `on_train_end` expect one positional argument:
      `logs`

  Arguments:
      on_epoch_begin: called at the beginning of every epoch.
      on_epoch_end: called at the end of every epoch.
      on_batch_begin: called at the beginning of every batch.
      on_batch_end: called at the end of every batch.
      on_train_begin: called at the beginning of model training.
      on_train_end: called at the end of model training.

  Example:

  ```python
  # Print the batch number at the beginning of every batch.
  batch_print_callback = LambdaCallback(
      on_batch_begin=lambda batch,logs: print(batch))

  # Stream the epoch loss to a file in JSON format. The file content
  # is not well-formed JSON but rather has a JSON object per line.
  import json
  json_log = open('loss_log.json', mode='wt', buffering=1)
  json_logging_callback = LambdaCallback(
      on_epoch_end=lambda epoch, logs: json_log.write(
          json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
      on_train_end=lambda logs: json_log.close()
  )

  # Terminate some processes after having finished model training.
  processes = ...
  cleanup_callback = LambdaCallback(
      on_train_end=lambda logs: [
          p.terminate() for p in processes if p.is_alive()])

  model.fit(...,
            callbacks=[batch_print_callback,
                       json_logging_callback,
                       cleanup_callback])
  ```
  """

  def __init__(self,
               on_epoch_begin=None,
               on_epoch_end=None,
               on_batch_begin=None,
               on_batch_end=None,
               on_train_begin=None,
               on_train_end=None,
               **kwargs):
    super(LambdaCallback, self).__init__()
    self.__dict__.update(kwargs)
    if on_epoch_begin is not None:
      self.on_epoch_begin = on_epoch_begin
    else:
      self.on_epoch_begin = lambda epoch, logs: None
    if on_epoch_end is not None:
      self.on_epoch_end = on_epoch_end
    else:
      self.on_epoch_end = lambda epoch, logs: None
    if on_batch_begin is not None:
      self.on_batch_begin = on_batch_begin
    else:
      self.on_batch_begin = lambda batch, logs: None
    if on_batch_end is not None:
      self.on_batch_end = on_batch_end
    else:
      self.on_batch_end = lambda batch, logs: None
    if on_train_begin is not None:
      self.on_train_begin = on_train_begin
    else:
      self.on_train_begin = lambda logs: None
    if on_train_end is not None:
      self.on_train_end = on_train_end
    else:
      self.on_train_end = lambda logs: None
