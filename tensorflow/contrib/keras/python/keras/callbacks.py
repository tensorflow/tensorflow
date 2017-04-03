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
"""Keras callbacks: utilities called at certain points during model training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
from collections import Iterable
from collections import OrderedDict
import csv
import json
import os
import time
import warnings

import numpy as np

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.utils.generic_utils import Progbar
from tensorflow.python.ops import array_ops
from tensorflow.python.summary import summary as tf_summary


# pylint: disable=g-import-not-at-top
try:
  import requests
except ImportError:
  requests = None
# pylint: enable=g-import-not-at-top


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

  def append(self, callback):
    self.callbacks.append(callback)

  def set_params(self, params):
    for callback in self.callbacks:
      callback.set_params(params)

  def set_model(self, model):
    for callback in self.callbacks:
      callback.set_model(model)

  def on_epoch_begin(self, epoch, logs=None):
    """Called at the start of an epoch.

    Arguments:
        epoch: integer, index of epoch.
        logs: dictionary of logs.
    """
    logs = logs or {}
    for callback in self.callbacks:
      callback.on_epoch_begin(epoch, logs)
    self._delta_t_batch = 0.
    self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
    self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

  def on_epoch_end(self, epoch, logs=None):
    """Called at the end of an epoch.

    Arguments:
        epoch: integer, index of epoch.
        logs: dictionary of logs.
    """
    logs = logs or {}
    for callback in self.callbacks:
      callback.on_epoch_end(epoch, logs)

  def on_batch_begin(self, batch, logs=None):
    """Called right before processing a batch.

    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dictionary of logs.
    """
    logs = logs or {}
    t_before_callbacks = time.time()
    for callback in self.callbacks:
      callback.on_batch_begin(batch, logs)
    self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
    delta_t_median = np.median(self._delta_ts_batch_begin)
    if (self._delta_t_batch > 0. and
        delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
      warnings.warn('Method on_batch_begin() is slow compared '
                    'to the batch update (%f). Check your callbacks.' %
                    delta_t_median)
    self._t_enter_batch = time.time()

  def on_batch_end(self, batch, logs=None):
    """Called at the end of a batch.

    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dictionary of logs.
    """
    logs = logs or {}
    if not hasattr(self, '_t_enter_batch'):
      self._t_enter_batch = time.time()
    self._delta_t_batch = time.time() - self._t_enter_batch
    t_before_callbacks = time.time()
    for callback in self.callbacks:
      callback.on_batch_end(batch, logs)
    self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
    delta_t_median = np.median(self._delta_ts_batch_end)
    if (self._delta_t_batch > 0. and
        (delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1)):
      warnings.warn('Method on_batch_end() is slow compared '
                    'to the batch update (%f). Check your callbacks.' %
                    delta_t_median)

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


class Callback(object):
  """Abstract base class used to build new callbacks.

  # Properties
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

  def on_train_begin(self, logs=None):
    pass

  def on_train_end(self, logs=None):
    pass


class BaseLogger(Callback):
  """Callback that accumulates epoch averages of metrics.

  This callback is automatically applied to every Keras model.
  """

  def on_epoch_begin(self, epoch, logs=None):
    self.seen = 0
    self.totals = {}

  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    batch_size = logs.get('size', 0)
    self.seen += batch_size

    for k, v in logs.items():
      if k in self.totals:
        self.totals[k] += v * batch_size
      else:
        self.totals[k] = v * batch_size

  def on_epoch_end(self, epoch, logs=None):
    if logs is not None:
      for k in self.params['metrics']:
        if k in self.totals:
          # Make value available to next callbacks.
          logs[k] = self.totals[k] / self.seen


class ProgbarLogger(Callback):
  """Callback that prints metrics to stdout.

  Arguments:
      count_mode: One of "steps" or "samples".
          Whether the progress bar should
          count samples seens or steps (batches) seen.

  Raises:
      ValueError: In case of invalid `count_mode`.
  """

  def __init__(self, count_mode='samples'):
    super(ProgbarLogger, self).__init__()
    if count_mode == 'samples':
      self.use_steps = False
    elif count_mode == 'steps':
      self.use_steps = True
    else:
      raise ValueError('Unknown `count_mode`: ' + str(count_mode))

  def on_train_begin(self, logs=None):
    self.verbose = self.params['verbose']
    self.epochs = self.params['epochs']

  def on_epoch_begin(self, epoch, logs=None):
    if self.verbose:
      print('Epoch %d/%d' % (epoch + 1, self.epochs))
      if self.use_steps:
        target = self.params['steps']
      else:
        target = self.params['samples']
      self.target = target
      self.progbar = Progbar(target=self.target, verbose=self.verbose)
    self.seen = 0

  def on_batch_begin(self, batch, logs=None):
    if self.seen < self.target:
      self.log_values = []

  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    batch_size = logs.get('size', 0)
    if self.use_steps:
      self.seen += 1
    else:
      self.seen += batch_size

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
      self.progbar.update(self.seen, self.log_values, force=True)


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
      warnings.warn('ModelCheckpoint mode %s is unknown, '
                    'fallback to auto mode.' % (mode), RuntimeWarning)
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
      filepath = self.filepath.format(epoch=epoch, **logs)
      if self.save_best_only:
        current = logs.get(self.monitor)
        if current is None:
          warnings.warn('Can save best model only with %s available, '
                        'skipping.' % (self.monitor), RuntimeWarning)
        else:
          if self.monitor_op(current, self.best):
            if self.verbose > 0:
              print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                    ' saving model to %s' % (epoch, self.monitor, self.best,
                                             current, filepath))
            self.best = current
            if self.save_weights_only:
              self.model.save_weights(filepath, overwrite=True)
            else:
              self.model.save(filepath, overwrite=True)
          else:
            if self.verbose > 0:
              print('Epoch %05d: %s did not improve' % (epoch, self.monitor))
      else:
        if self.verbose > 0:
          print('Epoch %05d: saving model to %s' % (epoch, filepath))
        if self.save_weights_only:
          self.model.save_weights(filepath, overwrite=True)
        else:
          self.model.save(filepath, overwrite=True)


class EarlyStopping(Callback):
  """Stop training when a monitored quantity has stopped improving.

  Arguments:
      monitor: quantity to be monitored.
      min_delta: minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: number of epochs with no improvement
          after which training will be stopped.
      verbose: verbosity mode.
      mode: one of {auto, min, max}. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `max`
          mode it will stop when the quantity
          monitored has stopped increasing; in `auto`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
  """

  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto'):
    super(EarlyStopping, self).__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.min_delta = min_delta
    self.wait = 0
    self.stopped_epoch = 0

    if mode not in ['auto', 'min', 'max']:
      warnings.warn('EarlyStopping mode %s is unknown, '
                    'fallback to auto mode.' % (self.mode), RuntimeWarning)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    self.wait = 0  # Allow instances to be re-used
    self.best = np.Inf if self.monitor_op == np.less else -np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get(self.monitor)
    if current is None:
      warnings.warn('Early stopping requires %s available!' % (self.monitor),
                    RuntimeWarning)

    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
    else:
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
      self.wait += 1

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch))


class RemoteMonitor(Callback):
  """Callback used to stream events to a server.

  Requires the `requests` library.
  Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
  HTTP POST, with a `data` argument which is a
  JSON-encoded dictionary of event data.

  Arguments:
      root: String; root url of the target server.
      path: String; path relative to `root` to which the events will be sent.
      field: String; JSON field under which the data will be stored.
      headers: Dictionary; optional custom HTTP headers.
          Defaults to:
          `{'Accept': 'application/json',
            'Content-Type': 'application/json'}`
  """

  def __init__(self,
               root='http://localhost:9000',
               path='/publish/epoch/end/',
               field='data',
               headers=None):
    super(RemoteMonitor, self).__init__()
    if headers is None:
      headers = {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
      }
    self.root = root
    self.path = path
    self.field = field
    self.headers = headers

  def on_epoch_end(self, epoch, logs=None):
    if requests is None:
      raise ImportError('RemoteMonitor requires ' 'the `requests` library.')
    logs = logs or {}
    send = {}
    send['epoch'] = epoch
    for k, v in logs.items():
      send[k] = v
    try:
      requests.post(
          self.root + self.path, {self.field: json.dumps(send)},
          headers=self.headers)
    except requests.exceptions.RequestException:
      warnings.warn('Warning: could not reach RemoteMonitor '
                    'root server at ' + str(self.root))


class LearningRateScheduler(Callback):
  """Learning rate scheduler.

  Arguments:
      schedule: a function that takes an epoch index as input
          (integer, indexed from 0) and returns a new
          learning rate as output (float).
  """

  def __init__(self, schedule):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    lr = self.schedule(epoch)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    K.set_value(self.model.optimizer.lr, lr)


class TensorBoard(Callback):
  """Tensorboard basic visualizations.

  This callback writes a log for TensorBoard, which allows
  you to visualize dynamic graphs of your training and test
  metrics, as well as activation histograms for the different
  layers in your model.

  Arguments:
      log_dir: the path of the directory where to save the log
          files to be parsed by Tensorboard.
      histogram_freq: frequency (in epochs) at which to compute activation
          histograms for the layers of the model. If set to 0,
          histograms won't be computed.
      write_graph: whether to visualize the graph in Tensorboard.
          The log file can become quite large when
          write_graph is set to True.
      write_images: whether to write model weights to visualize as
          image in Tensorboard.
  """

  def __init__(self,
               log_dir='./logs',
               histogram_freq=0,
               write_graph=True,
               write_images=False):
    super(TensorBoard, self).__init__()
    self.log_dir = log_dir
    self.histogram_freq = histogram_freq
    self.merged = None
    self.write_graph = write_graph
    self.write_images = write_images

  def set_model(self, model):
    self.model = model
    self.sess = K.get_session()
    if self.histogram_freq and self.merged is None:
      for layer in self.model.layers:

        for weight in layer.weights:
          tf_summary.histogram(weight.name, weight)
          if self.write_images:
            w_img = array_ops.squeeze(weight)
            shape = w_img.get_shape()
            if len(shape) > 1 and shape[0] > shape[1]:
              w_img = array_ops.transpose(w_img)
            if len(shape) == 1:
              w_img = array_ops.expand_dims(w_img, 0)
            w_img = array_ops.expand_dims(array_ops.expand_dims(w_img, 0), -1)
            tf_summary.image(weight.name, w_img)

        if hasattr(layer, 'output'):
          tf_summary.histogram('{}_out'.format(layer.name), layer.output)
    self.merged = tf_summary.merge_all()

    if self.write_graph:
      self.writer = tf_summary.FileWriter(self.log_dir, self.sess.graph)
    else:
      self.writer = tf_summary.FileWriter(self.log_dir)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}

    if self.validation_data and self.histogram_freq:
      if epoch % self.histogram_freq == 0:
        # TODO(fchollet): implement batched calls to sess.run
        # (current call will likely go OOM on GPU)
        if self.model.uses_learning_phase:
          cut_v_data = len(self.model.inputs)
          val_data = self.validation_data[:cut_v_data] + [0]
          tensors = self.model.inputs + [K.learning_phase()]
        else:
          val_data = self.validation_data
          tensors = self.model.inputs
        feed_dict = dict(zip(tensors, val_data))
        result = self.sess.run([self.merged], feed_dict=feed_dict)
        summary_str = result[0]
        self.writer.add_summary(summary_str, epoch)

    for name, value in logs.items():
      if name in ['batch', 'size']:
        continue
      summary = tf_summary.Summary()
      summary_value = summary.value.add()
      summary_value.simple_value = value.item()
      summary_value.tag = name
      self.writer.add_summary(summary, epoch)
    self.writer.flush()

  def on_train_end(self, _):
    self.writer.close()


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
      epsilon: threshold for measuring the new optimum,
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
               epsilon=1e-4,
               cooldown=0,
               min_lr=0):
    super(ReduceLROnPlateau, self).__init__()

    self.monitor = monitor
    if factor >= 1.0:
      raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
    self.factor = factor
    self.min_lr = min_lr
    self.epsilon = epsilon
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
      warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                    'fallback to auto mode.' % (self.mode), RuntimeWarning)
      self.mode = 'auto'
    if (self.mode == 'min' or
        (self.mode == 'auto' and 'acc' not in self.monitor)):
      self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
      self.best = np.Inf
    else:
      self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
      self.best = -np.Inf
    self.cooldown_counter = 0
    self.wait = 0
    self.lr_epsilon = self.min_lr * 1e-4

  def on_train_begin(self, logs=None):
    self._reset()

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = K.get_value(self.model.optimizer.lr)
    current = logs.get(self.monitor)
    if current is None:
      warnings.warn('Learning Rate Plateau Reducing requires %s available!' %
                    self.monitor, RuntimeWarning)
    else:
      if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0

      if self.monitor_op(current, self.best):
        self.best = current
        self.wait = 0
      elif not self.in_cooldown():
        if self.wait >= self.patience:
          old_lr = float(K.get_value(self.model.optimizer.lr))
          if old_lr > self.min_lr + self.lr_epsilon:
            new_lr = old_lr * self.factor
            new_lr = max(new_lr, self.min_lr)
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
              print('\nEpoch %05d: reducing learning rate to %s.' % (epoch,
                                                                     new_lr))
            self.cooldown_counter = self.cooldown
            self.wait = 0
        self.wait += 1

  def in_cooldown(self):
    return self.cooldown_counter > 0


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
    super(CSVLogger, self).__init__()

  def on_train_begin(self, logs=None):
    if self.append:
      if os.path.exists(self.filename):
        with open(self.filename) as f:
          self.append_header = not bool(len(f.readline()))
      self.csv_file = open(self.filename, 'a')
    else:
      self.csv_file = open(self.filename, 'w')

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}

    def handle_value(k):
      is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
      if isinstance(k, Iterable) and not is_zero_dim_ndarray:
        return '"[%s]"' % (', '.join(map(str, k)))
      else:
        return k

    if not self.writer:
      self.keys = sorted(logs.keys())

      class CustomDialect(csv.excel):
        delimiter = self.sep

      self.writer = csv.DictWriter(
          self.csv_file,
          fieldnames=['epoch'] + self.keys,
          dialect=CustomDialect)
      if self.append_header:
        self.writer.writeheader()

    row_dict = OrderedDict({'epoch': epoch})
    row_dict.update((key, handle_value(logs[key])) for key in self.keys)
    self.writer.writerow(row_dict)
    self.csv_file.flush()

  def on_train_end(self, logs=None):
    self.csv_file.close()
    self.writer = None


class LambdaCallback(Callback):
  """Callback for creating simple, custom callbacks on-the-fly.

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

      # Plot the loss after every epoch.
      import numpy as np
      import matplotlib.pyplot as plt
      plot_loss_callback = LambdaCallback(
          on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch),
                                                    logs['loss']))

      # Terminate some processes after having finished model training.
      processes = ...
      cleanup_callback = LambdaCallback(
          on_train_end=lambda logs: [
              p.terminate() for p in processes if p.is_alive()])

      model.fit(...,
                callbacks=[batch_print_callback,
                           plot_loss_callback,
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
