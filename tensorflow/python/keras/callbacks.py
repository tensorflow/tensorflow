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
# pylint: disable=g-classes-have-attributes
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
import re
import time

import numpy as np
import six

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.distribute import worker_training_state
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.training import checkpoint_management
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

try:
  import requests
except ImportError:
  requests = None


def configure_callbacks(callbacks,
                        model,
                        do_validation=False,
                        batch_size=None,
                        epochs=None,
                        steps_per_epoch=None,
                        samples=None,
                        verbose=1,
                        count_mode='steps',
                        mode=ModeKeys.TRAIN):
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
      mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
        Which loop mode to configure callbacks for.

  Returns:
      Instance of CallbackList used to control all Callbacks.
  """
  # Check if callbacks have already been configured.
  if isinstance(callbacks, CallbackList):
    return callbacks

  if not callbacks:
    callbacks = []

  # Add additional callbacks during training.
  if mode == ModeKeys.TRAIN:
    model.history = History()
    callbacks = [BaseLogger()] + (callbacks or []) + [model.history]
    if verbose:
      callbacks.append(ProgbarLogger(count_mode))
  callback_list = CallbackList(callbacks)

  # Set callback model
  callback_model = model._get_callback_model()  # pylint: disable=protected-access
  callback_list.set_model(callback_model)

  set_callback_parameters(
      callback_list,
      model,
      do_validation=do_validation,
      batch_size=batch_size,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      samples=samples,
      verbose=verbose,
      mode=mode)

  callback_list.model.stop_training = False
  return callback_list


def set_callback_parameters(callback_list,
                            model,
                            do_validation=False,
                            batch_size=None,
                            epochs=None,
                            steps_per_epoch=None,
                            samples=None,
                            verbose=1,
                            mode=ModeKeys.TRAIN):
  """Sets callback parameters.

  Arguments:
      callback_list: CallbackList instance.
      model: Model being trained.
      do_validation: Whether or not validation loop will be run.
      batch_size: Number of samples per batch.
      epochs: Number of epoch to train.
      steps_per_epoch: Number of batches to run per training epoch.
      samples: Number of training samples.
      verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
      mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
        Which loop mode to configure callbacks for.
  """
  metric_names = model.metrics_names
  for cbk in callback_list:
    if isinstance(cbk, (BaseLogger, ProgbarLogger)):
      cbk.stateful_metrics = metric_names[1:]  # Exclude `loss`

  # Set callback parameters
  callback_metrics = []
  # When we have deferred build scenario with iterator input, we will compile
  # when we standardize first batch of data.
  if mode != ModeKeys.PREDICT:
    callback_metrics = copy.copy(metric_names)
    if do_validation:
      callback_metrics += ['val_' + n for n in metric_names]
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


def _is_generator_like(data):
  """Checks if data is a generator, Sequence, or Iterator."""
  return (hasattr(data, 'next') or hasattr(data, '__next__') or isinstance(
      data, (Sequence, iterator_ops.Iterator, iterator_ops.OwnedIterator)))


def make_logs(model, logs, outputs, mode, prefix=''):
  """Computes logs for sending to `on_batch_end` methods."""
  metric_names = model.metrics_names
  if mode in {ModeKeys.TRAIN, ModeKeys.TEST} and metric_names:
    for label, output in zip(metric_names, outputs):
      logs[prefix + label] = output
  else:
    logs['outputs'] = outputs
  return logs


@keras_export('keras.callbacks.CallbackList')
class CallbackList(object):
  """Container abstracting a list of callbacks."""

  def __init__(self,
               callbacks=None,
               add_history=False,
               add_progbar=False,
               model=None,
               **params):
    """Container for `Callback` instances.

    This object wraps a list of `Callback` instances, making it possible
    to call them all at once via a single endpoint
    (e.g. `callback_list.on_epoch_end(...)`).

    Arguments:
      callbacks: List of `Callback` instances.
      add_history: Whether a `History` callback should be added, if one does not
        already exist in the `callbacks` list.
      add_progbar: Whether a `ProgbarLogger` callback should be added, if one
        does not already exist in the `callbacks` list.
      model: The `Model` these callbacks are used with.
      **params: If provided, parameters will be passed to each `Callback` via
        `Callback.set_params`.
    """
    self.callbacks = nest.flatten(callbacks) if callbacks else []
    self._add_default_callbacks(add_history, add_progbar)

    if model:
      self.set_model(model)
    if params:
      self.set_params(params)

    # Performance optimization: determines if batch hooks need to be called.
    # pylint: disable=protected-access
    self._should_call_train_batch_hooks = any(
        cb._implements_train_batch_hooks() for cb in self.callbacks)
    self._should_call_test_batch_hooks = any(
        cb._implements_test_batch_hooks() for cb in self.callbacks)
    self._should_call_predict_batch_hooks = any(
        cb._implements_predict_batch_hooks() for cb in self.callbacks)
    # pylint: enable=protected-access

    # Performance check: Check batch hooks for slowness compared to batch time.
    self._timing = {}
    self._check_timing = False
    self._batch_start_time = None

  def _add_default_callbacks(self, add_history, add_progbar):
    """Adds `Callback`s that are always present."""
    self._progbar = None
    self._history = None

    for cb in self.callbacks:
      if isinstance(cb, ProgbarLogger):
        self._progbar = cb
      elif isinstance(cb, History):
        self._history = cb

    if self._progbar is None and add_progbar:
      self._progbar = ProgbarLogger(count_mode='steps')
      self.callbacks.append(self._progbar)

    if self._history is None and add_history:
      self._history = History()
      self.callbacks.append(self._history)

  def append(self, callback):
    self.callbacks.append(callback)

  def set_params(self, params):
    self.params = params
    for callback in self.callbacks:
      callback.set_params(params)

  def set_model(self, model):
    self.model = model
    if self._history:
      model.history = self._history
    for callback in self.callbacks:
      callback.set_model(model)

  def _call_batch_hook(self, mode, hook, batch, logs=None):
    """Helper function for all batch_{begin | end} methods."""
    if not self.callbacks:
      return

    if hook == 'begin':
      self._call_batch_begin_hook(mode, batch, logs)
    elif hook == 'end':
      self._call_batch_end_hook(mode, batch, logs)
    else:
      raise ValueError('Unrecognized hook: {}'.format(hook))

  def _call_batch_begin_hook(self, mode, batch, logs):
    """Helper function for `on_*_batch_begin` methods."""
    hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
    self._check_timing = batch == 1 and hook_name not in self._timing
    self._call_batch_hook_helper(hook_name, batch, logs)

    if self._check_timing:
      self._batch_start_time = time.time()

  def _call_batch_end_hook(self, mode, batch, logs):
    """Helper function for `on_*_batch_end` methods."""
    hook_name = 'on_{mode}_batch_end'.format(mode=mode)

    if self._check_timing:
      batch_time = time.time() - self._batch_start_time

    self._call_batch_hook_helper(hook_name, batch, logs)

    if self._check_timing:
      end_hook_name = hook_name
      begin_hook_name = 'on_{mode}_batch_begin'.format(mode=mode)

      threshold_time = 1.5 * batch_time
      warning_msg = ('Callbacks method `{hook}` is slow compared to '
                     'the batch time (batch time: {batch_time:.4f}s vs '
                     '`{hook}` time: {cbk_time:.4f}s). Check your callbacks.')
      if self._timing[begin_hook_name] > threshold_time:
        logging.warning(warning_msg.format(
            hook=begin_hook_name,
            batch_time=batch_time,
            cbk_time=self._timing[begin_hook_name]))
      if self._timing[end_hook_name] > threshold_time:
        logging.warning(warning_msg.format(
            hook=end_hook_name,
            batch_time=batch_time,
            cbk_time=self._timing[end_hook_name]))
      self._check_timing = False
      self._batch_start_time = None

  def _call_batch_hook_helper(self, hook_name, batch, logs):
    """Helper function for `on_*_batch_*` methods."""
    logs = logs or {}
    numpy_logs = None
    if self._check_timing:
      start_time = time.time()

    for callback in self.callbacks:
      hook = getattr(callback, hook_name)
      if getattr(callback, '_supports_tf_logs', False):
        hook(batch, logs)
      else:
        if numpy_logs is None:  # Only convert once.
          numpy_logs = tf_utils.to_numpy_or_python_type(logs)
        hook(batch, numpy_logs)

    if self._check_timing:
      self._timing[hook_name] = time.time() - start_time

  def _call_begin_hook(self, mode):
    """Helper function for on_{train|test|predict}_begin methods."""
    if mode == ModeKeys.TRAIN:
      self.on_train_begin()
    elif mode == ModeKeys.TEST:
      self.on_test_begin()
    else:
      self.on_predict_begin()

  def _call_end_hook(self, mode):
    """Helper function for on_{train|test|predict}_end methods."""
    if mode == ModeKeys.TRAIN:
      self.on_train_end()
    elif mode == ModeKeys.TEST:
      self.on_test_end()
    else:
      self.on_predict_end()

  def on_batch_begin(self, batch, logs=None):
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

  def on_batch_end(self, batch, logs=None):
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None):
    """Calls the `on_epoch_begin` methods of its callbacks.

    This function should only be called during TRAIN mode.

    Arguments:
        epoch: Integer, index of epoch.
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = logs or {}
    numpy_logs = None
    for callback in self.callbacks:
      if getattr(callback, '_supports_tf_logs', False):
        callback.on_epoch_begin(epoch, logs)
      else:
        if numpy_logs is None:  # Only convert once.
          numpy_logs = tf_utils.to_numpy_or_python_type(logs)
        callback.on_epoch_begin(epoch, numpy_logs)

  def on_epoch_end(self, epoch, logs=None):
    """Calls the `on_epoch_end` methods of its callbacks.

    This function should only be called during TRAIN mode.

    Arguments:
        epoch: Integer, index of epoch.
        logs: Dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`.
    """
    logs = logs or {}
    numpy_logs = None
    for callback in self.callbacks:
      if getattr(callback, '_supports_tf_logs', False):
        callback.on_epoch_end(epoch, logs)
      else:
        if numpy_logs is None:  # Only convert once.
          numpy_logs = tf_utils.to_numpy_or_python_type(logs)
        callback.on_epoch_end(epoch, numpy_logs)

  def on_train_batch_begin(self, batch, logs=None):
    """Calls the `on_train_batch_begin` methods of its callbacks.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    """
    # TODO(b/150629188): Make ProgBarLogger callback not use batch hooks
    # when verbose != 1
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

  def on_train_batch_end(self, batch, logs=None):
    """Calls the `on_train_batch_end` methods of its callbacks.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

  def on_test_batch_begin(self, batch, logs=None):
    """Calls the `on_test_batch_begin` methods of its callbacks.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    """
    if self._should_call_test_batch_hooks:
      self._call_batch_hook(ModeKeys.TEST, 'begin', batch, logs=logs)

  def on_test_batch_end(self, batch, logs=None):
    """Calls the `on_test_batch_end` methods of its callbacks.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    if self._should_call_test_batch_hooks:
      self._call_batch_hook(ModeKeys.TEST, 'end', batch, logs=logs)

  def on_predict_batch_begin(self, batch, logs=None):
    """Calls the `on_predict_batch_begin` methods of its callbacks.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    """
    if self._should_call_predict_batch_hooks:
      self._call_batch_hook(ModeKeys.PREDICT, 'begin', batch, logs=logs)

  def on_predict_batch_end(self, batch, logs=None):
    """Calls the `on_predict_batch_end` methods of its callbacks.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    if self._should_call_predict_batch_hooks:
      self._call_batch_hook(ModeKeys.PREDICT, 'end', batch, logs=logs)

  def on_train_begin(self, logs=None):
    """Calls the `on_train_begin` methods of its callbacks.

    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = logs or {}
    numpy_logs = None
    for callback in self.callbacks:
      if getattr(callback, '_supports_tf_logs', False):
        callback.on_train_begin(logs)
      else:
        if numpy_logs is None:  # Only convert once.
          numpy_logs = tf_utils.to_numpy_or_python_type(logs)
        callback.on_train_begin(numpy_logs)

  def on_train_end(self, logs=None):
    """Calls the `on_train_end` methods of its callbacks.

    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = logs or {}
    numpy_logs = None
    for callback in self.callbacks:
      if getattr(callback, '_supports_tf_logs', False):
        callback.on_train_end(logs)
      else:
        if numpy_logs is None:  # Only convert once.
          numpy_logs = tf_utils.to_numpy_or_python_type(logs)
        callback.on_train_end(numpy_logs)

  def on_test_begin(self, logs=None):
    """Calls the `on_test_begin` methods of its callbacks.

    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = logs or {}
    numpy_logs = None
    for callback in self.callbacks:
      if getattr(callback, '_supports_tf_logs', False):
        callback.on_test_begin(logs)
      else:
        if numpy_logs is None:  # Only convert once.
          numpy_logs = tf_utils.to_numpy_or_python_type(logs)
        callback.on_test_begin(numpy_logs)

  def on_test_end(self, logs=None):
    """Calls the `on_test_end` methods of its callbacks.

    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = logs or {}
    numpy_logs = None
    for callback in self.callbacks:
      if getattr(callback, '_supports_tf_logs', False):
        callback.on_test_end(logs)
      else:
        if numpy_logs is None:  # Only convert once.
          numpy_logs = tf_utils.to_numpy_or_python_type(logs)
        callback.on_test_end(numpy_logs)

  def on_predict_begin(self, logs=None):
    """Calls the 'on_predict_begin` methods of its callbacks.

    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = logs or {}
    numpy_logs = None
    for callback in self.callbacks:
      if getattr(callback, '_supports_tf_logs', False):
        callback.on_predict_begin(logs)
      else:
        if numpy_logs is None:  # Only convert once.
          numpy_logs = tf_utils.to_numpy_or_python_type(logs)
        callback.on_predict_begin(numpy_logs)

  def on_predict_end(self, logs=None):
    """Calls the `on_predict_end` methods of its callbacks.

    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = logs or {}
    numpy_logs = None
    for callback in self.callbacks:
      if getattr(callback, '_supports_tf_logs', False):
        callback.on_predict_end(logs)
      else:
        if numpy_logs is None:  # Only convert once.
          numpy_logs = tf_utils.to_numpy_or_python_type(logs)
        callback.on_predict_end(numpy_logs)

  def __iter__(self):
    return iter(self.callbacks)


@keras_export('keras.callbacks.Callback')
class Callback(object):
  """Abstract base class used to build new callbacks.

  Attributes:
      params: Dict. Training parameters
          (eg. verbosity, batch size, number of epochs...).
      model: Instance of `keras.models.Model`.
          Reference of the model being trained.

  The `logs` dictionary that callback methods
  take as argument will contain keys for quantities relevant to
  the current batch or epoch (see method-specific docstrings).
  """

  def __init__(self):
    self.validation_data = None  # pylint: disable=g-missing-from-attributes
    self.model = None
    # Whether this Callback should only run on the chief worker in a
    # Multi-Worker setting.
    # TODO(omalleyt): Make this attr public once solution is stable.
    self._chief_worker_only = None
    self._supports_tf_logs = False

  def set_params(self, params):
    self.params = params

  def set_model(self, model):
    self.model = model

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_batch_begin(self, batch, logs=None):
    """A backwards compatibility alias for `on_train_batch_begin`."""

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_batch_end(self, batch, logs=None):
    """A backwards compatibility alias for `on_train_batch_end`."""

  @doc_controls.for_subclass_implementers
  def on_epoch_begin(self, epoch, logs=None):
    """Called at the start of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Arguments:
        epoch: Integer, index of epoch.
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_epoch_end(self, epoch, logs=None):
    """Called at the end of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Arguments:
        epoch: Integer, index of epoch.
        logs: Dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_train_batch_begin(self, batch, logs=None):
    """Called at the beginning of a training batch in `fit` methods.

    Subclasses should override for any actions to run.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    """
    # For backwards compatibility.
    self.on_batch_begin(batch, logs=logs)

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_train_batch_end(self, batch, logs=None):
    """Called at the end of a training batch in `fit` methods.

    Subclasses should override for any actions to run.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    # For backwards compatibility.
    self.on_batch_end(batch, logs=logs)

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_test_batch_begin(self, batch, logs=None):
    """Called at the beginning of a batch in `evaluate` methods.

    Also called at the beginning of a validation batch in the `fit`
    methods, if validation data is provided.

    Subclasses should override for any actions to run.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_test_batch_end(self, batch, logs=None):
    """Called at the end of a batch in `evaluate` methods.

    Also called at the end of a validation batch in the `fit`
    methods, if validation data is provided.

    Subclasses should override for any actions to run.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_predict_batch_begin(self, batch, logs=None):
    """Called at the beginning of a batch in `predict` methods.

    Subclasses should override for any actions to run.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_predict_batch_end(self, batch, logs=None):
    """Called at the end of a batch in `predict` methods.

    Subclasses should override for any actions to run.

    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """

  @doc_controls.for_subclass_implementers
  def on_train_begin(self, logs=None):
    """Called at the beginning of training.

    Subclasses should override for any actions to run.

    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_train_end(self, logs=None):
    """Called at the end of training.

    Subclasses should override for any actions to run.

    Arguments:
        logs: Dict. Currently the output of the last call to `on_epoch_end()`
          is passed to this argument for this method but that may change in
          the future.
    """

  @doc_controls.for_subclass_implementers
  def on_test_begin(self, logs=None):
    """Called at the beginning of evaluation or validation.

    Subclasses should override for any actions to run.

    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_test_end(self, logs=None):
    """Called at the end of evaluation or validation.

    Subclasses should override for any actions to run.

    Arguments:
        logs: Dict. Currently the output of the last call to
          `on_test_batch_end()` is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_predict_begin(self, logs=None):
    """Called at the beginning of prediction.

    Subclasses should override for any actions to run.

    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_predict_end(self, logs=None):
    """Called at the end of prediction.

    Subclasses should override for any actions to run.

    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  def _implements_train_batch_hooks(self):
    """Determines if this Callback should be called for each train batch."""
    return (not generic_utils.is_default(self.on_batch_begin) or
            not generic_utils.is_default(self.on_batch_end) or
            not generic_utils.is_default(self.on_train_batch_begin) or
            not generic_utils.is_default(self.on_train_batch_end))

  def _implements_test_batch_hooks(self):
    """Determines if this Callback should be called for each test batch."""
    return (not generic_utils.is_default(self.on_test_batch_begin) or
            not generic_utils.is_default(self.on_test_batch_end))

  def _implements_predict_batch_hooks(self):
    """Determines if this Callback should be called for each predict batch."""
    return (not generic_utils.is_default(self.on_predict_batch_begin) or
            not generic_utils.is_default(self.on_predict_batch_end))


@keras_export('keras.callbacks.BaseLogger')
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


@keras_export('keras.callbacks.TerminateOnNaN')
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


@keras_export('keras.callbacks.ProgbarLogger')
class ProgbarLogger(Callback):
  """Callback that prints metrics to stdout.

  Arguments:
      count_mode: One of `"steps"` or `"samples"`.
          Whether the progress bar should
          count samples seen or steps (batches) seen.
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over an epoch.
          Metrics in this list will be logged as-is.
          All others will be averaged over time (e.g. loss, etc).
          If not provided, defaults to the `Model`'s metrics.

  Raises:
      ValueError: In case of invalid `count_mode`.
  """

  def __init__(self, count_mode='samples', stateful_metrics=None):
    super(ProgbarLogger, self).__init__()
    self._supports_tf_logs = True
    if count_mode == 'samples':
      self.use_steps = False
    elif count_mode == 'steps':
      self.use_steps = True
    else:
      raise ValueError('Unknown `count_mode`: ' + str(count_mode))
    # Defaults to all Model's metrics except for loss.
    self.stateful_metrics = set(stateful_metrics) if stateful_metrics else None

    self.seen = 0
    self.progbar = None
    self.target = None
    self.verbose = 1
    self.epochs = 1

    self._called_in_fit = False

  def set_params(self, params):
    self.verbose = params['verbose']
    self.epochs = params['epochs']
    if self.use_steps and 'steps' in params:
      self.target = params['steps']
    elif not self.use_steps and 'samples' in params:
      self.target = params['samples']
    else:
      self.target = None  # Will be inferred at the end of the first epoch.

  def on_train_begin(self, logs=None):
    # When this logger is called inside `fit`, validation is silent.
    self._called_in_fit = True

  def on_test_begin(self, logs=None):
    if not self._called_in_fit:
      self._reset_progbar()

  def on_predict_begin(self, logs=None):
    self._reset_progbar()

  def on_epoch_begin(self, epoch, logs=None):
    self._reset_progbar()
    if self.verbose and self.epochs > 1:
      print('Epoch %d/%d' % (epoch + 1, self.epochs))

  def on_train_batch_end(self, batch, logs=None):
    self._batch_update_progbar(batch, logs)

  def on_test_batch_end(self, batch, logs=None):
    if not self._called_in_fit:
      self._batch_update_progbar(batch, logs)

  def on_predict_batch_end(self, batch, logs=None):
    # Don't pass prediction results.
    self._batch_update_progbar(batch, None)

  def on_epoch_end(self, epoch, logs=None):
    self._finalize_progbar(logs)

  def on_test_end(self, logs=None):
    if not self._called_in_fit:
      self._finalize_progbar(logs)

  def on_predict_end(self, logs=None):
    self._finalize_progbar(logs)

  def _reset_progbar(self):
    self.seen = 0
    self.progbar = None

  def _maybe_init_progbar(self):
    if self.stateful_metrics is None:
      if self.model:
        self.stateful_metrics = (set(m.name for m in self.model.metrics))
      else:
        self.stateful_metrics = set()

    if self.progbar is None:
      self.progbar = Progbar(
          target=self.target,
          verbose=self.verbose,
          stateful_metrics=self.stateful_metrics,
          unit_name='step' if self.use_steps else 'sample')

  def _batch_update_progbar(self, batch, logs=None):
    """Updates the progbar."""
    logs = logs or {}
    self._maybe_init_progbar()
    if self.use_steps:
      self.seen = batch + 1  # One-indexed.
    else:
      # v1 path only.
      logs = copy.copy(logs)
      batch_size = logs.pop('size', 0)
      num_steps = logs.pop('num_steps', 1)
      logs.pop('batch', None)
      add_seen = num_steps * batch_size
      self.seen += add_seen

    if self.verbose == 1:
      # Only block async when verbose = 1.
      logs = tf_utils.to_numpy_or_python_type(logs)
      self.progbar.update(self.seen, list(logs.items()), finalize=False)

  def _finalize_progbar(self, logs):
    logs = logs or {}
    self._maybe_init_progbar()
    if self.target is None:
      self.target = self.seen
      self.progbar.target = self.seen
    logs = tf_utils.to_numpy_or_python_type(logs)
    self.progbar.update(self.seen, list(logs.items()), finalize=True)


@keras_export('keras.callbacks.History')
class History(Callback):
  """Callback that records events into a `History` object.

  This callback is automatically applied to
  every Keras model. The `History` object
  gets returned by the `fit` method of models.
  """

  def __init__(self):
    super(History, self).__init__()
    self.history = {}

  def on_train_begin(self, logs=None):
    self.epoch = []

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    self.epoch.append(epoch)
    for k, v in logs.items():
      self.history.setdefault(k, []).append(v)

    # Set the history attribute on the model after the epoch ends. This will
    # make sure that the state which is set is the latest one.
    self.model.history = self


@keras_export('keras.callbacks.ModelCheckpoint')
class ModelCheckpoint(Callback):
  """Callback to save the Keras model or model weights at some frequency.

  `ModelCheckpoint` callback is used in conjunction with training using
  `model.fit()` to save a model or weights (in a checkpoint file) at some
  interval, so the model or weights can be loaded later to continue the training
  from the state saved.

  A few options this callback provides include:

  - Whether to only keep the model that has achieved the "best performance" so
    far, or whether to save the model at the end of every epoch regardless of
    performance.
  - Definition of 'best'; which quantity to monitor and whether it should be
    maximized or minimized.
  - The frequency it should save at. Currently, the callback supports saving at
    the end of every epoch, or after a fixed number of training batches.
  - Whether only weights are saved, or the whole model is saved.

  Example:

  ```python
  EPOCHS = 10
  checkpoint_filepath = '/tmp/checkpoint'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_acc',
      mode='max',
      save_best_only=True)

  # Model weights are saved at the end of every epoch, if it's the best seen
  # so far.
  model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

  # The model weights (that are considered the best) are loaded into the model.
  model.load_weights(checkpoint_filepath)
  ```

  Arguments:
      filepath: string or `PathLike`, path to save the model file. `filepath`
        can contain named formatting options, which will be filled the value of
        `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
        `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
        checkpoints will be saved with the epoch number and the validation loss
        in the filename.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`, the latest best model according
        to the quantity monitored will not be overwritten.
        If `filepath` doesn't contain formatting options like `{epoch}` then
        `filepath` will be overwritten by each new better model.
      mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
        overwrite the current save file is made based on either the maximization
        or the minimization of the monitored quantity. For `val_acc`, this
        should be `max`, for `val_loss` this should be `min`, etc. In `auto`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      save_weights_only: if True, then only the model's weights will be saved
        (`model.save_weights(filepath)`), else the full model is saved
        (`model.save(filepath)`).
      save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
        the model after each epoch. When using integer, the callback saves the
        model at end of this many batches. If the `Model` is compiled with
        `experimental_steps_per_execution=N`, then the saving criteria will be
        checked every Nth batch. Note that if the saving isn't aligned to
        epochs, the monitored metric may potentially be less reliable (it
        could reflect as little as 1 batch, since the metrics get reset every
        epoch). Defaults to `'epoch'`.
      **kwargs: Additional arguments for backwards compatibility. Possible key
        is `period`.
  """

  def __init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               save_freq='epoch',
               **kwargs):
    super(ModelCheckpoint, self).__init__()
    self._supports_tf_logs = True
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = path_to_string(filepath)
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.save_freq = save_freq
    self.epochs_since_last_save = 0
    self._batches_seen_since_last_saving = 0
    self._last_batch_seen = 0

    # Deprecated field `load_weights_on_restart` is for loading the checkpoint
    # file from `filepath` at the start of `model.fit()`
    # TODO(rchao): Remove the arg during next breaking release.
    if 'load_weights_on_restart' in kwargs:
      self.load_weights_on_restart = kwargs['load_weights_on_restart']
      logging.warning('`load_weights_on_restart` argument is deprecated. '
                      'Please use `model.load_weights()` for loading weights '
                      'before the start of `model.fit()`.')
    else:
      self.load_weights_on_restart = False

    # Deprecated field `period` is for the number of epochs between which
    # the model is saved.
    if 'period' in kwargs:
      self.period = kwargs['period']
      logging.warning('`period` argument is deprecated. Please use `save_freq` '
                      'to specify the frequency in number of batches seen.')
    else:
      self.period = 1

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

    if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
      raise ValueError('Unrecognized save_freq: {}'.format(self.save_freq))

    # Only the chief worker writes model checkpoints, but all workers
    # restore checkpoint at on_train_begin().
    self._chief_worker_only = False

  def set_model(self, model):
    self.model = model
    # Use name matching rather than `isinstance` to avoid circular dependencies.
    if (not self.save_weights_only and
        not model._is_graph_network and  # pylint: disable=protected-access
        model.__class__.__name__ != 'Sequential'):
      self.save_weights_only = True

  def on_train_begin(self, logs=None):
    if self.load_weights_on_restart:
      filepath_to_load = (
          self._get_most_recently_modified_file_matching_pattern(self.filepath))
      if (filepath_to_load is not None and
          self._checkpoint_exists(filepath_to_load)):
        try:
          # `filepath` may contain placeholders such as `{epoch:02d}`, and
          # thus it attempts to load the most recently modified file with file
          # name matching the pattern.
          self.model.load_weights(filepath_to_load)
        except (IOError, ValueError) as e:
          raise ValueError('Error loading file from {}. Reason: {}'.format(
              filepath_to_load, e))

  def on_train_batch_end(self, batch, logs=None):
    if self._should_save_on_batch(batch):
      self._save_model(epoch=self._current_epoch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None):
    self._current_epoch = epoch

  def on_epoch_end(self, epoch, logs=None):
    self.epochs_since_last_save += 1
    # pylint: disable=protected-access
    if self.save_freq == 'epoch':
      self._save_model(epoch=epoch, logs=logs)

  def _should_save_on_batch(self, batch):
    """Handles batch-level saving logic, supports steps_per_execution."""
    if self.save_freq == 'epoch':
      return False

    if batch <= self._last_batch_seen:  # New epoch.
      add_batches = batch + 1  # batches are zero-indexed.
    else:
      add_batches = batch - self._last_batch_seen
    self._batches_seen_since_last_saving += add_batches
    self._last_batch_seen = batch

    if self._batches_seen_since_last_saving >= self.save_freq:
      self._batches_seen_since_last_saving = 0
      return True
    return False

  def _save_model(self, epoch, logs):
    """Saves the model.

    Arguments:
        epoch: the epoch this iteration is in.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
    logs = logs or {}

    if isinstance(self.save_freq,
                  int) or self.epochs_since_last_save >= self.period:
      # Block only when saving interval is reached.
      logs = tf_utils.to_numpy_or_python_type(logs)
      self.epochs_since_last_save = 0
      filepath = self._get_file_path(epoch, logs)

      try:
        if self.save_best_only:
          current = logs.get(self.monitor)
          if current is None:
            logging.warning('Can save best model only with %s available, '
                            'skipping.', self.monitor)
          else:
            if self.monitor_op(current, self.best):
              if self.verbose > 0:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s' % (epoch + 1, self.monitor,
                                               self.best, current, filepath))
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

        self._maybe_remove_file()
      except IOError as e:
        # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
        if 'is a directory' in six.ensure_str(e.args[0]).lower():
          raise IOError('Please specify a non-directory filepath for '
                        'ModelCheckpoint. Filepath used is an existing '
                        'directory: {}'.format(filepath))

  def _get_file_path(self, epoch, logs):
    """Returns the file path for checkpoint."""
    # pylint: disable=protected-access
    try:
      # `filepath` may contain placeholders such as `{epoch:02d}` and
      # `{mape:.2f}`. A mismatch between logged metrics and the path's
      # placeholders can cause formatting to fail.
      file_path = self.filepath.format(epoch=epoch + 1, **logs)
    except KeyError as e:
      raise KeyError('Failed to format this callback filepath: "{}". '
                     'Reason: {}'.format(self.filepath, e))
    self._write_filepath = distributed_file_utils.write_filepath(
        file_path, self.model.distribute_strategy)
    return self._write_filepath

  def _maybe_remove_file(self):
    # Remove the checkpoint directory in multi-worker training where this worker
    # should not checkpoint. It is a dummy directory previously saved for sync
    # distributed training.
    distributed_file_utils.remove_temp_dir_with_filepath(
        self._write_filepath, self.model.distribute_strategy)

  def _checkpoint_exists(self, filepath):
    """Returns whether the checkpoint `filepath` refers to exists."""
    if filepath.endswith('.h5'):
      return file_io.file_exists(filepath)
    tf_saved_model_exists = file_io.file_exists(filepath)
    tf_weights_only_checkpoint_exists = file_io.file_exists(filepath + '.index')
    return tf_saved_model_exists or tf_weights_only_checkpoint_exists

  def _get_most_recently_modified_file_matching_pattern(self, pattern):
    """Returns the most recently modified filepath matching pattern.

    Pattern may contain python formatting placeholder. If
    `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
    check for most recently modified one that matches the pattern.

    In the rare case where there are more than one pattern-matching file having
    the same modified time that is most recent among all, return the filepath
    that is largest (by `>` operator, lexicographically using the numeric
    equivalents). This provides a tie-breaker when multiple files are most
    recent. Note that a larger `filepath` can sometimes indicate a later time of
    modification (for instance, when epoch/batch is used as formatting option),
    but not necessarily (when accuracy or loss is used). The tie-breaker is
    put in the logic as best effort to return the most recent, and to avoid
    undeterministic result.

    Modified time of a file is obtained with `os.path.getmtime()`.

    This utility function is best demonstrated via an example:

    ```python
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    file_paths = [
        os.path.join(test_dir, file_name) for file_name in
        ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
    ]
    for file_path in file_paths:
      # Write something to each of the files
    self.assertEqual(
        _get_most_recently_modified_file_matching_pattern(path_pattern),
        file_paths[-1])
    ```

    Arguments:
        pattern: The file pattern that may optionally contain python placeholder
            such as `{epoch:02d}`.

    Returns:
        The most recently modified file's full filepath matching `pattern`. If
        `pattern` does not contain any placeholder, this returns the filepath
        that
        exactly matches `pattern`. Returns `None` if no match is found.
    """
    dir_name = os.path.dirname(pattern)
    base_name = os.path.basename(pattern)
    base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

    # If tf.train.latest_checkpoint tells us there exists a latest checkpoint,
    # use that as it is more robust than `os.path.getmtime()`.
    latest_tf_checkpoint = checkpoint_management.latest_checkpoint(dir_name)
    if latest_tf_checkpoint is not None and re.match(
        base_name_regex, os.path.basename(latest_tf_checkpoint)):
      return latest_tf_checkpoint

    latest_mod_time = 0
    file_path_with_latest_mod_time = None
    n_file_with_latest_mod_time = 0
    file_path_with_largest_file_name = None

    if file_io.file_exists(dir_name):
      for file_name in os.listdir(dir_name):
        # Only consider if `file_name` matches the pattern.
        if re.match(base_name_regex, file_name):
          file_path = os.path.join(dir_name, file_name)
          mod_time = os.path.getmtime(file_path)
          if (file_path_with_largest_file_name is None or
              file_path > file_path_with_largest_file_name):
            file_path_with_largest_file_name = file_path
          if mod_time > latest_mod_time:
            latest_mod_time = mod_time
            file_path_with_latest_mod_time = file_path
            # In the case a file with later modified time is found, reset
            # the counter for the number of files with latest modified time.
            n_file_with_latest_mod_time = 1
          elif mod_time == latest_mod_time:
            # In the case a file has modified time tied with the most recent,
            # increment the counter for the number of files with latest modified
            # time by 1.
            n_file_with_latest_mod_time += 1

    if n_file_with_latest_mod_time == 1:
      # Return the sole file that has most recent modified time.
      return file_path_with_latest_mod_time
    else:
      # If there are more than one file having latest modified time, return
      # the file path with the largest file name.
      return file_path_with_largest_file_name


@keras_export('keras.callbacks.experimental.BackupAndRestore', v1=[])
class BackupAndRestore(Callback):
  """Callback to back up and restore the training state.

  `BackupAndRestore` callback is intended to recover from interruptions that
  happened in the middle of a model.fit execution by backing up the
  training states in a temporary checkpoint file (based on TF CheckpointManager)
  at the end of each epoch. If training restarted before completion, the
  training state and model are restored to the most recently saved state at the
  beginning of a new model.fit() run.
  Note that user is responsible to bring jobs back up.
  This callback is important for the backup and restore mechanism for fault
  tolerance purpose. And the model to be restored from an previous checkpoint is
  expected to be the same as the one used to back up. If user changes arguments
  passed to compile or fit, the checkpoint saved for fault tolerance can become
  invalid.

  Note:
  1. This callback is not compatible with disabling eager execution.
  2. A checkpoint is saved at the end of each epoch, when restoring we'll redo
  any partial work from an unfinished epoch in which the training got restarted
  (so the work done before a interruption doesn't affect the final model state).
  3. This works for both single worker and multi-worker mode, only
  MirroredStrategy and MultiWorkerMirroredStrategy are supported for now.

  Example:

  >>> class InterruptingCallback(tf.keras.callbacks.Callback):
  ...   def on_epoch_begin(self, epoch, logs=None):
  ...     if epoch == 4:
  ...       raise RuntimeError('Interrupting!')
  >>> callback = tf.keras.callbacks.experimental.BackupAndRestore(
  ... backup_dir="/tmp")
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> try:
  ...   model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
  ...             batch_size=1, callbacks=[callback, InterruptingCallback()],
  ...             verbose=0)
  ... except:
  ...   pass
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
  ...             batch_size=1, callbacks=[callback], verbose=0)
  >>> # Only 6 more epochs are run, since first trainning got interrupted at
  >>> # zero-indexed epoch 4, second training will continue from 4 to 9.
  >>> len(history.history['loss'])
  6

  Arguments:
      backup_dir: String, path to save the model file. This is the directory in
        which the system stores temporary files to recover the model from jobs
        terminated unexpectedly. The directory cannot be reused elsewhere to
        store other checkpoints, e.g. by BackupAndRestore callback of another
        training, or by another callback (ModelCheckpoint) of the same training.
  """

  def __init__(self, backup_dir):
    super(BackupAndRestore, self).__init__()
    self.backup_dir = backup_dir
    self._supports_tf_logs = True
    self._supported_strategies = (
        distribute_lib._DefaultDistributionStrategy,
        mirrored_strategy.MirroredStrategy,
        collective_all_reduce_strategy.CollectiveAllReduceStrategy)

    if not context.executing_eagerly():
      if ops.inside_function():
        raise ValueError('This Callback\'s method contains Python state and '
                         'should be called outside of `tf.function`s.')
      else:  # Legacy graph mode:
        raise ValueError(
            'BackupAndRestore only supports eager mode. In graph '
            'mode, consider using ModelCheckpoint to manually save '
            'and restore weights with `model.load_weights()` and by '
            'providing `initial_epoch` in `model.fit()` for fault tolerance.')

    # Only the chief worker writes model checkpoints, but all workers
    # restore checkpoint at on_train_begin().
    self._chief_worker_only = False

  def set_model(self, model):
    self.model = model

  def on_train_begin(self, logs=None):
    # TrainingState is used to manage the training state needed for
    # failure-recovery of a worker in training.
    # pylint: disable=protected-access

    if not isinstance(self.model.distribute_strategy,
                      self._supported_strategies):
      raise NotImplementedError(
          'Currently only support empty strategy, MirroredStrategy and '
          'MultiWorkerMirroredStrategy.')
    self.model._training_state = (
        worker_training_state.WorkerTrainingState(self.model, self.backup_dir))
    self._training_state = self.model._training_state
    self._training_state.restore()

  def on_train_end(self, logs=None):
    # pylint: disable=protected-access
    # On exit of training, delete the training state backup file that was saved
    # for the purpose of worker recovery.
    self._training_state.delete_backup()

    # Clean up the training state.
    del self._training_state
    del self.model._training_state

  def on_epoch_end(self, epoch, logs=None):
    # Back up the model and current epoch for possible future recovery.
    self._training_state.back_up(epoch)


@keras_export('keras.callbacks.EarlyStopping')
class EarlyStopping(Callback):
  """Stop training when a monitored metric has stopped improving.

  Assuming the goal of a training is to minimize the loss. With this, the
  metric to be monitored would be `'loss'`, and mode would be `'min'`. A
  `model.fit()` training loop will check at end of every epoch whether
  the loss is no longer decreasing, considering the `min_delta` and
  `patience` if applicable. Once it's found no longer decreasing,
  `model.stop_training` is marked True and the training terminates.

  The quantity to be monitored needs to be available in `logs` dict.
  To make it so, pass the loss or metrics at `model.compile()`.

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
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
    restore_best_weights: Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used.

  Example:

  >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
  >>> # This callback will stop the training when there is no improvement in
  >>> # the validation loss for three consecutive epochs.
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=10, batch_size=1, callbacks=[callback],
  ...                     verbose=0)
  >>> len(history.history['loss'])  # Only 4 epochs are run.
  4
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
    self.best_weights = None

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


@keras_export('keras.callbacks.RemoteMonitor')
class RemoteMonitor(Callback):
  """Callback used to stream events to a server.

  Requires the `requests` library.
  Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
  HTTP POST, with a `data` argument which is a
  JSON-encoded dictionary of event data.
  If `send_as_json=True`, the content type of the request will be
  `"application/json"`.
  Otherwise the serialized JSON will be sent within a form.

  Arguments:
    root: String; root url of the target server.
    path: String; path relative to `root` to which the events will be sent.
    field: String; JSON field under which the data will be stored.
        The field is used only if the payload is sent within a form
        (i.e. send_as_json is set to False).
    headers: Dictionary; optional custom HTTP headers.
    send_as_json: Boolean; whether the request should be
        sent as `"application/json"`.
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
      # np.ndarray and np.generic are not scalar types
      # therefore we must unwrap their scalar values and
      # pass to the json-serializable dict 'send'
      if isinstance(v, (np.ndarray, np.generic)):
        send[k] = v.item()
      else:
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


@keras_export('keras.callbacks.LearningRateScheduler')
class LearningRateScheduler(Callback):
  """Learning rate scheduler.

  At the beginning of every epoch, this callback gets the updated learning rate
  value from `schedule` function provided at `__init__`, with the current epoch
  and current learning rate, and applies the updated learning rate
  on the optimizer.

  Arguments:
    schedule: a function that takes an epoch index (integer, indexed from 0)
        and current learning rate (float) as inputs and returns a new
        learning rate as output (float).
    verbose: int. 0: quiet, 1: update messages.

  Example:

  >>> # This function keeps the initial learning rate for the first ten epochs
  >>> # and decreases it exponentially after that.
  >>> def scheduler(epoch, lr):
  ...   if epoch < 10:
  ...     return lr
  ...   else:
  ...     return lr * tf.math.exp(-0.1)
  >>>
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> round(model.optimizer.lr.numpy(), 5)
  0.01

  >>> callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=15, callbacks=[callback], verbose=0)
  >>> round(model.optimizer.lr.numpy(), 5)
  0.00607

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
    if not isinstance(lr, (ops.Tensor, float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
      raise ValueError('The dtype of Tensor should be float')
    K.set_value(self.model.optimizer.lr, K.get_value(lr))
    if self.verbose > 0:
      print('\nEpoch %05d: LearningRateScheduler reducing learning '
            'rate to %s.' % (epoch + 1, lr))

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = K.get_value(self.model.optimizer.lr)


@keras_export('keras.callbacks.TensorBoard', v1=[])
class TensorBoard(Callback, version_utils.TensorBoardVersionSelector):
  # pylint: disable=line-too-long
  """Enable visualizations for TensorBoard.

  TensorBoard is a visualization tool provided with TensorFlow.

  This callback logs events for TensorBoard, including:

  * Metrics summary plots
  * Training graph visualization
  * Activation histograms
  * Sampled profiling

  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:

  ```
  tensorboard --logdir=path_to_your_logs
  ```

  You can find more information about TensorBoard
  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

  Example (Basic):

  ```python
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
  # run the tensorboard command to view the visualizations.
  ```

  Example (Profile):

  ```python
  # profile a single batch, e.g. the 5th batch.
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                                        profile_batch=5)
  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
  # Now run the tensorboard command to view the visualizations (profile plugin).

  # profile a range of batches, e.g. from 10 to 20.
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                                        profile_batch='10,20')
  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
  # Now run the tensorboard command to view the visualizations (profile plugin).
  ```

  Arguments:
      log_dir: the path of the directory where to save the log files to be
        parsed by TensorBoard.
      histogram_freq: frequency (in epochs) at which to compute activation and
        weight histograms for the layers of the model. If set to 0, histograms
        won't be computed. Validation data (or split) must be specified for
        histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard. The log file
        can become quite large when write_graph is set to True.
      write_images: whether to write model weights to visualize as image in
        TensorBoard.
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
        writes the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `1000`, the
        callback will write the metrics and losses to TensorBoard every 1000
        batches. Note that writing too frequently to TensorBoard can slow down
        your training.
      profile_batch: Profile the batch(es) to sample compute characteristics.
        profile_batch must be a non-negative integer or a tuple of integers.
        A pair of positive integers signify a range of batches to profile.
        By default, it will profile the second batch. Set profile_batch=0
        to disable profiling.
      embeddings_freq: frequency (in epochs) at which embedding layers will be
        visualized. If set to 0, embeddings won't be visualized.
      embeddings_metadata: a dictionary which maps layer name to a file name in
        which metadata for this embedding layer is saved. See the
        [details](
          https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
        about metadata files format. In case if the same metadata file is
        used for all embedding layers, string can be passed.

  Raises:
      ValueError: If histogram_freq is set and no validation data is provided.
  """

  # pylint: enable=line-too-long

  def __init__(self,
               log_dir='logs',
               histogram_freq=0,
               write_graph=True,
               write_images=False,
               update_freq='epoch',
               profile_batch=2,
               embeddings_freq=0,
               embeddings_metadata=None,
               **kwargs):
    super(TensorBoard, self).__init__()
    self._supports_tf_logs = True
    self._validate_kwargs(kwargs)

    self.log_dir = path_to_string(log_dir)
    self.histogram_freq = histogram_freq
    self.write_graph = write_graph
    self.write_images = write_images
    self.update_freq = 1 if update_freq == 'batch' else update_freq
    self.embeddings_freq = embeddings_freq
    self.embeddings_metadata = embeddings_metadata
    self._init_profile_batch(profile_batch)
    self._epoch = 0
    self._global_train_batch = 0

    # Lazily initialized in order to avoid creating event files when
    # not needed.
    self._writers = {}

    # Used to restore any existing `SummaryWriter` after training ends.
    self._prev_summary_state = []

  def _validate_kwargs(self, kwargs):
    """Handle arguments were supported in V1."""
    if kwargs.get('write_grads', False):
      logging.warning('`write_grads` will be ignored in TensorFlow 2.0 '
                      'for the `TensorBoard` Callback.')
    if kwargs.get('batch_size', False):
      logging.warning('`batch_size` is no longer needed in the '
                      '`TensorBoard` Callback and will be ignored '
                      'in TensorFlow 2.0.')
    if kwargs.get('embeddings_layer_names', False):
      logging.warning('`embeddings_layer_names` is not supported in '
                      'TensorFlow 2.0. Instead, all `Embedding` layers '
                      'will be visualized.')
    if kwargs.get('embeddings_data', False):
      logging.warning('`embeddings_data` is not supported in TensorFlow '
                      '2.0. Instead, all `Embedding` variables will be '
                      'visualized.')

    unrecognized_kwargs = set(kwargs.keys()) - {
        'write_grads', 'embeddings_layer_names', 'embeddings_data', 'batch_size'
    }

    # Only allow kwargs that were supported in V1.
    if unrecognized_kwargs:
      raise ValueError('Unrecognized arguments in `TensorBoard` '
                       'Callback: ' + str(unrecognized_kwargs))

  def set_model(self, model):
    """Sets Keras model and writes graph if specified."""
    self.model = model
    self._log_write_dir = self._get_log_write_dir()

    self._train_dir = os.path.join(self._log_write_dir, 'train')
    self._train_step = self.model._train_counter  # pylint: disable=protected-access

    self._val_dir = os.path.join(self._log_write_dir, 'validation')
    self._val_step = self.model._test_counter  # pylint: disable=protected-access

    self._writers = {}  # Resets writers.

    if self.write_graph:
      self._write_keras_model_graph()
    if self.embeddings_freq:
      self._configure_embeddings()

  @property
  def _train_writer(self):
    if 'train' not in self._writers:
      self._writers['train'] = summary_ops_v2.create_file_writer_v2(
          self._train_dir)
    return self._writers['train']

  @property
  def _val_writer(self):
    if 'val' not in self._writers:
      self._writers['val'] = summary_ops_v2.create_file_writer_v2(self._val_dir)
    return self._writers['val']

  def _get_log_write_dir(self):
    """For multi-worker, only chief should write, others write to '/tmp'."""
    return distributed_file_utils.write_dirpath(self.log_dir,
                                                self.model.distribute_strategy)

  def _delete_tmp_write_dir(self):
    """Deletes tmp write directories for multi-worker."""
    distributed_file_utils.remove_temp_dirpath(self.log_dir,
                                               self.model.distribute_strategy)

  def _write_keras_model_graph(self):
    """Writes Keras graph networks to TensorBoard."""
    with self._train_writer.as_default():
      with summary_ops_v2.always_record_summaries():
        if not self.model.run_eagerly:
          summary_ops_v2.graph(K.get_graph(), step=0)

        summary_writable = (
            self.model._is_graph_network or  # pylint: disable=protected-access
            self.model.__class__.__name__ == 'Sequential')  # pylint: disable=protected-access
        if summary_writable:
          summary_ops_v2.keras_model('keras', self.model, step=0)

  def _configure_embeddings(self):
    """Configure the Projector for embeddings."""
    # TODO(omalleyt): Add integration tests.
    from google.protobuf import text_format
    from tensorflow.python.keras.layers import embeddings
    from tensorflow.python.keras.protobuf import projector_config_pb2

    config = projector_config_pb2.ProjectorConfig()
    for layer in self.model.layers:
      if isinstance(layer, embeddings.Embedding):
        embedding = config.embeddings.add()
        # Embeddings are always the first layer, so this naming should be
        # consistent in any keras models checkpoints.
        name = 'layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE'
        embedding.tensor_name = name

        if self.embeddings_metadata is not None:
          if isinstance(self.embeddings_metadata, str):
            embedding.metadata_path = self.embeddings_metadata
          else:
            if layer.name in self.embeddings_metadata.keys():
              embedding.metadata_path = self.embeddings_metadata.pop(layer.name)

    if self.embeddings_metadata and not isinstance(self.embeddings_metadata,
                                                   str):
      raise ValueError('Unrecognized `Embedding` layer names passed to '
                       '`keras.callbacks.TensorBoard` `embeddings_metadata` '
                       'argument: ' + str(self.embeddings_metadata.keys()))

    config_pbtxt = text_format.MessageToString(config)
    path = os.path.join(self._log_write_dir, 'projector_config.pbtxt')
    with open(path, 'w') as f:
      f.write(config_pbtxt)

  def _push_writer(self, writer, step):
    """Sets the default writer for custom batch-level summaries."""
    if self.update_freq == 'epoch':
      return

    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    self._prev_summary_state.append({
        'is_recording': summary_state.is_recording,
        'writer': summary_state.writer,
        'step': summary_state.step
    })

    if self.update_freq == 'epoch':
      should_record = False
      writer = None
    else:
      should_record = lambda: math_ops.equal(step % self.update_freq, 0)

    summary_state.is_recording = should_record
    summary_state.writer = writer
    # TODO(b/151339474): Fix deadlock when not using .value() here.
    summary_ops_v2.set_step(step.value())

  def _pop_writer(self):
    """Pops the current writer."""
    if self.update_freq == 'epoch':
      return

    prev_state = self._prev_summary_state.pop()

    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    summary_state.is_recording = prev_state['is_recording']
    summary_state.writer = prev_state['writer']
    summary_ops_v2.set_step(prev_state['step'])

  def _close_writers(self):
    for writer in self._writers.values():
      writer.close()

  def _init_profile_batch(self, profile_batch):
    """Validate profile_batch value and set the range of batches to profile.

    Arguments:
      profile_batch: The range of batches to profile. Should be a non-negative
        integer or a comma separated string of pair of positive integers. A pair
        of positive integers signify a range of batches to profile.

    Returns:
      A pair of non-negative integers specifying the start and stop batch to
      profile.

    Raises:
      ValueError: If profile_batch is not an integer or a comma seperated pair
                  of positive integers.

    """
    profile_batch_error_message = (
        'profile_batch must be a non-negative integer or 2-tuple of positive '
        'integers. A pair of positive integers signifies a range of batches '
        'to profile. Found: {}'.format(profile_batch))

    # Support legacy way of specifying "start,stop" or "start" as str.
    if isinstance(profile_batch, six.string_types):
      profile_batch = str(profile_batch).split(',')
      profile_batch = nest.map_structure(int, profile_batch)

    if isinstance(profile_batch, int):
      self._start_batch = profile_batch
      self._stop_batch = profile_batch
    elif isinstance(profile_batch, (tuple, list)) and len(profile_batch) == 2:
      self._start_batch, self._stop_batch = profile_batch
    else:
      raise ValueError(profile_batch_error_message)

    if self._start_batch < 0 or self._stop_batch < self._start_batch:
      raise ValueError(profile_batch_error_message)

    if self._start_batch > 0:
      profiler.warmup()  # Improve the profiling accuracy.
    # True when a trace is running.
    self._is_tracing = False

    # Setting `profile_batch=0` disables profiling.
    self._should_trace = not (self._start_batch == 0 and self._stop_batch == 0)

  def on_train_begin(self, logs=None):
    self._global_train_batch = 0
    self._push_writer(self._train_writer, self._train_step)

  def on_train_end(self, logs=None):
    self._pop_writer()

    if self._is_tracing:
      self._stop_trace()

    self._close_writers()
    self._delete_tmp_write_dir()

  def on_test_begin(self, logs=None):
    self._push_writer(self._val_writer, self._val_step)

  def on_test_end(self, logs=None):
    self._pop_writer()

  def on_train_batch_begin(self, batch, logs=None):
    self._global_train_batch += 1
    if not self._should_trace:
      return

    if self._global_train_batch == self._start_batch:
      self._start_trace()

  def on_train_batch_end(self, batch, logs=None):
    if not self._should_trace:
      return

    if self._is_tracing and self._global_train_batch >= self._stop_batch:
      self._stop_trace()

  def on_epoch_begin(self, epoch, logs=None):
    # Keeps track of epoch for profiling.
    self._epoch = epoch

  def on_epoch_end(self, epoch, logs=None):
    """Runs metrics and histogram summaries at epoch end."""
    self._log_epoch_metrics(epoch, logs)

    if self.histogram_freq and epoch % self.histogram_freq == 0:
      self._log_weights(epoch)

    if self.embeddings_freq and epoch % self.embeddings_freq == 0:
      self._log_embeddings(epoch)

  def _start_trace(self):
    summary_ops_v2.trace_on(graph=True, profiler=False)
    profiler.start(logdir=self._train_dir)
    self._is_tracing = True

  def _stop_trace(self, batch=None):
    """Logs the trace graph to TensorBoard."""
    if batch is None:
      batch = self._stop_batch
    with self._train_writer.as_default():
      with summary_ops_v2.always_record_summaries():
        # TODO(b/126388999): Remove step info in the summary name.
        summary_ops_v2.trace_export(name='batch_%d' % batch, step=batch)
    profiler.stop()
    self._is_tracing = False

  def _log_epoch_metrics(self, epoch, logs):
    """Writes epoch metrics out as scalar summaries.

    Arguments:
        epoch: Int. The global step to use for TensorBoard.
        logs: Dict. Keys are scalar summary names, values are scalars.
    """
    if not logs:
      return

    train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
    val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}

    with summary_ops_v2.always_record_summaries():
      if train_logs:
        with self._train_writer.as_default():
          for name, value in train_logs.items():
            summary_ops_v2.scalar('epoch_' + name, value, step=epoch)
      if val_logs:
        with self._val_writer.as_default():
          for name, value in val_logs.items():
            name = name[4:]  # Remove 'val_' prefix.
            summary_ops_v2.scalar('epoch_' + name, value, step=epoch)

  def _log_weights(self, epoch):
    """Logs the weights of the Model to TensorBoard."""
    with self._train_writer.as_default():
      with summary_ops_v2.always_record_summaries():
        for layer in self.model.layers:
          for weight in layer.weights:
            weight_name = weight.name.replace(':', '_')
            summary_ops_v2.histogram(weight_name, weight, step=epoch)
            if self.write_images:
              self._log_weight_as_image(weight, weight_name, epoch)
        self._train_writer.flush()

  def _log_weight_as_image(self, weight, weight_name, epoch):
    """Logs a weight as a TensorBoard image."""
    w_img = array_ops.squeeze(weight)
    shape = K.int_shape(w_img)
    if len(shape) == 1:  # Bias case
      w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
    elif len(shape) == 2:  # Dense layer kernel case
      if shape[0] > shape[1]:
        w_img = array_ops.transpose(w_img)
        shape = K.int_shape(w_img)
      w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
    elif len(shape) == 3:  # ConvNet case
      if K.image_data_format() == 'channels_last':
        # Switch to channels_first to display every kernel as a separate
        # image.
        w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
        shape = K.int_shape(w_img)
      w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])

    shape = K.int_shape(w_img)
    # Not possible to handle 3D convnets etc.
    if len(shape) == 4 and shape[-1] in [1, 3, 4]:
      summary_ops_v2.image(weight_name, w_img, step=epoch)

  def _log_embeddings(self, epoch):
    embeddings_ckpt = os.path.join(self._log_write_dir, 'train',
                                   'keras_embedding.ckpt-{}'.format(epoch))
    self.model.save_weights(embeddings_ckpt)


@keras_export('keras.callbacks.ReduceLROnPlateau')
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
      factor: factor by which the learning rate will be reduced.
        `new_lr = lr * factor`.
      patience: number of epochs with no improvement after which learning rate
        will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
        the learning rate will be reduced when the
        quantity monitored has stopped decreasing; in `'max'` mode it will be
        reduced when the quantity monitored has stopped increasing; in `'auto'`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      min_delta: threshold for measuring the new optimum, to only focus on
        significant changes.
      cooldown: number of epochs to wait before resuming normal operation after
        lr has been reduced.
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


@keras_export('keras.callbacks.CSVLogger')
class CSVLogger(Callback):
  """Callback that streams epoch results to a CSV file.

  Supports all values that can be represented as a string,
  including 1D iterables such as `np.ndarray`.

  Example:

  ```python
  csv_logger = CSVLogger('training.log')
  model.fit(X_train, Y_train, callbacks=[csv_logger])
  ```

  Arguments:
      filename: Filename of the CSV file, e.g. `'run/log.csv'`.
      separator: String used to separate elements in the CSV file.
      append: Boolean. True: append if file exists (useful for continuing
          training). False: overwrite existing file.
  """

  def __init__(self, filename, separator=',', append=False):
    self.sep = separator
    self.filename = path_to_string(filename)
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
      if file_io.file_exists(self.filename):
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
      elif isinstance(k, collections_abc.Iterable) and not is_zero_dim_ndarray:
        return '"[%s]"' % (', '.join(map(str, k)))
      else:
        return k

    if self.keys is None:
      self.keys = sorted(logs.keys())

    if self.model.stop_training:
      # We set NA so that csv parsers do not fail for this last epoch.
      logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys)

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


@keras_export('keras.callbacks.LambdaCallback')
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
