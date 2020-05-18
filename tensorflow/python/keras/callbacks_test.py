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
"""Tests for Keras callbacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import os
import re
import shutil
import sys
import threading
import time
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import adam
from tensorflow.python.training import checkpoint_management

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None

try:
  import requests  # pylint:disable=g-import-not-at-top
except ImportError:
  requests = None


TRAIN_SAMPLES = 10
TEST_SAMPLES = 10
NUM_CLASSES = 2
INPUT_DIM = 3
NUM_HIDDEN = 5
BATCH_SIZE = 5


class Counter(keras.callbacks.Callback):
  """Counts the number of times each callback method was run.

  Attributes:
    method_counts: dict. Contains the counts of time  each callback method was
      run.
  """

  def __init__(self):
    self.method_counts = collections.defaultdict(int)
    methods_to_count = [
        'on_batch_begin', 'on_batch_end', 'on_epoch_begin', 'on_epoch_end',
        'on_predict_batch_begin', 'on_predict_batch_end', 'on_predict_begin',
        'on_predict_end', 'on_test_batch_begin', 'on_test_batch_end',
        'on_test_begin', 'on_test_end', 'on_train_batch_begin',
        'on_train_batch_end', 'on_train_begin', 'on_train_end'
    ]
    for method_name in methods_to_count:
      setattr(self, method_name,
              self.wrap_with_counts(method_name, getattr(self, method_name)))

  def wrap_with_counts(self, method_name, method):

    def _call_and_count(*args, **kwargs):
      self.method_counts[method_name] += 1
      return method(*args, **kwargs)

    return _call_and_count


def _get_numpy():
  return np.ones((10, 10)), np.ones((10, 1))


def _get_sequence():

  class MySequence(keras.utils.data_utils.Sequence):

    def __getitem__(self, _):
      return np.ones((2, 10)), np.ones((2, 1))

    def __len__(self):
      return 5

  return MySequence(), None


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class CallbackCountsTest(keras_parameterized.TestCase):

  def _check_counts(self, counter, expected_counts):
    """Checks that the counts registered by `counter` are those expected."""
    for method_name, expected_count in expected_counts.items():
      self.assertEqual(
          counter.method_counts[method_name],
          expected_count,
          msg='For method {}: expected {}, got: {}'.format(
              method_name, expected_count, counter.method_counts[method_name]))

  def _get_model(self):
    layers = [
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(10,))
    model.compile(
        adam.AdamOptimizer(0.001),
        'binary_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  @parameterized.named_parameters(('with_numpy', _get_numpy()),
                                  ('with_sequence', _get_sequence()))
  def test_callback_hooks_are_called_in_fit(self, data):
    if not context.executing_eagerly():
      self.skipTest('Behavior changed in v2.')
    x, y = data
    val_x, val_y = np.ones((4, 10)), np.ones((4, 1))

    model = self._get_model()
    counter = Counter()
    model.fit(
        x,
        y,
        validation_data=(val_x, val_y),
        batch_size=2,
        steps_per_epoch=5,
        epochs=5,
        callbacks=[counter])

    self._check_counts(
        counter, {
            'on_batch_begin': 25,
            'on_batch_end': 25,
            'on_epoch_begin': 5,
            'on_epoch_end': 5,
            'on_predict_batch_begin': 0,
            'on_predict_batch_end': 0,
            'on_predict_begin': 0,
            'on_predict_end': 0,
            'on_test_batch_begin': 10,
            'on_test_batch_end': 10,
            'on_test_begin': 5,
            'on_test_end': 5,
            'on_train_batch_begin': 25,
            'on_train_batch_end': 25,
            'on_train_begin': 1,
            'on_train_end': 1
        })

  @parameterized.named_parameters(('with_numpy', _get_numpy()),
                                  ('with_sequence', _get_sequence()))
  def test_callback_hooks_are_called_in_evaluate(self, data):
    x, y = data
    is_sequence = isinstance(x, keras.utils.data_utils.Sequence)

    model = self._get_model()
    counter = Counter()
    model.evaluate(
        x,
        y,
        batch_size=2 if not is_sequence else None,
        steps=5 if is_sequence else None,
        callbacks=[counter])
    self._check_counts(
        counter, {
            'on_test_batch_begin': 5,
            'on_test_batch_end': 5,
            'on_test_begin': 1,
            'on_test_end': 1
        })

  @parameterized.named_parameters(('with_numpy', _get_numpy()),
                                  ('with_sequence', _get_sequence()))
  def test_callback_hooks_are_called_in_predict(self, data):
    x = data[0]
    is_sequence = isinstance(x, keras.utils.data_utils.Sequence)

    model = self._get_model()
    counter = Counter()
    model.predict(
        x,
        batch_size=2 if not is_sequence else None,
        steps=5 if is_sequence else None,
        callbacks=[counter])
    self._check_counts(
        counter, {
            'on_predict_batch_begin': 5,
            'on_predict_batch_end': 5,
            'on_predict_begin': 1,
            'on_predict_end': 1
        })

  def test_callback_list_methods(self):
    counter = Counter()
    callback_list = keras.callbacks.CallbackList([counter])

    batch = 0
    callback_list.on_test_batch_begin(batch)
    callback_list.on_test_batch_end(batch)
    callback_list.on_predict_batch_begin(batch)
    callback_list.on_predict_batch_end(batch)

    self._check_counts(
        counter, {
            'on_test_batch_begin': 1,
            'on_test_batch_end': 1,
            'on_predict_batch_begin': 1,
            'on_predict_batch_end': 1
        })


class KerasCallbacksTest(keras_parameterized.TestCase):

  def _get_model(self, input_shape=None):
    layers = [
        keras.layers.Dense(3, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=input_shape)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=[keras.metrics.CategoricalAccuracy(name='my_acc')],
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_progbar_logging(self):
    model = self._get_model(input_shape=(3,))

    x = array_ops.ones((200, 3))
    y = array_ops.zeros((200, 2))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(10)
    expected_log = r'(.*- loss:.*- my_acc:.*)+'

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(dataset, epochs=2, steps_per_epoch=10)
      self.assertRegexpMatches(printed.contents(), expected_log)

  @keras_parameterized.run_all_keras_modes
  def test_callback_warning(self):

    class SleepCallback(keras.callbacks.Callback):

      def on_train_batch_end(self, batch, logs=None):
        time.sleep(1)

    model = sequential.Sequential()
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        'sgd',
        loss='binary_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly())

    warning_messages = []

    def warning(msg):
      warning_messages.append(msg)

    with test.mock.patch.object(logging, 'warning', warning):
      model.fit(
          np.ones((10, 10), 'float32'),
          np.ones((10, 1), 'float32'),
          batch_size=5,
          epochs=10,
          callbacks=[SleepCallback()])
    warning_msg = ('Callbacks method `on_train_batch_end` is slow compared '
                   'to the batch time')
    self.assertIn(warning_msg, '\n'.join(warning_messages))

  @keras_parameterized.run_with_all_model_types(exclude_models='functional')
  @keras_parameterized.run_all_keras_modes
  def test_progbar_logging_deferred_model_build(self):
    model = self._get_model()
    self.assertFalse(model.built)

    x = array_ops.ones((200, 3))
    y = array_ops.zeros((200, 2))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(10)
    expected_log = r'(.*- loss:.*- my_acc:.*)+'

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(dataset, epochs=2, steps_per_epoch=10)
      self.assertRegexpMatches(printed.contents(), expected_log)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_progbar_logging_validation_data(self):
    model = self._get_model(input_shape=(3,))

    x = array_ops.ones((50, 3))
    y = array_ops.zeros((50, 2))
    training_dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(10)
    val_dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(10)
    expected_log = r'(.*5/5.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:.*)+'

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(training_dataset, epochs=2, validation_data=val_dataset)
      self.assertRegexpMatches(printed.contents(), expected_log)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_progbar_logging_validation_split(self):
    model = self._get_model(input_shape=(3,))

    x = np.ones((100, 3))
    y = np.zeros((100, 2))
    expected_log = (
        r'(?s).*1/2.*8/8.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:'
        r'.*2/2.*8/8.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:.*')

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(x, y, batch_size=10, epochs=2, validation_split=0.2)
      self.assertRegexpMatches(printed.contents(), expected_log)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_progbar_logging_training_validation(self):
    model = self._get_model(input_shape=(2,))

    def generator():
      for _ in range(100):
        yield [1, 1], 1

    training = dataset_ops.Dataset \
        .from_generator(
            generator=generator,
            output_types=('float64', 'float64'),
            output_shapes=([2], [])) \
        .batch(2) \
        .repeat()
    validation = dataset_ops.Dataset \
        .from_generator(
            generator=generator,
            output_types=('float64', 'float64'),
            output_shapes=([2], [])) \
        .batch(2)
    expected_log = (
        r'(?s).*1/2.*20/20.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:'
        r'.*2/2.*20/20.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:.*')

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(
          x=training, validation_data=validation, epochs=2, steps_per_epoch=20)
      self.assertRegexpMatches(printed.contents(), expected_log)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_progbar_logging_with_dataset_and_partial_batch(self):
    model = self._get_model(input_shape=(2,))

    def generator():
      # Have a partial batch at the end.
      for _ in range(9):
        yield np.random.random(2), 1

    training = dataset_ops.Dataset \
      .from_generator(
          generator=generator,
          output_types=('float64', 'float64'),
          output_shapes=([2], [])) \
      .batch(2)
    validation = dataset_ops.Dataset \
      .from_generator(
          generator=generator,
          output_types=('float64', 'float64'),
          output_shapes=([2], [])) \
      .batch(2)

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(x=training, validation_data=validation)

      # Make sure the value of val_ metrics are not zeros.
      log_content = printed.contents()
      val_loss = re.findall(r'val_loss: (\d\.\d+)', log_content)
      self.assertLen(val_loss, 1)
      self.assertGreater(float(val_loss[0]), 0.0)

  @keras_parameterized.run_with_all_model_types
  def test_ModelCheckpoint(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    layers = [
        keras.layers.Dense(NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(10,))
    model.compile(
        loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

    filepath = os.path.join(temp_dir, 'checkpoint.h5')
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    # case 1
    monitor = 'val_loss'
    save_best_only = False
    mode = 'auto'

    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    cbks = [
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode)
    ]
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=1,
        verbose=0)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 2
    mode = 'min'
    cbks = [
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode)
    ]
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=1,
        verbose=0)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 3
    mode = 'max'
    monitor = 'val_acc'
    cbks = [
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode)
    ]
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=1,
        verbose=0)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 4
    save_best_only = True
    cbks = [
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode)
    ]
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=1,
        verbose=0)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # Case: metric not available.
    cbks = [
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='unknown',
            save_best_only=True)
    ]
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=1,
        verbose=0)
    # File won't be written.
    assert not os.path.exists(filepath)

    # case 5
    save_best_only = False
    period = 2
    mode = 'auto'

    filepath = os.path.join(temp_dir, 'checkpoint.{epoch:02d}.h5')
    cbks = [
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode,
            period=period)
    ]
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=4,
        verbose=1)
    assert os.path.exists(filepath.format(epoch=2))
    assert os.path.exists(filepath.format(epoch=4))
    os.remove(filepath.format(epoch=2))
    os.remove(filepath.format(epoch=4))
    assert not os.path.exists(filepath.format(epoch=1))
    assert not os.path.exists(filepath.format(epoch=3))

    # Invalid use: this will raise a warning but not an Exception.
    keras.callbacks.ModelCheckpoint(
        filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        mode='unknown')

    # Case 6: `ModelCheckpoint` with a combination of `save_freq` and `period`.
    # Though `period` is deprecated, we're testing it for
    # backward-compatibility.
    filepath = os.path.join(temp_dir, 'checkpoint.epoch{epoch:02d}.h5')
    cbks = [
        keras.callbacks.ModelCheckpoint(
            filepath, monitor=monitor, mode=mode, save_freq='epoch', period=5)
    ]
    assert not os.path.exists(filepath.format(epoch=0))
    assert not os.path.exists(filepath.format(epoch=5))
    model.fit(
        x_train,
        y_train,
        batch_size=2,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=10,
        verbose=1)
    assert not os.path.exists(filepath.format(epoch=1))
    assert not os.path.exists(filepath.format(epoch=2))
    assert not os.path.exists(filepath.format(epoch=3))
    assert not os.path.exists(filepath.format(epoch=4))
    assert os.path.exists(filepath.format(epoch=5))
    assert not os.path.exists(filepath.format(epoch=6))
    assert os.path.exists(filepath.format(epoch=10))
    os.remove(filepath.format(epoch=5))
    os.remove(filepath.format(epoch=10))

    # Case 7: `ModelCheckpoint` with an integer `save_freq`
    filepath = os.path.join(temp_dir, 'checkpoint.epoch{epoch:02d}.h5')
    cbks = [
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode,
            save_freq=15,
            period=100)  # The period should be ignored (this test tests this).
    ]
    assert not os.path.exists(filepath.format(epoch=3))
    model.fit(
        x_train,
        y_train,
        batch_size=2,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=10,
        verbose=1)
    assert not os.path.exists(filepath.format(epoch=1))
    assert not os.path.exists(filepath.format(epoch=2))
    assert os.path.exists(filepath.format(epoch=3))
    assert not os.path.exists(filepath.format(epoch=4))
    assert not os.path.exists(filepath.format(epoch=5))
    assert os.path.exists(filepath.format(epoch=6))
    assert not os.path.exists(filepath.format(epoch=7))
    assert not os.path.exists(filepath.format(epoch=8))
    assert os.path.exists(filepath.format(epoch=9))
    os.remove(filepath.format(epoch=3))
    os.remove(filepath.format(epoch=6))
    os.remove(filepath.format(epoch=9))

    # Case 8: `ModelCheckpoint` with valid and invalid save_freq argument.
    with self.assertRaisesRegexp(ValueError, 'Unrecognized save_freq'):
      keras.callbacks.ModelCheckpoint(
          filepath,
          monitor=monitor,
          save_best_only=save_best_only,
          mode=mode,
          save_freq='invalid_save_freq')
    # The following should not raise ValueError.
    keras.callbacks.ModelCheckpoint(
        filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        mode=mode,
        save_freq='epoch')
    keras.callbacks.ModelCheckpoint(
        filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        mode=mode,
        save_freq=3)

  def _get_dummy_resource_for_model_checkpoint_testing(self):

    def get_input_datasets():
      # Simple training input.
      train_input = [[1.]] * 16
      train_label = [[0.]] * 16
      ds = dataset_ops.Dataset.from_tensor_slices((train_input, train_label))
      return ds.batch(8, drop_remainder=True)

    # Very simple bias model to eliminate randomness.
    optimizer = gradient_descent.SGD(0.1)
    model = sequential.Sequential()
    model.add(testing_utils.Bias(input_shape=(1,)))
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    train_ds = get_input_datasets()

    temp_dir = self.get_temp_dir()
    filepath = os.path.join(temp_dir, 'checkpoint.epoch{epoch:02d}.h5')

    # The filepath shouldn't exist at the beginning.
    self.assertFalse(os.path.exists(filepath))
    callback = keras.callbacks.ModelCheckpoint(
        filepath=filepath, save_weights_only=True)

    return model, train_ds, callback, filepath

  def _run_load_weights_on_restart_test_common_iterations(self):

    (model, train_ds, callback,
     filepath) = self._get_dummy_resource_for_model_checkpoint_testing()
    initial_epochs = 3
    model.fit(train_ds, epochs=initial_epochs, callbacks=[callback])

    # The files should exist after fitting with callback.
    for epoch in range(initial_epochs):
      self.assertTrue(os.path.exists(filepath.format(epoch=epoch + 1)))
    self.assertFalse(os.path.exists(filepath.format(epoch=initial_epochs + 1)))
    self.assertEqual(
        callback._get_most_recently_modified_file_matching_pattern(filepath),
        filepath.format(epoch=initial_epochs))

    model.fit(train_ds, epochs=1)
    weights_after_one_more_epoch = model.get_weights()

    # The filepath should continue to exist after fitting without callback.
    for epoch in range(initial_epochs):
      self.assertTrue(os.path.exists(filepath.format(epoch=epoch + 1)))

    return model, train_ds, filepath, weights_after_one_more_epoch

  @staticmethod
  def get_ModelCheckpoint_load_weights_on_restart_true_test(save_weights_only):

    def func(self):
      (model, train_ds, filepath, weights_after_one_more_epoch
      ) = self._run_load_weights_on_restart_test_common_iterations()

      # Sleep for some short time period ensuring the files are created with
      # a different time (in MacOS OSS the granularity is only 1 second).
      time.sleep(2)
      callback = keras.callbacks.ModelCheckpoint(
          filepath=filepath,
          save_weights_only=save_weights_only,
          load_weights_on_restart=True)
      model.fit(train_ds, epochs=1, callbacks=[callback])
      weights_after_model_restoring_and_one_more_epoch = model.get_weights()

      self.assertEqual(
          callback._get_most_recently_modified_file_matching_pattern(filepath),
          filepath.format(epoch=1))

      model.fit(
          train_ds,
          epochs=1,
          callbacks=[
              keras.callbacks.ModelCheckpoint(
                  filepath=filepath,
                  save_weights_only=save_weights_only,
                  load_weights_on_restart=True)
          ])
      weights_with_one_final_extra_epoch = model.get_weights()

      # Asserting the weights one epoch after initial fitting and another epoch
      # after that are closed, if a ModelCheckpoint with
      # load_weights_on_restart=True is given (so the model is restored at the
      # beginning of training).
      self.assertAllClose(weights_after_one_more_epoch,
                          weights_after_model_restoring_and_one_more_epoch)

      self.assertNotAllClose(weights_after_one_more_epoch,
                             weights_with_one_final_extra_epoch)

    return func

  @staticmethod
  def get_ModelCheckpoint_load_weights_on_restart_false_test(save_weights_only):

    def func(self):
      (model, train_ds, filepath, weights_after_one_more_epoch
      ) = self._run_load_weights_on_restart_test_common_iterations()

      model.fit(
          train_ds,
          epochs=1,
          callbacks=[
              keras.callbacks.ModelCheckpoint(
                  filepath=filepath, save_weights_only=save_weights_only)
          ])
      weights_after_model_restoring_and_one_more_epoch = model.get_weights()

      # Asserting the weights one epoch after initial fitting and another epoch
      # after that are different, if a ModelCheckpoint with
      # load_weights_on_restart=False is given (so the model is not restored at
      # the beginning of training).
      self.assertNotAllClose(weights_after_one_more_epoch,
                             weights_after_model_restoring_and_one_more_epoch)

    return func

  test_model_checkpoint_load_weights_on_restart_true_save_weights_only_true = \
        get_ModelCheckpoint_load_weights_on_restart_true_test.__func__(True)

  test_model_checkpoint_load_weights_on_restart_true_save_weights_only_false = \
        get_ModelCheckpoint_load_weights_on_restart_true_test.__func__(False)

  test_model_checkpoint_load_weights_on_restart_false_save_weights_only_true = \
        get_ModelCheckpoint_load_weights_on_restart_false_test.__func__(True)

  test_model_checkpoint_load_weights_on_restart_false_save_weights_only_false \
        = get_ModelCheckpoint_load_weights_on_restart_false_test.__func__(False)

  def test_ModelCheckpoint_override_if_file_exist(self):
    (model, train_ds, filepath,
     _) = self._run_load_weights_on_restart_test_common_iterations()

    # Sleep for some short time period to ensure the files are created with
    # a different time (in MacOS OSS the granularity is only 1 second).
    time.sleep(2)
    callback = keras.callbacks.ModelCheckpoint(
        filepath=filepath, save_weights_only=True)
    model.load_weights(
        callback._get_most_recently_modified_file_matching_pattern(filepath))
    weights_before_additional_fit = model.get_weights()
    model.fit(train_ds, epochs=1, callbacks=[callback])
    model.load_weights(
        callback._get_most_recently_modified_file_matching_pattern(filepath))
    weights_after_additional_fit = model.get_weights()

    self.assertNotAllClose(weights_before_additional_fit,
                           weights_after_additional_fit)

  def test_fit_with_ModelCheckpoint_with_tf_config(self):
    (model, train_ds, callback,
     _) = self._get_dummy_resource_for_model_checkpoint_testing()

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ['localhost:23333']
        },
        'task': {
            'type': 'worker',
            'index': 0
        }
    })

    # `model.fit()` should work regardless of the presence of `TF_CONFIG`.
    model.fit(train_ds, epochs=1, callbacks=[callback])

  def test_fit_with_ModelCheckpoint_with_dir_as_h5_filepath(self):
    (model, train_ds, callback,
     filepath) = self._get_dummy_resource_for_model_checkpoint_testing()

    temp_dir = self.get_temp_dir()
    filepath = os.path.join(temp_dir, 'temp.h5')

    self.assertFalse(os.path.exists(filepath))
    os.mkdir(filepath)
    self.assertTrue(os.path.exists(filepath))

    callback = keras.callbacks.ModelCheckpoint(filepath=filepath)

    with self.assertRaisesRegexp(IOError, 'Please specify a non-directory '
                                          'filepath for ModelCheckpoint.'):
      model.fit(train_ds, epochs=1, callbacks=[callback])

  def test_ModelCheckpoint_with_bad_path_placeholders(self):
    (model, train_ds, callback,
     filepath) = self._get_dummy_resource_for_model_checkpoint_testing()

    temp_dir = self.get_temp_dir()
    filepath = os.path.join(temp_dir, 'chkpt_{epoch:02d}_{mape:.2f}.h5')
    callback = keras.callbacks.ModelCheckpoint(filepath=filepath)

    with self.assertRaisesRegexp(KeyError, 'Failed to format this callback '
                                           'filepath.*'):
      model.fit(train_ds, epochs=1, callbacks=[callback])

  def test_ModelCheckpoint_nonblocking(self):
    filepath = self.get_temp_dir()
    # Should only cause a sync block when saving is actually performed.
    callback = keras.callbacks.ModelCheckpoint(filepath=filepath, save_freq=100)
    self.assertTrue(callback._supports_tf_logs)

    model = keras.Sequential([keras.layers.Dense(1)])
    cb_list = keras.callbacks.CallbackList([callback],
                                           model=model,
                                           epochs=1,
                                           steps=10,
                                           verbose=0)

    with context.eager_mode():
      tensor = ops.convert_to_tensor(1.)

    def mock_numpy():
      raise RuntimeError(
          'If this error is seen, ModelCheckpoint is causing a blocking '
          'NumPy conversion even when not checkpointing.')

    with test.mock.patch.object(tensor, 'numpy', mock_numpy):
      logs = {'metric': tensor}

      cb_list.on_train_begin(logs)
      cb_list.on_epoch_begin(0, logs)
      cb_list.on_train_batch_begin(0, logs)
      cb_list.on_train_batch_end(0, logs)
      cb_list.on_epoch_end(0, logs)
      cb_list.on_train_end(logs)

      cb_list.on_test_begin(logs)
      cb_list.on_test_batch_begin(0, logs)
      cb_list.on_test_batch_end(0, logs)
      cb_list.on_test_end(logs)

      cb_list.on_predict_begin(logs)
      cb_list.on_predict_batch_begin(logs)
      cb_list.on_predict_batch_end(logs)
      cb_list.on_predict_end(logs)

  def test_ProgbarLogger_verbose_2_nonblocking(self):
    # Should only cause a sync block on epoch end methods.
    callback = keras.callbacks.ProgbarLogger(count_mode='steps')
    self.assertTrue(callback._supports_tf_logs)

    model = keras.Sequential([keras.layers.Dense(1)])
    cb_list = keras.callbacks.CallbackList([callback],
                                           model=model,
                                           epochs=1,
                                           steps=10,
                                           verbose=2)

    with context.eager_mode():
      tensor = ops.convert_to_tensor(1.)

    def mock_numpy():
      raise RuntimeError(
          'If this error is seen, ModelCheckpoint is causing a blocking '
          'NumPy conversion even when not checkpointing.')

    with test.mock.patch.object(tensor, 'numpy', mock_numpy):
      logs = {'metric': tensor}

      cb_list.on_train_begin(logs)
      cb_list.on_epoch_begin(0, logs)
      cb_list.on_train_batch_begin(0, logs)
      cb_list.on_train_batch_end(0, logs)

      cb_list.on_test_begin(logs)
      cb_list.on_test_batch_begin(0, logs)
      cb_list.on_test_batch_end(0, logs)
      cb_list.on_test_end(logs)

      with self.assertRaisesRegexp(RuntimeError, 'NumPy conversion'):
        # on_epoch_end should still block.
        cb_list.on_epoch_end(0, logs)
      cb_list.on_train_end(logs)

  def test_EarlyStopping(self):
    with self.cached_session():
      np.random.seed(123)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
      model.compile(
          loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

      cases = [
          ('max', 'val_acc'),
          ('min', 'val_loss'),
          ('auto', 'val_acc'),
          ('auto', 'loss'),
          ('unknown', 'unknown')
      ]
      for mode, monitor in cases:
        patience = 0
        cbks = [
            keras.callbacks.EarlyStopping(
                patience=patience, monitor=monitor, mode=mode)
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=5,
            verbose=0)

  def test_EarlyStopping_reuse(self):
    with self.cached_session():
      np.random.seed(1337)
      patience = 3
      data = np.random.random((100, 1))
      labels = np.where(data > 0.5, 1, 0)
      model = keras.models.Sequential((keras.layers.Dense(
          1, input_dim=1, activation='relu'), keras.layers.Dense(
              1, activation='sigmoid'),))
      model.compile(
          optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
      weights = model.get_weights()

      # This should allow training to go for at least `patience` epochs
      model.set_weights(weights)

      stopper = keras.callbacks.EarlyStopping(monitor='acc', patience=patience)
      hist = model.fit(data, labels, callbacks=[stopper], verbose=0, epochs=20)
      assert len(hist.epoch) >= patience

  def test_EarlyStopping_with_baseline(self):
    with self.cached_session():
      np.random.seed(1337)
      baseline = 0.6
      (data, labels), _ = testing_utils.get_test_data(
          train_samples=100,
          test_samples=50,
          input_shape=(1,),
          num_classes=NUM_CLASSES)
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=1, num_classes=1, input_dim=1)
      model.compile(
          optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])

      stopper = keras.callbacks.EarlyStopping(monitor='acc',
                                              baseline=baseline)
      hist = model.fit(data, labels, callbacks=[stopper], verbose=0, epochs=20)
      assert len(hist.epoch) == 1

      patience = 3
      stopper = keras.callbacks.EarlyStopping(monitor='acc',
                                              patience=patience,
                                              baseline=baseline)
      hist = model.fit(data, labels, callbacks=[stopper], verbose=0, epochs=20)
      assert len(hist.epoch) >= patience

  def test_EarlyStopping_final_weights_when_restoring_model_weights(self):

    class DummyModel(object):

      def __init__(self):
        self.stop_training = False
        self.weights = -1

      def get_weights(self):
        return self.weights

      def set_weights(self, weights):
        self.weights = weights

      def set_weight_to_epoch(self, epoch):
        self.weights = epoch

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=2,
                                               restore_best_weights=True)
    early_stop.model = DummyModel()
    losses = [0.2, 0.15, 0.1, 0.11, 0.12]
    # The best configuration is in the epoch 2 (loss = 0.1000).
    epochs_trained = 0
    early_stop.on_train_begin()
    for epoch in range(len(losses)):
      epochs_trained += 1
      early_stop.model.set_weight_to_epoch(epoch=epoch)
      early_stop.on_epoch_end(epoch, logs={'val_loss': losses[epoch]})
      if early_stop.model.stop_training:
        break
    # The best configuration is in epoch 2 (loss = 0.1000),
    # and while patience = 2, we're restoring the best weights,
    # so we end up at the epoch with the best weights, i.e. epoch 2
    self.assertEqual(early_stop.model.get_weights(), 2)

  def test_RemoteMonitor(self):
    if requests is None:
      self.skipTest('`requests` required to run this test')
      return None

    monitor = keras.callbacks.RemoteMonitor()
    # This will raise a warning since the default address in unreachable:
    monitor.on_epoch_end(0, logs={'loss': 0.})

  def test_LearningRateScheduler(self):
    with self.cached_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])

      cbks = [keras.callbacks.LearningRateScheduler(lambda x: 1. / (1. + x))]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=5,
          verbose=0)
      assert (
          float(keras.backend.get_value(
              model.optimizer.lr)) - 0.2) < keras.backend.epsilon()

      cbks = [keras.callbacks.LearningRateScheduler(lambda x, lr: lr / 2)]
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=0)
      assert (
          float(keras.backend.get_value(
              model.optimizer.lr)) - 0.01 / 4) < keras.backend.epsilon()

      cbks = [
          keras.callbacks.LearningRateScheduler(
              lambda epoch, _: learning_rate_schedule.CosineDecay(0.01, 2)
              (epoch))
      ]
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=0)

      cosine_decay_np = 0.5 * (1 + np.cos(np.pi * (1 / 2)))
      decayed_learning_rate = 0.01 * cosine_decay_np

      assert (float(keras.backend.get_value(model.optimizer.lr)) -
              decayed_learning_rate) < keras.backend.epsilon()

  def test_ReduceLROnPlateau(self):
    with self.cached_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)

      def make_model():
        random_seed.set_random_seed(1234)
        np.random.seed(1337)
        model = testing_utils.get_small_sequential_mlp(
            num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=gradient_descent.SGD(lr=0.1))
        return model

      # TODO(psv): Make sure the callback works correctly when min_delta is
      # set as 0. Test fails when the order of this callback and assertion is
      # interchanged.
      model = make_model()
      cbks = [
          keras.callbacks.ReduceLROnPlateau(
              monitor='val_loss',
              factor=0.1,
              min_delta=0,
              patience=1,
              cooldown=5)
      ]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=0)
      self.assertAllClose(
          float(keras.backend.get_value(model.optimizer.lr)), 0.1, atol=1e-4)

      model = make_model()
      # This should reduce the LR after the first epoch (due to high epsilon).
      cbks = [
          keras.callbacks.ReduceLROnPlateau(
              monitor='val_loss',
              factor=0.1,
              min_delta=10,
              patience=1,
              cooldown=5)
      ]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=2)
      self.assertAllClose(
          float(keras.backend.get_value(model.optimizer.lr)), 0.01, atol=1e-4)

  def test_ReduceLROnPlateau_patience(self):

    class DummyOptimizer(object):

      def __init__(self):
        self.lr = keras.backend.variable(1.0)

    class DummyModel(object):

      def __init__(self):
        self.optimizer = DummyOptimizer()

    reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=2)
    reduce_on_plateau.model = DummyModel()

    losses = [0.0860, 0.1096, 0.1040]
    lrs = []

    for epoch in range(len(losses)):
      reduce_on_plateau.on_epoch_end(epoch, logs={'val_loss': losses[epoch]})
      lrs.append(keras.backend.get_value(reduce_on_plateau.model.optimizer.lr))

    # The learning rates should be 1.0 except the last one
    for lr in lrs[:-1]:
      self.assertEqual(lr, 1.0)
    self.assertLess(lrs[-1], 1.0)

  def test_ReduceLROnPlateau_backwards_compatibility(self):
    with test.mock.patch.object(logging, 'warning') as mock_log:
      reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(epsilon=1e-13)
      self.assertRegexpMatches(
          str(mock_log.call_args), '`epsilon` argument is deprecated')
    self.assertFalse(hasattr(reduce_on_plateau, 'epsilon'))
    self.assertTrue(hasattr(reduce_on_plateau, 'min_delta'))
    self.assertEqual(reduce_on_plateau.min_delta, 1e-13)

  def test_CSVLogger(self):
    with self.cached_session():
      np.random.seed(1337)
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
      filepath = os.path.join(temp_dir, 'log.tsv')

      sep = '\t'
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)

      def make_model():
        np.random.seed(1337)
        model = testing_utils.get_small_sequential_mlp(
            num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=gradient_descent.SGD(lr=0.1),
            metrics=['accuracy'])
        return model

      # case 1, create new file with defined separator
      model = make_model()
      cbks = [keras.callbacks.CSVLogger(filepath, separator=sep)]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=1,
          verbose=0)

      assert os.path.exists(filepath)
      with open(filepath) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read())
      assert dialect.delimiter == sep
      del model
      del cbks

      # case 2, append data to existing file, skip header
      model = make_model()
      cbks = [keras.callbacks.CSVLogger(filepath, separator=sep, append=True)]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=1,
          verbose=0)

      # case 3, reuse of CSVLogger object
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=0)

      with open(filepath) as csvfile:
        list_lines = csvfile.readlines()
        for line in list_lines:
          assert line.count(sep) == 4
        assert len(list_lines) == 5
        output = ' '.join(list_lines)
        assert len(re.findall('epoch', output)) == 1

      os.remove(filepath)

  def test_stop_training_csv(self):
    # Test that using the CSVLogger callback with the TerminateOnNaN callback
    # does not result in invalid CSVs.
    np.random.seed(1337)
    tmpdir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

    with self.cached_session():
      fp = os.path.join(tmpdir, 'test.csv')
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)

      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)
      cbks = [keras.callbacks.TerminateOnNaN(), keras.callbacks.CSVLogger(fp)]
      model = keras.models.Sequential()
      for _ in range(5):
        model.add(keras.layers.Dense(2, input_dim=INPUT_DIM, activation='relu'))
      model.add(keras.layers.Dense(NUM_CLASSES, activation='linear'))
      model.compile(loss='mean_squared_error',
                    optimizer='rmsprop')

      def data_generator():
        i = 0
        max_batch_index = len(x_train) // BATCH_SIZE
        tot = 0
        while 1:
          if tot > 3 * len(x_train):
            yield (np.ones([BATCH_SIZE, INPUT_DIM]) * np.nan,
                   np.ones([BATCH_SIZE, NUM_CLASSES]) * np.nan)
          else:
            yield (x_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                   y_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE])
          i += 1
          tot += 1
          i %= max_batch_index

      history = model.fit_generator(data_generator(),
                                    len(x_train) // BATCH_SIZE,
                                    validation_data=(x_test, y_test),
                                    callbacks=cbks,
                                    epochs=20)
      loss = history.history['loss']
      assert len(loss) > 1
      assert loss[-1] == np.inf or np.isnan(loss[-1])

      values = []
      with open(fp) as f:
        for x in csv.reader(f):
          # In windows, due to \r\n line ends we may end up reading empty lines
          # after each line. Skip empty lines.
          if x:
            values.append(x)
      assert 'nan' in values[-1], 'The last epoch was not logged.'

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_TerminateOnNaN(self):
    np.random.seed(1337)
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES)

    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    cbks = [keras.callbacks.TerminateOnNaN()]
    model = keras.models.Sequential()
    initializer = keras.initializers.Constant(value=1e5)
    for _ in range(5):
      model.add(
          keras.layers.Dense(
              2,
              input_dim=INPUT_DIM,
              activation='relu',
              kernel_initializer=initializer))
    model.add(keras.layers.Dense(NUM_CLASSES))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=20)
    loss = history.history['loss']
    self.assertEqual(len(loss), 1)
    self.assertTrue(np.isnan(loss[0]))

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  def test_LambdaCallback(self):
    with self.cached_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)
      model = keras.models.Sequential()
      model.add(
          keras.layers.Dense(
              NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
      model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])

      # Start an arbitrary process that should run during model
      # training and be terminated after training has completed.
      e = threading.Event()

      def target():
        e.wait()

      t = threading.Thread(target=target)
      t.start()
      cleanup_callback = keras.callbacks.LambdaCallback(
          on_train_end=lambda logs: e.set())

      cbks = [cleanup_callback]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=5,
          verbose=0)
      t.join()
      assert not t.is_alive()

  def test_RemoteMonitor_np_array(self):
    if requests is None:
      self.skipTest('`requests` required to run this test')
    with test.mock.patch.object(requests, 'post') as requests_post:
      monitor = keras.callbacks.RemoteMonitor(send_as_json=True)
      a = np.arange(1)  # a 1 by 1 array
      logs = {'loss': 0., 'val': a}
      monitor.on_epoch_end(0, logs=logs)
      send = {'loss': 0., 'epoch': 0, 'val': 0}
      requests_post.assert_called_once_with(
          monitor.root + monitor.path, json=send, headers=monitor.headers)

  def test_RemoteMonitor_np_float32(self):
    if requests is None:
      self.skipTest('`requests` required to run this test')

    with test.mock.patch.object(requests, 'post') as requests_post:
      monitor = keras.callbacks.RemoteMonitor(send_as_json=True)
      a = np.float32(1.0)  # a float32 generic type
      logs = {'loss': 0., 'val': a}
      monitor.on_epoch_end(0, logs=logs)
      send = {'loss': 0., 'epoch': 0, 'val': 1.0}
      requests_post.assert_called_once_with(
          monitor.root + monitor.path, json=send, headers=monitor.headers)

  def test_RemoteMonitorWithJsonPayload(self):
    if requests is None:
      self.skipTest('`requests` required to run this test')
      return None
    with self.cached_session():
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = keras.utils.np_utils.to_categorical(y_test)
      y_train = keras.utils.np_utils.to_categorical(y_train)
      model = keras.models.Sequential()
      model.add(
          keras.layers.Dense(
              NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
      model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])
      cbks = [keras.callbacks.RemoteMonitor(send_as_json=True)]

      with test.mock.patch.object(requests, 'post'):
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1)

  def test_progbar_infers_steps(self):
    x, y = np.ones((10, 1)), np.ones((10, 1))
    data = dataset_ops.DatasetV2.from_tensor_slices((x, y)).batch(2)
    data = data.filter(lambda x, y: True)  # Unknown cardinality.

    progbar = keras.callbacks.ProgbarLogger('steps')
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile('sgd', 'mse')
    self.assertIsNone(progbar.target)
    model.fit(data, epochs=2, callbacks=[progbar])
    self.assertEqual(progbar.target, 5)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_callback_passed_floats(self):

    class MyCallback(keras.callbacks.Callback):

      def on_batch_end(self, batch, logs=None):
        assert isinstance(batch, int)
        assert isinstance(logs['loss'], float)
        self.on_batch_end_called = True

      def on_epoch_end(self, batch, logs=None):
        assert isinstance(batch, int)
        assert isinstance(logs['loss'], float)
        self.on_epoch_end_called = True

    x, y = np.ones((10, 1)), np.ones((10, 1))
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())

    callback = MyCallback()
    model.fit(x, y, epochs=2, callbacks=[callback])
    self.assertTrue(callback.on_batch_end_called)
    self.assertTrue(callback.on_batch_end_called)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_implements_batch_hooks(self):

    class MyCallbackWithBatchHooks(keras.callbacks.Callback):

      def __init__(self):
        self.train_batches = 0
        self.test_batches = 0
        self.predict_batches = 0

      def on_train_batch_end(self, batch, logs=None):
        self.train_batches += 1

      def on_test_batch_end(self, batch, logs=None):
        self.test_batches += 1

      def on_predict_batch_end(self, batch, logs=None):
        self.predict_batches += 1

    class MyCallbackWithoutBatchHooks(keras.callbacks.Callback):

      def __init__(self):
        self.epochs = 0

      def on_epoch_end(self, epoch, logs=None):
        self.epochs += 1

    x, y = np.ones((10, 1)), np.ones((10, 1))
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile('sgd', 'mse')

    my_cb = MyCallbackWithBatchHooks()
    cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
    self.assertTrue(cb_list._should_call_train_batch_hooks)
    self.assertTrue(cb_list._should_call_test_batch_hooks)
    self.assertTrue(cb_list._should_call_predict_batch_hooks)

    model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
    model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
    model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

    self.assertEqual(my_cb.train_batches, 2)
    self.assertEqual(my_cb.test_batches, 1)
    self.assertEqual(my_cb.predict_batches, 1)

    my_cb = MyCallbackWithoutBatchHooks()
    cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
    self.assertLen(cb_list.callbacks, 1)
    self.assertFalse(cb_list._should_call_train_batch_hooks)
    self.assertFalse(cb_list._should_call_test_batch_hooks)
    self.assertFalse(cb_list._should_call_predict_batch_hooks)

    model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
    model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
    model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_implements_batch_hooks_override(self):

    class MyCallback(keras.callbacks.Callback):

      def __init__(self, should_run=True):
        self.should_run = should_run
        self.train_batches = 0
        self.test_batches = 0
        self.predict_batches = 0

      def on_train_batch_end(self, batch, logs=None):
        self.train_batches += 1

      def on_test_batch_end(self, batch, logs=None):
        self.test_batches += 1

      def on_predict_batch_end(self, batch, logs=None):
        self.predict_batches += 1

      def _implements_train_batch_hooks(self):
        return self.should_run

      def _implements_test_batch_hooks(self):
        return self.should_run

      def _implements_predict_batch_hooks(self):
        return self.should_run

    x, y = np.ones((10, 1)), np.ones((10, 1))
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile('sgd', 'mse')

    my_cb = MyCallback(should_run=True)
    cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
    self.assertTrue(cb_list._should_call_train_batch_hooks)
    self.assertTrue(cb_list._should_call_test_batch_hooks)
    self.assertTrue(cb_list._should_call_predict_batch_hooks)

    model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
    model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
    model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

    self.assertEqual(my_cb.train_batches, 2)
    self.assertEqual(my_cb.test_batches, 1)
    self.assertEqual(my_cb.predict_batches, 1)

    my_cb = MyCallback(should_run=False)
    cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
    self.assertFalse(cb_list._should_call_train_batch_hooks)
    self.assertFalse(cb_list._should_call_test_batch_hooks)
    self.assertFalse(cb_list._should_call_predict_batch_hooks)

    model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
    model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
    model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

    self.assertEqual(my_cb.train_batches, 0)
    self.assertEqual(my_cb.test_batches, 0)
    self.assertEqual(my_cb.predict_batches, 0)


# A summary that was emitted during a test. Fields:
#   logdir: str. The logdir of the FileWriter to which the summary was
#     written.
#   tag: str. The name of the summary.
_ObservedSummary = collections.namedtuple('_ObservedSummary', ('logdir', 'tag'))


class _SummaryFile(object):
  """A record of summary tags and the files to which they were written.

  Fields `scalars`, `images`, `histograms`, and `tensors` are sets
  containing `_ObservedSummary` values.
  """

  def __init__(self):
    self.scalars = set()
    self.images = set()
    self.histograms = set()
    self.tensors = set()


def list_summaries(logdir):
  """Read all summaries under the logdir into a `_SummaryFile`.

  Args:
    logdir: A path to a directory that contains zero or more event
      files, either as direct children or in transitive subdirectories.
      Summaries in these events must only contain old-style scalars,
      images, and histograms. Non-summary events, like `graph_def`s, are
      ignored.

  Returns:
    A `_SummaryFile` object reflecting all summaries written to any
    event files in the logdir or any of its descendant directories.

  Raises:
    ValueError: If an event file contains an summary of unexpected kind.
  """
  result = _SummaryFile()
  for (dirpath, _, filenames) in os.walk(logdir):
    for filename in filenames:
      if not filename.startswith('events.out.'):
        continue
      path = os.path.join(dirpath, filename)
      for event in summary_iterator.summary_iterator(path):
        if not event.summary:  # (e.g., it's a `graph_def` event)
          continue
        for value in event.summary.value:
          tag = value.tag
          # Case on the `value` rather than the summary metadata because
          # the Keras callback uses `summary_ops_v2` to emit old-style
          # summaries. See b/124535134.
          kind = value.WhichOneof('value')
          container = {
              'simple_value': result.scalars,
              'image': result.images,
              'histo': result.histograms,
              'tensor': result.tensors,
          }.get(kind)
          if container is None:
            raise ValueError(
                'Unexpected summary kind %r in event file %s:\n%r'
                % (kind, path, event))
          elif kind == 'tensor' and tag != 'keras':
            # Check for V2 scalar summaries, which have a different PB
            # structure.
            if event.summary.value[
                0].metadata.plugin_data.plugin_name == 'scalars':
              container = result.scalars
          container.add(_ObservedSummary(logdir=dirpath, tag=tag))
  return result


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class TestTensorBoardV2(keras_parameterized.TestCase):

  def setUp(self):
    super(TestTensorBoardV2, self).setUp()
    self.logdir = os.path.join(self.get_temp_dir(), 'tb')
    self.train_dir = os.path.join(self.logdir, 'train')
    self.validation_dir = os.path.join(self.logdir, 'validation')

  def _get_model(self):
    layers = [
        keras.layers.Conv2D(8, (3, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(1)
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(10, 10, 1))
    opt = gradient_descent.SGD(learning_rate=0.001)
    model.compile(
        opt,
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  def test_TensorBoard_default_logdir(self):
    """Regression test for cross-platform pathsep in default logdir."""
    os.chdir(self.get_temp_dir())

    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard()  # no logdir specified

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])

    summary_file = list_summaries(logdir='.')
    train_dir = os.path.join('.', 'logs', 'train')
    validation_dir = os.path.join('.', 'logs', 'validation')
    self.assertEqual(
        summary_file.scalars, {
            _ObservedSummary(logdir=train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=validation_dir, tag='epoch_loss'),
        })

  def test_TensorBoard_basic(self):
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(self.logdir)

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])

    summary_file = list_summaries(self.logdir)
    self.assertEqual(
        summary_file.scalars, {
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
        })

  def test_TensorBoard_across_invocations(self):
    """Regression test for summary writer resource use-after-free.

    See: <https://github.com/tensorflow/tensorflow/issues/25707>
    """
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(self.logdir)

    for _ in (1, 2):
      model.fit(
          x,
          y,
          batch_size=2,
          epochs=2,
          validation_data=(x, y),
          callbacks=[tb_cbk])

    summary_file = list_summaries(self.logdir)
    self.assertEqual(
        summary_file.scalars, {
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
        })

  def test_TensorBoard_no_spurious_event_files(self):
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(self.logdir)

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        callbacks=[tb_cbk])

    events_file_run_basenames = set()
    for (dirpath, _, filenames) in os.walk(self.logdir):
      if any(fn.startswith('events.out.') for fn in filenames):
        events_file_run_basenames.add(os.path.basename(dirpath))
    self.assertEqual(events_file_run_basenames, {'train'})

  def test_TensorBoard_batch_metrics(self):
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(self.logdir, update_freq=1)

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])

    summary_file = list_summaries(self.logdir)
    self.assertEqual(
        summary_file.scalars,
        {
            _ObservedSummary(logdir=self.train_dir, tag='batch_loss'),
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
        },
    )

  def test_TensorBoard_weight_histograms(self):
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(self.logdir, histogram_freq=1)
    model_type = testing_utils.get_model_type()

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])
    summary_file = list_summaries(self.logdir)

    self.assertEqual(
        summary_file.scalars,
        {
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
        },
    )
    self.assertEqual(
        self._strip_layer_names(summary_file.histograms, model_type),
        {
            _ObservedSummary(logdir=self.train_dir, tag='bias_0'),
            _ObservedSummary(logdir=self.train_dir, tag='kernel_0'),
        },
    )

  def test_TensorBoard_weight_images(self):
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(
        self.logdir, histogram_freq=1, write_images=True)
    model_type = testing_utils.get_model_type()

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])
    summary_file = list_summaries(self.logdir)

    self.assertEqual(
        summary_file.scalars,
        {
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
        },
    )
    self.assertEqual(
        self._strip_layer_names(summary_file.histograms, model_type),
        {
            _ObservedSummary(logdir=self.train_dir, tag='bias_0'),
            _ObservedSummary(logdir=self.train_dir, tag='kernel_0'),
        },
    )
    self.assertEqual(
        self._strip_layer_names(summary_file.images, model_type),
        {
            _ObservedSummary(logdir=self.train_dir, tag='bias_0/image/0'),
            _ObservedSummary(logdir=self.train_dir, tag='kernel_0/image/0'),
            _ObservedSummary(logdir=self.train_dir, tag='kernel_0/image/1'),
            _ObservedSummary(logdir=self.train_dir, tag='kernel_0/image/2'),
        },
    )

  def test_TensorBoard_projector_callback(self):
    layers = [
        keras.layers.Embedding(10, 10, name='test_embedding'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(10,))
    model.compile(
        optimizer='adam',
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        run_eagerly=testing_utils.should_run_eagerly())
    x, y = np.ones((10, 10)), np.ones((10, 10))
    tb_cbk = keras.callbacks.TensorBoard(
        self.logdir,
        embeddings_freq=1,
        embeddings_metadata={'test_embedding': 'metadata.tsv'})

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])

    with open(os.path.join(self.logdir, 'projector_config.pbtxt')) as f:
      self.assertEqual(
          f.readlines(), [
              'embeddings {\n',
              '  tensor_name: "test_embedding/.ATTRIBUTES/VARIABLE_VALUE"\n',
              '  metadata_path: "metadata.tsv"\n',
              '}\n'])

  def test_custom_summary(self):
    if not context.executing_eagerly():
      self.skipTest('Custom summaries only supported in V2 code path.')

    def scalar_v2_mock(name, data, step=None):
      """A reimplementation of the scalar plugin to avoid circular deps."""
      metadata = summary_pb2.SummaryMetadata()
      # Should match value in tensorboard/plugins/scalar/metadata.py.
      metadata.plugin_data.plugin_name = 'scalars'
      with summary_ops_v2.summary_scope(
          name, 'scalar_summary', values=[data, step]) as (tag, _):
        return summary_ops_v2.write(
            tag=tag,
            tensor=math_ops.cast(data, 'float32'),
            step=step,
            metadata=metadata)

    class LayerWithSummary(keras.layers.Layer):

      def call(self, x):
        scalar_v2_mock('custom_summary', math_ops.reduce_sum(x))
        return x

    model = testing_utils.get_model_from_layers([LayerWithSummary()],
                                                input_shape=(5,),
                                                name='model')

    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    tb_cbk = keras.callbacks.TensorBoard(self.logdir, update_freq=1)
    x, y = np.ones((10, 5)), np.ones((10, 5))
    model.fit(x, y, batch_size=2, validation_data=(x, y), callbacks=[tb_cbk])
    summary_file = list_summaries(self.logdir)
    self.assertEqual(
        summary_file.scalars,
        {
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.train_dir, tag='batch_loss'),
            _ObservedSummary(
                logdir=self.train_dir,
                tag='model/layer_with_summary/custom_summary'),
            _ObservedSummary(
                logdir=self.validation_dir,
                tag='model/layer_with_summary/custom_summary')
        },
    )

  def _strip_layer_names(self, summaries, model_type):
    """Deduplicate summary names modulo layer prefix.

    This removes the first slash-component of each tag name: for
    instance, "foo/bar/baz" becomes "bar/baz".

    Args:
      summaries: A `set` of `_ObservedSummary` values.
      model_type: The model type currently being tested.

    Returns:
      A new `set` of `_ObservedSummary` values with layer prefixes
      removed.
    """
    result = set()
    for summary in summaries:
      if '/' not in summary.tag:
        raise ValueError('tag has no layer name: %r' % summary.tag)
      start_from = 2 if 'subclass' in model_type else 1
      new_tag = '/'.join(summary.tag.split('/')[start_from:])
      result.add(summary._replace(tag=new_tag))
    return result

  def test_TensorBoard_invalid_argument(self):
    with self.assertRaisesRegexp(ValueError, 'Unrecognized arguments'):
      keras.callbacks.TensorBoard(wwrite_images=True)

  def test_TensorBoard_non_blocking(self):
    model = keras.Sequential([keras.layers.Dense(1)])
    tb = keras.callbacks.TensorBoard(self.logdir)
    self.assertTrue(tb._supports_tf_logs)
    cb_list = keras.callbacks.CallbackList([tb],
                                           model=model,
                                           epochs=1,
                                           steps=100,
                                           verbose=0)

    tensor = ops.convert_to_tensor(1.)

    def mock_numpy():
      raise RuntimeError(
          'If this error is seen, TensorBoard is causing a blocking '
          'NumPy conversion.')

    with test.mock.patch.object(tensor, 'numpy', mock_numpy):
      logs = {'metric': tensor}

      cb_list.on_train_begin(logs)
      cb_list.on_epoch_begin(0, logs)
      cb_list.on_train_batch_begin(0, logs)
      cb_list.on_train_batch_end(0, logs)
      cb_list.on_epoch_end(0, logs)
      cb_list.on_train_end(logs)

      cb_list.on_test_begin(logs)
      cb_list.on_test_batch_begin(0, logs)
      cb_list.on_test_batch_end(0, logs)
      cb_list.on_test_end(logs)

      cb_list.on_predict_begin(logs)
      cb_list.on_predict_batch_begin(logs)
      cb_list.on_predict_batch_end(logs)
      cb_list.on_predict_end(logs)


# Note that this test specifies model_type explicitly.
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class TestTensorBoardV2NonParameterizedTest(keras_parameterized.TestCase):

  def setUp(self):
    super(TestTensorBoardV2NonParameterizedTest, self).setUp()
    self.logdir = os.path.join(self.get_temp_dir(), 'tb')
    self.train_dir = os.path.join(self.logdir, 'train')
    self.validation_dir = os.path.join(self.logdir, 'validation')

  def _get_seq_model(self):
    model = keras.models.Sequential([
        keras.layers.Conv2D(8, (3, 3), input_shape=(10, 10, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(1),
    ])
    opt = gradient_descent.SGD(learning_rate=0.001)
    model.compile(
        opt,
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  def _count_trace_file(self, logdir):
    profile_dir = os.path.join(logdir, 'plugins', 'profile')
    count = 0
    for (dirpath, dirnames, filenames) in os.walk(profile_dir):
      del dirpath  # unused
      del dirnames  # unused
      for filename in filenames:
        if filename.endswith('.trace.json.gz'):
          count += 1
    return count

  def fitModelAndAssertKerasModelWritten(self, model):
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(self.logdir,
                                         write_graph=True,
                                         profile_batch=0)
    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])
    summary_file = list_summaries(self.logdir)
    self.assertEqual(
        summary_file.tensors,
        {
            _ObservedSummary(logdir=self.train_dir, tag='keras'),
        },
    )

  def test_TensorBoard_writeSequentialModel_noInputShape(self):
    model = keras.models.Sequential([
        keras.layers.Conv2D(8, (3, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(1),
    ])
    model.compile('sgd', 'mse', run_eagerly=False)
    self.fitModelAndAssertKerasModelWritten(model)

  def test_TensorBoard_writeSequentialModel_withInputShape(self):
    model = keras.models.Sequential([
        keras.layers.Conv2D(8, (3, 3), input_shape=(10, 10, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(1),
    ])
    model.compile('sgd', 'mse', run_eagerly=False)
    self.fitModelAndAssertKerasModelWritten(model)

  def test_TensoriBoard_writeModel(self):
    inputs = keras.layers.Input([10, 10, 1])
    x = keras.layers.Conv2D(8, (3, 3), activation='relu')(inputs)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs=inputs, outputs=[x])
    model.compile('sgd', 'mse', run_eagerly=False)
    self.fitModelAndAssertKerasModelWritten(model)

  def test_TensorBoard_autoTrace(self):
    model = self._get_seq_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(
        self.logdir, histogram_freq=1, profile_batch=1, write_graph=False)

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])
    summary_file = list_summaries(self.logdir)

    self.assertEqual(
        summary_file.tensors,
        {
            _ObservedSummary(logdir=self.train_dir, tag=u'batch_1'),
        },
    )
    self.assertEqual(1, self._count_trace_file(logdir=self.train_dir))

  def test_TensorBoard_autoTrace_tagNameWithBatchNum(self):
    model = self._get_seq_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(
        self.logdir, histogram_freq=1, profile_batch=2, write_graph=False)

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])
    summary_file = list_summaries(self.logdir)

    self.assertEqual(
        summary_file.tensors,
        {
            _ObservedSummary(logdir=self.train_dir, tag=u'batch_2'),
        },
    )
    self.assertEqual(1, self._count_trace_file(logdir=self.train_dir))

  def test_TensorBoard_autoTrace_profileBatchRangeSingle(self):
    model = self._get_seq_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(
        self.logdir, histogram_freq=1, profile_batch='2,2', write_graph=False)

    model.fit(
        x,
        y,
        batch_size=3,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])
    summary_file = list_summaries(self.logdir)

    self.assertEqual(
        summary_file.tensors,
        {
            # Trace will be logged once at the batch it stops profiling.
            _ObservedSummary(logdir=self.train_dir, tag=u'batch_2'),
        },
    )
    self.assertEqual(1, self._count_trace_file(logdir=self.train_dir))

  def test_TensorBoard_autoTrace_profileBatchRangeTwice(self):
    model = self._get_seq_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(
        self.logdir, histogram_freq=1, profile_batch='10,10', write_graph=False)

    model.fit(
        x,
        y,
        batch_size=3,
        epochs=10,
        validation_data=(x, y),
        callbacks=[tb_cbk])

    time.sleep(1)  # Avoids the second profile over-writing the first.

    model.fit(
        x,
        y,
        batch_size=3,
        epochs=10,
        validation_data=(x, y),
        callbacks=[tb_cbk])
    self.assertEqual(2, self._count_trace_file(logdir=self.train_dir))

  # Test case that replicates a Github issue.
  # https://github.com/tensorflow/tensorflow/issues/37543
  def test_TensorBoard_autoTrace_profileTwiceGraphMode(self):
    ops.disable_eager_execution()
    inp = keras.Input((1,))
    out = keras.layers.Dense(units=1)(inp)
    model = keras.Model(inp, out)

    model.compile(gradient_descent.SGD(1), 'mse')

    logdir = os.path.join(self.get_temp_dir(), 'tb1')
    model.fit(
        np.zeros((64, 1)),
        np.zeros((64, 1)),
        batch_size=32,
        callbacks=[keras.callbacks.TensorBoard(logdir, profile_batch=1)],
    )
    # Verifies trace exists in the first logdir.
    self.assertEqual(1, self._count_trace_file(logdir=logdir))
    logdir = os.path.join(self.get_temp_dir(), 'tb2')
    model.fit(
        np.zeros((64, 1)),
        np.zeros((64, 1)),
        batch_size=32,
        callbacks=[keras.callbacks.TensorBoard(logdir, profile_batch=2)],
    )
    # Verifies trace exists in the second logdir.
    self.assertEqual(1, self._count_trace_file(logdir=logdir))

  def test_TensorBoard_autoTrace_profileBatchRange(self):
    model = self._get_seq_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(
        self.logdir, histogram_freq=1, profile_batch='1,3', write_graph=False)

    model.fit(
        x,
        y,
        batch_size=4,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])
    summary_file = list_summaries(self.logdir)

    self.assertEqual(
        summary_file.tensors,
        {
            # Trace will be logged once at the batch it stops profiling.
            _ObservedSummary(logdir=self.train_dir, tag=u'batch_3'),
        },
    )
    self.assertEqual(1, self._count_trace_file(logdir=self.train_dir))

  def test_TensorBoard_autoTrace_profileInvalidBatchRange(self):
    with self.assertRaises(ValueError):
      keras.callbacks.TensorBoard(
          self.logdir,
          histogram_freq=1,
          profile_batch='-1,3',
          write_graph=False)

    with self.assertRaises(ValueError):
      keras.callbacks.TensorBoard(
          self.logdir,
          histogram_freq=1,
          profile_batch='1,None',
          write_graph=False)

    with self.assertRaises(ValueError):
      keras.callbacks.TensorBoard(
          self.logdir, histogram_freq=1, profile_batch='6,5', write_graph=False)

    with self.assertRaises(ValueError):
      keras.callbacks.TensorBoard(
          self.logdir, histogram_freq=1, profile_batch=-1, write_graph=False)

  def test_TensorBoard_autoTrace_profile_batch_largerThanBatchCount(self):
    model = self._get_seq_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(
        self.logdir, histogram_freq=1, profile_batch=10000, write_graph=False)

    model.fit(
        x,
        y,
        batch_size=2,
        epochs=2,
        validation_data=(x, y),
        callbacks=[tb_cbk])
    summary_file = list_summaries(self.logdir)

    # Enabled trace only on the 10000th batch, thus it should be empty.
    self.assertEmpty(summary_file.tensors)
    self.assertEqual(0, self._count_trace_file(logdir=self.train_dir))


class MostRecentlyModifiedFileMatchingPatternTest(test.TestCase):

  def test_get_most_recently_modified_file_matching_pattern(self):
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    file_paths = [
        os.path.join(test_dir, file_name) for file_name in
        ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
    ]
    for file_path in file_paths:
      with open(file_path, 'w') as f:
        # Ensure there are some intervals between file creation.
        time.sleep(2)
        f.write('foo bar')
    # Ensure the files have been actually written.
    self.assertEqual(
        set([
            os.path.join(test_dir, file_name)
            for file_name in os.listdir(test_dir)
        ]), set(file_paths))
    self.assertEqual(
        keras.callbacks.ModelCheckpoint(None)
        ._get_most_recently_modified_file_matching_pattern(path_pattern),
        file_paths[-1])

  def test_some_file_not_matching_pattern(self):
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    file_paths = [
        os.path.join(test_dir, file_name) for file_name in
        ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.baatch01epoch01.h5']
    ]
    for file_path in file_paths:
      with open(file_path, 'w') as f:
        # Ensure there are some intervals between file creation.
        time.sleep(2)
        f.write('foo bar')
    self.assertEqual(
        keras.callbacks.ModelCheckpoint(None)
        ._get_most_recently_modified_file_matching_pattern(path_pattern),
        file_paths[-2])

  def test_get_same_file_if_file_name_equals_pattern(self):
    file_name = 'f.batch02.h5'
    test_dir = self.get_temp_dir()
    file_path = os.path.join(test_dir, file_name)
    with open(file_path, 'w') as f:
      f.write('foo bar')
    self.assertEqual(os.path.join(test_dir, os.listdir(test_dir)[0]), file_path)
    self.assertEqual(
        keras.callbacks.ModelCheckpoint(
            None)._get_most_recently_modified_file_matching_pattern(file_path),
        file_path)

  def test_get_none_if_file_does_not_exist(self):
    file_name = 'f.batch02.h5'
    test_dir = self.get_temp_dir()
    file_path = os.path.join(test_dir, file_name)
    self.assertLen(os.listdir(test_dir), 0)
    self.assertEqual(
        keras.callbacks.ModelCheckpoint(
            None)._get_most_recently_modified_file_matching_pattern(file_path),
        None)

  def test_using_checkpoint_management_latest_checkpoint(self):
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}'
    ckpt_file_name = 'f.batchXepochY'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    ckpt_file_path = os.path.join(test_dir, ckpt_file_name)
    with open(ckpt_file_path, 'w') as f:
      f.write('dummy ckpt')
    checkpoint_management.update_checkpoint_state_internal(
        test_dir, ckpt_file_path)

    file_paths = [
        os.path.join(test_dir, file_name)
        for file_name in ['f.batch03epoch02', 'f.batch02epoch02']
    ]
    for file_path in file_paths:
      with open(file_path, 'w') as f:
        f.write('foo bar')

    # The result returned from checkpoint_management.latest_checkpoint takes
    # priority, so even if it was written earlier, we should still return that.
    self.assertEqual(
        keras.callbacks.ModelCheckpoint(None)
        ._get_most_recently_modified_file_matching_pattern(path_pattern),
        ckpt_file_path)


if __name__ == '__main__':
  test.main()
