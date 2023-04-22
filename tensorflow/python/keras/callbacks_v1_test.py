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

import os
import shutil
import tempfile

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import ops
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import callbacks_v1
from tensorflow.python.keras import combinations
from tensorflow.python.keras import layers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.platform import test
from tensorflow.python.training import adam


TRAIN_SAMPLES = 10
TEST_SAMPLES = 10
NUM_CLASSES = 2
INPUT_DIM = 3
NUM_HIDDEN = 5
BATCH_SIZE = 5


class TestTensorBoardV1(test.TestCase, parameterized.TestCase):

  def test_TensorBoard(self):
    np.random.seed(1337)

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    def data_generator(train):
      if train:
        max_batch_index = len(x_train) // BATCH_SIZE
      else:
        max_batch_index = len(x_test) // BATCH_SIZE
      i = 0
      while 1:
        if train:
          yield (x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                 y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        else:
          yield (x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                 y_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        i += 1
        i %= max_batch_index

    # case: Sequential
    with ops.Graph().as_default(), self.cached_session():
      model = sequential.Sequential()
      model.add(
          layers.Dense(
              NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
      # non_trainable_weights: moving_variance, moving_mean
      model.add(layers.BatchNormalization())
      model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
      tsb = callbacks_v1.TensorBoard(
          log_dir=temp_dir,
          histogram_freq=1,
          write_images=True,
          write_grads=True,
          batch_size=5)
      cbks = [tsb]

      # fit with validation data
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=3,
          verbose=0)

      # fit with validation data and accuracy
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=0)

      # fit generator with validation data
      model.fit_generator(
          data_generator(True),
          len(x_train),
          epochs=2,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          verbose=0)

      # fit generator without validation data
      # histogram_freq must be zero
      tsb.histogram_freq = 0
      model.fit_generator(
          data_generator(True),
          len(x_train),
          epochs=2,
          callbacks=cbks,
          verbose=0)

      # fit generator with validation data and accuracy
      tsb.histogram_freq = 1
      model.fit_generator(
          data_generator(True),
          len(x_train),
          epochs=2,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          verbose=0)

      # fit generator without validation data and accuracy
      tsb.histogram_freq = 0
      model.fit_generator(
          data_generator(True), len(x_train), epochs=2, callbacks=cbks)
      assert os.path.exists(temp_dir)

  def test_TensorBoard_multi_input_output(self):
    np.random.seed(1337)
    tmpdir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

    with ops.Graph().as_default(), self.cached_session():
      filepath = os.path.join(tmpdir, 'logs')

      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)

      def data_generator(train):
        if train:
          max_batch_index = len(x_train) // BATCH_SIZE
        else:
          max_batch_index = len(x_test) // BATCH_SIZE
        i = 0
        while 1:
          if train:
            # simulate multi-input/output models
            yield ([x_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]] * 2,
                   [y_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]] * 2)
          else:
            yield ([x_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]] * 2,
                   [y_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]] * 2)
          i += 1
          i %= max_batch_index

      inp1 = input_layer.Input((INPUT_DIM,))
      inp2 = input_layer.Input((INPUT_DIM,))
      inp = layers.add([inp1, inp2])
      hidden = layers.Dense(2, activation='relu')(inp)
      hidden = layers.Dropout(0.1)(hidden)
      output1 = layers.Dense(NUM_CLASSES, activation='softmax')(hidden)
      output2 = layers.Dense(NUM_CLASSES, activation='softmax')(hidden)
      model = training.Model([inp1, inp2], [output1, output2])
      model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])

      # we must generate new callbacks for each test, as they aren't stateless
      def callbacks_factory(histogram_freq):
        return [
            callbacks_v1.TensorBoard(
                log_dir=filepath,
                histogram_freq=histogram_freq,
                write_images=True,
                write_grads=True,
                batch_size=5)
        ]

      # fit without validation data
      model.fit([x_train] * 2, [y_train] * 2, batch_size=BATCH_SIZE,
                callbacks=callbacks_factory(histogram_freq=0), epochs=3)

      # fit with validation data and accuracy
      model.fit([x_train] * 2, [y_train] * 2, batch_size=BATCH_SIZE,
                validation_data=([x_test] * 2, [y_test] * 2),
                callbacks=callbacks_factory(histogram_freq=1), epochs=2)

      # fit generator without validation data
      model.fit_generator(data_generator(True), len(x_train), epochs=2,
                          callbacks=callbacks_factory(histogram_freq=0))

      # fit generator with validation data and accuracy
      model.fit_generator(data_generator(True), len(x_train), epochs=2,
                          validation_data=([x_test] * 2, [y_test] * 2),
                          callbacks=callbacks_factory(histogram_freq=1))
      assert os.path.isdir(filepath)

  def test_Tensorboard_histogram_summaries_in_test_function(self):

    class FileWriterStub(object):

      def __init__(self, logdir, graph=None):
        self.logdir = logdir
        self.graph = graph
        self.steps_seen = []

      def add_summary(self, summary, global_step):
        summary_obj = summary_pb2.Summary()

        # ensure a valid Summary proto is being sent
        if isinstance(summary, bytes):
          summary_obj.ParseFromString(summary)
        else:
          assert isinstance(summary, summary_pb2.Summary)
          summary_obj = summary

        # keep track of steps seen for the merged_summary op,
        # which contains the histogram summaries
        if len(summary_obj.value) > 1:
          self.steps_seen.append(global_step)

      def flush(self):
        pass

      def close(self):
        pass

    def _init_writer(obj, _):
      obj.writer = FileWriterStub(obj.log_dir)

    np.random.seed(1337)
    tmpdir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    with ops.Graph().as_default(), self.cached_session():
      model = sequential.Sequential()
      model.add(
          layers.Dense(
              NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
      # non_trainable_weights: moving_variance, moving_mean
      model.add(layers.BatchNormalization())
      model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
      callbacks_v1.TensorBoard._init_writer = _init_writer
      tsb = callbacks_v1.TensorBoard(
          log_dir=tmpdir,
          histogram_freq=1,
          write_images=True,
          write_grads=True,
          batch_size=5)
      cbks = [tsb]

      # fit with validation data
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=3,
          verbose=0)

      self.assertAllEqual(tsb.writer.steps_seen, [0, 1, 2, 3, 4, 5])

  def test_Tensorboard_histogram_summaries_with_generator(self):
    np.random.seed(1337)
    tmpdir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

    def generator():
      x = np.random.randn(10, 100).astype(np.float32)
      y = np.random.randn(10, 10).astype(np.float32)
      while True:
        yield x, y

    with ops.Graph().as_default(), self.cached_session():
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=10, num_classes=10, input_dim=100)
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
      tsb = callbacks_v1.TensorBoard(
          log_dir=tmpdir,
          histogram_freq=1,
          write_images=True,
          write_grads=True,
          batch_size=5)
      cbks = [tsb]

      # fit with validation generator
      model.fit_generator(
          generator(),
          steps_per_epoch=2,
          epochs=2,
          validation_data=generator(),
          validation_steps=2,
          callbacks=cbks,
          verbose=0)

      with self.assertRaises(ValueError):
        # fit with validation generator but no
        # validation_steps
        model.fit_generator(
            generator(),
            steps_per_epoch=2,
            epochs=2,
            validation_data=generator(),
            callbacks=cbks,
            verbose=0)

      self.assertTrue(os.path.exists(tmpdir))

  def test_TensorBoard_with_ReduceLROnPlateau(self):
    with self.cached_session():
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

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
          loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

      cbks = [
          callbacks.ReduceLROnPlateau(
              monitor='val_loss', factor=0.5, patience=4, verbose=1),
          callbacks_v1.TensorBoard(log_dir=temp_dir)
      ]

      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=0)

      assert os.path.exists(temp_dir)

  def test_Tensorboard_batch_logging(self):

    class FileWriterStub(object):

      def __init__(self, logdir, graph=None):
        self.logdir = logdir
        self.graph = graph
        self.batches_logged = []
        self.summary_values = []
        self.summary_tags = []

      def add_summary(self, summary, step):
        self.summary_values.append(summary.value[0].simple_value)
        self.summary_tags.append(summary.value[0].tag)
        self.batches_logged.append(step)

      def flush(self):
        pass

      def close(self):
        pass

    with ops.Graph().as_default():
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

      tb_cbk = callbacks_v1.TensorBoard(temp_dir, update_freq='batch')
      tb_cbk.writer = FileWriterStub(temp_dir)

      for batch in range(5):
        tb_cbk.on_batch_end(batch, {'acc': batch})
      self.assertEqual(tb_cbk.writer.batches_logged, [0, 1, 2, 3, 4])
      self.assertEqual(tb_cbk.writer.summary_values, [0., 1., 2., 3., 4.])
      self.assertEqual(tb_cbk.writer.summary_tags, ['batch_acc'] * 5)

  def test_Tensorboard_epoch_and_batch_logging(self):

    class FileWriterStub(object):

      def __init__(self, logdir, graph=None):
        self.logdir = logdir
        self.graph = graph

      def add_summary(self, summary, step):
        if 'batch_' in summary.value[0].tag:
          self.batch_summary = (step, summary)
        elif 'epoch_' in summary.value[0].tag:
          self.epoch_summary = (step, summary)

      def flush(self):
        pass

      def close(self):
        pass

    with ops.Graph().as_default():
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

      tb_cbk = callbacks_v1.TensorBoard(temp_dir, update_freq='batch')
      tb_cbk.writer = FileWriterStub(temp_dir)

      tb_cbk.on_batch_end(0, {'acc': 5.0})
      tb_cbk.on_train_end()
      batch_step, batch_summary = tb_cbk.writer.batch_summary
      self.assertEqual(batch_step, 0)
      self.assertEqual(batch_summary.value[0].simple_value, 5.0)

      tb_cbk = callbacks_v1.TensorBoard(temp_dir, update_freq='epoch')
      tb_cbk.writer = FileWriterStub(temp_dir)
      tb_cbk.on_epoch_end(0, {'acc': 10.0})
      tb_cbk.on_train_end()
      epoch_step, epoch_summary = tb_cbk.writer.epoch_summary
      self.assertEqual(epoch_step, 0)
      self.assertEqual(epoch_summary.value[0].simple_value, 10.0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_Tensorboard_eager(self):
    temp_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

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
        loss='binary_crossentropy',
        optimizer=adam.AdamOptimizer(0.01),
        metrics=['accuracy'])

    cbks = [callbacks_v1.TensorBoard(log_dir=temp_dir)]

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=2,
        verbose=0)

    self.assertTrue(os.path.exists(temp_dir))

  def test_TensorBoard_update_freq(self):

    class FileWriterStub(object):

      def __init__(self, logdir, graph=None):
        self.logdir = logdir
        self.graph = graph
        self.batch_summaries = []
        self.epoch_summaries = []

      def add_summary(self, summary, step):
        if 'batch_' in summary.value[0].tag:
          self.batch_summaries.append((step, summary))
        elif 'epoch_' in summary.value[0].tag:
          self.epoch_summaries.append((step, summary))

      def flush(self):
        pass

      def close(self):
        pass

    with ops.Graph().as_default():
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

      # Epoch mode
      tb_cbk = callbacks_v1.TensorBoard(temp_dir, update_freq='epoch')
      tb_cbk.writer = FileWriterStub(temp_dir)

      tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 1})
      self.assertEqual(tb_cbk.writer.batch_summaries, [])
      tb_cbk.on_epoch_end(0, {'acc': 10.0, 'size': 1})
      self.assertLen(tb_cbk.writer.epoch_summaries, 1)
      tb_cbk.on_train_end()

      # Batch mode
      tb_cbk = callbacks_v1.TensorBoard(temp_dir, update_freq='batch')
      tb_cbk.writer = FileWriterStub(temp_dir)

      tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 1})
      self.assertLen(tb_cbk.writer.batch_summaries, 1)
      tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 1})
      self.assertLen(tb_cbk.writer.batch_summaries, 2)
      self.assertFalse(tb_cbk.writer.epoch_summaries)
      tb_cbk.on_train_end()

      # Integer mode
      tb_cbk = callbacks_v1.TensorBoard(temp_dir, update_freq=20)
      tb_cbk.writer = FileWriterStub(temp_dir)

      tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 10})
      self.assertFalse(tb_cbk.writer.batch_summaries)
      tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 10})
      self.assertLen(tb_cbk.writer.batch_summaries, 1)
      tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 10})
      self.assertLen(tb_cbk.writer.batch_summaries, 1)
      tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 10})
      self.assertLen(tb_cbk.writer.batch_summaries, 2)
      tb_cbk.on_batch_end(0, {'acc': 10.0, 'size': 10})
      self.assertLen(tb_cbk.writer.batch_summaries, 2)
      self.assertFalse(tb_cbk.writer.epoch_summaries)
      tb_cbk.on_train_end()


if __name__ == '__main__':
  test.main()
