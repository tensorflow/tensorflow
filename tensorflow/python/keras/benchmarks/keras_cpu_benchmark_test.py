# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark tests for CPU performance of Keras models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit

import numpy as np
import six

import tensorflow as tf

from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

_NUM_EPOCHS = 4

# Dataset for benchmark
_MLP_X = np.random.random((5000, 784))
_MLP_Y = np.random.random((5000, 10))

_CONVNET_X = np.random.random((5000, 28, 28, 1))
_CONVNET_Y = np.random.random((5000, 10))

_LSTM_X = np.random.randint(0, 1999, size=(2500, 100))
_LSTM_Y = np.random.random((2500, 1))


class TimerCallback(tf.keras.callbacks.Callback):

  def __init__(self):
    self.times = []
    self.timer = timeit.default_timer
    self.startup_time = timeit.default_timer()
    self.recorded_startup = False

  def on_epoch_begin(self, e, logs):
    self.epoch_start_time = self.timer()

  def on_batch_end(self, e, logs):
    if not self.recorded_startup:
      self.startup_time = self.timer() - self.startup_time
      self.recorded_startup = True

  def on_epoch_end(self, e, logs):
    self.times.append(self.timer() - self.epoch_start_time)


class KerasModelCPUBenchmark(
    six.with_metaclass(benchmark.ParameterizedBenchmark, test.Benchmark)):

  # Set parameters for paramerized benchmark.
  _benchmark_parameters = [
      ('bs_32', 32, 3), ('bs_64', 64, 2), ('bs_128', 128, 2),
      ('bs_256', 256, 1), ('bs_512', 512, 1)]

  def _measure_performance(self, model_fn, x, y, batch_size=32,
                           run_iters=4):
    build_time_list, compile_time_list, startup_time_list = [], [], []
    avg_epoch_time_list, wall_time_list, exp_per_sec_list = [], [], []
    total_num_examples = y.shape[0] * _NUM_EPOCHS

    for _ in range(run_iters):
      timer = timeit.default_timer
      t0 = timer()
      model = model_fn()
      build_time = timer() - t0

      t1 = timer()
      model.compile('rmsprop', 'binary_crossentropy')
      compile_time = timer() - t1

      cbk = TimerCallback()
      t2 = timer()
      model.fit(x, y, epochs=_NUM_EPOCHS, batch_size=batch_size,
                callbacks=[cbk], verbose=0)
      end_time = timer()

      build_time_list.append(build_time)
      compile_time_list.append(compile_time)
      startup_time_list.append(cbk.startup_time)
      avg_epoch_time_list.append(np.mean(cbk.times[1:]))
      wall_time_list.append(end_time - t0)
      exp_per_sec_list.append(total_num_examples / (end_time - t2))

    results = {'build_time': np.mean(build_time_list),
               'compile_time': np.mean(compile_time_list),
               'startup_time': np.mean(startup_time_list),
               'avg_epoch_time': np.mean(avg_epoch_time_list),
               'wall_time': np.mean(wall_time_list),
               'exp_per_sec': np.mean(exp_per_sec_list)}

    self.report_benchmark(
        iters=_NUM_EPOCHS,
        wall_time=results['wall_time'],
        extras=results)

  def _mnist_mlp(self):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

  def _mnist_convnet(self):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

  def _imdb_lstm(self):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(20000, 128))
    model.add(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

  def benchmark_mnist_mlp(self, batch_size, run_iters):
    self._measure_performance(self._mnist_mlp, _MLP_X, _MLP_Y,
                              batch_size=batch_size, run_iters=run_iters)

  def benchmark_mnist_convnet(self, batch_size, run_iters):
    self._measure_performance(self._mnist_convnet, _CONVNET_X, _CONVNET_Y,
                              batch_size=batch_size, run_iters=run_iters)

  def benchmark_imdb_lstm(self, batch_size, run_iters):
    self._measure_performance(self._imdb_lstm, _LSTM_X, _LSTM_Y,
                              batch_size=batch_size, run_iters=run_iters)


if __name__ == '__main__':
  test.main()
