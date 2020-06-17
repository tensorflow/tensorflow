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
"""Benchmark for Bidirectional LSTM on IMDB"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit

import numpy as np
import six

from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import preprocessing
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

_NUM_EPOCHS = 2
_MAX_FEATURE = 20000
_MAX_LEN = 200


class TimerCallBack(callbacks.Callback):
  def __init__(self):
    self.times = []
    self.timer = timeit.default_timer
    self.startup_time = timeit.default_timer()
    self.recorded_startup = False

  def on_epoch_begin(self, e, logs):
    self.epoch_start_time = self.timer()

  def on_epoch_end(self, e, logs):
    self.times.append(self.timer() - self.epoch_start_time)

  def on_batch_end(self, e, logs):
    if not self.recorded_startup:
      self.startup_time = self.timer() - self.startup_time
      self.recorded_startup = True


class BidirectLSTMBenchmark(
    six.with_metaclass(benchmark.ParameterizedBenchmark, test.Benchmark)):
  # Set different batch_size and iteration as needed.
  _benchmark_parameters = [
      ('bs_32', 32, 2), ('bs_64', 64, 2), ('bs_128', 128, 1),
      ('bs_256', 256, 1), ('bs_512', 512, 3)]

  def _measure_performance(self, model_fn, x_train, y_train, x_val, y_val,
                           batch_size=32, run_iters=4):
    build_time_list, compile_time_list, startup_time_list = [], [], []
    avg_epoch_time_list, wall_time_list, exp_per_sec_list = [], [], []
    total_num_examples = (y_train.shape[0] + y_val.shape[0]) * _NUM_EPOCHS

    for _ in range(run_iters):
      timer = timeit.default_timer
      t0 = timer()
      model = model_fn()
      build_time = timer() - t0

      t1 = timer()
      model.compile(
          optimizer='adam',
          loss="binary_crossentropy",
          metrics=["accuracy"],
      )
      compile_time = timer() - t1

      cbk = TimerCallBack()
      t2 = timer()
      model.fit(x_train,
                y_train,
                batch_size=batch_size,
                epochs=_NUM_EPOCHS,
                callbacks=[cbk],
                verbose=0,
                validation_data=(x_val, y_val))
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

  def _model(self):
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(_MAX_FEATURE, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model

  # real data
  def benchmark_bidirect_lstm_imdb(self, batch_size, run_iters):
    (x_train, y_train), (x_val, y_val) = imdb.load_data(
        num_words=_MAX_FEATURE
    )
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=_MAX_LEN)
    x_val = preprocessing.sequence.pad_sequences(x_val, maxlen=_MAX_LEN)
    self._measure_performance(self._model, x_train, y_train, x_val, y_val,
                              batch_size=batch_size, run_iters=run_iters)


if __name__ == '__main__':
  test.main()
