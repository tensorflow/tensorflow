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
"""Benchmark for examples on https://keras.io/examples."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.benchmarks import benchmark_util


_MAX_FEATURE = 20000
_MAX_LEN = 200

# Load common dataset.
(_IMDB_X, _IMDB_Y), _ = tf.keras.datasets.imdb.load_data(
    num_words=_MAX_FEATURE)
_IMDB_X = tf.keras.preprocessing.sequence.pad_sequences(
    _IMDB_X, maxlen=_MAX_LEN)


class KerasExamplesBenchmark(tf.test.Benchmark):
  """Required Arguments for measure_performance:

      x: Input data, it could be Numpy or load from tfds.
      y: Target data. If `x` is a dataset, generator instance,
         `y` should not be specified.
      loss: Loss function for model.
      optimizer: Optimizer for model.
      Other details can see in `measure_performance()` method of
      benchmark_util.
  """

  def _lstm_imdb_model(self):
    """LSTM model from https://keras.io/examples/nlp/bidirectional_lstm_imdb/."""
    inputs = tf.keras.Input(shape=(None,), dtype='int32')
    x = tf.keras.layers.Embedding(_MAX_FEATURE, 128)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

  def benchmark_bidirect_lstm_imdb_bs_32(self):
    """Benchmark for Bidirectional LSTM on IMDB,
       batch_size=32, run_iters=1."""
    batch_size = 32
    run_iters = 1
    results = benchmark_util.measure_performance(
        self._lstm_imdb_model,
        x=_IMDB_X,
        y=_IMDB_Y,
        batch_size=batch_size,
        run_iters=run_iters,
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    self.report_benchmark(
        iters=run_iters, wall_time=results['wall_time'], extras=results)

  def benchmark_bidirect_lstm_imdb_bs_64(self):
    """Benchmark for Bidirectional LSTM on IMDB,
       batch_size=64, run_iters=3."""
    batch_size = 64
    run_iters = 3
    results = benchmark_util.measure_performance(
        self._lstm_imdb_model,
        x=_IMDB_X,
        y=_IMDB_Y,
        batch_size=batch_size,
        run_iters=run_iters,
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    self.report_benchmark(
        iters=run_iters, wall_time=results['wall_time'], extras=results)

  def benchmark_bidirect_lstm_imdb_bs_128(self):
    """Benchmark for Bidirectional LSTM on IMDB,
       batch_size=128, run_iters=2."""
    batch_size = 128
    run_iters = 2
    results = benchmark_util.measure_performance(
        self._lstm_imdb_model,
        x=_IMDB_X,
        y=_IMDB_Y,
        batch_size=batch_size,
        run_iters=run_iters,
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    self.report_benchmark(
        iters=run_iters, wall_time=results['wall_time'], extras=results)

  def benchmark_bidirect_lstm_imdb_bs_256(self):
    """Benchmark for Bidirectional LSTM on IMDB,
       batch_size: 256, run_iters=3."""
    batch_size = 256
    run_iters = 3
    results = benchmark_util.measure_performance(
        self._lstm_imdb_model,
        x=_IMDB_X,
        y=_IMDB_Y,
        batch_size=batch_size,
        run_iters=run_iters,
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    self.report_benchmark(
        iters=run_iters, wall_time=results['wall_time'], extras=results)

  def benchmark_bidirect_lstm_imdb_bs_512(self):
    """Benchmark for Bidirectional LSTM on IMDB,
       batch_size=512, run_iters=2."""
    batch_size = 512
    run_iters = 2
    results = benchmark_util.measure_performance(
        self._lstm_imdb_model,
        x=_IMDB_X,
        y=_IMDB_Y,
        batch_size=batch_size,
        run_iters=run_iters,
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    self.report_benchmark(
        iters=run_iters, wall_time=results['wall_time'], extras=results)


if __name__ == '__main__':
  tf.test.main()
