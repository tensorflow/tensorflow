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
"""Benchmarks on Bidirectional LSTM on IMDB."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.benchmarks import benchmark_util


class BidirectionalLSTMBenchmark(tf.test.Benchmark):
  """Benchmarks for Bidirectional LSTM using `tf.test.Benchmark`."""

  def __init__(self):
    super(BidirectionalLSTMBenchmark, self).__init__()
    self.max_feature = 20000
    self.max_len = 200
    (self.imdb_x, self.imdb_y), _ = tf.keras.datasets.imdb.load_data(
        num_words=self.max_feature)
    self.imdb_x = tf.keras.preprocessing.sequence.pad_sequences(
        self.imdb_x, maxlen=self.max_len)

  def _build_model(self):
    """Model from https://keras.io/examples/nlp/bidirectional_lstm_imdb/."""
    inputs = tf.keras.Input(shape=(None,), dtype='int32')
    x = tf.keras.layers.Embedding(self.max_feature, 128)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True))(
            x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

  # In each benchmark test, the required arguments for the
  # method `measure_performance` include:
  #   x: Input data, it could be Numpy or loaded from tfds.
  #   y: Target data. If `x` is a dataset or generator instance,
  #      `y` should not be specified.
  #   loss: Loss function for model.
  #   optimizer: Optimizer for model.
  #   Check more details in `measure_performance()` method of
  #   benchmark_util.
  def benchmark_bidirect_lstm_imdb_bs_128(self):
    """Measure performance with batch_size=128."""
    batch_size = 128
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.imdb_x,
        y=self.imdb_y,
        batch_size=batch_size,
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata(
        'bidirectional_lstm', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_bidirect_lstm_imdb_bs_256(self):
    """Measure performance with batch_size=256."""
    batch_size = 256
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.imdb_x,
        y=self.imdb_y,
        batch_size=batch_size,
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata(
        'bidirectional_lstm', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_bidirect_lstm_imdb_bs_512(self):
    """Measure performance with batch_size=512."""
    batch_size = 512
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.imdb_x,
        y=self.imdb_y,
        batch_size=batch_size,
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata(
        'bidirectional_lstm', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_bidirect_lstm_imdb_bs_512_gpu_2(self):
    """Measure performance with batch_size=512, gpu=2 and

    distribution_strategy=`mirrored`.
    """
    batch_size = 512
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.imdb_x,
        y=self.imdb_y,
        batch_size=batch_size,
        num_gpus=2,
        distribution_strategy='mirrored',
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata(
        'bidirectional_lstm', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)


if __name__ == '__main__':
  tf.test.main()
