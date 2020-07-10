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

import six

import tensorflow as tf

from tensorflow.python.platform import benchmark
from tensorflow.python.keras.benchmarks import benchmark_util


class BidirectionalLSTMBenchmark(
    six.with_metaclass(benchmark.ParameterizedBenchmark, tf.test.Benchmark)):
  """Benchmarks for Bidirectional LSTM using `ParameterizedBenchmark`."""
  def __init__(self):
    super(BidirectionalLSTMBenchmark, self).__init__()
    self.max_feature = 20000
    self.max_len = 200
    (self.imdb_x, self.imdb_y), _ = tf.keras.datasets.imdb.load_data(
        num_words=self.max_feature)
    self.imdb_x = tf.keras.preprocessing.sequence.pad_sequences(
        self.imdb_x, maxlen=self.max_len)

  """The parameters of each benchmark is a tuple:

     (benchmark_name_suffix, batch_size, run_iters).
     benchmark_name_suffix: The suffix of the benchmark test name with
     convention `{bs}_{batch_size}`.
     batch_size: Integer. Number of samples per gradient update.
     run_iters: Integer. Number of iterations to run the
         performance measurement.
  """
  _benchmark_parameters = [
      ('bs_32', 32, 2), ('bs_64', 64, 2),
      ('bs_128', 128, 1), ('bs_256', 256, 1),
      ('bs_512', 512, 3)]

  def _build_model(self):
    """model from https://keras.io/examples/nlp/bidirectional_lstm_imdb/"""
    inputs = tf.keras.Input(shape=(None,), dtype='int32')
    x = tf.keras.layers.Embedding(self.max_feature, 128)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

  def benchmark_bidirect_lstm_imdb(self, batch_size, run_iters):
    """ Required Arguments for measure_performance:

      x: Input data, it could be Numpy or load from tfds.
      y: Target data. If `x` is a dataset, generator instance,
         `y` should not be specified.
      loss: Loss function for model.
      optimizer: Optimizer for model.
      Other details can see in `measure_performance()` method of
      benchmark_util.
    """
    results = benchmark_util.measure_performance(
        self._build_model,
        x=self.imdb_x,
        y=self.imdb_y,
        batch_size=batch_size,
        run_iters=run_iters,
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    self.report_benchmark(
        iters=run_iters, wall_time=results['wall_time'], extras=results)


if __name__ == '__main__':
  tf.test.main()
