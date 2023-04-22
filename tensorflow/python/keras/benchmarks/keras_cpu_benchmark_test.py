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

import numpy as np

import tensorflow as tf

from tensorflow.python.keras.benchmarks import benchmark_util
from tensorflow.python.platform import benchmark   # pylint: disable=unused-import

# Loss function and optimizer.
_LOSS = 'binary_crossentropy'
_OPTIMIZER = 'rmsprop'


class KerasModelCPUBenchmark(  # pylint: disable=undefined-variable
    tf.test.Benchmark, metaclass=benchmark.ParameterizedBenchmark):
  """Required Arguments for measure_performance.

      x: Input data, it could be Numpy or load from tfds.
      y: Target data. If `x` is a dataset, generator instance,
         `y` should not be specified.
      loss: Loss function for model.
      optimizer: Optimizer for model.
      Other details can see in `measure_performance()` method of
      benchmark_util.
  """
  # The parameters of each benchmark is a tuple:

  # (benchmark_name_suffix, batch_size, run_iters).
  # benchmark_name_suffix: The suffix of the benchmark test name with
  # convention `{bs}_{batch_size}`.
  # batch_size: Integer. Number of samples per gradient update.
  # run_iters: Integer. Number of iterations to run the
  # performance measurement.

  _benchmark_parameters = [
      ('bs_32', 32, 3), ('bs_64', 64, 2), ('bs_128', 128, 2),
      ('bs_256', 256, 1), ('bs_512', 512, 1)]

  def _mnist_mlp(self):
    """Simple MLP model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

  def _mnist_convnet(self):
    """Simple Convnet model."""
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
    """Simple LSTM model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(20000, 128))
    model.add(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

  def benchmark_mnist_mlp(self, batch_size, run_iters):
    """Benchmark for MLP model on synthetic mnist data."""
    mlp_x = np.random.random((5000, 784))
    mlp_y = np.random.random((5000, 10))
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._mnist_mlp,
        x=mlp_x,
        y=mlp_y,
        batch_size=batch_size,
        run_iters=run_iters,
        optimizer=_OPTIMIZER,
        loss=_LOSS)
    self.report_benchmark(
        iters=run_iters, wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_mnist_convnet(self, batch_size, run_iters):
    """Benchmark for Convnet model on synthetic mnist data."""
    convnet_x = np.random.random((5000, 28, 28, 1))
    convnet_y = np.random.random((5000, 10))
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._mnist_convnet,
        x=convnet_x,
        y=convnet_y,
        batch_size=batch_size,
        run_iters=run_iters,
        optimizer=_OPTIMIZER,
        loss=_LOSS)
    self.report_benchmark(
        iters=run_iters, wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_imdb_lstm(self, batch_size, run_iters):
    """Benchmark for LSTM model on synthetic imdb review dataset."""
    lstm_x = np.random.randint(0, 1999, size=(2500, 100))
    lstm_y = np.random.random((2500, 1))
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._imdb_lstm,
        x=lstm_x,
        y=lstm_y,
        batch_size=batch_size,
        run_iters=run_iters,
        optimizer=_OPTIMIZER,
        loss=_LOSS)
    self.report_benchmark(
        iters=run_iters, wall_time=wall_time, metrics=metrics, extras=extras)


if __name__ == '__main__':
  tf.test.main()
