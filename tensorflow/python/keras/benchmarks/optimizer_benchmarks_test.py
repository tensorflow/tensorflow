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
"""Benchmark tests for Keras optimizers."""

import tensorflow as tf

from tensorflow.python.keras.benchmarks import benchmark_util
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.platform.benchmark import ParameterizedBenchmark


def bidirect_imdb_lstm_config():
  """Bidirectional LSTM model and IMDB data."""

  def model_fn():
    inputs = tf.keras.Input(shape=(None,), dtype="int32")
    x = tf.keras.layers.Embedding(20000, 128)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True))(
            x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

  (x_train, y_train), _ = tf.keras.datasets.imdb.load_data(num_words=20000)
  x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)

  return model_fn, x_train, y_train


class KerasOptimizerBenchmark(
    tf.test.Benchmark, metaclass=ParameterizedBenchmark):
  """Keras optimizer benchmarks."""

  # The parameter of each benchmark test is a tuple, and the first one is
  # the optimizer name.
  _benchmark_parameters = benchmark_util.generate_benchmark_params_cpu_gpu([
      ("Adam", tf.keras.optimizers.Adam(), 10),
      ("NonFusedAdam", adam.NonFusedAdam(), 10),
  ])

  def benchmark_optimizer(self, optimizer, num_iters):
    """Optimizer benchmark with Bidirectional LSTM model on IMDB data.

    Args:
      optimizer: The optimizer instance to be benchmarked.
      num_iters: The number of iterations to run for performance measurement.
    """
    model, train_x, train_y = bidirect_imdb_lstm_config()
    metrics, wall_time, extras = benchmark_util.measure_performance(
        model,
        x=train_x,
        y=train_y,
        batch_size=512,
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"])
    name = benchmark_util.get_benchmark_name(self._get_name())
    metadata = {
        "implementation": name[0],
        "model_name": "optimizers",
        "parameters": "lstm.512",
    }
    extras.update(metadata)
    self.report_benchmark(
        iters=num_iters, wall_time=wall_time, metrics=metrics, extras=extras)


if __name__ == "__main__":
  tf.test.main()
