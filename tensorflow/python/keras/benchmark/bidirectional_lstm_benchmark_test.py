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

import six

from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import preprocessing
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.benchmark import benchmark_util
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

_MAX_FEATURE = 20000
_MAX_LEN = 200


class BidirectLSTMBenchmark(
    six.with_metaclass(benchmark.ParameterizedBenchmark, test.Benchmark)):
  # Set different batch_size and iteration as needed.
  _benchmark_parameters = [
      ('bs_32', 32, 2), ('bs_64', 64, 2), ('bs_128', 128, 1),
      ('bs_256', 256, 1), ('bs_512', 512, 3)]

  # Built same model in https://keras.io/examples/nlp/bidirectional_lstm_imdb/
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

  def set_config(self, batch_size, run_iters):
    test_config = {
      'batch_size': batch_size,
      'run_iters': run_iters,
      'epoch': 3,
      'optimizer': 'adam',
      'loss': 'binary_crossentropy',
      'metrics': ['accuracy'],
      'warmup_epoch': 1,
      'distribution_strategy': 'mirrored',
      'num_gpus': 2,
    }
    return test_config

  # load real data
  def benchmark_bidirect_lstm_imdb(self, batch_size, run_iters):
    (x_train, y_train), (x_val, y_val) = imdb.load_data(
        num_words=_MAX_FEATURE
    )
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=_MAX_LEN)
    x_val = preprocessing.sequence.pad_sequences(x_val, maxlen=_MAX_LEN)
    test_config = self.set_config(batch_size, run_iters)
    results = benchmark_util._measure_performance(self._model, x_train, y_train,
                                                  x_val, y_val,
                                                  test_config=test_config)
    self.report_benchmark(
        iters=test_config['epoch'],
        wall_time=results['wall_time'],
        extras=results)


if __name__ == '__main__':
  test.main()
