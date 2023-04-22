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
"""Benchmarks on MLP on Reuters dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow.python.keras.benchmarks import benchmark_util


class MLPReutersBenchmark(tf.test.Benchmark):
  """Benchmarks for MLP using `tf.test.Benchmark`."""

  def __init__(self):
    super(MLPReutersBenchmark, self).__init__()
    self.max_words = 1000
    (self.x_train, self.y_train), _ = tf.keras.datasets.reuters.load_data(
        num_words=self.max_words)
    self.num_classes = np.max(self.y_train) + 1
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_words)
    self.x_train = tokenizer.sequences_to_matrix(self.x_train, mode='binary')
    self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
    self.epochs = 5

  def _build_model(self):
    """Model from https://github.com/keras-team/keras/blob/master/

    examples/reuters_mlp.py.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, input_shape=(self.max_words,)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(self.num_classes))
    model.add(tf.keras.layers.Activation('softmax'))
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
  def benchmark_mlp_reuters_bs_128(self):
    """Measure performance with batch_size=128."""
    batch_size = 128
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.x_train,
        y=self.y_train,
        batch_size=batch_size,
        epochs=self.epochs,
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata('mlp', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_mlp_reuters_bs_256(self):
    """Measure performance with batch_size=256."""
    batch_size = 256
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.x_train,
        y=self.y_train,
        batch_size=batch_size,
        epochs=self.epochs,
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata('mlp', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_mlp_reuters_bs_512(self):
    """Measure performance with batch_size=512."""
    batch_size = 512
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.x_train,
        y=self.y_train,
        batch_size=batch_size,
        epochs=self.epochs,
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata('mlp', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_mlp_reuters_bs_512_gpu_2(self):
    """Measure performance with batch_size=512, gpu=2 and

    distribution_strategy='mirrored'
    """
    batch_size = 512
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.x_train,
        y=self.y_train,
        batch_size=batch_size,
        num_gpus=2,
        distribution_strategy='mirrored',
        epochs=self.epochs,
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    metadata = benchmark_util.get_keras_examples_metadata('mlp', batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)


if __name__ == '__main__':
  tf.test.main()
