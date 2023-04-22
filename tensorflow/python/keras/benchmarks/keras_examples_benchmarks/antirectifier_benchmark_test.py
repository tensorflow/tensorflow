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
"""Benchmarks on Antirectifier."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.benchmarks import benchmark_util


class AntirectifierBenchmark(tf.test.Benchmark):
  """Benchmarks for Antirectifier using `tf.test.Benchmark`."""

  def __init__(self):
    super(AntirectifierBenchmark, self).__init__()
    (self.x_train, self.y_train), _ = tf.keras.datasets.mnist.load_data()
    self.x_train = self.x_train.reshape(-1, 784)
    self.x_train = self.x_train.astype("float32") / 255

  def _build_model(self):
    """Model from https://keras.io/examples/keras_recipes/antirectifier/."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(256),
        Antirectifier(),
        tf.keras.layers.Dense(256),
        Antirectifier(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10),
    ])
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
  def benchmark_antirectifier_bs_128(self):
    """Measure performance with batch_size=128."""
    batch_size = 128
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.x_train,
        y=self.y_train,
        batch_size=batch_size,
        optimizer="rmsprop",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"])

    metadata = benchmark_util.get_keras_examples_metadata(
        "antirectifier", batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_antirectifier_bs_256(self):
    """Measure performance with batch_size=256."""
    batch_size = 256
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.x_train,
        y=self.y_train,
        batch_size=batch_size,
        optimizer="rmsprop",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"])

    metadata = benchmark_util.get_keras_examples_metadata(
        "antirectifier", batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_antirectifier_bs_512(self):
    """Measure performance with batch_size=512."""
    batch_size = 512
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.x_train,
        y=self.y_train,
        batch_size=batch_size,
        optimizer="rmsprop",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"])

    metadata = benchmark_util.get_keras_examples_metadata(
        "antirectifier", batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_antirectifier_bs_512_gpu_2(self):
    """Measure performance with batch_size=512, gpu=2 and

    distribution_strategy=`mirrored`.
    """
    batch_size = 512
    metrics, wall_time, extras = benchmark_util.measure_performance(
        self._build_model,
        x=self.x_train,
        y=self.y_train,
        batch_size=batch_size,
        num_gpus=2,
        distribution_strategy="mirrored",
        optimizer="rmsprop",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"])

    metadata = benchmark_util.get_keras_examples_metadata(
        "antirectifier", batch_size)
    extras.update(metadata)
    self.report_benchmark(wall_time=wall_time, metrics=metrics, extras=extras)


class Antirectifier(tf.keras.layers.Layer):
  """Build simple custome layer."""

  def __init__(self, initializer="he_normal", **kwargs):
    super(Antirectifier, self).__init__(**kwargs)
    self.initializer = tf.keras.initializers.get(initializer)

  def build(self, input_shape):
    output_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(output_dim * 2, output_dim),
        initializer=self.initializer,
        name="kernel",
        trainable=True,
    )

  def call(self, inputs):  #pylint: disable=arguments-differ
    inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
    pos = tf.nn.relu(inputs)
    neg = tf.nn.relu(-inputs)
    concatenated = tf.concat([pos, neg], axis=-1)
    mixed = tf.matmul(concatenated, self.kernel)
    return mixed

  def get_config(self):
    # Implement get_config to enable serialization. This is optional.
    base_config = super(Antirectifier, self).get_config()
    config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
    return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
  tf.test.main()
