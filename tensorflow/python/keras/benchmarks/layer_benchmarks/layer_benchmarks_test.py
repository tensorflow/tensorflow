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
"""Benchmarks on Keras layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

import tensorflow as tf
from tensorflow.python.keras.benchmarks.layer_benchmarks import layer_benchmarks_test_base
from tensorflow.python.platform import benchmark


def _layer_call_backward(layer, x):
  with tf.GradientTape() as tape:
    y = layer(x)
    loss = tf.reduce_mean(y**2)

  _ = tape.gradient(loss, layer.trainable_variables)


class KerasLayerBenchmarks(six.with_metaclass(
    benchmark.ParameterizedBenchmark,
    layer_benchmarks_test_base.LayerBenchmarksBase)):

  _benchmark_parameters = [
      ("Conv2D_small_shape", tf.keras.layers.Conv2D,
       {"filters": 1, "kernel_size": 1, "activation": "relu"},
       (1, 1, 1, 1), 10000),
      ("Conv2D_normal_shape", tf.keras.layers.Conv2D,
       {"filters": 1, "kernel_size": 1, "activation": "relu"},
       (64, 28, 28, 3), 10000),
      ("LSTM_small_shape", tf.keras.layers.LSTM,
       {"units": 1}, (1, 1, 1), 10000),
      ("LSTM_normal_shape", tf.keras.layers.LSTM,
       {"units": 4}, (32, 10, 8), 10000),
  ]

  def benchmark_layer_call(self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)

    fn = functools.partial(layer, x)
    self.run_report(fn, num_iters)

  def benchmark_layer_call_with_function(
      self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)
    layer.call = tf.function(layer.call)

    fn = functools.partial(layer, x)
    self.run_report(fn, num_iters)

  def benchmark_layer_call_with_xla(
      self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)
    layer.call = tf.function(
        layer.call, experimental_compile=True)

    fn = functools.partial(layer, x)
    self.run_report(fn, num_iters)

  def benchmark_layer_call_backward(
      self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)

    fn = functools.partial(_layer_call_backward, layer, x)
    self.run_report(fn, num_iters)

  def benchmark_layer_call_backward_with_function(
      self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)
    layer.call = tf.function(layer.call)

    fn = functools.partial(_layer_call_backward, layer, x)
    self.run_report(fn, num_iters)


class KerasLayerBenchmarksBackwardXLA(six.with_metaclass(
    benchmark.ParameterizedBenchmark,
    layer_benchmarks_test_base.LayerBenchmarksBase)):

  _benchmark_parameters = [
      ("Conv2D_small_shape", tf.keras.layers.Conv2D,
       {"filters": 1, "kernel_size": 1, "activation": "relu"},
       (1, 1, 1, 1), 10000),
      ("Conv2D_normal_shape", tf.keras.layers.Conv2D,
       {"filters": 1, "kernel_size": 1, "activation": "relu"},
       (64, 28, 28, 3), 10000),
      # TODO(b/153480400)
      # ("LSTM_small_shape", tf.keras.layers.LSTM,
      #  {"units": 1}, (1, 1, 1), 10000),
      # ("LSTM_normal_shape", tf.keras.layers.LSTM,
      #  {"units": 4}, (32, 10, 8), 10000),
  ]

  def benchmark_layer_call_backward_with_xla(
      self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)
    layer.call = tf.function(
        layer.call, experimental_compile=True)

    fn = functools.partial(_layer_call_backward, layer, x)
    self.run_report(fn, num_iters)


if __name__ == "__main__":
  tf.test.main()
